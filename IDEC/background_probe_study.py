"""Fit matched IDEC encoders and save their responses to a fixed-site brightness change."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from threadpoolctl import threadpool_limits

from adaptive_saliency_study import MAPS, combined_weights
from clean_neighborhood_study import save_predictions
from external_study import manifest
from gaussian_refinement_study import fit, spatial_weights
from geometry_study import SCALE, SIZE, placements
from idec_final.data import FILES, load_mnist
from idec_final.experiment import assignments
from idec_final.utils import ROOT, predict, sha256
from varied_backgrounds import render, template_bank


DOMAINS = ("varied", "clean")
ARMS = ("uniform", "adaptive")
CALIBRATION = 4000
SOURCES = ("background_probe/design.md", "background_probe_study.py", "adaptive_saliency_study.py",
           "gaussian_refinement_study.py", "dec_background_study.py", "geometry_study.py",
           "varied_backgrounds.py", "clean_neighborhood_study.py", "external_study.py",
           "idec_final/data.py", "idec_final/experiment.py", "idec_final/model.py",
           "idec_final/utils.py", "protocol.json")


def brightness_pair(clean):
    bank, _, attributes = template_bank()
    boxes = placements(clean, 8101)
    # The last attribute varies fastest in the released template bank.
    base = 3 * np.random.default_rng(8102).integers(len(bank) // 3, size=(len(clean), 2))
    np.testing.assert_array_equal(attributes[base, :3], attributes[base + 2, :3])
    np.testing.assert_allclose(attributes[base, 3], 0.6)
    np.testing.assert_allclose(attributes[base + 2, 3], 1.)
    dim = render(clean, boxes, base, bank)
    bright = render(clean, boxes, base + 2, bank)
    guard = maximum_filter(clean.reshape(-1, 28, 28) > 0, size=(1, 3, 3),
                           mode="constant").reshape(clean.shape)
    np.testing.assert_array_equal(clean[guard], bright[guard])
    np.testing.assert_array_equal(clean[guard], dim[guard])
    np.testing.assert_allclose(dim - clean, 0.6 * (bright - clean), atol=1e-7)
    gap = (bright - dim).sum(1, dtype=np.float64)
    assert np.all(gap > 0)
    return dim, bright, boxes, base, gap


def eligible(clean):
    blocked = maximum_filter(clean.reshape(-1, 28, 28) > 0,
                             size=(1, SIZE + 2, SIZE + 2), mode="constant")
    valid = []
    for mask in ~blocked[:, 2:-2, 2:-2]:
        sites = np.argwhere(mask)
        valid.append(len(sites) > 1 and np.ptp(sites, axis=0).max() >= SIZE)
    return np.asarray(valid)


def prepare(cfg):
    pool, labels, _, _ = load_mnist()
    order = np.random.default_rng(cfg["split_seed"]).permutation(len(pool))
    fit_ids, probe_ids = order[:cfg["train_size"]], order[32000:40000].copy()
    missing = np.flatnonzero(~eligible(pool[probe_ids]))
    excluded = probe_ids[missing].copy()
    offset = 40000
    pending = missing.copy()
    while len(pending):
        candidates = order[offset:offset + len(pending)]
        if not len(candidates):
            raise ValueError("Insufficient eligible reserve digits")
        selected = candidates[eligible(pool[candidates])]
        probe_ids[pending[:len(selected)]] = selected
        pending, offset = pending[len(selected):], offset + len(candidates)
    clean = pool[fit_ids]
    with np.load(ROOT / "varied/results/evaluation.npz") as e:
        np.testing.assert_array_equal(fit_ids, e["train_ids"])
        np.testing.assert_array_equal(labels[fit_ids], e["train_digit"])
        bank, _, _ = template_bank()
        np.testing.assert_array_equal(bank, e["bank"])
        marked = render(clean, e["train_boxes"], e["train_types"][0], bank)
    unmarked = pool[probe_ids]
    dim, bright, boxes, base, gap = brightness_pair(unmarked)
    metadata = dict(fit_ids=fit_ids, fit_digit=labels[fit_ids].astype(np.uint8), probe_ids=probe_ids,
                    probe_digit=labels[probe_ids].astype(np.uint8), boxes=boxes.astype(np.uint8),
                    dim_types=base.astype(np.uint8), pixel_intensity_gap=gap,
                    replaced_slots=missing, excluded_ids=excluded)
    return {"clean": clean, "varied": marked}, {"clean": unmarked, "dim": dim, "bright": bright}, metadata


def run(output, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    manifest(output, cfg, device, SOURCES, dataset="mnist_background_brightness",
             arms=list(ARMS), domains=list(DOMAINS), calibration=CALIBRATION, file_md5=FILES,
             split_sha256=sha256(ROOT / "varied/results/evaluation.npz"),
             map_sha256={d: sha256(MAPS / f"{d}_maps.npz") for d in DOMAINS})
    training, views, meta = prepare(cfg)
    save_predictions(output / "evaluation.npz", **meta)
    views = {key: torch.from_numpy(value).to(device) * SCALE for key, value in views.items()}
    for domain in DOMAINS:
        train = torch.from_numpy(training[domain]).to(device) * SCALE
        with np.load(MAPS / f"{domain}_maps.npz") as saved:
            sal = torch.from_numpy(saved["saliency"]).to(device)
        for arm in ARMS:
            weights = (torch.ones_like(train) if arm == "uniform"
                       else combined_weights(spatial_weights(train, "centroid"), sal))
            for seed in cfg["seeds"]:
                path = output / f"{domain}_{arm}_{seed}.npz"
                if path.exists():
                    continue
                model, centers = fit(train, weights, cfg, seed)
                fit_cluster = assignments(model, centers, train).astype(np.uint8)
                codes = {k: predict(model.encoder, v).cpu().numpy() for k, v in views.items()}
                save_predictions(path, centers=centers.detach().cpu().numpy(), fit_cluster=fit_cluster, **codes)
                print(f"{domain} {arm} seed {seed} saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/background_probe")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
