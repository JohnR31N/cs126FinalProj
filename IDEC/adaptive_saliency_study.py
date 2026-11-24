"""Train adaptive-center Gaussian IDEC with aligned and shuffled cluster saliency."""
import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import save_predictions
from dec_background_study import DOMAINS, store_assignments
from external_study import manifest
from gaussian_refinement_study import fit, spatial_weights
from gaussian_saliency_study import SOURCES as BASE, donor_order
from geometry_study import SCALE
from idec_final.utils import ROOT, save_json, sha256
from varied_backgrounds import dataset


ARMS = ("adaptive_aligned", "adaptive_shuffled")
SOURCES = (*BASE, "adaptive_saliency_study.py", "adaptive_saliency/design.md")
MAPS = ROOT / "gaussian_saliency/results"


def combined_weights(gaussian, saliency):
    w = gaussian * (0.5 + saliency.float() / 255)
    return w / w.mean(1, keepdim=True)


def run(out, device):
    cfg, data, metadata = dataset()
    with np.load(ROOT / "varied/results/evaluation.npz") as archived:
        for name in archived.files:
            np.testing.assert_array_equal(metadata[name], archived[name])
    manifest(out, cfg, device, SOURCES, dataset="marked_mnist", arms=list(ARMS),
             map_sha256={d: sha256(MAPS / f"{d}_maps.npz") for d in DOMAINS})
    clean = torch.from_numpy(data["train"]["clean"]).to(device) * SCALE
    marked = torch.from_numpy(data["train"]["varied"][0]).to(device) * SCALE
    test = torch.from_numpy(data["test"]["clean"]).to(device) * SCALE
    views = torch.from_numpy(data["test"]["varied"]).to(device) * SCALE
    audit = {}
    for domain in DOMAINS:
        train = clean if domain == "clean" else marked
        gaussian = spatial_weights(train, "centroid")
        with np.load(MAPS / f"{domain}_maps.npz") as f:
            np.testing.assert_array_equal(f["donor"], donor_order(len(train)))
            donor = torch.from_numpy(f["donor"]).to(device)
            maps = torch.from_numpy(f["saliency"]).to(device)
        for arm in ARMS:
            weights = combined_weights(gaussian, maps if arm == "adaptive_aligned" else maps[donor])
            torch.testing.assert_close(weights.mean(1), torch.ones_like(train[:, 0]), atol=1e-6, rtol=0)
            if domain == "varied":
                audit[arm] = {"ink": weights[clean > 0].mean().item(), "symbol": weights[clean != marked].mean().item()}
            for seed in cfg["seeds"]:
                path = out / f"{domain}_{arm}_{seed}.npz"
                if path.exists():
                    continue
                model, centers = fit(train, weights, cfg, seed)
                saved = {}
                store_assignments(saved, arm, model, centers, train, test, views)
                save_predictions(path, **saved)
                print(domain, arm, seed, flush=True)
    save_json(out / "weights.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/adaptive_saliency")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
