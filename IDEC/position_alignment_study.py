"""Evaluate frozen circular Gaussian × saliency students with and without input centering."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from adaptive_saliency_study import MAPS
from clean_neighborhood_study import save_predictions
from external_study import manifest
from gaussian_refinement_study import fit
from gaussian_shape_study import SOURCES as SHAPE_SOURCES, reconstruction_weights, shifted_pairs
from geometry_study import SCALE, placements
from idec_final.data import load_mnist
from idec_final.experiment import assignments
from idec_final.utils import ROOT, sha256
from position_alignment import align, calibrate
from varied_backgrounds import render, template_bank


VIEWS = ("clean", "marked", "shifted_clean", "shifted_marked")
MODES = ("raw", "aligned")
SOURCES = (*SHAPE_SOURCES, "position_alignment/design.md", "position_alignment.py",
           "position_alignment_study.py", "spatial_augmentation.py")


def prepare(cfg):
    original, labels, _, _ = load_mnist()
    order = np.random.default_rng(cfg["split_seed"]).permutation(len(original))
    train_ids = order[:cfg["train_size"]]
    ids = order[22000:32000]
    with np.load(ROOT / "gaussian_shape/results/evaluation.npz") as prior:
        np.testing.assert_array_equal(train_ids, prior["train_ids"])
        assert np.intersect1d(ids, np.r_[prior["train_ids"], prior["holdout_ids"]]).size == 0
    x = original[ids]
    boxes = placements(x, 81129)
    bank, _, _ = template_bank()
    types = np.random.default_rng(91129).permutation(np.arange(2*len(x)) % len(bank)).reshape(-1, 2)
    marked = render(x, boxes, types, bank)
    shifted, eligible, directions = shifted_pairs(x, marked, 101129)
    # The known inverse is an evaluation-only oracle. It must recover both views exactly.
    for direction in np.unique(directions, axis=0):
        subset = (directions[eligible] == direction).all(1)
        for moved, centered in zip(shifted, (x, marked), strict=True):
            undo = np.roll(moved[subset].reshape(-1, 28, 28), -direction, axis=(1, 2))
            np.testing.assert_array_equal(undo.reshape(-1, 784), centered[eligible[subset]])
    meta = dict(train_ids=train_ids, train_y=labels[train_ids], test_ids=ids, test_y=labels[ids],
                eligible=eligible, directions=directions, symbol_boxes=boxes, symbol_types=types)
    return original[train_ids], dict(zip(VIEWS, (x, marked, *shifted), strict=True)), meta


def run(output, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    map_path = MAPS / "clean_maps.npz"
    digest = sha256(map_path)
    assert digest == json.loads((ROOT / "adaptive_saliency/results/manifest.json").read_text())["map_sha256"]["clean"]
    manifest(output, cfg, device, SOURCES, dataset="mnist_position_alignment", modes=list(MODES),
             clean_map_sha256=digest)
    train, views, meta = prepare(cfg)
    train = torch.from_numpy(train).to(device) * SCALE
    ref, threshold = calibrate(train)
    meta.update(reference_yx=ref.cpu().numpy(), threshold_px=threshold.cpu().numpy())
    inputs = {}
    for name, array in views.items():
        observed = torch.from_numpy(array).to(device) * SCALE
        corrected, offsets = align(observed, ref, threshold)
        inputs[f"raw_{name}"], inputs[f"aligned_{name}"] = observed, corrected
        meta[f"offset_{name}"] = offsets.cpu().numpy().astype(np.int8)
    path = output / "evaluation.npz"
    if path.exists():
        with np.load(path) as saved:
            assert set(meta) == set(saved.files)
            for key, value in meta.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        save_predictions(path, **meta)
    with np.load(map_path) as archive:
        saliency = torch.from_numpy(archive["saliency"]).to(device)
    w = reconstruction_weights(train, saliency, "circle")
    for seed in cfg["seeds"]:
        dest = output / f"circle_{seed}.npz"
        if dest.exists():
            continue
        model, centers = fit(train, w, cfg, seed)
        model.eval()
        pred = {"train": assignments(model, centers, train).astype(np.uint8)}
        for name, x in inputs.items():
            pred[name] = assignments(model, centers, x).astype(np.uint8)
        save_predictions(dest, **pred)
        print(f"Completed position study, seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/position_alignment")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
