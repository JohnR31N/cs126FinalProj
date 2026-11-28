"""Test whether the direction of an adaptive Gaussian helps clean-trained IDEC."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.ndimage import maximum_filter
from threadpoolctl import threadpool_limits

from adaptive_saliency_study import MAPS, combined_weights
from clean_neighborhood_study import save_predictions
from external_study import manifest
from gaussian_refinement_study import fit, spatial_weights
from geometry_study import SCALE, gaussian_weights, placements
from idec_final.data import load_mnist
from idec_final.experiment import assignments
from idec_final.model import student_q
from idec_final.utils import ROOT, sha256
from varied_backgrounds import render, template_bank


ARMS = ("fixed", "circle", "ellipse", "rotated")
SPLITS = ("holdout", "official", "shifted")
SOURCES = ("gaussian_shape/design.md", "gaussian_shape_study.py", "adaptive_saliency_study.py",
           "gaussian_refinement_study.py", "gaussian_saliency_study.py", "dec_background_study.py",
           "geometry_study.py", "varied_backgrounds.py", "idec_final/data.py", "idec_final/utils.py",
           "idec_final/model.py", "idec_final/experiment.py", "protocol.json", "external_study.py",
           "clean_neighborhood_study.py")


def elliptical_gaussian(images, rotated=False):
    """Return Gaussian weights with the released centroid and constant covariance determinant."""
    axis = torch.arange(28, dtype=images.dtype, device=images.device)
    xy = torch.cartesian_prod(axis, axis)
    mass = images * gaussian_weights(images.device)
    total = mass.sum(1, keepdim=True)
    center = torch.where(total > 0, (mass @ xy) / total.clamp_min(1e-12), 13.5)
    offset = xy[None] - center[:, None]
    density = mass / total.clamp_min(1e-12)
    cov = offset.transpose(1, 2) @ (offset * density[:, :, None])
    eigenvalues, axes = torch.linalg.eigh(cov + torch.eye(2, dtype=images.dtype, device=images.device))
    ratio = (eigenvalues[:, 1] / eigenvalues[:, 0]).pow(0.25).clamp(1, 2)
    variances = 49 * torch.stack((ratio.reciprocal(), ratio), dim=1)
    if rotated:
        variances = variances.flip(1)
    along_axes = offset @ axes
    distance = (along_axes.square() / variances[:, None]).sum(2)
    weights = torch.exp(-0.5 * distance)
    return weights / weights.mean(1, keepdim=True)


def reconstruction_weights(images, saliency, arm):
    if arm == "fixed":
        gaussian = gaussian_weights(images.device)
    elif arm == "circle":
        gaussian = spatial_weights(images, "centroid")
    elif arm in ("ellipse", "rotated"):
        gaussian = elliptical_gaussian(images, rotated=(arm == "rotated"))
    else:
        raise ValueError(f"Unknown spatial weighting: {arm}")
    return combined_weights(gaussian, saliency)


def shifted_pairs(clean, marked, seed):
    """Apply one assigned cardinal shift to both views when no nonzero pixel would be lost."""
    directions = np.array([(-3, 0), (3, 0), (0, -3), (0, 3)], dtype=np.int8)
    groups = np.random.default_rng(seed).integers(len(directions), size=len(clean))
    pair = np.stack((clean, marked), axis=1).reshape(-1, 2, 28, 28)
    keep, moved = [], []
    for group, (dy, dx) in enumerate(directions):
        idx = np.flatnonzero(groups == group)
        subset = pair[idx]
        rows = slice(max(0, -int(dy)), min(28, 28-int(dy)))
        cols = slice(max(0, -int(dx)), min(28, 28-int(dx)))
        legal = np.count_nonzero(subset[:, :, rows, cols], axis=(1, 2, 3)) == np.count_nonzero(subset, axis=(1, 2, 3))
        keep.append(idx[legal])
        moved.append(np.roll(subset[legal], (dy, dx), axis=(2, 3)))
    kept = np.concatenate(keep)
    order = np.argsort(kept)
    shifted = np.concatenate(moved)[order].reshape(-1, 2, 784)
    return (shifted[:, 0], shifted[:, 1]), kept[order], directions[groups]


def prepare(cfg):
    original, digits, test, test_y = load_mnist()
    order = np.random.default_rng(cfg["split_seed"]).permutation(len(original))
    n = cfg["train_size"]
    train_ids, holdout_ids = order[:n], order[n:n+10000]
    assert not np.intersect1d(train_ids, holdout_ids).size
    bank, _, _ = template_bank()
    with np.load(ROOT / "varied/results/evaluation.npz") as old:
        np.testing.assert_array_equal(train_ids, old["train_ids"])
        np.testing.assert_array_equal(digits[train_ids], old["train_digit"])
        np.testing.assert_array_equal(test_y, old["test_digit"])
        np.testing.assert_array_equal(bank, old["bank"])
        official_marks = render(test, old["test_boxes"], old["test_types"][0], bank)
    holdout = original[holdout_ids]
    boxes = placements(holdout, 81127)
    types = np.random.default_rng(91127).permutation(np.arange(2*len(holdout)) % len(bank)).reshape(-1, 2)
    marked = render(holdout, boxes, types, bank)
    guard = maximum_filter(holdout.reshape(-1, 28, 28) > 0, size=(1, 3, 3), mode="constant")
    assert not ((holdout != marked) & guard.reshape(-1, 784)).any()
    shifted, eligible, directions = shifted_pairs(holdout, marked, 101127)
    evaluation = dict(train_ids=train_ids, train_y=digits[train_ids], holdout_ids=holdout_ids,
                      holdout_y=digits[holdout_ids], official_y=test_y, shifted_ids=eligible,
                      shift_directions=directions, symbol_boxes=boxes, symbol_types=types)
    return original[train_ids], dict(holdout=(holdout, marked), official=(test, official_marks),
                                     shifted=shifted), evaluation


@torch.no_grad()
def read_pair(model, centers, clean, marked):
    outputs = {"clean": [], "marked": [], "response": []}
    for x, noisy in zip(clean.split(512), marked.split(512), strict=True):
        z, zn = model.encoder(x), model.encoder(noisy)
        diff = noisy - x
        response = ((model.decoder(zn) - model.decoder(z)) * diff).sum(1) / diff.square().sum(1)
        for key, code in (("clean", z), ("marked", zn)):
            outputs[key].append(student_q(code, centers).argmax(1).byte().cpu().numpy())
        outputs["response"].append(response.cpu().numpy())
    return {key: np.concatenate(parts) for key, parts in outputs.items()}


def run(output, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    maps_path = MAPS / "clean_maps.npz"
    maps_digest = sha256(maps_path)
    reference = json.loads((ROOT / "adaptive_saliency/results/manifest.json").read_text())
    if maps_digest != reference["map_sha256"]["clean"]:
        raise ValueError("Clean teacher maps differ from the released run")
    manifest(output, cfg, device, SOURCES, dataset="mnist_gaussian_shape", arms=list(ARMS),
             clean_map_sha256=maps_digest, old_evaluation_sha256=sha256(ROOT / "varied/results/evaluation.npz"))
    train, pairs, metadata = prepare(cfg)
    meta_path = output / "evaluation.npz"
    if meta_path.exists():
        with np.load(meta_path) as saved:
            assert set(saved.files) == set(metadata)
            for key, value in metadata.items():
                np.testing.assert_array_equal(saved[key], value)
    else:
        save_predictions(meta_path, **metadata)
    train = torch.from_numpy(train).to(device) * SCALE
    pairs = {name: tuple(torch.from_numpy(x).to(device) * SCALE for x in pair) for name, pair in pairs.items()}
    with np.load(maps_path) as saved:
        saliency = torch.from_numpy(saved["saliency"]).to(device)
    for arm in ARMS:
        weights = reconstruction_weights(train, saliency, arm)
        torch.testing.assert_close(weights.mean(1), torch.ones_like(train[:, 0]), atol=1e-6, rtol=0)
        for seed in cfg["seeds"]:
            dest = output / f"{arm}_{seed}.npz"
            if dest.exists():
                continue
            start = perf_counter()
            model, centers = fit(train, weights, cfg, seed)
            model.eval()
            predictions = {"train": assignments(model, centers, train).astype(np.uint8)}
            for name, pair in pairs.items():
                for key, values in read_pair(model, centers, *pair).items():
                    predictions[f"{name}_{key}"] = values
            save_predictions(dest, **predictions)
            print(f"Completed {arm}, seed {seed}, {perf_counter()-start:.0f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/gaussian_shape")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
