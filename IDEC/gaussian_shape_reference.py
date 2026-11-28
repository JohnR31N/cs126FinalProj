"""Fit ordinary IDEC on the split and image pairs used for the Gaussian shape comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import save_predictions
from external_study import manifest
from gaussian_refinement_study import fit
from gaussian_shape_study import SOURCES, prepare, read_pair
from geometry_study import SCALE
from idec_final.experiment import assignments
from idec_final.utils import ROOT, sha256


REFERENCE_SOURCES = (*SOURCES, "gaussian_shape/reference_design.md", "gaussian_shape_reference.py")


def run(output, evaluation_path, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    train, pairs, metadata = prepare(cfg)
    with np.load(evaluation_path) as saved:
        assert set(metadata) == set(saved.files)
        for key, value in metadata.items():
            np.testing.assert_array_equal(saved[key], value)
    manifest(output, cfg, device, REFERENCE_SOURCES, dataset="mnist_gaussian_shape_reference",
             arms=["uniform"], evaluation_sha256=sha256(evaluation_path))
    train = torch.from_numpy(train).to(device) * SCALE
    pairs = {name: tuple(torch.from_numpy(x).to(device)*SCALE for x in pair) for name, pair in pairs.items()}
    weights = torch.ones_like(train)
    for seed in cfg["seeds"]:
        path = output / f"uniform_{seed}.npz"
        if path.exists():
            continue
        model, centers = fit(train, weights, cfg, seed)
        model.eval()
        pred = {"train": assignments(model, centers, train).astype(np.uint8)}
        for name, pair in pairs.items():
            pred.update({f"{name}_{key}": v for key, v in read_pair(model, centers, *pair).items()})
        save_predictions(path, **pred)
        print(f"Completed uniform, seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/shape_reference")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "gaussian_shape/results/evaluation.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, args.evaluation, torch.device(args.device))
