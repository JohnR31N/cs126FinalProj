"""Replay pretraining and measure how much the decoder reproduces a moved background mark."""
import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from geometry_study import ARMS, SCALE, data, pretrain_model, stable_target
from idec_final.utils import ROOT, predict


def run(out, device):
    cfg, (training, testing) = data()
    raw = training[-1]
    views = torch.from_numpy(raw).to(device) * SCALE
    target = torch.from_numpy(stable_target(raw[0], raw[2])).to(device) * SCALE
    first, moved = [torch.from_numpy(testing[-1][i]).to(device) * SCALE for i in (0, 2)]
    delta = first - moved
    norm = delta.square().sum(1)
    assert torch.all(norm > 0)
    values = {}
    for seed in cfg["seeds"]:
        for arm in ARMS:
            model = pretrain_model(arm, seed, views, target, cfg)
            response = predict(model, first) - predict(model, moved)
            values[f"{arm}_{seed}"] = ((response * delta).sum(1) / norm).cpu().numpy()
        print(f"Measured decoder response for seed {seed}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/geometry/decoder.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
