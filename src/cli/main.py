import argparse
import importlib
import json
import platform
import subprocess
import sys

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from src.data.mnist import load_mnist, split_indices
from src.utils.common import ROOT, event, save_json, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["download", "run", "check", "report"])
    p.add_argument("--method", choices=["all", "cae", "idec", "svdd"], default="all")
    p.add_argument("--config", default="configs/pilot.json")
    p.add_argument("--seeds", nargs="+", type=int)
    p.add_argument("--output", default="results/pilot")
    args = p.parse_args()
    if args.command == "check":
        return subprocess.call([sys.executable, "-m", "pytest", "tests", "-q"], cwd=ROOT)
    if args.command == "report":
        from src.eval.reporting import build_report
        with threadpool_limits(limits=4):
            build_report(ROOT / args.output)
        return 0
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if args.seeds:
        cfg["seeds"] = args.seeds
    arrays = load_mnist()
    event(f"MNIST verified train={len(arrays[0])} test={len(arrays[2])}")
    if args.command == "download":
        return 0
    assert cfg["device"] != "cuda" or torch.cuda.is_available()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        existing = json.loads((out / "config.json").read_text(encoding="utf-8"))
        comparable = lambda d: {k: v for k, v in d.items() if k != "seeds"}
        if comparable(existing) != comparable(cfg):
            raise ValueError("Output directory has a different protocol; select a new --output directory")
    train_ids, val_ids = split_indices(cfg)
    np.savez(out / "split.npz", train=train_ids, validation=val_ids)
    if not (out / "config.json").exists():
        save_json(out / "config.json", cfg)
    sources = {str(f.relative_to(ROOT)): sha256(f) for f in (ROOT/"src").rglob("*.py")}
    environment_path = out / f"environment_{args.method}.json"
    if not environment_path.exists():
        save_json(environment_path, dict(python=platform.python_version(),
              torch=torch.__version__, numpy=np.__version__, cuda=torch.version.cuda,
              gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
              source_hashes=sources))
    methods = ["cae", "idec", "svdd"] if args.method == "all" else [args.method]
    with threadpool_limits(limits=4):
        for method in methods:
            module = importlib.import_module("src.method." + method)
            for seed in cfg["seeds"]:
                module.run(cfg, arrays, train_ids, val_ids, seed, out/method)
    return 0
