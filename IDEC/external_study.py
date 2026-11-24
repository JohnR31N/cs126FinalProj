"""Run the frozen nine-feature IDEC experiment on official external training/test splits."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import SCALE, normalized_digest, save_predictions
from cluster_feature_selection import (coordinates, fit_selector, graph_predictions,
                                       reconstruction_target, stroke_features, target_embedding)
from dec_background_study import fit_clusters
from external_data import BASE, FILES, SIZES, load_external
from idec_final.experiment import assignments
from idec_final.utils import ROOT, save_json
from varied_study import finetune, pretrain


SOURCES = ("protocol.json", "external_benchmarks/design.md", "cluster_feature_selection.py",
           "external_study.py", "external_data.py", "clean_neighborhood_study.py",
           "dec_background_study.py", "varied_study.py", "idec_final/experiment.py",
           "idec_final/model.py", "idec_final/utils.py")


def manifest(out, cfg, device, sources, **details):
    out.mkdir(parents=True, exist_ok=True)
    desc = dict(details, config=cfg, source_sha256_lf={k: normalized_digest(ROOT / k) for k in sources},
                device=str(device), hardware=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                torch=str(torch.__version__), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
    path = out / "manifest.json"
    if any(out.iterdir()) and (not path.exists() or json.loads(path.read_text()) != desc):
        raise ValueError(f"Existing files belong to a different protocol or environment: {out}")
    save_json(path, desc)


def select_target(images, features, selector, device):
    pixel_rms = np.sqrt(np.square(images).mean(dtype=np.float64)) * SCALE
    values = coordinates(features, selector, "selected")
    target = reconstruction_target(values, target_embedding(), pixel_rms)
    np.testing.assert_allclose(np.square(target).mean(dtype=np.float64), pixel_rms**2, rtol=3e-6)
    return torch.from_numpy(target).to(device)


def fit_or_load_selector(path, features, device):
    if path.exists():
        with np.load(path) as f:
            return dict(f)
    selector = fit_selector(features, device)
    save_predictions(path, **selector)
    return selector


def predictions(name, model, centers, tensors):
    return {f"{name}_{s}": assignments(model, centers, x).astype(np.uint8) for s, x in tensors.items()}


def run(dataset, cache, out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    manifest(out, cfg, device, SOURCES, dataset=dataset, source_url=BASE[dataset], file_md5=FILES[dataset])
    images, evaluation = load_external(dataset, cache, cfg)
    save_predictions(out / "evaluation.npz", **evaluation)
    tensors = {s: torch.from_numpy(x).to(device) * SCALE for s, x in images.items()}
    train = tensors["train"]
    targets = {"pixels": train}
    features = stroke_features(images["train"], device)
    try:
        selector = fit_or_load_selector(out / "selector.npz", features, device)
    except ValueError as error:
        if str(error) not in {"Every consensus group needs at least two training examples.",
                              "The consensus partition does not support nine discriminant directions."}:
            raise
        save_json(out / "unsupported.json", {"reason": str(error)})
        print(f"{dataset}: {error}", flush=True)
    else:
        targets["selected"] = select_target(images["train"], features, selector, device)
        half = len(train) // 2
        graph_train = np.r_[selector["second"][:half], selector["first"][half:]]
        test_features = stroke_features(images["test"], device)
        graph_test = graph_predictions(features, selector, test_features, device)
        save_predictions(out / "graph.npz", train=graph_train, test=graph_test)
        print(f"{dataset}: {int(selector['agreed'].sum())}/{len(train)} training inputs agree", flush=True)
    for seed in cfg["seeds"]:
        for name, target in targets.items():
            path = out / f"{name}_{seed}.npz"
            if path.exists():
                continue
            model, centers = pretrain(train, target, "none", cfg, seed)
            saved = predictions("initial", model, centers, tensors)
            dec = copy.deepcopy(model)
            dec_centers, _ = fit_clusters(dec, centers, train, "kl", cfg)
            saved.update(predictions("dec", dec, dec_centers, tensors))
            idec_centers, _ = finetune(model, centers, train, target, cfg)
            saved.update(predictions("idec", model, idec_centers, tensors))
            save_predictions(path, **saved)
            print(f"{dataset}: {name}, seed {seed} finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=list(SIZES), default=list(SIZES))
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "external")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "external")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        for dataset in args.datasets:
            run(dataset, args.cache, args.output / dataset, torch.device(args.device))
