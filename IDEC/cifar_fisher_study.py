"""Fit supported-rank Fisher and equal-width PCA reconstruction on grayscale CIFAR-10."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy import linalg
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import SCALE, save_predictions
from cluster_feature_selection import stroke_features
from external_data import load_external
from external_study import SOURCES, manifest, predictions, select_target
from idec_final.utils import ROOT, sha256
from varied_study import finetune, pretrain


REFERENCE = ROOT / "external_benchmarks/results/cifar10_gray"
FOLLOWUP = ROOT / "external_benchmarks/cifar_fisher"
TRAIN_SOURCES = (*SOURCES, "cifar_fisher_study.py", "external_benchmarks/cifar_fisher/design.md")
ARMS = ("fisher", "pca")


def supported_bases(features, groups):
    """Return Fisher and PCA bases fitted on the same consensus observations."""
    x = features.astype(np.float64)
    group_ids, inverse, counts = np.unique(groups, return_inverse=True, return_counts=True)
    rank = len(group_ids) - 1
    if rank < 1 or rank > x.shape[1] or counts.min() < 2:
        raise ValueError("The occupied consensus groups cannot support the requested subspace.")
    overall = x.mean(axis=0)
    means = np.stack([x[inverse == k].mean(axis=0) for k in range(len(counts))])
    residual = x - means[inverse]
    within = residual.T @ residual / len(x)
    within.flat[::x.shape[1]+1] += 0.1 * np.trace(within) / x.shape[1]
    offsets = means - overall
    between = (offsets.T * (counts / len(x))) @ offsets
    values, directions = linalg.eigh(between, within, driver="gvd")
    spectrum = values[-rank:][::-1]
    if spectrum.min() <= spectrum.max() * np.finfo(x.dtype).eps * x.shape[1]:
        raise ValueError("The between-group scatter has fewer positive directions than expected.")
    fisher, _ = np.linalg.qr(directions[:, -rank:][:, ::-1])
    centered = x - overall
    _, pca = linalg.eigh(centered.T @ centered, subset_by_index=(x.shape[1]-rank, x.shape[1]-1))
    return {"fisher": fisher.astype(np.float32), "pca": pca[:, ::-1].astype(np.float32),
            "eigenvalues": spectrum, "group_ids": group_ids, "group_sizes": counts}


def reference_files(seeds):
    names = ["graph_groups.npz", "evaluation.npz", *(f"pixels_{seed}.npz" for seed in seeds)]
    return {name: sha256(REFERENCE / name) for name in names}


def run(out, cache, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    manifest(out, cfg, device, TRAIN_SOURCES, dataset="cifar10_gray", arms=list(ARMS),
             reference_commit="7ffbcc8abe7566175bef27e48db5e0ac8312d9db", reference_sha256=reference_files(cfg["seeds"]))
    images, evaluation = load_external("cifar10_gray", cache, cfg)
    with np.load(REFERENCE / "evaluation.npz") as old:
        np.testing.assert_array_equal(evaluation["train_ids"], old["train_ids"])
    tensors = {s: torch.from_numpy(x).to(device) * SCALE for s, x in images.items()}
    features = stroke_features(images["train"], device)
    with np.load(REFERENCE / "graph_groups.npz") as graph:
        first, second = graph["first"], graph["second"]
        agreed = first == second
        bases = supported_bases(features[agreed], first[agreed])
        np.testing.assert_array_equal(bases["group_sizes"], graph["agreed_group_sizes"][bases["group_ids"]])
    mean = features.mean(0, dtype=np.float64).astype(np.float32)
    save_predictions(out / "bases.npz", **bases, mean=mean)
    targets = {arm: select_target(images["train"], features, {"mean": mean, "selected": bases[arm]}, device)
               for arm in ARMS}
    print(f"Fitting {len(bases['eigenvalues'])} directions using {agreed.sum()} consensus examples.", flush=True)
    for seed in cfg["seeds"]:
        for arm in ARMS:
            path = out / f"{arm}_{seed}.npz"
            if path.exists():
                continue
            target = targets[arm]
            model, centers = pretrain(tensors["train"], target, "none", cfg, seed)
            saved = predictions("initial", model, centers, tensors)
            centers, _ = finetune(model, centers, tensors["train"], target, cfg)
            saved.update(predictions("idec", model, centers, tensors))
            save_predictions(path, **saved)
            print(f"Finished {arm}, seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/cifar_fisher")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache/external")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, args.cache, torch.device(args.device))
