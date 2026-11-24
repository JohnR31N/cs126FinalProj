"""Evaluate hard selection of specific location/orientation bins from two unlabeled training halves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import SCALE, normalized_digest, save_predictions
from cluster_feature_selection import reconstruction_target, stroke_features, target_embedding
from cluster_feature_study import SPLITS, model_assignments, original_splits
from idec_final.utils import ROOT, save_json, sha256
from varied_study import finetune, pretrain


CHOICES = ("top", "bottom", "location_matched", "random")
TRAINED = CHOICES[:3]


def variance_shares(x, groups):
    x = x.astype(np.float64)
    mean = x.mean(0)
    total = np.square(x - mean).sum(0)
    between = np.zeros_like(total)
    for group in np.unique(groups):
        members = x[groups == group]
        between += len(members) * np.square(members.mean(0) - mean)
    return np.divide(between, total, out=np.zeros_like(total), where=total > 1e-12)


def select_bins(x, selector):
    half = len(x) // 2
    shares = []
    for ids, labels in ((np.arange(half), selector["second"]),
                        (np.arange(half, len(x)), selector["first"])):
        ids = ids[selector["agreed"][ids]]
        shares.append(variance_shares(x[ids], labels[ids]))
    score = np.minimum(*shares)
    take = x.shape[1] // 4
    ranked = np.lexsort((np.arange(len(score)), -score))
    top = np.sort(ranked[:take])
    bottom = np.sort(ranked[-take:])
    counts = np.bincount(top // 8, minlength=49)
    rng = np.random.default_rng(17126)
    matched = np.sort(np.concatenate([8*c + rng.permutation(8)[:n] for c, n in enumerate(counts)]))
    random = np.sort(np.random.default_rng(18126).permutation(len(score))[:take])
    return {"half_a": shares[0], "half_b": shares[1], "score": score, "top": top, "bottom": bottom,
            "location_matched": matched, "random": random}


def save_table(path, selection):
    flags = {n: set(selection[n].tolist()) for n in CHOICES}
    with path.open("w", encoding="utf-8", newline="") as file:
        fields = ("id", "cell_row", "cell_col", "gradient_degrees", "half_a", "half_b", "repeatable_share", *CHOICES)
        writer = csv.writer(file)
        writer.writerow(fields)
        for i, score in enumerate(selection["score"]):
            writer.writerow((i, i//56, i//8 % 7, (i % 8)*22.5, selection["half_a"][i],
                             selection["half_b"][i], score, *(int(i in flags[n]) for n in CHOICES)))


def run(parent, out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"seeds": cfg["seeds"], "trained": list(TRAINED), "frozen": list(CHOICES),
                "parent_selector_sha256": sha256(parent / "selector.npz"),
                "parent_evaluation_sha256": sha256(parent / "evaluation.npz"),
                "source_sha256_lf": {n: normalized_digest(ROOT/n) for n in (
                    "cluster_coordinate_study.py", "cluster_features/coordinate_design.md", "cluster_feature_selection.py",
                    "cluster_feature_study.py", "varied_study.py")}, "torch": str(torch.__version__), "device": str(device)}
    mpath = out / "manifest.json"
    if any(out.iterdir()) and (not mpath.exists() or json.loads(mpath.read_text()) != manifest):
        raise ValueError(f"Output directory belongs to a different run: {out}")
    save_json(mpath, manifest)
    images, evaluation = original_splits(cfg)
    with np.load(parent / "evaluation.npz") as saved:
        for key, array in evaluation.items():
            np.testing.assert_array_equal(array, saved[key])
    h = {s: stroke_features(images[s], device) for s in SPLITS}
    with np.load(parent / "selector.npz") as stored:
        selected = select_bins(h["train"], stored)
    save_predictions(out / "bins.npz", **selected)
    save_table(out / "bins.csv", selected)
    if not (out / "frozen.npz").exists():
        frozen = {}
        for name in CHOICES:
            ix = selected[name]
            km = KMeans(n_clusters=10, n_init=20, random_state=126).fit(np.ascontiguousarray(h["train"][:, ix]))
            for s in SPLITS:
                frozen[f"{name}_{s}"] = km.predict(np.ascontiguousarray(h[s][:, ix])).astype(np.uint8)
        save_predictions(out / "frozen.npz", **frozen)
    tensors = {s: torch.from_numpy(images[s]).to(device)*SCALE for s in SPLITS}
    basis = target_embedding()
    rms = np.sqrt(np.square(images["train"]).mean(dtype=np.float64))*SCALE
    targets = {n: torch.from_numpy(reconstruction_target(h["train"][:, selected[n]], basis, rms)).to(device)
               for n in TRAINED}
    for seed in cfg["seeds"]:
        for name, target in targets.items():
            path = out / f"{name}_{seed}.npz"
            if path.exists():
                continue
            model, centers = pretrain(tensors["train"], target, "none", cfg, seed)
            saved = model_assignments(model, centers, tensors, "initial")
            centers, _ = finetune(model, centers, tensors["train"], target, cfg)
            saved.update(model_assignments(model, centers, tensors, "idec"))
            save_predictions(path, **saved)
            print(f"Completed literal bin selection {name}, seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "cluster_features" / "results")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "coordinate_features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.parent, args.output, torch.device(args.device))
