"""Run label-free feature selection and feature-reconstruction IDEC on the original MNIST split."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import scipy
import sklearn
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import SCALE, normalized_digest, save_predictions
from cluster_feature_selection import (FEATURE_NAMES, coordinates, fit_selector, graph_predictions,
                                       reconstruction_target, stroke_features, target_embedding)
from dec_background_study import fit_clusters
from idec_final.data import FILES, load_mnist
from idec_final.experiment import assignments
from idec_final.utils import ROOT, save_json
from varied_study import finetune, pretrain


ARMS = ("pixels", *FEATURE_NAMES)
DEC_FORKS = ("pixels", "selected")
SPLITS = ("train", "test", "extra")
SOURCES = ("cluster_feature_selection.py", "cluster_feature_study.py", "cluster_features/design.md",
           "clean_neighborhood_study.py", "varied_study.py", "dec_background_study.py",
           "idec_final/data.py", "idec_final/experiment.py", "idec_final/model.py", "idec_final/utils.py")


def original_splits(cfg):
    original, original_digit, test, test_digit = load_mnist()
    order = np.random.default_rng(cfg["split_seed"]).permutation(len(original))
    train_ids, extra_ids = np.split(order, [cfg["train_size"]])
    images = {"train": original[train_ids], "test": test, "extra": original[extra_ids]}
    evaluation = {"train_ids": train_ids, "extra_ids": extra_ids,
                  "train_digit": original_digit[train_ids], "test_digit": test_digit,
                  "extra_digit": original_digit[extra_ids]}
    return images, evaluation


def frozen_assignments(features, selector, device, cfg):
    """Fit descriptor-only K-means on training and transfer both graph partitions by voting."""
    train = features["train"]
    half = len(train) // 2
    saved = {"graph_train": np.concatenate((selector["second"][:half], selector["first"][half:]))}
    for split in ("test", "extra"):
        saved[f"graph_{split}"] = graph_predictions(train, selector, features[split], device)
    for name in FEATURE_NAMES:
        measured = coordinates(train, selector, name)
        km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=126).fit(measured)
        saved[f"{name}_train"] = km.labels_.astype(np.uint8)
        for split in ("test", "extra"):
            coords = coordinates(features[split], selector, name)
            saved[f"{name}_{split}"] = km.predict(coords).astype(np.uint8)
    return saved


def model_assignments(model, centers, tensors, method):
    return {f"{method}_{s}": assignments(model, centers, tensors[s]).astype(np.uint8) for s in SPLITS}


def run(out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"arms": list(ARMS), "dec_forks": list(DEC_FORKS), "seeds": cfg["seeds"],
                "protocol": {k: cfg[k] for k in ("split_seed", "train_size", "dims", "clusters", "pretrain_epochs",
                                                 "batch_size", "lr", "gamma", "target_interval", "updates", "kmeans_restarts")},
                "source_sha256_lf": {k: normalized_digest(ROOT / k) for k in SOURCES}, "mnist_md5": FILES,
                "versions": {"torch": str(torch.__version__), "numpy": np.__version__, "scipy": scipy.__version__,
                             "sklearn": sklearn.__version__},
                "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    path = out / "manifest.json"
    if any(out.iterdir()) and (not path.exists() or json.loads(path.read_text()) != manifest):
        raise ValueError(f"Existing files are from a different protocol or environment: {out}")
    save_json(path, manifest)
    images, evaluation = original_splits(cfg)
    save_predictions(out / "evaluation.npz", **evaluation)
    features = {s: stroke_features(images[s], device) for s in SPLITS}
    path = out / "selector.npz"
    if path.exists():
        with np.load(path) as stored:
            selector = {k: stored[k] for k in stored.files}
    else:
        selector = fit_selector(features["train"], device)
        save_predictions(path, **selector)
    print(f"Agreement on {selector['agreed'].sum()} / {len(selector['agreed'])} training images", flush=True)
    if not (out / "frozen.npz").exists():
        frozen = frozen_assignments(features, selector, device, cfg)
        save_predictions(out / "frozen.npz", **frozen)
    tensors = {s: torch.from_numpy(images[s]).to(device) * SCALE for s in SPLITS}
    train = tensors["train"]
    target_map = target_embedding()
    pixel_rms = np.sqrt(np.square(images["train"]).mean(dtype=np.float64)) * SCALE
    targets = {"pixels": train}
    for name in FEATURE_NAMES:
        measured = coordinates(features["train"], selector, name)
        dst = reconstruction_target(measured, target_map, pixel_rms)
        np.testing.assert_allclose(np.square(dst).mean(dtype=np.float64), pixel_rms**2, rtol=3e-6)
        targets[name] = torch.from_numpy(dst).to(device)
    for seed in cfg["seeds"]:
        for name, target in targets.items():
            path = out / f"{name}_{seed}.npz"
            if path.exists():
                continue
            model, centers = pretrain(train, target, "none", cfg, seed)
            saved = model_assignments(model, centers, tensors, "initial")
            if name in DEC_FORKS:
                dec = copy.deepcopy(model)
                dec_centers, _ = fit_clusters(dec, centers, train, "kl", cfg)
                saved.update(model_assignments(dec, dec_centers, tensors, "dec"))
            idec_centers, _ = finetune(model, centers, train, target, cfg)
            saved.update(model_assignments(model, idec_centers, tensors, "idec"))
            save_predictions(path, **saved)
            print(f"Completed {name}, seed {seed}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "cluster_features")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
