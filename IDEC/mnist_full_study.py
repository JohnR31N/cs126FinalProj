"""Cross initialization and reconstruction target on all 70,000 clean MNIST images."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import SCALE, save_predictions
from cluster_feature_selection import stroke_features
from external_study import SOURCES, fit_or_load_selector, manifest, select_target
from idec_final.data import FILES, load_mnist
from idec_final.experiment import assignments
from idec_final.utils import ROOT, predict, seed_all
from reproduce_paper import DIGEST, initialized_model
from varied_study import finetune, pretrain


def run(out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    cfg.update(seeds=[0], train_size=70000, updates=20000)
    sources = (*SOURCES, "mnist_full_study.py", "reproduce_paper.py", "idec_final/data.py")
    manifest(out, cfg, device, sources, dataset="mnist_full", file_md5=FILES, author_weight_sha256=DIGEST)
    train, train_y, test, test_y = load_mnist()
    images = np.concatenate((train, test))
    save_predictions(out / "evaluation.npz", labels=np.r_[train_y, test_y].astype(np.uint8))
    features = stroke_features(images, device)
    graph_order = np.random.default_rng(cfg["split_seed"]).permutation(len(images))
    selector = fit_or_load_selector(out / "selector.npz", features[graph_order], device)
    print(f"Full MNIST: {selector['agreed'].sum()}/{len(images)} training inputs agree", flush=True)
    x = torch.from_numpy(images).to(device) * SCALE
    targets = {"pixels": x, "selected": select_target(images, features, selector, device)}
    seed_all(0)
    author, _ = initialized_model(device, images[:8] * SCALE)
    codes = predict(author.encoder, x).cpu().numpy()
    km = KMeans(n_clusters=10, n_init=20, random_state=0).fit(codes)
    shared_centers = torch.from_numpy(km.cluster_centers_).to(device)
    for recipe in ("author", "scratch"):
        for name, target in targets.items():
            path = out / f"{recipe}_{name}.npz"
            if path.exists():
                continue
            if recipe == "author":
                model, centers = copy.deepcopy(author), shared_centers
            else:
                model, centers = pretrain(x, target, "none", cfg, 0)
            initial = assignments(model, centers, x).astype(np.uint8)
            centers, _ = finetune(model, centers, x, target, cfg)
            save_predictions(path, initial=initial, final=assignments(model, centers, x).astype(np.uint8))
            print(f"Full MNIST: {recipe} / {name} finished", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "mnist_full")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
