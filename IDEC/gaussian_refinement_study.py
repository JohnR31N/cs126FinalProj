"""Compare a uniform floor and an image-dependent center in Gaussian IDEC pretraining."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import save_predictions
from dec_background_study import DOMAINS, fit_clusters, store_assignments
from external_study import manifest
from geometry_study import SCALE, gaussian_weights
from idec_final.model import Autoencoder
from idec_final.utils import ROOT, batches, predict, save_json, seed_all
from varied_backgrounds import dataset


ARMS = ("floor", "centroid")
SOURCES = ("gaussian_refinement/design.md", "gaussian_refinement_study.py", "protocol.json",
           "dec_background_study.py", "geometry_study.py", "varied_backgrounds.py", "external_study.py",
           "idec_final/model.py", "idec_final/utils.py", "idec_final/experiment.py")


def spatial_weights(images, arm):
    fixed = gaussian_weights(images.device)
    if arm == "floor":
        return ((1 + fixed) / 2).expand_as(images)
    if arm != "centroid":
        raise ValueError(arm)
    axis = torch.arange(28, device=images.device, dtype=images.dtype)
    rows, cols = torch.meshgrid(axis, axis, indexing="ij")
    xy = torch.stack((rows.flatten(), cols.flatten()), dim=1)
    mass = images * fixed
    total = mass.sum(1, keepdim=True)
    center = torch.where(total > 0, (mass @ xy) / total.clamp_min(1e-12), 13.5)
    squared = (xy[None] - center[:, None]).square().sum(2)
    weights = torch.exp(-squared / (2 * 7**2))
    return weights / weights.mean(1, keepdim=True)


def fit(train, weights, cfg, seed):
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(train.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for _ in range(cfg["pretrain_epochs"]):
        for ids in batches(len(train), cfg["batch_size"], train.device):
            observed = train[ids]
            loss = ((model(observed) - observed).square() * weights[ids]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    codes = predict(model.encoder, train).cpu().numpy()
    km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed).fit(codes)
    centers = torch.from_numpy(km.cluster_centers_).to(train.device)
    centers, _ = fit_clusters(model, centers, train, "uniform", cfg)
    return model, centers


def run(out, device):
    cfg, data, meta = dataset()
    with np.load(ROOT / "varied/results/evaluation.npz") as original:
        assert set(original.files) == set(meta)
        for name in meta:
            np.testing.assert_array_equal(meta[name], original[name])
    manifest(out, cfg, device, SOURCES, dataset="marked_mnist", arms=list(ARMS))
    test = torch.from_numpy(data["test"]["clean"]).to(device) * SCALE
    views = torch.from_numpy(data["test"]["varied"]).to(device) * SCALE
    handwriting = torch.from_numpy(data["train"]["clean"]).to(device) * SCALE
    marked = torch.from_numpy(data["train"]["varied"][0]).to(device) * SCALE
    audit = {}
    for domain in DOMAINS:
        train = handwriting if domain == "clean" else marked
        for arm in ARMS:
            weights = spatial_weights(train, arm)
            torch.testing.assert_close(weights.mean(1), torch.ones_like(train[:, 0]), atol=1e-6, rtol=0)
            if domain == "varied":
                audit[arm] = {"ink": weights[handwriting > 0].mean().item(),
                              "symbol": weights[marked != handwriting].mean().item()}
            for seed in cfg["seeds"]:
                path = out / f"{domain}_{arm}_{seed}.npz"
                if path.exists():
                    continue
                model, centers = fit(train, weights, cfg, seed)
                saved = {}
                store_assignments(saved, arm, model, centers, train, test, views)
                save_predictions(path, **saved)
                print(domain, arm, seed, flush=True)
    save_json(out / "weights.json", audit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/gaussian_refinement")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
