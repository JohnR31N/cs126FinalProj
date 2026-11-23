"""Train matched IDEC models with detached geometric marks on MNIST backgrounds."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.ndimage import maximum_filter
from sklearn.cluster import KMeans
import torch
from threadpoolctl import threadpool_limits

from idec_final.data import load_mnist
from idec_final.experiment import assignments, finetune
from idec_final.model import Autoencoder
from idec_final.stripes import orientations
from idec_final.utils import ROOT, batches, predict, save_json, seed_all, sync

ARMS = ("fixed", "augmented", "paired", "gaussian")
SCALE = 255 / 50
SIZE = 5


def placements(clean, seed):
    """Find two disjoint 5x5 boxes, each with a one-pixel ink-free border."""
    rng = np.random.default_rng(seed)
    images = clean.reshape(-1, 28, 28)
    blocked = maximum_filter(images > 0, size=(1, SIZE + 2, SIZE + 2), mode="constant")
    legal = ~blocked[:, 2:-2, 2:-2]
    boxes = np.empty((len(clean), 2, 2), dtype=np.int64)
    for i, mask in enumerate(legal):
        sites = np.argwhere(mask)
        for first in rng.permutation(sites):
            others = sites[np.max(np.abs(sites - first), axis=1) >= SIZE]
            if len(others):
                boxes[i] = first, others[rng.integers(len(others))]
                break
        else:
            raise ValueError(f"No pair of legal sites for image {i}")
    return boxes


def render_views(clean, boxes, shape):
    row, col = np.indices((SIZE, SIZE))
    symbols = np.stack([((row == 2) | (col == 2)), ((row == col) | (row + col == 4))])
    views = []
    for location in (0, 1):
        y, x = boxes[:, location].T
        for swap in (0, 1):
            image = clean.reshape(-1, 28, 28).copy()
            i, dy, dx = np.nonzero(symbols[shape ^ swap])
            image[i, y[i] + dy, x[i] + dx] = 1
            views.append(image.reshape(-1, 784))
    return np.stack(views)


def stable_target(first, moved):
    return np.where(first == moved, first, 0)


def data():
    x, y, test, labels = load_mnist()
    cfg = json.loads((ROOT / "protocol.json").read_text())
    ids = np.random.default_rng(cfg["split_seed"]).permutation(len(x))[:cfg["train_size"]]
    splits = []
    for clean, digits, site_seed, shape_seed in (
        (x[ids], y[ids], 2701, cfg["train_orientation_seed"]),
        (test, labels, 2702, cfg["test_orientation_seed"]),
    ):
        shape = orientations(len(clean), shape_seed)
        boxes = placements(clean, site_seed)
        views = render_views(clean, boxes, shape)
        guard = maximum_filter(clean.reshape(-1, 28, 28) > 0, size=(1, 3, 3), mode="constant").reshape(-1, 784)
        for view in views:
            changed = view != clean
            assert not (guard & changed).any() and np.all(changed.sum(1) == 9)
        target = stable_target(views[0], views[2])
        np.testing.assert_array_equal(target, clean)
        splits.append((clean, digits, boxes, shape, views))
    return cfg, splits


def gaussian_weights(device):
    axis = torch.arange(28, device=device) - 13.5
    weights = torch.exp(-(axis[:, None].square() + axis.square()) / (2 * 7**2))
    return (weights / weights.mean()).flatten()


def pretrain_model(arm, seed, views, target, cfg):
    device = views.device
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    weights = gaussian_weights(device)
    schedule = torch.Generator(device=device).manual_seed(10000 + seed)
    n = views.size(1)
    for _ in range(cfg["pretrain_epochs"]):
        choice = torch.randint(4, (n,), generator=schedule, device=device)
        for ids in batches(n, cfg["batch_size"], device):
            batch = views[0, ids] if arm == "fixed" else views[choice[ids], ids]
            goal = target[ids] if arm == "paired" else batch
            error = (model(batch) - goal).square()
            loss = (error * weights).mean() if arm == "gaussian" else error.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return model


def fit(arm, seed, views, target, cfg):
    model = pretrain_model(arm, seed, views, target, cfg)
    codes = predict(model.encoder, views[0]).cpu().numpy()
    km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed).fit(codes)
    centers = torch.from_numpy(km.cluster_centers_).to(views.device)
    centers, trace = finetune(model, centers, views[0], None, "idec", cfg)
    return model, centers, trace


def run(out, device):
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose an empty directory for a new run.")
    out.mkdir(parents=True, exist_ok=True)
    cfg, (training, testing) = data()
    clean, _, _, _, raw = training
    target = torch.from_numpy(stable_target(raw[0], raw[2])).to(device) * SCALE
    views = torch.from_numpy(raw).to(device) * SCALE
    clean_test, _, _, _, test_views = testing
    eval_inputs = {"train": views[0], "clean": torch.from_numpy(clean_test).to(device) * SCALE}
    for name, idx in (("marked", 0), ("shape", 1), ("moved", 2)):
        eval_inputs[name] = torch.from_numpy(test_views[idx]).to(device) * SCALE
    save_json(out / "run.json", {"config": cfg, "arms": ARMS, "torch": str(torch.__version__),
                                 "device": str(device), "hardware": torch.cuda.get_device_name(device)
                                 if device.type == "cuda" else "CPU", "design_commit": "62b24eb"})
    for seed in cfg["seeds"]:
        for arm in ARMS:
            sync()
            start = perf_counter()
            model, centers, trace = fit(arm, seed, views, target, cfg)
            sync()
            pred = {k: assignments(model, centers, v).astype(np.uint8) for k, v in eval_inputs.items()}
            np.savez_compressed(out / f"{arm}_{seed}.npz", **pred)
            save_json(out / f"{arm}_{seed}.json", {"trace": trace, "seconds": perf_counter() - start})
            print(f"Completed seed {seed}, {arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/geometry")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
