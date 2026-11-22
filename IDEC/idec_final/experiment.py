"""One shared initial model per condition, then three paired fine-tuning runs."""
import argparse
import copy
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from .data import load_mnist
from .metrics import evaluate
from .model import Autoencoder, student_q, target_distribution
from .stripes import observe, orientations, projected_mse, templates
from .utils import ROOT, batches, predict, save_json, seed_all, sha256, sync


def pretrain(train, cfg, seed, device):
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for _ in range(cfg["pretrain_epochs"]):
        for ids in batches(len(train), cfg["batch_size"], device):
            batch = train[ids]
            loss = (model(batch) - batch).square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    z = predict(model.encoder, train).cpu().numpy()
    km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed).fit(z)
    return model, torch.tensor(km.cluster_centers_, device=device)


@torch.no_grad()
def assignments(model, centers, x):
    return student_q(predict(model.encoder, x), centers).argmax(1).cpu().numpy()


def finetune(model, initial_centers, train, direction, method, cfg):
    centers = torch.nn.Parameter(initial_centers.clone())
    optimizer = torch.optim.Adam([*model.parameters(), centers], lr=cfg["lr"])
    size = cfg["batch_size"]
    num_batches = (len(train) + size - 1) // size
    history, previous = [], None
    for step in range(cfg["updates"] + 1):
        if step % cfg["target_interval"] == 0 or step == cfg["updates"]:
            with torch.no_grad():
                q = student_q(predict(model.encoder, train), centers)
                target = target_distribution(q)
                labels = q.argmax(1)
                delta = None if previous is None else (labels != previous).float().mean().item()
                history.append({"step": step, "assignment_change": delta,
                                "occupied": labels.unique().numel()})
                previous = labels
        if step == cfg["updates"]:
            break
        start = (step % num_batches) * size
        batch, goal = train[start:start + size], target[start:start + size]
        model.train()
        z = model.encoder(batch)
        q = student_q(z, centers)
        loss = cfg["gamma"] * torch.nn.functional.kl_div(q.clamp_min(1e-12).log(), goal, reduction="batchmean")
        if method != "cluster_only":
            residual = model.decoder(z) - batch
            loss = loss + (projected_mse(residual, direction) if method == "projected" else residual.square().mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return centers.detach(), history


def run(out, device, cfg):
    out.mkdir(parents=True, exist_ok=True)
    train_x, y, test_x, test_y = load_mnist()
    ids = np.random.default_rng(cfg["split_seed"]).permutation(len(y))[:cfg["train_size"]]
    ot = orientations(len(ids), cfg["train_orientation_seed"])
    ov = orientations(len(test_y), cfg["test_orientation_seed"])
    clean = torch.from_numpy(train_x[ids]).to(device)
    clean_test = torch.from_numpy(test_x).to(device)
    t = torch.from_numpy(ot).to(device)
    v = torch.from_numpy(ov).to(device)
    patterns, direction = templates(device)
    signatures = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py")) if p.name != "analysis.py"}
    manifest = {"protocol":cfg, "source_sha256":signatures, "torch":str(torch.__version__),
                "device":str(device), "hardware":torch.cuda.get_device_name() if device.type=="cuda" else platform.processor(),
                "versions": {"numpy":np.__version__, "python":platform.python_version()}}
    path = out / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Output belongs to a different implementation or environment; choose a fresh --output.")
    save_json(path, manifest)
    np.savez_compressed(out / "evaluation.npz", train_ids=ids, train_digit=y[ids], test_digit=test_y,
                        train_orientation=ot, test_orientation=ov)
    for strength in cfg["strengths"]:
        train = observe(clean, t, strength, patterns)
        test = observe(clean_test, v, strength, patterns)
        flipped = observe(clean_test, 1-v, strength, patterns)
        for seed in cfg["seeds"]:
            folder = out / f"strength_{strength:.1f}" / f"seed_{seed}"
            if (folder / "scores.json").exists():
                continue
            folder.mkdir(parents=True, exist_ok=True)
            sync()
            start = time.perf_counter()
            initial, centers = pretrain(train, cfg, seed, device)
            sync()
            init_seconds = time.perf_counter() - start
            records, saved, traces = {}, {}, {}
            for method in ["initial", *cfg["methods"]]:
                model = copy.deepcopy(initial)
                start = time.perf_counter()
                if method == "initial":
                    final_centers, trace = centers, []
                else:
                    final_centers, trace = finetune(model, centers, train, direction, method, cfg)
                sync()
                seconds = init_seconds if method == "initial" else time.perf_counter()-start
                predictions = [assignments(model, final_centers, data) for data in (train, test, flipped)]
                for name, values in zip(("train","test","flipped"), predictions):
                    saved[f"{method}_{name}"] = values.astype(np.uint8)
                records[method] = dict(**evaluate(y[ids], predictions[0], test_y, *predictions[1:], ov),
                                       seconds=seconds)
                traces[method] = trace
                print(f"strength={strength} seed={seed} {method}: test ACC={records[method]['digit_acc']:.4f}", flush=True)
            np.savez_compressed(folder / "predictions.npz", **saved)
            save_json(folder / "training.json", traces)
            save_json(folder / "scores.tmp", records)
            os.replace(folder / "scores.tmp", folder / "scores.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device), json.loads((ROOT/"protocol.json").read_text()))


if __name__ == "__main__":
    main()
