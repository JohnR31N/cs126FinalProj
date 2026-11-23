"""Test whether reconstruction of random geometric backgrounds harms digit clusters."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from geometry_study import SCALE, gaussian_weights
from idec_final.experiment import assignments
from idec_final.model import Autoencoder, student_q, target_distribution
from idec_final.utils import ROOT, batches, predict, save_json, seed_all, sha256
from spatial_augmentation import Translation
from varied_backgrounds import dataset


# Domain, pretraining target, intervention, fine-tuning targets.
GROUPS = (
    ("clean", "full", "none", ("full",)),
    ("fixed", "full", "none", ("full",)),
    ("fixed", "clean", "none", ("clean",)),
    ("varied", "full", "none", ("full", "clean")),
    ("varied", "clean", "none", ("full", "clean")),
    ("varied", "full", "translate", ("full",)),
    ("varied", "full", "gaussian", ("full",)),
)
ARMS = [f"{domain}_{pre}_{fine}_{intervention}" for domain, pre, intervention, fine_targets in GROUPS for fine in fine_targets]


def pretrain(images, target, intervention, cfg, seed):
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(images.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    translation = Translation(images) if intervention == "translate" else None
    generator = torch.Generator(device=images.device).manual_seed(34000+seed)
    weights = gaussian_weights(images.device) if intervention == "gaussian" else None
    for _ in range(cfg["pretrain_epochs"]):
        observed = translation(generator) if translation else images
        goals = observed if translation else target
        for ids in batches(len(images), cfg["batch_size"], images.device):
            error = (model(observed[ids])-goals[ids]).square()
            loss = error.mean() if weights is None else (error*weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    codes = predict(model.encoder, images).cpu().numpy()
    km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed).fit(codes)
    return model, torch.from_numpy(km.cluster_centers_).to(images.device)


def finetune(model, initial_centers, images, reconstruction_target, cfg):
    centers = torch.nn.Parameter(initial_centers.clone())
    optimizer = torch.optim.Adam([*model.parameters(), centers], lr=cfg["lr"])
    size = cfg["batch_size"]
    num_batches = (len(images)+size-1)//size
    history, previous = [], None
    for step in range(cfg["updates"]+1):
        if step % cfg["target_interval"] == 0 or step == cfg["updates"]:
            with torch.no_grad():
                q = student_q(predict(model.encoder, images), centers)
                target = target_distribution(q)
                labels = q.argmax(1)
                delta = None if previous is None else (labels != previous).float().mean().item()
                history.append({"step": step, "assignment_change": delta, "occupied": labels.unique().numel()})
                previous = labels
        if step == cfg["updates"]:
            break
        start = (step % num_batches)*size
        batch, goal = images[start:start+size], target[start:start+size]
        model.train()
        z = model.encoder(batch)
        q = student_q(z, centers)
        loss = cfg["gamma"]*torch.nn.functional.kl_div(q.clamp_min(1e-12).log(), goal, reduction="batchmean")
        loss = loss+(model.decoder(z)-reconstruction_target[start:start+size]).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return centers.detach(), history


@torch.no_grad()
def readout(model, views, clean):
    recon = predict(model, views[0])/SCALE
    other = predict(model, views[1])/SCALE
    first, second, digit = views[0]/SCALE, views[1]/SCALE, clean/SCALE
    difference = first-second
    norm = difference.square().sum(1)
    response = ((recon-other)*difference).sum(1)/norm.clamp_min(1e-12)
    mask = (first-digit)>0
    energy = (first-digit).square().sum(1)
    background_error = ((recon-first).square()*mask).sum(1)/energy.clamp_min(1e-12)
    ink = digit>0
    ink_error = ((recon-digit).square()*ink).sum(1)/ink.sum(1)
    return {name: value.cpu().numpy() for name, value in {
        "response": response, "response_valid": norm > 1e-12,
        "background_error": background_error, "ink_error": ink_error}.items()}


def run(out, device):
    cfg, data, metadata = dataset()
    manifest = {"design_commit": "e1cf43f", "config": cfg, "arms": ARMS,
                "source_sha256": {name: sha256(ROOT/name) for name in
                                  ("varied/design.md", "varied_study.py", "varied_backgrounds.py",
                                   "spatial_augmentation.py", "geometry_study.py", "idec_final/model.py")},
                "torch": str(torch.__version__), "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/"run.json"
    if any(out.iterdir()) and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError("This output contains a different experiment.")
    save_json(manifest_path, manifest)
    np.savez_compressed(out/"evaluation.npz", **metadata)
    train_clean = torch.from_numpy(data["train"]["clean"]).to(device)*SCALE
    test_clean = torch.from_numpy(data["test"]["clean"]).to(device)*SCALE
    train = {domain: torch.from_numpy(data["train"][domain][0]).to(device)*SCALE for domain in ("fixed","varied")}
    test = {domain: torch.from_numpy(data["test"][domain]).to(device)*SCALE for domain in ("fixed","varied")}
    train["clean"] = train_clean
    for seed in cfg["seeds"]:
        for domain, pre, intervention, fine_targets in GROUPS:
            stems = [f"{domain}_{pre}_{fine}_{intervention}_{seed}" for fine in fine_targets]
            if all((out/f"{stem}.npz").exists() for stem in stems):
                continue
            images = train[domain]
            pre_target = images if pre == "full" else train_clean
            base_model, initial_centers = pretrain(images, pre_target, intervention, cfg, seed)
            views = test["varied" if domain == "clean" else domain]
            initial_readout = readout(base_model, views, test_clean)
            for fine, stem in zip(fine_targets, stems):
                dest = out/f"{stem}.npz"
                if dest.exists():
                    continue
                model = copy.deepcopy(base_model)
                fine_target = images if fine == "full" else train_clean
                centers, trace = finetune(model, initial_centers, images, fine_target, cfg)
                predictions = {"train": assignments(model, centers, images).astype(np.uint8),
                               "clean": assignments(model, centers, test_clean).astype(np.uint8),
                               "backgrounds": np.stack([assignments(model, centers, view).astype(np.uint8) for view in views])}
                predictions.update({"pre_"+key: value for key, value in initial_readout.items()})
                predictions.update({"final_"+key: value for key, value in readout(model, views, test_clean).items()})
                save_json(dest.with_suffix(".json"), trace)
                temp = dest.with_suffix(".tmp.npz")
                np.savez_compressed(temp, **predictions)
                temp.replace(dest)
                print(f"Finished {stem}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"scratch/varied")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
