"""Fit matched DEC and IDEC models with and without geometric backgrounds."""
import argparse
import copy
import hashlib
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import sklearn
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from geometry_study import SCALE, gaussian_weights
from idec_final.experiment import assignments
from idec_final.model import Autoencoder, student_q, target_distribution
from idec_final.utils import ROOT, predict, save_json, seed_all, sha256
from varied_backgrounds import dataset
from varied_study import pretrain, readout


DOMAINS = ("clean", "varied")
# Each group trains one encoder and fits K-means once before forking.
GROUPS = {
    "uniform": {"dec": "kl", "idec": "uniform", "gaussian_finetune": "gaussian"},
    "gaussian": {"gaussian_pretrain": "uniform", "gaussian_both": "gaussian"},
    "random": {"kl_from_scratch": "kl"},
}
SOURCES = (
    "dec_background/design.md", "dec_background_study.py", "protocol.json",
    "varied/results/evaluation.npz", "varied_study.py", "varied_backgrounds.py",
    "geometry_study.py", "idec_final/model.py", "idec_final/experiment.py",
    "idec_final/data.py", "idec_final/utils.py",
)


def fingerprint(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def initialize(images, cfg, seed, init):
    if init != "random":
        treatment = "gaussian" if init == "gaussian" else "none"
        return pretrain(images, images, treatment, cfg, seed)
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(images.device)
    codes = predict(model.encoder, images).cpu().numpy()
    centers = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"],
                     random_state=seed).fit(codes).cluster_centers_
    return model, torch.from_numpy(centers).to(images.device)


def fit_clusters(model, initial_centers, images, kind, cfg):
    """Use the same cluster optimizer and batches with each reconstruction penalty."""
    if kind not in {"kl", "uniform", "gaussian"}:
        raise ValueError(f"Unknown fine-tuning loss {kind}")
    centers = torch.nn.Parameter(initial_centers.clone())
    optimizer = torch.optim.Adam([*model.parameters(), centers], lr=cfg["lr"])
    weights = gaussian_weights(images.device) if kind == "gaussian" else None
    size = cfg["batch_size"]
    count = (len(images) + size - 1) // size
    trace = []
    previous = None
    for step in range(cfg["updates"] + 1):
        if step % cfg["target_interval"] == 0 or step == cfg["updates"]:
            with torch.no_grad():
                q = student_q(predict(model.encoder, images), centers)
                target = target_distribution(q)
                labels = q.argmax(1)
                changes = None if previous is None else (labels != previous).float().mean().item()
                trace.append({"step": step, "assignment_change": changes,
                              "occupied": labels.unique().numel()})
                previous = labels
        if step == cfg["updates"]:
            break
        start = (step % count) * size
        batch = images[start:start + size]
        model.train()
        codes = model.encoder(batch)
        q = student_q(codes, centers)
        loss = cfg["gamma"] * torch.nn.functional.kl_div(
            q.clamp_min(1e-12).log(), target[start:start + size], reduction="batchmean")
        if kind != "kl":
            error = (model.decoder(codes) - batch).square()
            loss = loss + (error.mean() if weights is None else (error * weights).mean())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return centers.detach(), trace


def store_assignments(saved, name, model, centers, train, clean, views):
    for split, data in (("train", train), ("clean", clean)):
        saved[f"{name}_{split}"] = assignments(model, centers, data).astype(np.uint8)
    saved[f"{name}_backgrounds"] = np.stack(
        [assignments(model, centers, view).astype(np.uint8) for view in views])


def run_cell(folder, domain, seed, cfg, train, clean, views):
    saved = {}
    record = {"seed": seed, "domain": domain, "initializations": {}, "methods": {}}
    for init, methods in GROUPS.items():
        model, initial_centers = initialize(train, cfg, seed, init)
        signature = fingerprint([*model.parameters(), initial_centers])
        decoder_before = fingerprint(model.decoder.parameters())
        record["initializations"][init] = signature
        store_assignments(saved, "init_" + init, model, initial_centers, train, clean, views)
        for name, loss in methods.items():
            arm = copy.deepcopy(model)
            start = fingerprint([*arm.parameters(), initial_centers])
            assert start == signature
            begin = perf_counter()
            centers, trace = fit_clusters(arm, initial_centers, train, loss, cfg)
            store_assignments(saved, name, arm, centers, train, clean, views)
            if loss == "kl":
                assert fingerprint(arm.decoder.parameters()) == decoder_before
            else:
                diagnostics = readout(arm, views, clean)
                saved[name + "_response"] = diagnostics["response"]
                valid = diagnostics["response_valid"]
                if "valid_response" in saved:
                    np.testing.assert_array_equal(valid, saved["valid_response"])
                saved["valid_response"] = valid
            record["methods"][name] = {"init": init, "start_sha256": start, "loss": loss,
                                       "trace": trace, "seconds": perf_counter() - begin}
        print(f"Finished seed {seed}, {domain}, {init} initialization", flush=True)
    with TemporaryDirectory(dir=folder.parent) as tmp:
        cell = Path(tmp) / folder.name
        cell.mkdir()
        np.savez_compressed(cell / "predictions.npz", **saved)
        save_json(cell / "record.json", record)
        cell.rename(folder)


def check_spatial_prior(data):
    weights = gaussian_weights("cpu").numpy().astype(np.float64)
    summary = {}
    for split, images in data.items():
        clean, marked = images["clean"], images["varied"][0]
        counts = {"ink": (clean > 0).sum(axis=0), "symbol": (marked != clean).sum(axis=0)}
        summary[split] = {region: float(count @ weights / count.sum())
                          for region, count in counts.items()}
    return summary


def run(out, device):
    cfg, data, metadata = dataset()
    with np.load(ROOT / "varied/results/evaluation.npz") as archived:
        assert set(metadata) == set(archived.files)
        for key, values in metadata.items():
            np.testing.assert_array_equal(values, archived[key])
    manifest = {"design_commit": "7e753dd", "config": cfg, "groups": GROUPS, "domains": list(DOMAINS),
                "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
                "python": platform.python_version(), "torch": str(torch.__version__),
                "numpy": np.__version__, "sklearn": sklearn.__version__, "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "mean_pixel_weight": check_spatial_prior(data)}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if any(out.iterdir()) and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError("Existing results use a different protocol or environment. Choose a fresh output.")
    save_json(manifest_path, manifest)
    training_inputs = {"clean": data["train"]["clean"], "varied": data["train"]["varied"][0]}
    train = {domain: torch.from_numpy(images).to(device) * SCALE for domain, images in training_inputs.items()}
    clean = torch.from_numpy(data["test"]["clean"]).to(device) * SCALE
    views = torch.from_numpy(data["test"]["varied"]).to(device) * SCALE
    for seed in cfg["seeds"]:
        for domain in DOMAINS:
            folder = out / f"{domain}_{seed}"
            if not folder.exists():
                run_cell(folder, domain, seed, cfg, train[domain], clean, views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/dec_background")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
