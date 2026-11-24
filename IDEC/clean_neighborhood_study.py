"""Fit and rescore three IDEC reconstruction objectives on original MNIST images."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from idec_final.data import FILES, load_mnist
from idec_final.experiment import assignments
from idec_final.model import Autoencoder, student_q, target_distribution
from idec_final.utils import ROOT, batches, predict, save_json, seed_all
from neighborhood_weights import training_weights
from photometric_analysis import digit_mapping


ARMS = ("uniform", "neighborhood", "shuffled")
SCALE = 255 / 50


def reconstruction_loss(decoded, observed, weights, ids):
    error = (decoded - observed).square()
    if weights is not None:
        error = error * weights[ids]
    return error.mean()


def fit(train, weights, cfg, seed):
    """Match the released clean baseline's initialization, batches, scaling and update counts."""
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    size = cfg["batch_size"]
    for _ in range(cfg["pretrain_epochs"]):
        for ids in batches(len(train), size, train.device):
            observed = train[ids]
            loss = reconstruction_loss(model(observed), observed, weights, ids)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    embeddings = predict(model.encoder, train).cpu().numpy()
    km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed).fit(embeddings)
    centers = torch.nn.Parameter(torch.from_numpy(km.cluster_centers_).to(train.device))
    optimizer = torch.optim.Adam([*model.parameters(), centers], lr=cfg["lr"])
    batch_count = (len(train) + size - 1) // size
    for step in range(cfg["updates"]):
        if step % cfg["target_interval"] == 0:
            with torch.no_grad():
                target = target_distribution(student_q(predict(model.encoder, train), centers))
        start = (step % batch_count) * size
        ids = slice(start, start + size)
        observed = train[ids]
        model.train()
        code = model.encoder(observed)
        q = student_q(code, centers)
        cluster_loss = F.kl_div(q.clamp_min(1e-12).log(), target[ids], reduction="batchmean")
        loss = cfg["gamma"] * cluster_loss + reconstruction_loss(model.decoder(code), observed, weights, ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model, centers.detach()


def save_predictions(path, **arrays):
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
        np.savez_compressed(temp, **arrays)
    Path(temp.name).replace(path)


def train_runs(out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    all_train, all_labels, test_images, test_labels = load_mnist()
    ids = np.random.default_rng(cfg["split_seed"]).permutation(len(all_train))[:cfg["train_size"]]
    images = all_train[ids]
    out.mkdir(parents=True, exist_ok=True)
    source_names = ("clean_neighborhood_study.py", "neighborhood_weights.py", "idec_final/model.py",
                    "idec_final/data.py", "idec_final/utils.py", "idec_final/experiment.py")
    manifest = {
        "arms": list(ARMS), "seeds": cfg["seeds"],
        "config": {k: cfg[k] for k in ("dims", "clusters", "split_seed", "train_size", "pretrain_epochs",
                                       "batch_size", "lr", "gamma", "target_interval", "updates", "kmeans_restarts")},
        "mnist_md5": FILES, "intensity_scale": SCALE,
        "source_sha256_lf": {k: normalized_digest(ROOT / k) for k in source_names},
        "torch": str(torch.__version__), "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
    }
    manifest_path = out / "manifest.json"
    if any(out.iterdir()) and (not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest):
        raise ValueError(f"Output directory contains another experiment: {out}")
    save_json(manifest_path, manifest)
    maps = training_weights(images, device)
    order = np.random.default_rng(cfg["split_seed"]).permutation(len(maps))
    donor = np.empty_like(order)
    donor[order] = np.roll(order, 1)
    assert np.all(donor != np.arange(len(donor)))
    save_predictions(out / "evaluation.npz", train_ids=ids, train_digit=all_labels[ids],
                     test_digit=test_labels, shuffled_donor=donor)
    train = torch.from_numpy(images).to(device) * SCALE
    test = torch.from_numpy(test_images).to(device) * SCALE
    arms = {"uniform": None, "neighborhood": torch.from_numpy(maps).to(device),
            "shuffled": torch.from_numpy(maps[donor]).to(device)}
    for seed in cfg["seeds"]:
        for name, weights in arms.items():
            path = out / f"{name}_{seed}.npz"
            if path.exists():
                continue
            model, centers = fit(train, weights, cfg, seed)
            save_predictions(path, train=assignments(model, centers, train).astype(np.uint8),
                             test=assignments(model, centers, test).astype(np.uint8))
            print(f"Finished original MNIST, {name}, seed {seed}", flush=True)


def normalized_digest(path):
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def score(out):
    manifest = json.loads((out / "manifest.json").read_text())
    seeds = manifest["seeds"]
    assert manifest["arms"] == list(ARMS) and seeds == [0, 1, 2, 3, 4]
    expected = {f"{arm}_{seed}" for arm in ARMS for seed in seeds} | {"evaluation"}
    assert {p.stem for p in out.glob("*.npz")} == expected
    with np.load(out / "evaluation.npz") as data:
        train_digit, test_digit = data["train_digit"], data["test_digit"]
        assert len(train_digit) == len(np.unique(data["train_ids"])) == 12000 and len(test_digit) == 10000
        donor = data["shuffled_donor"]
        np.testing.assert_array_equal(np.sort(donor), np.arange(len(train_digit)))
        assert np.all(donor != np.arange(len(train_digit)))
    rows = {arm: [] for arm in ARMS}
    for arm in ARMS:
        for seed in seeds:
            with np.load(out / f"{arm}_{seed}.npz") as p:
                train, test = p["train"], p["test"]
            assert train.shape == train_digit.shape and test.shape == test_digit.shape
            assert train.dtype == test.dtype == np.uint8 and max(train.max(), test.max()) < 10
            mapping = digit_mapping(train, train_digit)
            rows[arm].append({"seed": seed, "acc_pct": float(100 * np.mean(mapping[test] == test_digit)),
                              "nmi": float(normalized_mutual_info_score(test_digit, test)),
                              "ari": float(adjusted_rand_score(test_digit, test)),
                              "occupied": int(np.unique(test).size)})
    means = {arm: {key: float(np.mean([r[key] for r in values])) for key in ("acc_pct", "nmi", "ari")}
             for arm, values in rows.items()}
    paired = {}
    for left, right in (("neighborhood", "uniform"), ("shuffled", "uniform"), ("neighborhood", "shuffled")):
        delta = np.array([a["acc_pct"] - b["acc_pct"] for a, b in zip(rows[left], rows[right])])
        paired[f"{left}_minus_{right}_pp"] = {"mean": float(delta.mean()), "sd": float(delta.std(ddof=1)),
                                               "wins": int((delta > 0).sum()), "by_seed": delta.tolist()}
    return {"means": means, "paired": paired, "runs": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "clean_neighborhood")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    with threadpool_limits(4):
        if not args.evaluate_only:
            train_runs(args.output, torch.device(args.device))
        result = score(args.output)
        if args.evaluate_only:
            stored = json.loads((args.output / "summary.json").read_text())
            for arm in ARMS:
                for actual, expected in zip(result["runs"][arm], stored["runs"][arm]):
                    assert actual.keys() == expected.keys()
                    np.testing.assert_allclose(list(actual.values()), [expected[k] for k in actual], atol=1e-12, rtol=0)
            assert result["paired"] == stored["paired"]
            for arm in ARMS:
                for key, value in result["means"][arm].items():
                    np.testing.assert_allclose(value, stored["means"][arm][key], atol=1e-12, rtol=0)
        else:
            save_json(args.output / "summary.json", result)
        print(json.dumps({"means": result["means"], "paired": result["paired"]}, indent=2))
