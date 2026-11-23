"""Matched whole-image jitter and Gaussian-weighted IDEC pretraining."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from geometry_study import SCALE, data, gaussian_weights
from idec_final.experiment import assignments, finetune
from idec_final.model import Autoencoder
from idec_final.utils import ROOT, batches, predict, save_json, seed_all, sha256


# Training domain, whole-image jitter, Gaussian weighting.
ARMS = {
    "baseline": ("marked", False, False),
    "jitter": ("marked", True, False),
    "gaussian": ("marked", False, True),
    "both": ("marked", True, True),
    "clean_baseline": ("clean", False, False),
    "clean_jitter": ("clean", True, False),
}


def jitter_images(images, draws):
    """Apply per-image brightness/contrast to pixels in the model's [0, SCALE] range."""
    brightness, contrast = (0.7 + 0.6 * draws[:, :2]).split(1, dim=1)
    bright = (brightness * images).clamp(0, SCALE)
    mean = bright.mean(dim=1, keepdim=True)
    brightness_first = (mean + contrast * (bright - mean)).clamp(0, SCALE)
    mean = images.mean(dim=1, keepdim=True)
    contrasted = (mean + contrast * (images - mean)).clamp(0, SCALE)
    contrast_first = (brightness * contrasted).clamp(0, SCALE)
    return torch.where(draws[:, 2:] < 0.5, brightness_first, contrast_first)


def shifted_test(clean):
    images = clean.reshape(-1, 28, 28)
    left = np.arange(len(images)) % 2 == 0
    valid = np.where(left, ~(images[:, :, :4] > 0).any(axis=(1, 2)),
                     ~(images[:, :, -4:] > 0).any(axis=(1, 2)))
    shifted = np.zeros_like(images)
    shifted[left, :, :-4] = images[left, :, 4:]
    shifted[~left, :, 4:] = images[~left, :, :-4]
    np.testing.assert_array_equal(np.count_nonzero(shifted[valid], axis=(1, 2)),
                                  np.count_nonzero(images[valid], axis=(1, 2)))
    return shifted[valid].reshape(-1, 784), np.flatnonzero(valid)


def fit(images, jitter, gaussian, cfg, seed):
    device = images.device
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    generator = torch.Generator(device=device).manual_seed(34000 + seed)
    weights = gaussian_weights(device) if gaussian else None
    for _ in range(cfg["pretrain_epochs"]):
        epoch = images
        if jitter:
            draws = torch.rand((len(images), 3), device=device, generator=generator)
            epoch = jitter_images(images, draws)
        for ids in batches(len(images), cfg["batch_size"], device):
            batch = epoch[ids]
            error = (model(batch) - batch).square()
            loss = error.mean() if weights is None else (error * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    codes = predict(model.encoder, images).cpu().numpy()
    km = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed).fit(codes)
    centers = torch.from_numpy(km.cluster_centers_).to(device)
    centers, trace = finetune(model, centers, images, None, "idec", cfg)
    return model, centers, trace


def shape_probe(model, training_views, testing_views, train_shape):
    def encode(views):
        return np.concatenate([predict(model.encoder, view).cpu().numpy() for view in views])

    probe = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))
    probe.fit(encode(training_views), np.concatenate((train_shape, 1-train_shape)))
    return probe.predict(encode(testing_views)).astype(np.uint8)


def run(out, device):
    cfg, (training, testing) = data()
    shifted, eligible = shifted_test(testing[0])

    def tensor(array):
        return torch.from_numpy(array).to(device) * SCALE

    train = {"clean": tensor(training[0]), "marked": tensor(training[-1][0])}
    test = {name: tensor(images) for name, images in zip(
        ("clean", "marked", "shape", "moved", "shifted"),
        (testing[0], *testing[-1][:3], shifted))}
    probe_train = (train["marked"], tensor(training[-1][1]))
    manifest = {"design_commit": "715521a", "design_sha256": sha256(ROOT/"photometric/design.md"),
                "code_sha256": sha256(Path(__file__)), "config": cfg, "arms": list(ARMS),
                "torch": str(torch.__version__), "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"}
    out.mkdir(parents=True, exist_ok=True)
    metadata = out/"run.json"
    if any(out.iterdir()) and (not metadata.exists() or json.loads(metadata.read_text()) != manifest):
        raise ValueError("Output belongs to another run. Select a fresh directory.")
    save_json(metadata, manifest)
    np.savez_compressed(out/"evaluation.npz", train_digit=training[1], test_digit=testing[1],
                        train_shape=training[3], test_shape=testing[3], shifted_indices=eligible)
    for seed in cfg["seeds"]:
        for arm, (domain, jitter, gaussian) in ARMS.items():
            dest = out/f"{arm}_{seed}.npz"
            if dest.exists():
                continue
            images = train[domain]
            model, centers, trace = fit(images, jitter, gaussian, cfg, seed)
            pred = {name: assignments(model, centers, x).astype(np.uint8) for name, x in test.items()}
            pred["train"] = assignments(model, centers, images).astype(np.uint8)
            pred["shape_probe"] = shape_probe(model, probe_train, (test["marked"], test["shape"]), training[3])
            save_json(dest.with_suffix(".json"), trace)
            temp = dest.with_suffix(".tmp.npz")
            np.savez_compressed(temp, **pred)
            temp.replace(dest)
            print(f"Finished seed {seed}, {arm}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"scratch/photometric")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
