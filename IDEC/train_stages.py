"""Cross the pretraining and fine-tuning losses at the original strong stripe amplitude."""
import argparse
import copy
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import scipy
import sklearn
import torch
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from idec_final.data import load_mnist
from idec_final.experiment import assignments, finetune, pretrain
from idec_final.metrics import evaluate
from idec_final.model import Autoencoder
from idec_final.stripes import observe, orientations, projected_mse, templates
from idec_final.utils import ROOT, batches, predict, save_json, seed_all, sha256, sync


STRENGTH = 0.8
PRETRAINING = ("full", "projected")
FINETUNING = ("initial", "full", "projected")


def pretrain_projected(train, direction, cfg, seed):
    """Mirror the archived AE recipe, changing its reconstruction penalty."""
    seed_all(seed)
    model = Autoencoder(cfg["dims"]).to(train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    for _ in range(cfg["pretrain_epochs"]):
        for ids in batches(len(train), cfg["batch_size"], train.device):
            batch = train[ids]
            loss = projected_mse(model(batch) - batch, direction)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    embedding = predict(model.encoder, train).cpu().numpy()
    kmeans = KMeans(n_clusters=cfg["clusters"], n_init=cfg["kmeans_restarts"], random_state=seed)
    kmeans.fit(embedding)
    return model, torch.as_tensor(kmeans.cluster_centers_, device=train.device)


def run_seed(folder, seed, cfg, direction, train, test, flipped, train_y, test_y, test_orientation):
    scores, traces, saved = {}, {}, {}
    for early in PRETRAINING:
        sync()
        start = perf_counter()
        if early == "full":
            model, centers = pretrain(train, cfg, seed, train.device)
        else:
            model, centers = pretrain_projected(train, direction, cfg, seed)
        sync()
        pretrain_seconds = perf_counter() - start
        for late in FINETUNING:
            key = f"{early}_{late}"
            arm = copy.deepcopy(model)
            if late == "initial":
                final_centers, traces[key], seconds = centers, [], pretrain_seconds
            else:
                method = "idec" if late == "full" else "projected"
                sync()
                start = perf_counter()
                final_centers, traces[key] = finetune(arm, centers, train, direction, method, cfg)
                sync()
                seconds = perf_counter() - start

            pred = [assignments(arm, final_centers, x) for x in (train, test, flipped)]
            scores[key] = evaluate(train_y, pred[0], test_y, pred[1], pred[2], test_orientation)
            scores[key]["seconds"] = seconds
            for split, labels in zip(("train", "test", "flipped"), pred):
                saved[f"{key}_{split}"] = labels.astype(np.uint8)
        print(f"Finished seed {seed}, {early} pretraining", flush=True)

    # An incomplete seed is never published or used on resume.
    with TemporaryDirectory(dir=folder.parent) as tmp:
        pending = Path(tmp) / folder.name
        pending.mkdir()
        np.savez_compressed(pending / "predictions.npz", **saved)
        save_json(pending / "record.json", {"seed": seed, "scores": scores, "traces": traces})
        pending.rename(folder)


def run(out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    train_x, train_y, test_x, test_y = load_mnist()
    ids = np.random.default_rng(cfg["split_seed"]).permutation(len(train_y))[:cfg["train_size"]]
    train_orientation = orientations(len(ids), cfg["train_orientation_seed"])
    test_orientation = orientations(len(test_y), cfg["test_orientation_seed"])
    evaluation = ROOT / "results" / "evaluation.npz"
    with np.load(evaluation, allow_pickle=False) as old:
        for key, value in dict(train_ids=ids, train_digit=train_y[ids], test_digit=test_y,
                               train_orientation=train_orientation, test_orientation=test_orientation).items():
            np.testing.assert_array_equal(old[key], value)

    inputs = [ROOT / "protocol.json", ROOT / "stage" / "design.md", Path(__file__).resolve(),
              evaluation, *sorted((ROOT / "idec_final").glob("*.py"))]
    manifest = {"protocol": cfg, "strength": STRENGTH, "torch": str(torch.__version__),
                "numpy": np.__version__, "sklearn": sklearn.__version__, "scipy": scipy.__version__,
                "python": platform.python_version(), "device": str(device),
                "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
                "sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in inputs}}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing stage results use different inputs or software. Select a fresh --output.")
    else:
        if any(out.iterdir()):
            raise ValueError("The output directory has no stage manifest. Select a fresh --output.")
        save_json(manifest_path, manifest)

    patterns, direction = templates(device)
    train = observe(torch.from_numpy(train_x[ids]).to(device),
                    torch.from_numpy(train_orientation).to(device), STRENGTH, patterns)
    clean_test = torch.from_numpy(test_x).to(device)
    assigned = torch.from_numpy(test_orientation).to(device)
    test = observe(clean_test, assigned, STRENGTH, patterns)
    flipped = observe(clean_test, 1 - assigned, STRENGTH, patterns)
    for seed in cfg["seeds"]:
        folder = out / f"seed_{seed}"
        if not folder.exists():
            run_seed(folder, seed, cfg, direction, train, test, flipped, train_y[ids], test_y, test_orientation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch" / "stages")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
