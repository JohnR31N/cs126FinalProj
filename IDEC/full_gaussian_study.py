"""Run or independently rescore the final pretraining weights on full transductive MNIST."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from analyze import csv_write
from clean_neighborhood_study import save_predictions
from external_analysis import check_manifest
from external_study import manifest
from fit_final import train_student
from gaussian_refinement_study import fit
from geometry_study import SCALE
from idec_final.data import FILES, load_mnist
from idec_final.experiment import assignments
from idec_final.utils import ROOT
from reproduce_paper import metrics


ARMS = ("uniform", "adaptive")
SOURCES = ("full_gaussian/design.md", "full_gaussian_study.py", "fit_final.py",
           "adaptive_saliency_study.py", "gaussian_refinement_study.py", "gaussian_saliency_study.py",
           "dec_background_study.py", "geometry_study.py", "varied_study.py", "idec_final/data.py",
           "idec_final/model.py", "idec_final/experiment.py", "idec_final/utils.py", "protocol.json",
           "clean_neighborhood_study.py", "external_study.py", "reproduce_paper.py")


def protocol():
    cfg = json.loads((ROOT / "protocol.json").read_text())
    cfg.update(seeds=[0], train_size=70000, updates=20000)
    return cfg


def run(results, device):
    cfg = protocol()
    manifest(results, cfg, device, SOURCES, dataset="mnist_full_adaptive_gaussian",
             arms=list(ARMS), teacher_seed=126, file_md5=FILES)
    train, first, test, last = load_mnist()
    images = torch.from_numpy(np.concatenate((train, test))).to(device) * SCALE
    labels = np.concatenate((first, last)).astype(np.uint8)
    del train, test
    save_predictions(results / "evaluation.npz", digit=labels)
    for arm in ARMS:
        path = results / f"{arm}_0.npz"
        if path.exists():
            continue
        print(f"Fitting full MNIST, {arm}, seed 0", flush=True)
        if arm == "uniform":
            model, centers = fit(images, torch.ones_like(images), cfg, 0)
        else:
            model, centers = train_student(images, cfg, 0)
        pred = assignments(model, centers, images).astype(np.uint8)
        save_predictions(path, final=pred)
        print(f"Saved {path.name}", flush=True)


def score(results, output):
    meta = check_manifest(results, "mnist_full_adaptive_gaussian", SOURCES)
    assert meta["config"] == protocol() and meta["arms"] == list(ARMS) and meta["file_md5"] == FILES
    assert meta["teacher_seed"] == 126
    assert {p.name for p in results.glob("*.npz")} == {"evaluation.npz", *(f"{arm}_0.npz" for arm in ARMS)}
    with np.load(results / "evaluation.npz") as data:
        labels = data["digit"]
    with np.load(ROOT / "reference/predictions.npz") as ref:
        np.testing.assert_array_equal(labels, ref["digits"])
    rows = []
    for arm in ARMS:
        with np.load(results / f"{arm}_0.npz") as archive:
            ids = archive["final"]
        assert labels.shape == ids.shape == (70000,) and ids.dtype == np.uint8 and ids.max() < 10
        result = metrics(labels, ids)
        rows.append(dict(arm=arm, seed=0, n=len(labels), acc_pct=100 * result["acc"], nmi=result["nmi"]))
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "summary.csv", rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true", help="train both models before scoring their assignments")
    parser.add_argument("--results", type=Path, default=ROOT / "full_gaussian/results")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/full_gaussian_scores")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        if args.fit:
            run(args.results, torch.device(args.device))
        score(args.results, args.output)
