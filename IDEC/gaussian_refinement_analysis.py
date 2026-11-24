"""Recompute Gaussian refinement comparisons from the matched per-image predictions."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from dec_background_analysis import scores
from external_analysis import check_manifest, paired
from gaussian_refinement_study import ARMS, DOMAINS, SOURCES
from idec_final.utils import ROOT, save_json


def analyze(results, output):
    meta = check_manifest(results, "marked_mnist", SOURCES)
    cfg = json.loads((ROOT / "protocol.json").read_text())
    assert meta["config"] == cfg and meta["arms"] == list(ARMS) and cfg["seeds"] == list(range(5))
    expected = {f"{domain}_{arm}_{seed}.npz" for domain in DOMAINS for arm in ARMS for seed in cfg["seeds"]}
    assert expected == {p.name for p in results.glob("*.npz")}
    old = ROOT / "dec_background/results"
    with np.load(ROOT / "varied/results/evaluation.npz") as e:
        train_y, test_y = e["train_digit"], e["test_digit"]
    rows, means, series = [], {}, {}
    for arm in ("idec", "gaussian_pretrain", *ARMS):
        runs = []
        for seed in cfg["seeds"]:
            values = {}
            for domain in DOMAINS:
                cell = f"{domain}_{seed}"
                path = results / f"{domain}_{arm}_{seed}.npz" if arm in ARMS else old / cell / "predictions.npz"
                with np.load(path) as pred:
                    keys = [arm + "_" + key for key in ("train", "clean", "backgrounds")]
                    if arm in ARMS:
                        assert set(pred.files) == set(keys)
                    measured = scores({k: pred[k] for k in keys}, arm, domain, train_y, test_y, None)
                    values[domain] = measured["domain_acc_pct"]
                    if domain == "clean":
                        values["clean_trained_marked"] = measured["marked_test_acc_pct"]
                        values["clean_trained_penalty"] = measured["frozen_test_drop_pp"]
                    else:
                        values["redraw_flip"] = measured["redraw_flip_pct"]
            values["drop"] = values["clean"] - values["varied"]
            runs.append(values)
            rows.append(dict(arm=arm, seed=seed, **values))
        series[arm] = {k: [row[k] for row in runs] for k in runs[0]}
        means[arm] = {k: float(np.mean(v)) for k, v in series[arm].items()}
    contrasts = {arm: {k: paired(series[arm][k], series["gaussian_pretrain"][k])
                       for k in series[arm]} for arm in ARMS}
    with (old / "summary.csv").open(encoding="utf-8") as file:
        archived = {r["method"]: r for r in csv.DictReader(file)}
    for arm in ("idec", "gaussian_pretrain"):
        for key, col in (("clean", "clean_acc_pct"), ("varied", "marked_acc_pct"),
                         ("redraw_flip", "redraw_flip_pct"), ("drop", "drop_pp")):
            np.testing.assert_allclose(means[arm][key], float(archived[arm][col]))
    audit = json.loads((results / "weights.json").read_text())
    audit["gaussian_pretrain"] = json.loads((old / "manifest.json").read_text())["mean_pixel_weight"]["train"]
    audit["idec"] = {"ink": 1.0, "symbol": 1.0}
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "runs.csv", rows)
    save_json(output / "summary.json", {"means_pct": means, "minus_fixed_gaussian_pp": contrasts,
                                        "mean_training_weights": audit})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "gaussian_refinement/results")
    parser.add_argument("--output", type=Path, default=ROOT / "gaussian_refinement")
    args = parser.parse_args()
    analyze(args.results, args.output)
