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
from idec_final.utils import ROOT, save_json, sha256


def analyze(results, output, study="gaussian_refinement"):
    arms, sources = ARMS, SOURCES
    extra = set()
    controls = {}
    if study == "gaussian_saliency":
        from gaussian_saliency_study import ARMS as saliency_arms, SOURCES as saliency_sources, donor_order
        arms, sources = saliency_arms, saliency_sources
        for domain in DOMAINS:
            name = f"{domain}_maps.npz"
            extra.add(name)
            with np.load(results / name) as saved:
                maps = saved["saliency"]
                assert maps.shape == (12000, 784) and maps.dtype == np.uint8 and maps.max() > 0
                np.testing.assert_array_equal(saved["donor"], donor_order(len(maps)))
    if study == "adaptive_saliency":
        from adaptive_saliency_study import ARMS as adaptive_arms, SOURCES as adaptive_sources, MAPS
        arms, sources = adaptive_arms, adaptive_sources
        controls = {"centroid": ROOT / "gaussian_refinement/results", "combined": MAPS, "shuffled": MAPS}
    meta = check_manifest(results, "marked_mnist", sources)
    if study == "adaptive_saliency":
        assert meta["map_sha256"] == {d: sha256(MAPS / f"{d}_maps.npz") for d in DOMAINS}
    cfg = json.loads((ROOT / "protocol.json").read_text())
    assert meta["config"] == cfg and meta["arms"] == list(arms) and cfg["seeds"] == list(range(5))
    expected = {f"{domain}_{arm}_{seed}.npz" for domain in DOMAINS for arm in arms for seed in cfg["seeds"]}
    assert expected | extra == {p.name for p in results.glob("*.npz")}
    old = ROOT / "dec_background/results"
    with np.load(ROOT / "varied/results/evaluation.npz") as e:
        train_y, test_y = e["train_digit"], e["test_digit"]
    rows, means, series = [], {}, {}
    arm_dirs = {**controls, **dict.fromkeys(arms, results)}
    for arm in ("idec", "gaussian_pretrain", *arm_dirs):
        runs = []
        for seed in cfg["seeds"]:
            values = {}
            for domain in DOMAINS:
                cell = f"{domain}_{seed}"
                path = arm_dirs[arm] / f"{domain}_{arm}_{seed}.npz" if arm in arm_dirs else old / cell / "predictions.npz"
                with np.load(path) as pred:
                    keys = [arm + "_" + key for key in ("train", "clean", "backgrounds")]
                    if arm in arm_dirs:
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
                       for k in series[arm]} for arm in arms}
    with (old / "summary.csv").open(encoding="utf-8") as file:
        archived = {r["method"]: r for r in csv.DictReader(file)}
    for arm in ("idec", "gaussian_pretrain"):
        for key, col in (("clean", "clean_acc_pct"), ("varied", "marked_acc_pct"),
                         ("redraw_flip", "redraw_flip_pct"), ("drop", "drop_pp")):
            np.testing.assert_allclose(means[arm][key], float(archived[arm][col]))
    audit = json.loads((results / "weights.json").read_text())
    for arm, folder in controls.items():
        audit[arm] = json.loads((folder / "weights.json").read_text())[arm]
    audit["gaussian_pretrain"] = json.loads((old / "manifest.json").read_text())["mean_pixel_weight"]["train"]
    audit["idec"] = {"ink": 1.0, "symbol": 1.0}
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "runs.csv", rows)
    summary = {"means_pct": means, "minus_fixed_gaussian_pp": contrasts, "mean_training_weights": audit}
    if study == "gaussian_saliency":
        summary["combined_minus_shuffled_pp"] = {k: paired(v, series["shuffled"][k])
                                                  for k, v in series["combined"].items()}
    if study == "adaptive_saliency":
        aligned = series["adaptive_aligned"]
        summary["adaptive_aligned_minus"] = {arm: {k: paired(v, series[arm][k]) for k, v in aligned.items()}
                                              for arm in ("centroid", "combined", "adaptive_shuffled")}
        summary["center_by_saliency_interaction_pp"] = {
            k: paired(np.subtract(v, series["centroid"][k]),
                      np.subtract(series["combined"][k], series["gaussian_pretrain"][k]))
            for k, v in aligned.items()}
    save_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("gaussian_refinement", "gaussian_saliency", "adaptive_saliency"),
                        default="gaussian_refinement")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    analyze(args.results or ROOT / args.study / "results", args.output or ROOT / args.study, args.study)
