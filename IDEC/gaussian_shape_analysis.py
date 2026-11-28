"""Rescore the Gaussian shape test and apply its prespecified decision rule."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from external_analysis import check_manifest
from gaussian_shape_reference import REFERENCE_SOURCES
from gaussian_shape_study import ARMS, SOURCES, SPLITS
from idec_final.utils import ROOT, save_json, sha256
from photometric_analysis import digit_mapping


LABELS = {"uniform": "Ordinary IDEC", "fixed": "Fixed circle", "circle": "Adaptive circle", "ellipse": "Aligned ellipse",
          "rotated": "Rotated ellipse"}
DOMAINS = (*SPLITS, "eligible")
METRICS = ("clean_acc_pct", "marked_acc_pct", "mark_drop_pp", "cluster_flip_pct", "decoder_response")


def read_scores(pred, mapping, labels, split, ids=slice(None)):
    clean, marked, response = [pred[f"{split}_{key}"][ids] for key in ("clean", "marked", "response")]
    for array in (clean, marked):
        assert array.shape == labels.shape and array.dtype == np.uint8 and array.max() < 10
    assert response.shape == labels.shape and np.isfinite(response).all()
    clean_acc = float(100 * (mapping[clean] == labels).mean())
    marked_acc = float(100 * (mapping[marked] == labels).mean())
    return dict(clean_acc_pct=clean_acc, marked_acc_pct=marked_acc, mark_drop_pp=clean_acc-marked_acc,
                cluster_flip_pct=float(100 * (clean != marked).mean()),
                decoder_response=float(response.mean(dtype=np.float64)))


def collect(results, reference):
    meta = check_manifest(results, "mnist_gaussian_shape", SOURCES)
    cfg = json.loads((ROOT / "protocol.json").read_text())
    assert meta["config"] == cfg and meta["arms"] == list(ARMS) and cfg["seeds"] == list(range(5))
    assert meta["clean_map_sha256"] == sha256(ROOT / "gaussian_saliency/results/clean_maps.npz")
    assert meta["old_evaluation_sha256"] == sha256(ROOT / "varied/results/evaluation.npz")
    files = {"evaluation.npz", *(f"{arm}_{s}.npz" for arm in ARMS for s in cfg["seeds"])}
    assert {p.name for p in results.glob("*.npz")} == files
    ref = check_manifest(reference, "mnist_gaussian_shape_reference", REFERENCE_SOURCES)
    assert set(p.name for p in reference.glob("*.npz")) == {f"uniform_{s}.npz" for s in cfg["seeds"]}
    assert ref["config"] == cfg and ref["evaluation_sha256"] == sha256(results / "evaluation.npz")
    assert ref["arms"] == ["uniform"]
    for key in ("device", "hardware", "torch", "numpy", "scipy", "sklearn"):
        assert ref[key] == meta[key], key
    with np.load(results / "evaluation.npz") as e:
        train_y, eligible = e["train_y"], e["shifted_ids"]
        order = np.random.default_rng(cfg["split_seed"]).permutation(60000)
        np.testing.assert_array_equal(e["train_ids"], order[:12000])
        np.testing.assert_array_equal(e["holdout_ids"], order[12000:22000])
        assert 0 < len(eligible) < 10000 and np.all(np.diff(eligible) > 0)
        assert eligible.min() >= 0 and eligible.max() < 10000
        assert e["symbol_boxes"].shape == (10000, 2, 2) and e["symbol_types"].shape == (10000, 2)
        assert e["shift_directions"].shape == (10000, 2) and (np.abs(e["shift_directions"]).sum(1) == 3).all()
        labels = {"holdout": e["holdout_y"], "official": e["official_y"], "shifted": e["holdout_y"][eligible]}
    with np.load(ROOT / "varied/results/evaluation.npz") as old:
        np.testing.assert_array_equal(train_y, old["train_digit"])
        np.testing.assert_array_equal(labels["official"], old["test_digit"])
    assert labels["holdout"].shape == labels["official"].shape == (10000,)
    assert set(np.unique(labels["holdout"])) == set(range(10))
    keys = {"train", *(f"{s}_{k}" for s in SPLITS for k in ("clean", "marked", "response"))}
    runs = []
    for arm in LABELS:
        folder = reference if arm == "uniform" else results
        for seed in cfg["seeds"]:
            with np.load(folder / f"{arm}_{seed}.npz") as pred:
                assert set(pred.files) == keys
                train = pred["train"]
                assert train.shape == train_y.shape and train.dtype == np.uint8 and train.max() < 10
                mapping = digit_mapping(train, train_y)
                for split in SPLITS:
                    runs.append(dict(arm=arm, seed=seed, split=split,
                                     **read_scores(pred, mapping, labels[split], split)))
                runs.append(dict(arm=arm, seed=seed, split="eligible",
                                 **read_scores(pred, mapping, labels["shifted"], "holdout", eligible)))
    return runs, len(eligible)


def report(results, reference, output):
    runs, n_shift = collect(results, reference)
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "per_seed.csv", runs)
    by_key = {(r["arm"], r["split"], r["seed"]): r for r in runs}
    means, contrasts = [], []
    series = {}
    for arm in LABELS:
        for split in DOMAINS:
            for metric in METRICS:
                a = np.array([by_key[arm, split, seed][metric] for seed in range(5)])
                series[arm, split, metric] = a
                means.append(dict(arm=arm, split=split, metric=metric, mean=a.mean(), sample_sd=a.std(ddof=1)))
    for arm in ("circle", "fixed", "rotated", "uniform"):
        for split in DOMAINS:
            for metric in METRICS:
                diff = series["ellipse", split, metric] - series[arm, split, metric]
                contrasts.append(dict(comparator=arm, split=split, metric=metric, mean=diff.mean(),
                                      sample_sd=diff.std(ddof=1), positive=int((diff > 0).sum()),
                                      **{f"seed_{i}": v for i, v in enumerate(diff)}))
    csv_write(output / "means.csv", means)
    csv_write(output / "paired.csv", contrasts)
    circle = {(r["split"], r["metric"]): r for r in contrasts if r["comparator"] == "circle"}
    primary, official = [circle[s, "marked_acc_pct"] for s in ("holdout", "official")]
    checks = {
        "at_least_four_positive_primary_seeds": primary["positive"] >= 4,
        "positive_mean_marked_on_both_sets": bool(primary["mean"] > 0 and official["mean"] > 0),
        "clean_mean_loss_at_most_one_point_on_both": all(circle[s, "clean_acc_pct"]["mean"] >= -1
                                                          for s in ("holdout", "official")),
    }
    save_json(output / "decision.json", {"shift_eligible": n_shift, "paired_fits": len(LABELS)*5,
                                         "checks": checks, "selected": "ellipse" if all(checks.values()) else "circle"})
    table = [r"\begin{tabular}{lrrrrr}\toprule",
             r"& \multicolumn{2}{c}{New 10,000} & Official & \multicolumn{2}{c}{Shift-eligible}\\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){5-6}Gaussian & Clean & Marked & Marked & Marked & Shifted marked\\\midrule"]
    for arm in LABELS:
        columns = [("holdout", "clean_acc_pct"), ("holdout", "marked_acc_pct"), ("official", "marked_acc_pct"),
                   ("eligible", "marked_acc_pct"), ("shifted", "marked_acc_pct")]
        values = [f"{series[arm, s, m].mean():.2f}" for s, m in columns]
        table.append(" & ".join([LABELS[arm], *values]) + r"\\")
    table.append(r"\bottomrule\end{tabular}")
    (output / "table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "gaussian_shape/results")
    parser.add_argument("--reference", type=Path, default=ROOT / "gaussian_shape/reference")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/shape_scores")
    args = parser.parse_args()
    report(args.results, args.reference, args.output)
