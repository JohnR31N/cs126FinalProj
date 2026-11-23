"""Verify the stage study from image-level assignments and calculate paired contrasts."""
import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from analyze import csv_write, verify
from idec_final.data import load_mnist
from idec_final.utils import ROOT, save_json, sha256
from train_stages import FINETUNING, PRETRAINING, STRENGTH


METRICS = ("digit_acc", "digit_nmi", "digit_ari", "stripe_nmi", "flip_rate", "train_acc")
CONTRASTS = {
    "early_at_initialization": {"projected_initial": 1, "full_initial": -1},
    "early_with_full_finetuning": {"projected_full": 1, "full_full": -1},
    "early_with_projected_finetuning": {"projected_projected": 1, "full_projected": -1},
    "late_after_full_pretraining": {"full_projected": 1, "full_full": -1},
    "late_after_projected_pretraining": {"projected_projected": 1, "projected_full": -1},
    "interaction": {"projected_projected": 1, "full_projected": -1, "projected_full": -1, "full_full": 1},
}


def collect(out, cfg, meta):
    rows = []
    historical_matches = 0
    cells = {f"{early}_{late}" for early in PRETRAINING for late in FINETUNING}
    splits = ("train", "test", "flipped")
    steps = list(range(0, cfg["updates"] + 1, cfg["target_interval"]))
    seeds = cfg["seeds"]
    assert {p.name for p in out.glob("seed_*")} == {f"seed_{seed}" for seed in seeds}
    for seed in seeds:
        folder = out / f"seed_{seed}"
        record = json.loads((folder / "record.json").read_text())
        assert record["seed"] == seed and set(record["scores"]) == set(record["traces"]) == cells
        original = ROOT / "results" / f"strength_{STRENGTH:.1f}" / folder.name / "predictions.npz"
        with np.load(folder / "predictions.npz", allow_pickle=False) as data, np.load(original) as old:
            assert set(data.files) == {f"{cell}_{s}" for cell in cells for s in splits}
            for early in PRETRAINING:
                for late in FINETUNING:
                    cell = f"{early}_{late}"
                    scores = record["scores"][cell]
                    assert set(scores) == {*METRICS, "seconds"} and scores["seconds"] > 0
                    trace = record["traces"][cell]
                    assert [r["step"] for r in trace] == ([] if late == "initial" else steps)
                    train, test, flipped = [data[f"{cell}_{split}"] for split in splits]
                    for values, expected in ((train, len(meta["train_ids"])), (test, 10000), (flipped, 10000)):
                        assert values.shape == (expected,) and values.dtype == np.uint8
                        assert values.max() < cfg["clusters"]
                    lo, hi = verify(meta["train_digit"], meta["test_digit"], meta["test_orientation"],
                                    train, test, flipped, scores)
                    rows.append(dict(seed=seed, pretrain=early, finetune=late, **scores,
                                     test_clusters=len(np.unique(test)), digit_acc_tie_min=lo, digit_acc_tie_max=hi))
                    if early == "full":
                        old_key = "idec" if late == "full" else late
                        historical_matches += all(np.array_equal(data[f"{cell}_{s}"], old[f"{old_key}_{s}"])
                                                  for s in splits)
    return rows, historical_matches


def summarize(rows, seeds):
    by_seed = {s: {} for s in seeds}
    for r in rows:
        by_seed[r["seed"]][f"{r['pretrain']}_{r['finetune']}"] = r
    groups = []
    for early in PRETRAINING:
        for late in FINETUNING:
            group = dict(pretrain=early, finetune=late, n=len(seeds))
            cells = [by_seed[s][f"{early}_{late}"] for s in seeds]
            for metric in (*METRICS, "seconds"):
                values = [cell[metric] for cell in cells]
                group[f"{metric}_mean"], group[f"{metric}_sd"] = mean(values), stdev(values)
            group["min_test_clusters"] = min(cell["test_clusters"] for cell in cells)
            groups.append(group)

    contrasts = []
    for comparison, weights in CONTRASTS.items():
        for metric in ("flip_rate", "digit_acc", "stripe_nmi"):
            diffs = [sum(weight * by_seed[s][cell][metric] for cell, weight in weights.items()) for s in seeds]
            contrasts.append(dict(comparison=comparison, metric=metric, mean=mean(diffs), sd=stdev(diffs),
                                  differences=dict(zip(map(str, seeds), diffs))))
    return groups, contrasts


def write_table(out, groups):
    names = {"initial": "AE + K-means", "full": "Full pixel", "projected": "Projected"}
    lines = [r"\begin{tabular}{llrrr}", r"\toprule",
             r"Pretraining & Fine-tuning & Digit ACC (\%) & Flip rate (\%) & Stripe NMI\\", r"\midrule"]
    for row in groups:
        if row["pretrain"] == "projected" and row["finetune"] == "initial":
            lines.append(r"\addlinespace")
        values = [names[row["pretrain"]], names[row["finetune"]]]
        for metric, scale, digits in (("digit_acc", 100, 2), ("flip_rate", 100, 2), ("stripe_nmi", 1, 4)):
            avg, sd = row[f"{metric}_mean"] * scale, row[f"{metric}_sd"] * scale
            values.append(f"{avg:.{digits}f} " + r"$\pm$" + f" {sd:.{digits}f}")
        lines.append(" & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    flips = {(r["pretrain"], r["finetune"]): r["flip_rate_mean"] for r in groups}
    slide = [r"\begin{tabular}{lrrr}", r"\toprule",
             r" & Before & \multicolumn{2}{c}{After fine-tuning}\\", r"\cmidrule(lr){3-4}",
             r"Pretraining & fine-tuning & Full pixel & Projected\\", r"\midrule"]
    for early in PRETRAINING:
        values = [names[early], *(f"{flips[early, late]:.2%}".replace("%", r"\%") for late in FINETUNING)]
        slide.append(" & ".join(values) + r"\\")
    slide.extend([r"\bottomrule", r"\end{tabular}"])
    (out / "flip_table.tex").write_text("\n".join(slide) + "\n", encoding="utf-8")


def audit(out):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["protocol"] == cfg and manifest["strength"] == STRENGTH
    sources = [ROOT / "protocol.json", ROOT / "stage/design.md", ROOT / "train_stages.py",
               ROOT / "results/evaluation.npz", *sorted((ROOT / "idec_final").glob("*.py"))]
    assert manifest["sha256"] == {p.relative_to(ROOT).as_posix(): sha256(p) for p in sources}
    _, train_y, _, test_y = load_mnist()
    ids = np.random.default_rng(cfg["split_seed"]).permutation(len(train_y))[:cfg["train_size"]]
    with np.load(ROOT / "results/evaluation.npz") as meta:
        np.testing.assert_array_equal(meta["train_ids"], ids)
        np.testing.assert_array_equal(meta["train_digit"], train_y[ids])
        np.testing.assert_array_equal(meta["test_digit"], test_y)
        for split, size in (("train", len(ids)), ("test", len(test_y))):
            expected = np.random.default_rng(cfg[f"{split}_orientation_seed"]).permutation(np.arange(size) % 2)
            np.testing.assert_array_equal(meta[f"{split}_orientation"], expected)
        rows, matches = collect(out, cfg, meta)
    groups, contrasts = summarize(rows, cfg["seeds"])
    csv_write(out / "runs.csv", rows)
    csv_write(out / "summary.csv", groups)
    save_json(out / "paired.json", contrasts)
    write_table(out, groups)
    save_json(out / "verification.json", {
        "seeds": cfg["seeds"], "strength": STRENGTH, "initializations": 2 * len(cfg["seeds"]),
        "fine_tuned_models": 4 * len(cfg["seeds"]), "evaluations": len(rows), "metrics_checked": list(METRICS),
        "identical_ordinary_pretrain_evaluations_to_original": matches,
        "ordinary_pretrain_evaluations_compared": 3 * len(cfg["seeds"]),
        "min_occupied_test_clusters": min(r["test_clusters"] for r in rows),
    })
    print(f"Verified {len(rows)} evaluations across {len(cfg['seeds'])} paired seeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "stage" / "results")
    audit(parser.parse_args().input)
