"""Recompute the photometric ablation and paired factorial effects from predictions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze import csv_write
from idec_final.utils import ROOT, save_json
from photometric_study import ARMS, SPATIAL_ARMS


LABELS = {"baseline": "Ordinary IDEC", "jitter": "Jitter only", "gaussian": "Gaussian only",
          "both": "Jitter + Gaussian", "clean_baseline": "Unmarked, ordinary", "clean_jitter": "Unmarked, jitter"}
LABELS.update(translate="Translation", rotate="Rotation", translate_gaussian="Translation + Gaussian",
              rotate_gaussian="Rotation + Gaussian", clean_translate="Unmarked, translation", clean_rotate="Unmarked, rotation")
METRICS = ("marked_acc", "clean_acc", "mark_penalty", "move_flip", "shape_flip", "probe_acc",
           "eligible_acc", "shifted_acc", "shift_penalty")
CONTRASTS = {
    "jitter_vs_baseline": {"jitter": 1, "baseline": -1},
    "gaussian_vs_baseline": {"gaussian": 1, "baseline": -1},
    "gaussian_vs_jitter": {"gaussian": 1, "jitter": -1},
    "both_vs_gaussian": {"both": 1, "gaussian": -1},
    "both_vs_jitter": {"both": 1, "jitter": -1},
    "jitter_main": {"jitter": .5, "both": .5, "baseline": -.5, "gaussian": -.5},
    "gaussian_main": {"gaussian": .5, "both": .5, "baseline": -.5, "jitter": -.5},
    "interaction": {"both": 1, "baseline": 1, "gaussian": -1, "jitter": -1},
    "jitter_in_clean_training": {"clean_jitter": 1, "clean_baseline": -1},
}


def digit_mapping(clusters, labels):
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (clusters, labels), 1)
    return linear_sum_assignment(-counts)[1]


def scores(p, e):
    train_y, test_y = e["train_digit"], e["test_digit"]
    n = len(test_y)
    eligible = e["shifted_indices"]
    sizes = {"train": len(train_y), "clean": n, "marked": n, "shape": n,
             "moved": n, "shifted": len(eligible), "shape_probe": 2*n}
    assert set(p.files) == set(sizes)
    for key, size in sizes.items():
        array = p[key]
        assert array.shape == (size,) and array.dtype == np.uint8
        assert array.max() < (2 if key == "shape_probe" else 10)
    mapping = digit_mapping(p["train"], train_y)
    clean_acc = np.mean(mapping[p["clean"]] == test_y)
    marked_acc = np.mean(mapping[p["marked"]] == test_y)
    eligible_acc = np.mean(mapping[p["clean"][eligible]] == test_y[eligible])
    shifted_acc = np.mean(mapping[p["shifted"]] == test_y[eligible])
    shape_labels = np.concatenate((e["test_shape"], 1 - e["test_shape"]))
    return dict(marked_acc=marked_acc, clean_acc=clean_acc, mark_penalty=clean_acc-marked_acc,
                move_flip=np.mean(p["marked"] != p["moved"]), shape_flip=np.mean(p["marked"] != p["shape"]),
                probe_acc=np.mean(p["shape_probe"] == shape_labels), eligible_acc=eligible_acc,
                shifted_acc=shifted_acc, shift_penalty=eligible_acc-shifted_acc,
                occupied=len(np.unique(p["marked"])))


def read_runs(out, arms, cfg):
    meta = json.loads((out/"run.json").read_text())
    assert meta["config"] == cfg and meta["arms"] == list(arms)
    design = "spatial_design.md" if "rotate" in arms else "design.md"
    raw_design = (ROOT/"photometric"/design).read_bytes()
    assert hashlib.sha256(raw_design).hexdigest() == meta["design_sha256"]
    seeds = cfg["seeds"]
    keys = {f"{a}_{s}" for a in arms for s in seeds}
    assert {p.stem for p in out.glob("*.npz")} == keys | {"evaluation"}
    rows = []
    with np.load(out/"evaluation.npz") as evaluation, np.load(ROOT/"results/evaluation.npz") as original:
        for key in ("train_digit", "test_digit"):
            np.testing.assert_array_equal(evaluation[key], original[key])
        for split in ("train", "test"):
            np.testing.assert_array_equal(evaluation[split+"_shape"], original[split+"_orientation"])
        eligible = evaluation["shifted_indices"]
        assert np.all(np.diff(eligible) > 0) and 0 <= eligible.min() < eligible.max() < len(evaluation["test_digit"])
        for arm in arms:
            for seed in seeds:
                path = out/f"{arm}_{seed}.npz"
                trace = json.loads(path.with_suffix(".json").read_text())
                assert [step["step"] for step in trace] == list(range(0, cfg["updates"]+1, cfg["target_interval"]))
                with np.load(path) as pred:
                    rows.append({"arm": arm, "seed": seed, **scores(pred, evaluation)})
    return rows, eligible


def analyze(first, spatial, out):
    cfg = json.loads((ROOT/"protocol.json").read_text())
    rows, eligible = read_runs(first, ARMS, cfg)
    more, spatial_eligible = read_runs(spatial, SPATIAL_ARMS, cfg)
    np.testing.assert_array_equal(eligible, spatial_eligible)
    rows += more
    out.mkdir(parents=True, exist_ok=True)
    csv_write(out/"runs.csv", rows)
    by_arm = {arm: {m: np.array([r[m] for r in rows if r["arm"] == arm]) for m in METRICS} for arm in LABELS}
    summary = []
    for arm, metrics in by_arm.items():
        record = {"arm": arm}
        for metric, values in metrics.items():
            record[metric+"_mean"] = values.mean()
            record[metric+"_sd"] = values.std(ddof=1)
        summary.append(record)
    csv_write(out/"summary.csv", summary)
    common_rows = []
    with np.load(first/"evaluation.npz") as labels:
        for seed in cfg["seeds"]:
            pairs = {}
            common = np.ones(len(labels["test_digit"]), dtype=bool)
            for arm in ("baseline", "jitter", "translate", "rotate", "gaussian"):
                root = first if arm in ARMS else spatial
                with np.load(root/f"{arm}_{seed}.npz") as p:
                    mapping = digit_mapping(p["train"], labels["train_digit"])
                    common &= mapping[p["clean"]] == labels["test_digit"]
                    pairs[arm] = (p["marked"] != p["moved"], p["marked"] != p["shape"])
            for arm, (move, shape) in pairs.items():
                common_rows.append({"arm": arm, "seed": seed, "common_clean_correct": int(common.sum()),
                                    "move_flip": move[common].mean(), "shape_flip": shape[common].mean()})
    csv_write(out/"common_clean_correct.csv", common_rows)
    contrasts = []
    def add_contrast(name, metric, diffs):
        contrasts.append(dict(contrast=name, metric=metric, mean=float(diffs.mean()),
                              sd=float(diffs.std(ddof=1)), positive=int((diffs > 1e-12).sum()),
                              negative=int((diffs < -1e-12).sum()), differences=diffs.tolist()))

    comparisons = CONTRASTS.copy()
    for aug in ("translate", "rotate"):
        joint = aug+"_gaussian"
        comparisons.update({
            aug+"_vs_baseline": {aug: 1, "baseline": -1}, "gaussian_vs_"+aug: {"gaussian": 1, aug: -1},
            aug+"_vs_jitter": {aug: 1, "jitter": -1}, joint+"_vs_"+aug: {joint: 1, aug: -1},
            joint+"_vs_gaussian": {joint: 1, "gaussian": -1},
            aug+"_main": {aug: .5, joint: .5, "baseline": -.5, "gaussian": -.5},
            "gaussian_main_with_"+aug: {"gaussian": .5, joint: .5, "baseline": -.5, aug: -.5},
            aug+"_interaction": {joint: 1, "baseline": 1, "gaussian": -1, aug: -1},
            aug+"_in_clean_training": {"clean_"+aug: 1, "clean_baseline": -1},
        })
    comparisons["rotate_vs_translate"] = {"rotate": 1, "translate": -1}
    for contrast, weights in comparisons.items():
        for metric in METRICS:
            add_contrast(contrast, metric, sum(w * by_arm[a][metric] for a, w in weights.items()))
    for aug in ("jitter", "translate", "rotate"):
        gain_marked = by_arm[aug]["marked_acc"] - by_arm["baseline"]["marked_acc"]
        gain_clean = by_arm["clean_"+aug]["clean_acc"] - by_arm["clean_baseline"]["clean_acc"]
        add_contrast("marked_minus_clean_domain_"+aug, "domain_matched_acc", gain_marked-gain_clean)
    save_json(out/"paired.json", contrasts)
    save_json(out/"verification.json", {"models": len(rows), "seeds": cfg["seeds"], "seed_sd_ddof": 1,
              "eligible_shift_pairs": len(eligible), "min_occupied_marked_test_clusters": min(r["occupied"] for r in rows),
              "test_digits": 10000, "probe_test_views": 20000, "metrics_from_saved_predictions": True})
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Pretraining & Marked ACC & Clean ACC & Move flips & Shape flips & Shape probe\\", r"\midrule"]
    for arm in ("baseline","jitter","translate","rotate","gaussian"):
        cells = [LABELS[arm]] + [f"{by_arm[arm][m].mean()*100:.2f}" for m in
                                 ("marked_acc", "clean_acc", "move_flip", "shape_flip", "probe_acc")]
        lines.append(" & ".join(cells) + r"\\")
    (out/"table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"])+"\n", encoding="utf8")
    controls = [r"\begin{tabular}{lrrrr}", r"\toprule",
                r"Augmentation & Clean-only ACC & Clean-only gain & Joint marked ACC & Joint move flips\\",
                r"\midrule"]
    for aug, joint in (("jitter","both"), ("translate","translate_gaussian"), ("rotate","rotate_gaussian")):
        clean = by_arm["clean_"+aug]["clean_acc"]
        gain = (clean - by_arm["clean_baseline"]["clean_acc"]) * 100
        controls.append(f"{LABELS[aug]} & {100*clean.mean():.2f} & "
                        f"${gain.mean():+.2f}\\pm{gain.std(ddof=1):.2f}$ & "
                        f"{100*by_arm[joint]['marked_acc'].mean():.2f} & "
                        f"{100*by_arm[joint]['move_flip'].mean():.2f}" + r"\\")
    (out/"controls.tex").write_text("\n".join(controls + [r"\bottomrule", r"\end{tabular}"])+"\n", encoding="utf8")
    print("\n".join(lines[4:]))
    print(f"{len(rows)} models; {len(eligible)} eligible shift pairs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"photometric/results")
    parser.add_argument("--spatial-input", type=Path, default=ROOT/"photometric/spatial_results")
    parser.add_argument("--output", type=Path, default=ROOT/"photometric/analysis")
    parser.add_argument("--figure", type=Path, help="Also render data and loss examples using the original MNIST.")
    parser.add_argument("--effects-figure", type=Path, help="Plot matched seed differences for each whole-image augmentation.")
    args = parser.parse_args()
    analyze(args.input, args.spatial_input, args.output)
    if args.figure:
        from photometric_figure import render
        render(args.figure)
    if args.effects_figure:
        from photometric_figure import plot_paired_effects
        plot_paired_effects(args.output, args.effects_figure)
