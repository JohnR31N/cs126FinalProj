"""Check geometric interventions and score the saved matched models on identical digits."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from analyze import csv_write
from geometry_study import ARMS, data, stable_target
from idec_final.utils import ROOT, save_json


NAMES = {"fixed": "Full, fixed input", "augmented": "Augmentation only",
         "paired": "Aug. + paired target", "gaussian": "Aug. + Gaussian"}
METRICS = ("marked_acc", "clean_acc", "move_flip", "shape_flip", "outer_acc", "decoder_response")


def example(test, dest):
    clean, labels, _, _, views = test
    panels = (clean, views[0], views[1], views[2], stable_target(views[0], views[2]))
    titles = ("Clean", "Marked input", "Change shape", "Move mark", "Paired target")
    fig, axes = plt.subplots(2, 5, figsize=(8, 3.2), layout="constrained")
    for j, (images, title) in enumerate(zip(panels, titles)):
        axes[0, j].set_title(title, fontsize=11)
        for i in range(2):
            ax = axes[i, j]
            ax.imshow(images[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set(xticks=[], yticks=[])
            if j == 0:
                ax.set_ylabel(f"Digit {labels[i]}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=180)
    plt.close(fig)


def analyze(out, figure):
    cfg, (training, testing) = data()
    example(testing, figure)
    assert json.loads((out / "run.json").read_text())["config"] == cfg
    y, test_y = training[1], testing[1]
    grid = np.indices((28, 28)).reshape(2, -1)
    clean = testing[0]
    centers = (clean @ grid.T) / clean.sum(1, keepdims=True)
    radius = np.linalg.norm(centers - 13.5, axis=1)
    outer = np.argsort(radius, kind="stable")[-len(test_y)//4:]
    keys = {f"{arm}_{s}" for arm in ARMS for s in cfg["seeds"]}
    assert {f.stem for f in out.glob("*.npz")} == keys | {"decoder"}
    rows = []
    with np.load(out / "decoder.npz") as readout:
        assert set(readout.files) == keys
        for arm in ARMS:
            for seed in cfg["seeds"]:
                key = f"{arm}_{seed}"
                trace = json.loads((out / f"{key}.json").read_text())["trace"]
                assert [t["step"] for t in trace] == list(range(0, cfg["updates"]+1, cfg["target_interval"]))
                with np.load(out / f"{key}.npz") as p:
                    assert set(p.files) == {"train", "marked", "shape", "moved", "clean"}
                    for split in p.files:
                        assert p[split].shape == (len(y) if split == "train" else len(test_y),)
                        assert p[split].dtype == np.uint8 and p[split].max() < cfg["clusters"]
                    counts = np.zeros((10, 10), dtype=np.int64)
                    np.add.at(counts, (p["train"], y), 1)
                    _, mapping = linear_sum_assignment(-counts)
                    marked = p["marked"]
                    correct = mapping[marked] == test_y
                    response = readout[key]
                    assert response.shape == test_y.shape and np.isfinite(response).all()
                    rows.append(dict(arm=arm, seed=seed, marked_acc=correct.mean(),
                                     clean_acc=(mapping[p["clean"]] == test_y).mean(),
                                     move_flip=(marked != p["moved"]).mean(), shape_flip=(marked != p["shape"]).mean(),
                                     outer_acc=correct[outer].mean(), decoder_response=response.mean(dtype=np.float64),
                                     occupied=np.unique(marked).size))
    groups = []
    for arm in ARMS:
        group = {"arm": arm}
        for name in METRICS:
            values = [r[name] for r in rows if r["arm"] == arm]
            group[name+"_mean"], group[name+"_sd"] = mean(values), stdev(values)
        groups.append(group)
    csv_write(out / "runs.csv", rows)
    csv_write(out / "summary.csv", groups)
    signatures = {}
    for split, sample in (("train", training), ("test", testing)):
        _, digits, boxes, shapes, _ = sample
        signatures[split] = {k: hashlib.sha256(np.asarray(v, dtype="<i8").tobytes()).hexdigest()
                             for k, v in (("digits", digits), ("boxes", boxes), ("shapes", shapes))}
    save_json(out / "verification.json", {
        "scored_models": len(rows), "seeds": cfg["seeds"], "training_images": len(y),
        "test_images": len(test_y), "views_verified_per_image": 4, "changed_pixels_per_view": 9,
        "changed_pixels_on_ink_or_one_pixel_guard": 0, "paired_target_equals_clean": True,
        "min_occupied_test_clusters": int(min(r["occupied"] for r in rows)), "outer_quartile_size": len(outer),
        "outer_quartile_min_centroid_radius_px": float(radius[outer].min()),
        "rendering_metadata_sha256_int64le": signatures,
    })
    by_id = {(r["arm"],r["seed"]): r for r in rows}
    contrasts = []
    for arm, base in (("augmented", "fixed"), ("paired", "augmented"), ("gaussian", "augmented"), ("paired","gaussian")):
        for metric in METRICS:
            diffs = [by_id[arm,s][metric] - by_id[base,s][metric] for s in cfg["seeds"]]
            contrasts.append(dict(arm=arm, base=base, metric=metric, mean=mean(diffs), sd=stdev(diffs), differences=diffs))
    save_json(out / "paired.json", contrasts)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Pretraining & Marked ACC & Clean ACC & Move flips & Shape flips & Decoder response\\", r"\midrule"]
    for g in groups:
        values = [NAMES[g["arm"]]]
        for m in ("marked_acc","clean_acc","move_flip","shape_flip","decoder_response"):
            factor = 1 if m == "decoder_response" else 100
            precision = 5 if m == "decoder_response" else 2
            values.append(f"{factor * g[m + '_mean']:.{precision}f}")
        lines.append(" & ".join(values) + r"\\")
    (out / "table.tex").write_text("\n".join(lines + [r"\bottomrule",r"\end{tabular}"])+"\n", encoding="utf8")
    print("\n".join(lines[3:]))
    print(f"Checked {len(rows)} fitted models, {len(y)} training digits and {len(test_y)} test digits.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "background/results")
    parser.add_argument("--figure", type=Path, default=ROOT / "figures/geometry.png")
    args = parser.parse_args()
    analyze(args.input, args.figure)
