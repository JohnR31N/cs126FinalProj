"""Plot the retained directions, literal bin choices and held-out clustering results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from idec_final.utils import ROOT


def make_figure(parent, coords, output):
    summary = json.loads((parent / "summary.json").read_text())
    coordinate = json.loads((coords / "summary.json").read_text())
    with np.load(parent / "probes.npz") as f:
        images = f["tail_images"]
    with np.load(coords / "bins.npz") as f:
        top = f["top"]
    with (parent / "runs.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    fig = plt.figure(figsize=(14.0, 5.4), layout="constrained")
    left, mid, right = fig.subfigures(1, 3, width_ratios=(1.05, 1.16, 1.95), wspace=.045)
    left.suptitle("What the retained directions measure", fontsize=11.5, fontweight="bold", wrap=True)
    grid = left.subplots(3, 2)
    for i, pair in enumerate(grid):
        for j, ax in enumerate(pair):
            ax.imshow(images[i, j], cmap="Greys", interpolation="nearest", vmin=0, vmax=.88)
            ax.set(xticks=[], yticks=[])
            for spine in ax.spines.values():
                spine.set_visible(False)
            counts = summary["training_tail_digit_counts"][i][("low", "high")[j]]
            digit = int(np.argmax(counts))
            ax.set_xlabel(f"Digit {digit} · {100*counts[digit]/sum(counts):.0f}%", fontsize=8, labelpad=1)
            if not j:
                ax.set_ylabel(f"Axis {i+1}", fontsize=9)
            if not i:
                ax.set_title(("Lowest 5%", "Highest 5%")[j], fontsize=9)
    mid.suptitle("Where the 98 selected bins lie", fontsize=11.5, fontweight="bold", wrap=True)
    ax = mid.subplots()
    count = np.bincount(top//8, minlength=49).reshape(7, 7)
    ax.imshow(count, cmap="Blues", vmin=0, vmax=12, origin="upper", interpolation="nearest")
    for i in top:
        row, col = divmod(i//8, 7)
        ang = (i%8)*np.pi/8 + np.pi/2
        dx, dy = .39*np.cos(ang), .39*np.sin(ang)
        ax.plot((col-dx, col+dx), (row-dy, row+dy), color="#162b40", lw=1.05, solid_capstyle="round")
    ax.set_xticks(range(7), range(1, 8), fontsize=8)
    ax.set_yticks(range(7), range(1, 8), fontsize=8)
    ax.set_xticks(np.arange(-.5, 7, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="#cbd5e1", lw=.55)
    ax.tick_params(length=0)
    ax.set_xlabel("Cell column · lines show edge angles", fontsize=9)
    ax.set_ylabel("Cell row", fontsize=9)
    right.suptitle("Digit clustering accuracy on original MNIST", fontsize=11.5, fontweight="bold", wrap=True)
    ax = right.subplots()
    names = [("pixels", "Original pixels"), ("all_strokes", "All 392 stroke features"),
             ("selected", "Retained 9 directions"), ("complement", "Remaining directions"),
             ("pca", "PCA, 9 directions"), ("permuted", "Permuted groups, 9 directions"),
             ("top", "Top 98 named bins"), ("location_matched", "Same-location 98 bins"),
             ("bottom", "Bottom 98 named bins")]
    for y, (key, _) in enumerate(names):
        if key in ("top", "location_matched", "bottom"):
            vals = [r["idec"]["test"]["acc_pct"] for r in coordinate["runs"][key]]
            extra = coordinate["means"]["extra"][key]["acc_pct"]
        else:
            vals = [float(r["acc_pct"]) for r in rows if r["target"] == key and
                    r["stage"] == "idec" and r["split"] == "test"]
            extra = summary["means"]["extra"][key+"_idec"]["acc_pct"]
        ax.hlines(y, min(vals), max(vals), color="#778495", lw=1.5, zorder=1)
        ax.plot(np.mean(vals), y-.09, "o", color="#14635e", markersize=6, zorder=2,
                label="10,000 test digits" if y == 0 else None)
        ax.plot(extra, y+.09, "D", mfc="white", mec="#a34c21", ms=5.5, zorder=3,
                label="48,000 additional digits" if y == 0 else None)
    ax.set_yticks(np.arange(len(names)), [label for _, label in names], fontsize=9)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#d6dce2", lw=.65)
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=8)
    ax.set_xlabel("Accuracy (%) · lines span the five test-set runs", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), frameon=False, fontsize=8, ncol=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "cluster_features" / "results")
    parser.add_argument("--coordinates", type=Path, default=ROOT / "cluster_features" / "coordinates")
    parser.add_argument("--output", type=Path, default=ROOT / "cluster_features" / "feature_selection.png")
    args = parser.parse_args()
    make_figure(args.input, args.coordinates, args.output)
