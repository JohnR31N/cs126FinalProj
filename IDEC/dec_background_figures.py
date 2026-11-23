"""Plot the spatial reconstruction prior and the two paired degradation contrasts."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from geometry_study import gaussian_weights
from idec_final.data import load_mnist
from idec_final.utils import ROOT
from varied_backgrounds import render


BLUE, ORANGE, DARK = "#27638b", "#a34726", "#19344c"


def prior_figure(source, target):
    _, _, clean, _ = load_mnist()
    with np.load(ROOT / "varied/results/evaluation.npz") as meta:
        marked = render(clean[:1], meta["test_boxes"][:1], meta["test_types"][0, :1], meta["bank"])[0]
    weights = gaussian_weights("cpu").numpy().reshape(28, 28)
    means = json.loads((source / "manifest.json").read_text())["mean_pixel_weight"]["train"]
    fig, axes = plt.subplots(1, 3, figsize=(8.8, 2.55), width_ratios=(1, 1.15, 1.4), layout="constrained")
    axes[0].imshow(marked.reshape(28, 28), vmin=0, vmax=1, cmap="gray", interpolation="nearest")
    axes[0].set_title("Observed image")
    heat = axes[1].imshow(weights, vmin=0, vmax=3, cmap="viridis", interpolation="nearest")
    axes[1].set_title("Fixed loss weights")
    fig.colorbar(heat, ax=axes[1], ticks=[0, 1, 2, 3], shrink=.8)
    for axis in axes[:2]:
        axis.set_axis_off()
    ax = axes[2]
    ax.set_title("Weight per pixel")
    ax.barh([0, 1], [means["symbol"], means["ink"]], color=[ORANGE, BLUE], height=.45)
    ax.set_yticks([0, 1], ["Symbol", "Digit ink"])
    ax.set(xlim=(0, 2.14), ylim=(-.65, 1.65), xticks=[0, 1, 2])
    ax.axvline(1, ls="--", lw=1.2, color=DARK)
    for y, v in enumerate([means["symbol"], means["ink"]]):
        ax.text(v+.04, y, f"{v:.2f}", va="center")
    ax.set_xlabel("Uniform loss = 1")
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def effects_figure(source, target):
    with (source / "paired.csv").open(newline="", encoding="utf8") as f:
        rows = list(csv.DictReader(f))
    groups = [
        ("Accuracy lost after adding symbols", [
            ("DEC", "dec_drop_pp"), ("IDEC", "idec_drop_pp"), ("Gaussian\nboth stages", "gaussian_both_drop_pp")]),
        ("Difference in loss within each seed", [
            ("IDEC minus DEC", "idec_minus_dec_drop_pp"),
            ("IDEC minus Gaussian", "idec_minus_gaussian_both_drop_pp")]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8), layout="constrained")
    for ax, (title, comparisons) in zip(axes, groups):
        for i, (_, column) in enumerate(comparisons):
            y = len(comparisons) - i - 1
            values = np.array([float(r[column]) for r in rows])
            ax.scatter(values, y + np.linspace(-.15, .15, len(values)), color=BLUE, s=30, zorder=3)
            ax.scatter(values.mean(), y, color=ORANGE, edgecolor="white", marker="D", s=68, zorder=4)
        ax.set_yticks(np.arange(len(comparisons))[::-1], [label for label, _ in comparisons])
        ax.axvline(0, color=DARK, ls="--", lw=1)
        ax.grid(axis="x", color="#d5dfe7", lw=.65)
        ax.set_title(title, fontsize=11, fontweight="bold", color=DARK)
        ax.set_xlabel("Percentage points")
        ax.set_ylim(-.55, len(comparisons)-.45)
        ax.spines[["left","top","right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=210, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dec_background/results")
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--effects", type=Path)
    args = parser.parse_args()
    if not (args.prior or args.effects):
        parser.error("Supply --prior or --effects with the desired output path.")
    if args.prior:
        prior_figure(args.input, args.prior)
    if args.effects:
        effects_figure(args.input, args.effects)
