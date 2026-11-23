"""Show actual paired inputs and matched-seed effects for the random-background study."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from idec_final.utils import ROOT
from varied_backgrounds import dataset


def examples(path):
    _, data, _ = dataset()
    sample = data["test"]
    panels = (sample["clean"], sample["fixed"][0], sample["varied"][0], sample["varied"][1], sample["varied"][2])
    titles = ("Clean digit", "Fixed form", "Varied attributes", "Redraw background", "Another redraw")
    fig, axes = plt.subplots(3, 5, figsize=(9, 5.0), layout="constrained")
    for j, (panel, title) in enumerate(zip(panels, titles)):
        axes[0, j].set_title(title, fontsize=10)
        for i in range(3):
            axes[i, j].imshow(panel[i].reshape(28,28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[i, j].set(xticks=[], yticks=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def effects(out, path):
    comparisons = json.loads((out/"paired.json").read_text())
    values = {(r["contrast"], r["metric"]): np.array(r["differences"]) for r in comparisons}
    names = ("Full target", "Clean target", "Translation", "Gaussian")
    contrasts = ("varied_vs_clean", "varied_clean_target", "translation_vs_standard", "gaussian_vs_standard")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), layout="constrained")
    for axis, metric, title in zip(axes, ("domain_acc", "redraw_flip"),
                                   ("Change in digit accuracy", "Change in background flips")):
        for i, (label, contrast) in enumerate(zip(names[1:], contrasts[1:])):
            numbers = values[contrast, metric]*100
            axis.scatter(numbers, i+np.linspace(-.12,.12,len(numbers)), c="#397ca3", s=24, zorder=3)
            axis.scatter(numbers.mean(), i, marker="D", c="#9b4b25", s=55, edgecolor="white", zorder=4)
        axis.axvline(0, lw=1, color="#737b81")
        axis.set(yticks=range(3), yticklabels=names[1:], xlabel="Percentage points", title=title, ylim=(2.5,-.5))
        axis.grid(axis="x", color="#dde3e7")
        axis.spines[["top","right","left"]].set_visible(False)
    axes[1].set_yticklabels([])
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=ROOT/"figures/varied.png")
    parser.add_argument("--effects", type=Path)
    parser.add_argument("--input", type=Path, default=ROOT/"varied/results")
    args = parser.parse_args()
    examples(args.examples)
    if args.effects:
        effects(args.input, args.effects)
