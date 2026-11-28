"""Draw the first two held-out brightness pairs in their original pixel coordinates."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from background_probe_study import CALIBRATION
from idec_final.data import load_mnist
from idec_final.utils import ROOT
from varied_backgrounds import render, template_bank


def figure(results, output):
    pool = load_mnist()[0]
    rows = slice(CALIBRATION, CALIBRATION + 2)
    with np.load(results / "evaluation.npz") as meta:
        clean = pool[meta["probe_ids"][rows]]
        boxes, base = meta["boxes"][rows], meta["dim_types"][rows]
        expected_gap = meta["pixel_intensity_gap"][rows]
    bank, _, _ = template_bank()
    dim = render(clean, boxes, base, bank)
    bright = render(clean, boxes, base + 2, bank)
    diff = bright - dim
    np.testing.assert_allclose(diff.sum(1, dtype=np.float64), expected_gap)
    difference_cmap = LinearSegmentedColormap.from_list("difference", ["#111827", "#ffc383"])
    fig, axes = plt.subplots(2, 4, figsize=(6.4, 2.68), layout="constrained")
    for col, (view, title) in enumerate(zip((clean, dim, bright, diff),
            ("Original digit", "Dim symbols", "Bright symbols", "Bright minus dim"), strict=True)):
        axes[0, col].set_title(title, fontsize=10, color="#19344c", pad=4)
        for row, ax in enumerate(axes[:, col]):
            ax.imshow(view[row].reshape(28, 28), cmap=difference_cmap if col == 3 else "gray",
                      vmin=0, vmax=0.4 if col == 3 else 1, interpolation="nearest")
            ax.axis("off")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "background_probe/results")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/brightness_pairs.png")
    args = parser.parse_args()
    figure(args.results, args.output)
