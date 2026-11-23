"""Render the training interventions and the counterfactual background test."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from geometry_study import SCALE, data, gaussian_weights
from photometric_study import jitter_images
from spatial_augmentation import Rotation


def plot_paired_effects(summary_dir, path):
    import json
    contrasts = {(r["contrast"], r["metric"]): np.asarray(r["differences"]) * 100
                 for r in json.loads((Path(summary_dir)/"paired.json").read_text())}
    methods = (("jitter", "Brightness / contrast"), ("rotate", "Rotation"), ("translate", "Translation"))
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.75), layout="constrained")
    for ax, title in zip(axes, ("Clean-digit accuracy gain", "Fewer flips when marks move")):
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axvline(0, color="#808b94", lw=1)
        ax.grid(axis="x", color="#dde3e7", lw=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "left", "right"]].set_visible(False)
        ax.set(xlabel="Percentage points", ylim=(2.48, -.5))
        ax.tick_params(left=False)
    for i, (key, label) in enumerate(methods):
        gain = contrasts[key+"_in_clean_training", "clean_acc"]
        reduction = -contrasts[key+"_vs_baseline", "move_flip"]
        for ax, values in zip(axes, (gain, reduction)):
            ax.scatter(values, i + np.linspace(-.14,.14,len(values)), color="#5681a4", s=24, zorder=2)
            ax.scatter(values.mean(), i, color="#a34321", edgecolor="white", marker="D", s=58, zorder=3)
    axes[0].set_yticks(range(len(methods)), [label for _, label in methods])
    axes[1].set_yticks(range(len(methods)), [])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render(path):
    _, (_, testing) = data()
    examples = torch.from_numpy(testing[-1][0][:64])
    candidates = Rotation(examples, 1.)
    images = examples.reshape(-1, 28, 28)
    can_shift = ~(images[:, -2:, :].ne(0).any(dim=(1, 2)) | images[:, :, -2:].ne(0).any(dim=(1, 2)))
    i = int(torch.where(can_shift & candidates.legal[:, -1])[0][0])
    first = testing[-1][0][i]
    dimmed = jitter_images(torch.from_numpy(first[None]) * SCALE,
                           torch.tensor([[0., 0., 0.]])).numpy()[0] / SCALE
    w = gaussian_weights(torch.device("cpu")).numpy().reshape(28, 28)
    panels = [("Training image", first), ("Brightness + contrast", dimmed),
              ("Translate image", np.roll(first.reshape(28, 28), (2, 2), axis=(0, 1))),
              ("Rotate image", candidates.views[-1, i].numpy()), ("Clean digit", testing[0][i]),
              ("Loss weights", w), ("Move mark", testing[-1][2][i]), ("Swap mark", testing[-1][1][i])]
    fig, axes = plt.subplots(2, 4, figsize=(8, 3.75), layout="constrained")
    for ax, (title, values) in zip(axes.flat, panels):
        is_prior = title == "Loss weights"
        ax.imshow(np.reshape(values, (28, 28)), cmap="viridis" if is_prior else "gray",
                  vmin=0, vmax=w.max() if is_prior else 1, interpolation="nearest")
        ax.set(title=title, xticks=[], yticks=[])
        if is_prior:
            ax.set_xlabel(f"Corner {w[0, 0]:.2f}   Center {w[13, 13]:.2f}", fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
