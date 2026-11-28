"""Visualize the circular and elliptical reconstruction weights on real clean inputs."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from adaptive_saliency_study import MAPS
from gaussian_refinement_study import spatial_weights
from gaussian_shape_study import elliptical_gaussian, reconstruction_weights
from geometry_study import gaussian_weights
from idec_final.data import load_mnist
from idec_final.utils import ROOT, save_json


def summarize_geometry(images):
    xy = np.indices((28, 28), dtype=float).reshape(2, -1).T
    fixed = gaussian_weights("cpu").numpy()
    shifts, ratios = [], []
    for batch in np.array_split(images, 48):
        mass = batch.astype(float) * fixed
        total = mass.sum(1, keepdims=True)
        density = mass / np.maximum(total, 1e-12)
        center = np.where(total > 0, density @ xy, 13.5)
        offsets = xy[None]-center[:, None]
        cov = offsets.transpose(0, 2, 1) @ (offsets*density[:, :, None])
        ev = np.linalg.eigvalsh(cov + np.eye(2))
        shifts.extend(np.linalg.norm(center-13.5, axis=1))
        ratios.extend(np.clip((ev[:, 1]/ev[:, 0])**0.25, 1, 2))
    summary = {"clean_train_images": len(images)}
    for key, values in (("centroid_displacement_px", shifts), ("gaussian_axis_ratio", ratios)):
        summary[key] = {"mean": float(np.mean(values)), "median": float(np.median(values)),
                        "p95": float(np.quantile(values, .95))}
    return summary


def render(output):
    original, _, _, _ = load_mnist()
    with np.load(ROOT / "varied/results/evaluation.npz") as saved:
        images = original[saved["train_ids"]]
    x = torch.from_numpy(images[:2])
    with np.load(MAPS / "clean_maps.npz") as saved:
        sal = torch.from_numpy(saved["saliency"][:2])
    arms = ("circle", "ellipse", "rotated")
    weights = [reconstruction_weights(x, sal, arm).reshape(-1, 28, 28).numpy() for arm in arms]
    kernels = [spatial_weights(x, "centroid"), elliptical_gaussian(x), elliptical_gaussian(x, True)]
    maximum = max(w.max() for w in weights)
    fig, axes = plt.subplots(2, 4, figsize=(7.3, 3.4), layout="constrained")
    titles = ("Clean input", "Adaptive circle", "Aligned ellipse", "Rotated ellipse")
    for i, row in enumerate(axes):
        row[0].imshow(x[i].reshape(28, 28), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        for j, weight in enumerate(weights, 1):
            im = row[j].imshow(weight[i], cmap="viridis", vmin=0, vmax=maximum, interpolation="nearest")
            kernel = kernels[j-1][i].reshape(28, 28).numpy()
            row[j].contour(kernel, levels=[0.5*kernel.max()], colors="white", linewidths=.85)
        for j, ax in enumerate(row):
            ax.set_axis_off()
            if i == 0:
                ax.set_title(titles[j], fontsize=10)
    fig.colorbar(im, ax=axes[:, 1:].ravel().tolist(), shrink=.86, label="Reconstruction weight")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    save_json(output.with_suffix(".json"), summarize_geometry(images))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/shape_scores/weights.png")
    render(parser.parse_args().output)
