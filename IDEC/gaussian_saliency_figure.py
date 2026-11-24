"""Show the first three marked training inputs with their frozen teacher and reconstruction maps."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from gaussian_saliency_study import weights_for
from geometry_study import gaussian_weights
from idec_final.utils import ROOT
from varied_backgrounds import dataset


def plot(results, output):
    _, data, _ = dataset()
    imgs = data["train"]["varied"][0][:3]
    with np.load(results / "varied_maps.npz") as f:
        maps = torch.from_numpy(f["saliency"])
        ids = torch.arange(3)
        aligned = weights_for("combined", maps[:3], ids).numpy()
        shuffled = weights_for("combined", maps[f["donor"][:3]], ids).numpy()
    fixed = np.broadcast_to(gaussian_weights("cpu").numpy(), imgs.shape)
    columns = [imgs, maps[:3].numpy()/255, fixed, aligned, shuffled]
    limit = max(w.max() for w in columns[2:])
    fig, axes = plt.subplots(3, 5, figsize=(9, 5.6), layout="constrained")
    for j, (values, title) in enumerate(zip(columns, ("Training input", "Cluster saliency", "Gaussian",
                                                       "Gaussian × saliency", "Shuffled control"), strict=True)):
        for i, axis in enumerate(axes[:, j]):
            view = axis.imshow(values[i].reshape(28, 28), vmin=0, vmax=1 if j<2 else limit,
                               cmap="gray" if j==0 else ("magma" if j==1 else "viridis"),
                               interpolation="nearest")
            axis.set_axis_off()
            if i == 0:
                axis.set_title(title, fontsize=10)
    fig.colorbar(view, ax=axes[:, 2:].ravel().tolist(), location="bottom", shrink=.8, aspect=42,
                 label="Reconstruction weight (image mean = 1)")
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "gaussian_saliency/results")
    parser.add_argument("--output", type=Path, default=ROOT / "gaussian_saliency/saliency.png")
    args = parser.parse_args()
    plot(args.results, args.output)
