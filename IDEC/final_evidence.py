"""Build the final ablation and paired contrasts from the released per-image assignments."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze import csv_write
from gaussian_refinement_analysis import analyze
from idec_final.utils import ROOT


LABELS = {
    "idec": "Ordinary IDEC",
    "gaussian_pretrain": "Fixed Gaussian",
    "saliency": "Saliency only",
    "centroid": "Adaptive Gaussian",
    "combined": "Fixed Gaussian + saliency",
    "adaptive_shuffled": "Adaptive + shuffled saliency",
    "adaptive_aligned": "Adaptive + aligned saliency",
}
FINAL = "adaptive_aligned"
PRIMARY = "clean_trained_marked"


def rescore():
    runs = {}
    with TemporaryDirectory() as staging:
        for study in ("gaussian_saliency", "adaptive_saliency"):
            folder = Path(staging) / study
            analyze(ROOT / study / "results", folder, study)
            with (folder / "runs.csv").open(encoding="utf-8") as file:
                for row in csv.DictReader(file):
                    key = row["arm"], int(row["seed"])
                    if key[0] in LABELS:
                        if key in runs and runs[key] != row:
                            raise ValueError(f"Shared control differs between studies: {key}")
                        runs[key] = row
    if set(runs) != {(arm, seed) for arm in LABELS for seed in range(5)}:
        raise ValueError("Incomplete paired ablation")
    return runs


def build(output):
    runs = rescore()
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "per_seed.csv", [runs[arm, seed] for arm in LABELS for seed in range(5)])
    rows, contrasts = [], []
    table = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Clean fit} & Marked fit\\",
             r"\cmidrule(lr){2-3}Pretraining weights & Clean test & Marked test & Marked test\\\midrule"]
    for arm, label in LABELS.items():
        series = {k: np.array([float(runs[arm, s][k]) for s in range(5)])
                  for k in ("clean", PRIMARY, "varied", "redraw_flip")}
        means = {k: float(values.mean()) for k, values in series.items()}
        rows.append(dict(arm=arm, **means, primary_seed_sd=series[PRIMARY].std(ddof=1)))
        print_label = label.replace(" + ", r" $\times$ ")
        cells = [print_label, *(f"{means[k]:.2f}" for k in ("clean", PRIMARY, "varied"))]
        if arm == FINAL:
            table.append(r"\midrule")
            cells = [r"\textbf{" + cell + "}" for cell in cells]
        table.append(" & ".join(cells) + r"\\")
        if arm != FINAL:
            delta = np.array([float(runs[FINAL, s][PRIMARY]) for s in range(5)]) - series[PRIMARY]
            contrasts.append(dict(comparator=arm, mean_pp=delta.mean(), sample_sd_pp=delta.std(ddof=1),
                                  positive_seeds=int((delta > 0).sum()),
                                  **{f"seed_{i}_pp": v for i, v in enumerate(delta)}))
    csv_write(output / "ablation.csv", rows)
    csv_write(output / "paired.csv", contrasts)
    table.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "ablation.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    plot_contrasts(contrasts, output / "paired.png")


def plot_contrasts(contrasts, output):
    order = ["idec", "gaussian_pretrain", "saliency", "centroid", "combined", "adaptive_shuffled"]
    by_arm = {r["comparator"]: r for r in contrasts}
    fig, ax = plt.subplots(figsize=(7.6, 3.1), layout="constrained")
    for y, arm in enumerate(order):
        row = by_arm[arm]
        values = [row[f"seed_{s}_pp"] for s in range(5)]
        ax.scatter(values, y + np.linspace(-.13, .13, 5), s=23, color="#79939e", zorder=3)
        ax.scatter(row["mean_pp"], y, marker="D", s=52, color="#a44720", zorder=4)
    ax.axvline(0, color="#384954", lw=1)
    ax.set_yticks(range(len(order)), [LABELS[a].replace(" + ", " × ") for a in order], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Final minus comparison accuracy (percentage points)")
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.scatter([], [], s=23, color="#79939e", label="One student seed")
    ax.scatter([], [], s=52, color="#a44720", marker="D", label="Paired mean")
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.savefig(output, dpi=165)
    plt.close(fig)


def plot_weights(output):
    import torch
    from adaptive_saliency_study import MAPS, combined_weights
    from gaussian_refinement_study import spatial_weights
    from varied_backgrounds import dataset

    _, data, _ = dataset()
    fig, axes = plt.subplots(2, 4, figsize=(7.5, 3.6), layout="constrained")
    columns = ("Observed image", "Adaptive Gaussian", "Teacher saliency", "Final weight")
    records = []
    for domain in ("clean", "varied"):
        images = data["train"][domain]
        x = torch.from_numpy((images if domain == "clean" else images[0])[:1])
        gaussian = spatial_weights(x, "centroid")
        with np.load(MAPS / f"{domain}_maps.npz") as f:
            saliency = torch.from_numpy(f["saliency"][:1])
        records.append([x, gaussian, saliency.float() / 255, combined_weights(gaussian, saliency)])
    limit = max(record[j].max().item() for record in records for j in (1, 3))
    for i, record in enumerate(records):
        for j, field in enumerate(record):
            ax = axes[i, j]
            img = ax.imshow(field.reshape(28, 28), cmap=("gray", "viridis", "magma", "viridis")[j],
                            vmin=0, vmax=limit if j in (1, 3) else 1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(columns[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(("Clean fit", "Marked fit")[i], fontsize=10)
    fig.colorbar(img, ax=axes.ravel().tolist(), shrink=.74, aspect=20, label="Gaussian / final weight")
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/final_evidence")
    parser.add_argument("--weight-figure", action="store_true", help="Also render MNIST weight maps (requires IDX data).")
    args = parser.parse_args()
    build(args.output)
    if args.weight_figure:
        plot_weights(args.output / "weights.png")
