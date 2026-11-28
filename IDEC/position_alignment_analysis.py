"""Recompute position sensitivity from paired cluster assignments without fitting."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from external_analysis import check_manifest, paired
from idec_final.utils import ROOT, save_json, sha256
from photometric_analysis import digit_mapping
from position_alignment_study import MODES, SOURCES, VIEWS


CONDITIONS = {"clean": "Clean", "marked": "Marked", "eligible_clean": "Eligible clean",
              "shifted_clean": "Shifted clean", "eligible_marked": "Eligible marked", "shifted_marked": "Shifted marked"}


def analyze(results, output):
    meta = check_manifest(results, "mnist_position_alignment", SOURCES)
    cfg = json.loads((ROOT / "protocol.json").read_text())
    assert meta["config"] == cfg and meta["modes"] == list(MODES) and cfg["seeds"] == list(range(5))
    assert meta["clean_map_sha256"] == sha256(ROOT / "gaussian_saliency/results/clean_maps.npz")
    expected = {"evaluation.npz", *(f"circle_{seed}.npz" for seed in cfg["seeds"])}
    assert {p.name for p in results.glob("*.npz")} == expected
    with np.load(results / "evaluation.npz") as e:
        train_y, y, eligible = e["train_y"], e["test_y"], e["eligible"]
        ids = np.random.default_rng(cfg["split_seed"]).permutation(60000)
        np.testing.assert_array_equal(e["test_ids"], ids[22000:32000])
        np.testing.assert_array_equal(e["train_ids"], ids[:12000])
        assert len(train_y) == 12000 and len(y) == 10000
        assert np.all(np.diff(eligible)>0) and eligible.min() >= 0 and eligible.max() < len(y)
        threshold, reference = float(e["threshold_px"]), e["reference_yx"].tolist()
        assert np.isfinite(reference).all() and 0 < threshold < 28
        offsets = {view: e[f"offset_{view}"] for view in VIEWS}
    affected = {}
    for name, a in offsets.items():
        n = len(eligible) if name.startswith("shifted") else len(y)
        assert a.dtype == np.int8 and a.shape == (n, 2) and np.abs(a).max() <= 3
        affected[name] = 100 * np.any(a != 0, axis=1).mean()
    rows = []
    for seed in cfg["seeds"]:
        with np.load(results / f"circle_{seed}.npz") as pred:
            assert set(pred.files) == {"train", *(f"{m}_{v}" for m in MODES for v in VIEWS)}
            train = pred["train"]
            assert train.shape == train_y.shape and train.dtype == np.uint8 and train.max() < 10
            mapping = digit_mapping(train, train_y)
            for v in VIEWS:
                same = np.all(offsets[v] == 0, axis=1)
                np.testing.assert_array_equal(pred[f"raw_{v}"][same], pred[f"aligned_{v}"][same])
            for condition in CONDITIONS:
                view = condition.removeprefix("eligible_")
                labels = y if condition in ("clean", "marked") else y[eligible]
                values = {}
                for mode in MODES:
                    arr = pred[f"{mode}_{view}"]
                    if condition.startswith("eligible"):
                        arr = arr[eligible]
                    assert arr.shape == labels.shape and arr.dtype == np.uint8 and arr.max() < 10
                    values[mode] = float(100 * (mapping[arr] == labels).mean())
                rows.append(dict(condition=condition, seed=seed, **values, gain_pp=values["aligned"]-values["raw"]))
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "per_seed.csv", rows)
    summary = {}
    table = [r"\begin{tabular}{lrrr}\toprule", r"Input & Raw ACC & Corrected ACC & Change (points)\\\midrule"]
    for condition, label in CONDITIONS.items():
        subset = [r for r in rows if r["condition"] == condition]
        raw, corrected = [[r[m] for r in subset] for m in MODES]
        diff = paired(corrected, raw)
        summary[condition] = dict(raw_pct=float(np.mean(raw)), aligned_pct=float(np.mean(corrected)), **diff)
        table.append(f"{label} & {np.mean(raw):.2f} & {np.mean(corrected):.2f} & {diff['mean_pp']:+.2f}" + r"\\")
    table.append(r"\bottomrule\end{tabular}")
    (output / "table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
    rule = (summary["shifted_marked"]["wins"] >= 4 and
            min(summary["clean"]["mean_pp"], summary["marked"]["mean_pp"]) >= -1)
    save_json(output / "summary.json", {"eligible": len(eligible), "reference_yx": reference,
                                        "threshold_px": threshold, "actually_moved_pct": affected,
                                        "conditions": summary, "supports_position_option": rule})


def plot_examples(results, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from idec_final.data import load_mnist
    from varied_backgrounds import render, template_bank

    original, _, _, _ = load_mnist()
    with np.load(results / "evaluation.npz") as e:
        ids = e["eligible"][:2]
        clean = original[e["test_ids"][ids]]
        bank, _, _ = template_bank()
        marked = render(clean, e["symbol_boxes"][ids], e["symbol_types"][ids], bank).reshape(-1, 28, 28)
        shifts = e["directions"][ids]
        corrections = e["offset_shifted_marked"][:2]
    fig, axes = plt.subplots(2, 3, figsize=(5.1, 3.5), layout="constrained")
    for i, row in enumerate(axes):
        shifted = np.roll(marked[i], shifts[i], axis=(0, 1))
        corrected = np.roll(shifted, corrections[i], axis=(0, 1))
        for j, image in enumerate((marked[i], shifted, corrected)):
            ax = row[j]
            ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(("Centered", "Shifted", "Corrected")[j], fontsize=10)
            if j:
                dy, dx = (shifts[i] if j == 1 else corrections[i])
                ax.set_xlabel(f"({dy:+d}, {dx:+d}) px", fontsize=8)
    fig.savefig(output / "examples.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "position_alignment/results")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/position_scores")
    parser.add_argument("--figure", action="store_true", help="Render examples from the verified MNIST cache.")
    args = parser.parse_args()
    analyze(args.results, args.output)
    if args.figure:
        plot_examples(args.results, args.output)
