"""Rescore literal bin selection and matched controls using their saved original-MNIST assignments."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from cluster_coordinate_study import CHOICES, TRAINED
from cluster_feature_analysis import compare, eval_labels, score_assignments
from cluster_feature_study import SPLITS
from idec_final.utils import ROOT, save_json, sha256


def analyze(parent, out):
    labels = eval_labels(parent)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["seeds"] == list(range(5)) and manifest["trained"] == list(TRAINED)
    assert manifest["frozen"] == list(CHOICES)
    for k in ("selector", "evaluation"):
        assert sha256(parent/f"{k}.npz") == manifest[f"parent_{k}_sha256"]
    expected = {f"{arm}_{seed}" for arm in TRAINED for seed in range(5)} | {"bins", "frozen"}
    assert {p.stem for p in out.glob("*.npz")} == expected
    with np.load(out / "bins.npz") as selected:
        assert set(selected.files) == {"half_a", "half_b", "score", *CHOICES}
        shares = selected["score"]
        np.testing.assert_array_equal(shares, np.minimum(selected["half_a"], selected["half_b"]))
        assert shares.shape == (392,) and shares.min() >= 0 and shares.max() <= 1+1e-9
        ranked = np.lexsort((np.arange(392), -shares))
        np.testing.assert_array_equal(selected["top"], np.sort(ranked[:98]))
        np.testing.assert_array_equal(selected["bottom"], np.sort(ranked[-98:]))
        flags = {}
        for name in CHOICES:
            ix = selected[name]
            assert ix.shape == (98,) and np.unique(ix).size == 98 and ix.min() >= 0 and ix.max() < 392
            flags[name] = set(ix.tolist())
        top, matched = selected["top"], selected["location_matched"]
        np.testing.assert_array_equal(np.bincount(top//8, minlength=49), np.bincount(matched//8, minlength=49))
        coverage = {"kept": 98, "total": 392, "cells": len(np.unique(top//8)),
                    "matched_overlap": len(flags["top"] & flags["location_matched"]),
                    "top_five_ids": ranked[:5].tolist(), "top_five_scores": shares[ranked[:5]].tolist()}
        with (out / "bins.csv").open(newline="", encoding="utf-8") as f:
            table = list(csv.DictReader(f))
        assert len(table) == 392
        for i, row in enumerate(table):
            assert (int(row["id"]), int(row["cell_row"]), int(row["cell_col"])) == (i, i//56, i//8%7)
            assert float(row["gradient_degrees"]) == i%8*22.5
            for name in CHOICES:
                assert int(row[name]) == int(i in flags[name])
            for column, key in (("repeatable_share", "score"), ("half_a", "half_a"), ("half_b", "half_b")):
                assert np.isclose(float(row[column]), selected[key][i], rtol=0, atol=1e-14)
    with np.load(out / "frozen.npz") as p:
        assert set(p.files) == {f"{n}_{s}" for n in CHOICES for s in SPLITS}
        frozen = {n: score_assignments(p, n, labels) for n in CHOICES}
    runs = {}
    for name in TRAINED:
        runs[name] = []
        for seed in range(5):
            with np.load(out / f"{name}_{seed}.npz") as p:
                assert set(p.files) == {f"{stage}_{s}" for stage in ("initial", "idec") for s in SPLITS}
                runs[name].append({stage: score_assignments(p, stage, labels) for stage in ("initial", "idec")})
    uniform = []
    for seed in range(5):
        with np.load(parent / f"pixels_{seed}.npz") as p:
            uniform.append(score_assignments(p, "idec", labels))
    means, effects = {}, {}
    for s in ("test", "extra"):
        means[s] = {n: {k: float(np.mean([r["idec"][s][k] for r in runs[n]])) for k in ("acc_pct", "nmi", "ari")}
                    for n in TRAINED}
        acc = {n: np.array([r["idec"][s]["acc_pct"] for r in runs[n]]) for n in TRAINED}
        acc["pixels"] = np.array([r[s]["acc_pct"] for r in uniform])
        effects[s] = {}
        for n in ("location_matched", "bottom", "pixels"):
            diff = acc["top"] - acc[n]
            effects[s][f"top_minus_{n}"] = {"mean_pp": float(diff.mean()), "sd_pp": float(diff.std(ddof=1)),
                                           "wins": int((diff > 0).sum()), "by_seed_pp": diff.tolist()}
    return {"coverage": coverage, "means": means, "paired": effects, "frozen": frozen, "runs": runs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=ROOT / "cluster_features" / "results")
    parser.add_argument("--input", type=Path, default=ROOT / "cluster_features" / "coordinates")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = analyze(args.parent, args.input)
    path = args.input / "summary.json"
    if args.verify:
        compare(result, json.loads(path.read_text()))
    else:
        save_json(path, result)
    print(json.dumps(result["means"], indent=2))
