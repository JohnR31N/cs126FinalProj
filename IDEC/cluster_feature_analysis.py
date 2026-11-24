"""Score the feature selection experiment using only post-fit digit mappings and saved assignments."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import normalized_digest
from cluster_feature_probes import STYLE_NAMES, assignment_style_scores, fit_probes, score_probes
from cluster_feature_selection import FEATURE_NAMES
from cluster_feature_study import ARMS, DEC_FORKS, SOURCES, SPLITS
from idec_final.utils import ROOT, save_json
from photometric_analysis import digit_mapping


def check_runs(out, training_source=None):
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["arms"] == list(ARMS) and manifest["dec_forks"] == list(DEC_FORKS)
    assert manifest["seeds"] == list(range(5))
    assert set(manifest["source_sha256_lf"]) == set(SOURCES)
    for key, digest in manifest["source_sha256_lf"].items():
        assert len(bytes.fromhex(digest)) == 32
        if training_source is not None:
            assert normalized_digest(training_source / key) == digest, key
    expected = {f"{n}_{s}" for n in ARMS for s in range(5)} | {"evaluation", "selector", "frozen"}
    found = {p.stem for p in out.glob("*.npz")} - {"probes"}
    if found != expected:
        raise ValueError(f"Incomplete run collection. Missing {sorted(expected-found)}, extra {sorted(found-expected)}")


def eval_labels(out):
    with np.load(out / "evaluation.npz") as meta:
        assert set(meta.files) == {"train_ids", "extra_ids", "train_digit", "test_digit", "extra_digit"}
        labels = {s: meta[s+"_digit"] for s in SPLITS}
        order = np.random.default_rng(126).permutation(60000)
        np.testing.assert_array_equal(np.r_[meta["train_ids"], meta["extra_ids"]], order)
    assert [len(labels[s]) for s in SPLITS] == [12000, 10000, 48000]
    return labels


def score_assignments(p, prefix, labels):
    arrays = {s: p[f"{prefix}_{s}"] for s in SPLITS}
    for s, values in arrays.items():
        assert values.shape == labels[s].shape and values.dtype == np.uint8 and values.max() < 10
    mapping = digit_mapping(arrays["train"], labels["train"])
    results = {}
    for s in ("test", "extra"):
        pred, truth = arrays[s], labels[s]
        results[s] = {"acc_pct": float(100*np.mean(mapping[pred] == truth)),
                      "nmi": float(normalized_mutual_info_score(truth, pred)),
                      "ari": float(adjusted_rand_score(truth, pred)), "occupied": int(np.unique(pred).size)}
    return results


def analyze(out):
    check_runs(out)
    labels = eval_labels(out)
    probes, tails = score_probes(out, labels)
    with np.load(out / "selector.npz") as selector:
        assert selector["first"].shape == selector["second"].shape == (12000,)
        agree = selector["first"] == selector["second"]
        np.testing.assert_array_equal(agree, selector["agreed"])
        sizes = np.bincount(selector["first"][agree], minlength=10)
        assert sizes.min() >= 2
        for name in ("selected", "pca", "permuted"):
            basis = selector[name].astype(np.float64)
            assert basis.shape == (392, 9)
            np.testing.assert_allclose(basis.T @ basis, np.eye(9), atol=1e-6, rtol=0)
        selection = {"agreed": int(agree.sum()), "total": len(agree), "group_sizes": sizes.tolist(),
                     "fisher_eigenvalues": selector["eigenvalues"].tolist()}
        cross_fold = np.r_[selector["second"][:6000], selector["first"][6000:]]
    with np.load(out / "frozen.npz") as frozen_file:
        names = ("graph", *FEATURE_NAMES)
        assert set(frozen_file.files) == {f"{n}_{s}" for n in names for s in SPLITS}
        np.testing.assert_array_equal(frozen_file["graph_train"], cross_fold)
        frozen = {name: score_assignments(frozen_file, name, labels) for name in names}
    with np.load(out / "probes.npz") as p:
        residuals = {s: p[s+"_style"].astype(np.float64) for s in SPLITS}
    rows, scores, decision_styles = [], {}, {}
    for arm in ARMS:
        stages = ("initial", "idec", "dec") if arm in DEC_FORKS else ("initial", "idec")
        for seed in range(5):
            with np.load(out / f"{arm}_{seed}.npz") as saved:
                assert set(saved.files) == {f"{stage}_{s}" for stage in stages for s in SPLITS}
                decision_styles.setdefault(arm, []).append(assignment_style_scores(saved, labels, residuals))
                for stage in stages:
                    by_split = score_assignments(saved, stage, labels)
                    for s, values in by_split.items():
                        rows.append({"target": arm, "stage": stage, "seed": seed, "split": s, **values})
                        scores.setdefault((arm, stage, s), []).append(values)
    means, paired = {}, {}
    for s in ("test", "extra"):
        means[s] = {}
        for (arm, stage, split), runs in scores.items():
            if split == s:
                assert len(runs) == 5
                means[s][f"{arm}_{stage}"] = {k: float(np.mean([r[k] for r in runs]))
                                               for k in ("acc_pct", "nmi", "ari")}
                means[s][f"{arm}_{stage}"]["min_occupied"] = min(r["occupied"] for r in runs)
        selected = np.array([r["acc_pct"] for r in scores["selected", "idec", s]])
        paired[s] = {}
        for arm, stage in (("pixels", "idec"), ("all_strokes", "idec"), ("complement", "idec"),
                           ("pca", "idec"), ("permuted", "idec"), ("selected", "initial"),
                           ("selected", "dec"), ("pixels", "dec")):
            diff = selected - np.array([r["acc_pct"] for r in scores[arm, stage, s]])
            paired[s][f"selected_idec_minus_{arm}_{stage}"] = {
                "mean_pp": float(diff.mean()), "sd_pp": float(diff.std(ddof=1)),
                "wins": int((diff > 0).sum()), "by_seed_pp": diff.tolist()}
    style_by_split = {s: {arm: dict(zip(STYLE_NAMES, np.mean([r[s] for r in trials], axis=0).tolist()))
                          for arm, trials in decision_styles.items()} for s in ("test", "extra")}
    return {"selection": selection, "means": means, "paired": paired, "frozen": frozen, "probes": probes,
            "assignment_style_r2": style_by_split, "training_tail_digit_counts": tails}, rows


def compare(current, stored):
    if isinstance(current, dict):
        assert current.keys() == stored.keys()
        for key in current:
            compare(current[key], stored[key])
    elif isinstance(current, list):
        assert len(current) == len(stored)
        for actual, saved in zip(current, stored):
            compare(actual, saved)
    else:
        np.testing.assert_allclose(current, stored, atol=1e-10, rtol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "cluster_features" / "results")
    parser.add_argument("--prepare-probes", action="store_true", help="Fit post-training evaluation probes on original MNIST")
    parser.add_argument("--verify", action="store_true", help="Compare rescored outputs with the stored JSON and CSV")
    parser.add_argument("--training-source", type=Path, help="Verify the manifest against the IDEC directory used for fitting")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        check_runs(args.input, args.training_source)
        if args.prepare_probes:
            fit_probes(args.input, torch.device(args.device))
        result, rows = analyze(args.input)
        summary = args.input / "summary.json"
        csv_path = args.input / "runs.csv"
        if args.verify:
            compare(result, json.loads(summary.read_text()))
            with csv_path.open(newline="", encoding="utf-8") as f:
                stored = list(csv.DictReader(f))
            assert len(stored) == len(rows)
            for actual, saved in zip(rows, stored):
                assert actual.keys() == saved.keys()
                for k, v in actual.items():
                    if isinstance(v, str):
                        assert v == saved[k]
                    else:
                        np.testing.assert_allclose(v, float(saved[k]), atol=1e-10, rtol=0)
        else:
            save_json(summary, result)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
        for s in ("test", "extra"):
            print(s, {n: round(result["means"][s][n+"_idec"]["acc_pct"], 3) for n in ARMS})
