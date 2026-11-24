"""Recompute the supported-rank CIFAR-10 comparison using archived per-image predictions."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from cifar_fisher_study import ARMS, FOLLOWUP, REFERENCE, TRAIN_SOURCES, reference_files
from external_analysis import check_manifest, metrics, paired, vector
from idec_final.utils import ROOT, save_json
from photometric_analysis import digit_mapping


def score(results, output):
    manifest = check_manifest(results, "cifar10_gray", TRAIN_SOURCES)
    cfg = json.loads((ROOT / "protocol.json").read_text())
    seeds = cfg["seeds"]
    assert manifest["config"] == cfg and manifest["arms"] == list(ARMS) and seeds == list(range(5))
    assert manifest["reference_commit"] == "7ffbcc8abe7566175bef27e48db5e0ac8312d9db"
    assert manifest["reference_sha256"] == reference_files(seeds)
    with np.load(REFERENCE / "evaluation.npz") as e:
        expected = np.random.default_rng(cfg["split_seed"]).permutation(50000)[:cfg["train_size"]]
        np.testing.assert_array_equal(e["train_ids"], expected)
        train_y, test_y = vector(e["train_class"], len(expected)), vector(e["test_class"], 10000)
    with np.load(results / "bases.npz") as bases, np.load(REFERENCE / "graph_groups.npz") as graph:
        selected = graph["first"][graph["first"] == graph["second"]]
        groups, sizes = np.unique(selected, return_counts=True)
        rank = len(groups)-1
        np.testing.assert_array_equal(bases["group_ids"], groups)
        np.testing.assert_array_equal(bases["group_sizes"], sizes)
        assert len(selected) == 5881 and rank == 8 and sizes.min() >= 2
        assert bases["eigenvalues"].shape == (rank,) and np.all(bases["eigenvalues"] > 0)
        for arm in ARMS:
            basis = bases[arm]
            assert basis.shape == (392, rank)
            np.testing.assert_allclose(basis.T @ basis, np.eye(rank), atol=1e-6)
    rows, by_arm, acc = [], {}, {}
    expected_keys = {f"{stage}_{split}" for stage in ("initial", "idec") for split in ("train", "test")}
    for arm in ("pixels", *ARMS):
        source = REFERENCE if arm == "pixels" else results
        assert {p.name for p in source.glob(f"{arm}_*.npz")} == {f"{arm}_{s}.npz" for s in seeds}
        by_arm[arm] = {}
        for seed in seeds:
            with np.load(source / f"{arm}_{seed}.npz") as prediction:
                if arm != "pixels":
                    assert set(prediction.files) == expected_keys
                for stage in ("initial", "idec"):
                    train = vector(prediction[stage + "_train"], len(train_y))
                    test = vector(prediction[stage + "_test"], len(test_y))
                    values = metrics(test_y, test, digit_mapping(train, train_y))
                    rows.append(dict(target=arm, stage=stage, seed=seed, **values))
                    by_arm[arm].setdefault(stage, []).append(values)
        acc[arm] = [v["acc_pct"] for v in by_arm[arm]["idec"]]
    means = {}
    for name, stages in by_arm.items():
        means[name] = {}
        for stage, values in stages.items():
            means[name][stage] = {m: float(np.mean([v[m] for v in values])) for m in ("acc_pct", "nmi", "ari")}
            means[name][stage]["acc_sd"] = float(np.std([v["acc_pct"] for v in values], ddof=1))
            means[name][stage]["min_occupied"] = min(v["occupied"] for v in values)
    old = json.loads((ROOT / "external_benchmarks/summary.json").read_text())["external"]["cifar10_gray"]
    for m in ("acc_pct", "nmi", "ari"):
        np.testing.assert_allclose(means["pixels"]["idec"][m], old["methods"]["pixels_idec"][m])
    with np.load(REFERENCE / "graph.npz") as g:
        mapping = digit_mapping(g["train"], train_y)
        airplane_group = int(np.flatnonzero(mapping == 0)[0])
        class_counts = np.bincount(test_y[g["test"] == airplane_group], minlength=10)
    summary = {"n_fit": len(train_y), "n_test": len(test_y), "n_consensus": len(selected), "target_rank": rank,
               "student_clusters": cfg["clusters"], "seeds": seeds, "means": means,
               "fisher_minus_pixels": paired(acc["fisher"], acc["pixels"]),
               "fisher_minus_pca": paired(acc["fisher"], acc["pca"]),
               "graph_cluster_mapped_to_airplane": {"cluster_id": airplane_group, "n_test": int(class_counts.sum()),
                                                     "test_counts_official_class_order": class_counts.tolist()}}
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "runs.csv", rows)
    save_json(output / "summary.json", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=FOLLOWUP / "results")
    parser.add_argument("--output", type=Path, default=FOLLOWUP)
    args = parser.parse_args()
    score(args.results, args.output)
