"""Recompute external held-out scores and the full-MNIST comparison from saved cluster IDs."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from analyze import csv_write
from clean_neighborhood_study import normalized_digest
from external_data import FILES, SIZES
from external_study import SOURCES
from idec_final.utils import ROOT, save_json
from photometric_analysis import digit_mapping


def vector(values, count):
    if values.dtype != np.uint8 or values.shape != (count,) or values.max() > 9:
        raise ValueError("Invalid cluster or class IDs")
    return values


def metrics(y, predicted, mapping):
    return {"acc_pct": float(100 * np.mean(mapping[predicted] == y)),
            "nmi": float(normalized_mutual_info_score(y, predicted)),
            "ari": float(adjusted_rand_score(y, predicted)),
            "occupied": int(len(np.unique(predicted)))}


def check_manifest(folder, dataset, sources):
    m = json.loads((folder / "manifest.json").read_text())
    assert m["dataset"] == dataset and set(m["source_sha256_lf"]) == set(sources)
    for name in sources:
        assert m["source_sha256_lf"][name] == normalized_digest(ROOT / name), name
    return m


def check_selector(folder, n):
    with np.load(folder / "selector.npz") as s:
        basis = s["selected"]
        assert basis.shape == (392, 9)
        np.testing.assert_allclose(basis.T @ basis, np.eye(9), atol=1e-6)
        first, second = vector(s["first"], n), vector(s["second"], n)
        np.testing.assert_array_equal(s["agreed"], first == second)
        counts = np.bincount(first[first == second], minlength=10)
        assert counts.min() >= 2 and len(s["eigenvalues"]) == 9 and s["eigenvalues"].min() > 0
        return {"agreed": int(counts.sum()), "group_sizes": counts.tolist()}, np.r_[second[:n//2], first[n//2:]]


def paired(left, right):
    delta = np.asarray(left) - np.asarray(right)
    return {"mean_pp": float(delta.mean()), "sd_pp": float(delta.std(ddof=1)),
            "wins": int((delta > 0).sum()), "ties": int((delta == 0).sum()), "per_seed_pp": delta.tolist()}


def score_dataset(folder, dataset):
    m = check_manifest(folder, dataset, SOURCES)
    cfg = m["config"]
    assert cfg == json.loads((ROOT / "protocol.json").read_text()) and m["file_md5"] == FILES[dataset]
    ntrain, ntest = SIZES[dataset]
    ids = np.random.default_rng(cfg["split_seed"]).permutation(ntrain)[:cfg["train_size"]]
    n = len(ids)
    with np.load(folder / "evaluation.npz") as e:
        assert set(e.files) == {"train_ids", "train_class", "test_class"}
        np.testing.assert_array_equal(e["train_ids"], ids)
        train_y, test_y = vector(e["train_class"], n), vector(e["test_class"], ntest)
        assert len(np.unique(train_y)) == len(np.unique(test_y)) == 10
    def heldout(train_cluster, test_cluster):
        mapping = digit_mapping(vector(train_cluster, n), train_y)
        return metrics(test_y, vector(test_cluster, ntest), mapping)

    rows, by_method, acc = [], {}, {}
    desc = {"n_fit": n, "n_test": ntest}
    arms = ["pixels"]
    if (folder / "unsupported.json").exists():
        desc["unsupported"] = json.loads((folder / "unsupported.json").read_text())["reason"]
        assert not (folder / "selector.npz").exists() and not list(folder.glob("selected_*.npz"))
        info = json.loads((folder / "graph_audit.json").read_text())
        assert info["source_sha256_lf"] == normalized_digest(ROOT / "external_graph_audit.py")
        with np.load(folder / "graph_groups.npz") as groups:
            first, second = vector(groups["first"], n), vector(groups["second"], n)
            counts = np.bincount(first[first == second], minlength=10)
            np.testing.assert_array_equal(counts, groups["agreed_group_sizes"])
            assert info["group_sizes"] == counts.tolist() and info["agreed"] == counts.sum()
            if "at least two" in desc["unsupported"]:
                assert counts.min() < 2
            graph_train = np.r_[second[:n//2], first[n//2:]]
        desc["partition"] = {"agreed": info["agreed"], "group_sizes": info["group_sizes"]}
    else:
        arms.append("selected")
        selection, graph_train = check_selector(folder, n)
        desc["selection"] = selection
    with np.load(folder / "graph.npz") as graph:
        np.testing.assert_array_equal(graph["train"], graph_train)
        desc["graph"] = heldout(graph["train"], graph["test"])
    rows.append(dict(dataset=dataset, target="graph", stage="vote", seed="", **desc["graph"]))
    keys = {f"{stage}_{s}" for stage in ("initial", "dec", "idec") for s in ("train", "test")}
    for arm in arms:
        assert {p.name for p in folder.glob(f"{arm}_*.npz")} == {f"{arm}_{k}.npz" for k in cfg["seeds"]}
        for seed in cfg["seeds"]:
            with np.load(folder / f"{arm}_{seed}.npz") as saved:
                assert set(saved.files) == keys
                for stage in ("initial", "dec", "idec"):
                    score = heldout(saved[stage + "_train"], saved[stage + "_test"])
                    rows.append(dict(dataset=dataset, target=arm, stage=stage, seed=seed, **score))
                    by_method.setdefault(f"{arm}_{stage}", []).append(score)
    desc["methods"] = {}
    for method, runs in by_method.items():
        acc[method] = [r["acc_pct"] for r in runs]
        desc["methods"][method] = {key: float(np.mean([r[key] for r in runs])) for key in ("acc_pct", "nmi", "ari")}
        desc["methods"][method].update(acc_sd=float(np.std(acc[method], ddof=1)),
                                       min_occupied=min(r["occupied"] for r in runs))
    if "selected" in arms:
        desc["selected_idec_minus"] = {k: paired(acc["selected_idec"], acc[k])
                                       for k in ("pixels_idec", "pixels_dec", "selected_dec")}
    return desc, rows


def score_full(folder):
    from idec_final.data import FILES as mnist_files
    from reproduce_paper import DIGEST
    sources = (*SOURCES, "mnist_full_study.py", "reproduce_paper.py", "idec_final/data.py")
    m = check_manifest(folder, "mnist_full", sources)
    cfg = json.loads((ROOT / "protocol.json").read_text())
    cfg.update(seeds=[0], train_size=70000, updates=20000)
    assert m["config"] == cfg and m["file_md5"] == mnist_files and m["author_weight_sha256"] == DIGEST
    selection, _ = check_selector(folder, 70000)
    result = {"n_fit_and_score": 70000, "seed": 0, "selection": selection, "arms": {}}
    with np.load(folder / "evaluation.npz") as e, np.load(ROOT / "reference/predictions.npz") as reference:
        y = vector(e["labels"], 70000)
        np.testing.assert_array_equal(reference["digits"], y)
        author_initial = None
        for recipe in ("author", "scratch"):
            for name in ("pixels", "selected"):
                arm = f"{recipe}_{name}"
                result["arms"][arm] = {}
                with np.load(folder / f"{arm}.npz") as saved:
                    assert set(saved.files) == {"initial", "final"}
                    for stage in ("initial", "final"):
                        ids = vector(saved[stage], 70000)
                        result["arms"][arm][stage] = metrics(y, ids, digit_mapping(ids, y))
                    if arm == "author_pixels":
                        author_initial = saved["initial"]
                        result["author_pixel_vs_archived_ari"] = {
                            stage: float(adjusted_rand_score(reference[stage], saved[stage])) for stage in saved.files}
                    if arm == "author_selected":
                        np.testing.assert_array_equal(author_initial, saved["initial"])
        old = reference["final"]
        result["archived_pixel"] = metrics(y, old, digit_mapping(old, y))
        old_map = digit_mapping(old[:60000], y[:60000])
        result["archived_last_10000_already_in_fit"] = metrics(y[60000:], old[60000:], old_map)
    recorded = json.loads((ROOT / "reference/result.json").read_text())
    np.testing.assert_allclose(result["archived_pixel"]["acc_pct"], 100 * recorded["final"]["acc"], atol=1e-9)
    for recipe in ("author", "scratch"):
        arms = result["arms"]
        result[f"{recipe}_selected_minus_pixel_pp"] = arms[recipe + "_selected"]["final"]["acc_pct"] - arms[recipe + "_pixels"]["final"]["acc_pct"]
    return result


def archived_heldout():
    studies = {"background_pixel": [], "feature_pixel": [], "feature_selected": []}
    with np.load(ROOT / "varied/results/evaluation.npz") as a, np.load(ROOT / "cluster_features/results/evaluation.npz") as b:
        for field in ("train_ids", "train_digit", "test_digit"):
            np.testing.assert_array_equal(a[field], b[field])
        train_y, test_y = a["train_digit"], a["test_digit"]
    for seed in range(5):
        sources = {"background_pixel": (ROOT / f"dec_background/results/clean_{seed}/predictions.npz", "idec_clean"),
                   "feature_pixel": (ROOT / f"cluster_features/results/pixels_{seed}.npz", "idec_test"),
                   "feature_selected": (ROOT / f"cluster_features/results/selected_{seed}.npz", "idec_test")}
        for name, (path, key) in sources.items():
            with np.load(path) as result:
                mapping = digit_mapping(result["idec_train"], train_y)
                studies[name].append(float(100 * np.mean(mapping[result[key]] == test_y)))
    return {k: {"mean_test_acc_pct": float(np.mean(v)), "per_seed_pct": v} for k, v in studies.items()}


def analyze(results, output):
    by_dataset, rows = {}, []
    for dataset in SIZES:
        by_dataset[dataset], seed_rows = score_dataset(results / dataset, dataset)
        rows.extend(seed_rows)
    full = score_full(results / "mnist_full")
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "summary.json", {"external": by_dataset, "mnist_full": full, "mnist_heldout": archived_heldout()})
    csv_write(output / "runs.csv", rows)
    labels = {"usps": "USPS digits", "kmnist": "Kuzushiji-MNIST", "fashion": "Fashion-MNIST",
              "cifar10_gray": "CIFAR-10, grayscale"}
    lines = [r"\begin{tabular}{lrrrrr}\toprule",
             r"Dataset & DEC & Pixel IDEC & IDEC+ & Difference (points) & Wins\\\midrule"]
    for dataset, data in by_dataset.items():
        methods = data["methods"]
        selected = methods.get("selected_idec")
        delta = data.get("selected_idec_minus", {}).get("pixels_idec")
        columns = (labels[dataset], f"{methods['pixels_dec']['acc_pct']:.2f}", f"{methods['pixels_idec']['acc_pct']:.2f}",
                   f"{selected['acc_pct']:.2f}" if selected else "--",
                   f"{delta['mean_pp']:+.2f}" if delta else "--", f"{delta['wins']}/5" if delta else "--")
        lines.append(" & ".join(columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "heldout_table.tex").write_text("\n".join(lines) + "\n", encoding="ascii")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "external_benchmarks/results")
    parser.add_argument("--output", type=Path, default=ROOT / "external_benchmarks")
    args = parser.parse_args()
    analyze(args.results, args.output)
