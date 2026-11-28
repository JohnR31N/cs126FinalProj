"""Re-fit both brightness readouts on CPU and score the archived, frozen embeddings."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from analyze import csv_write
from background_probe_study import ARMS, CALIBRATION, DOMAINS, MAPS, SOURCES
from external_analysis import check_manifest
from idec_final.data import FILES
from idec_final.utils import ROOT, sha256
from photometric_analysis import digit_mapping


def ranking_pct(margins):
    return float(100 * ((margins > 0).mean() + 0.5 * (margins == 0).mean()))


def readouts(dim, bright, clean, ncal):
    scale = StandardScaler().fit(np.concatenate((dim[:ncal], bright[:ncal])))
    lo, hi = scale.transform(dim), scale.transform(bright)
    delta = hi - lo
    labels = np.repeat([0, 1], ncal)
    paired = LogisticRegression(C=1., fit_intercept=False, max_iter=1000)
    paired.fit(np.concatenate((-delta[:ncal], delta[:ncal])), labels)
    margins = paired.decision_function(delta[ncal:])
    null_rng = np.random.default_rng(8103)
    signs = null_rng.choice([-1, 1], size=(200, len(margins)))
    null = ranking_pct(signs * margins)
    identical = ranking_pct(paired.decision_function(np.zeros_like(delta[ncal:])))
    assert identical == 50.

    single = LogisticRegression(C=1., max_iter=1000)
    single.fit(np.concatenate((lo[:ncal], hi[:ncal])), labels)
    accuracy = 50 * ((single.predict(lo[ncal:]) == 0).mean() + (single.predict(hi[ncal:]) == 1).mean())

    cov = np.cov(clean[:ncal], rowvar=False)
    eig, axes = np.linalg.eigh(cov)
    ridge = 1e-4 * eig.mean()
    if ridge <= 0:
        raise ValueError("Calibration clean embeddings have zero overall variance")
    whitened = ((bright[ncal:] - dim[ncal:]) @ axes) / np.sqrt(eig + ridge)
    distance = np.linalg.norm(whitened, axis=1).mean()
    return dict(paired_pct=ranking_pct(margins), single_pct=float(accuracy), code_distance=float(distance),
                permuted_pct=null, identical_pct=identical)


def scores(saved, fit_digit, probe_digit):
    assert set(saved.files) == {"dim", "bright", "clean", "centers", "fit_cluster"}
    assert saved["fit_cluster"].shape == fit_digit.shape and saved["fit_cluster"].max() < 10
    centers = saved["centers"].astype(np.float64)
    assert centers.shape == (10, 10) and np.isfinite(centers).all()
    codes = {k: saved[k].astype(np.float64) for k in ("clean", "dim", "bright")}
    assert all(z.shape == (len(probe_digit), 10) and np.isfinite(z).all() for z in codes.values())
    outcome = readouts(codes["dim"], codes["bright"], codes["clean"], CALIBRATION)
    mapping = digit_mapping(saved["fit_cluster"], fit_digit)
    clusters = {key: ((z[CALIBRATION:, None] - centers) ** 2).sum(axis=2).argmin(axis=1)
                for key, z in codes.items()}
    truth = probe_digit[CALIBRATION:]
    acc = {key: float(100 * (mapping[c] == truth).mean()) for key, c in clusters.items()}
    return dict(**outcome, clean_acc_pct=acc["clean"], dim_acc_pct=acc["dim"],
                bright_acc_pct=acc["bright"], marked_acc_pct=(acc["dim"] + acc["bright"]) / 2,
                flip_pct=float(100 * (clusters["dim"] != clusters["bright"]).mean()))


def report(results, output):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    m = check_manifest(results, "mnist_background_brightness", SOURCES)
    assert m["config"] == cfg and m["file_md5"] == FILES and m["calibration"] == CALIBRATION
    assert m["arms"] == list(ARMS) and m["domains"] == list(DOMAINS) and cfg["seeds"] == list(range(5))
    assert m["map_sha256"] == {d: sha256(MAPS / f"{d}_maps.npz") for d in DOMAINS}
    assert m["split_sha256"] == sha256(ROOT / "varied/results/evaluation.npz")
    expected = {f"{d}_{a}_{s}.npz" for d in DOMAINS for a in ARMS for s in cfg["seeds"]}
    assert {p.name for p in results.glob("*.npz")} == expected | {"evaluation.npz"}
    with np.load(results / "evaluation.npz") as e:
        order = np.random.default_rng(cfg["split_seed"]).permutation(60000)
        ids = order[32000:40000].copy()
        slots = e["replaced_slots"]
        np.testing.assert_array_equal(ids[slots], e["excluded_ids"])
        assert set(e["probe_ids"][slots]).issubset(order[40000:])
        ids[slots] = e["probe_ids"][slots]
        np.testing.assert_array_equal(ids, e["probe_ids"])
        np.testing.assert_array_equal(order[:cfg["train_size"]], e["fit_ids"])
        assert len(np.unique(ids)) == 8000 and not np.intersect1d(ids, e["fit_ids"]).size
        assert e["boxes"].shape == (8000, 2, 2) and e["dim_types"].shape == (8000, 2)
        assert e["boxes"].max() <= 23 and (e["dim_types"] % 3 == 0).all() and e["dim_types"].max() < 96
        assert e["pixel_intensity_gap"].shape == (8000,) and (e["pixel_intensity_gap"] > 0).all()
        fit_digit, probe_digit = e["fit_digit"], e["probe_digit"]
        assert fit_digit.shape == (12000,) and probe_digit.shape == (8000,)
        with np.load(ROOT / "varied/results/evaluation.npz") as ref:
            np.testing.assert_array_equal(fit_digit, ref["train_digit"])

    rows, means, contrasts = [], [], []
    for domain in DOMAINS:
        arms = {}
        for arm in ARMS:
            arm_rows = []
            for seed in cfg["seeds"]:
                with np.load(results / f"{domain}_{arm}_{seed}.npz") as emb:
                    out = scores(emb, fit_digit, probe_digit)
                rows.append(dict(domain=domain, arm=arm, seed=seed, **out))
                arm_rows.append(out)
            series = {k: np.array([row[k] for row in arm_rows]) for k in arm_rows[0]}
            means.append(dict(domain=domain, arm=arm, **{k: v.mean() for k, v in series.items()}))
            arms[arm] = series
        for key in arms["adaptive"]:
            diff = arms["adaptive"][key] - arms["uniform"][key]
            contrasts.append(dict(domain=domain, metric=key, mean=float(diff.mean()),
                                  sample_sd=float(diff.std(ddof=1)), lower=int((diff < 0).sum()),
                                  higher=int((diff > 0).sum()), **{f"seed_{s}": d for s, d in enumerate(diff)}))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in (("per_seed", rows), ("summary", means), ("paired", contrasts)):
        csv_write(output / f"{name}.csv", data)
    for row in means:
        print(row["domain"], row["arm"], {k: round(float(v), 4) for k, v in row.items() if k not in ("domain", "arm")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "background_probe/results")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/brightness_scores")
    args = parser.parse_args()
    with threadpool_limits(1):
        report(args.results, args.output)
