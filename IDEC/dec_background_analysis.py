"""Score matched clean/marked fits and compute the paired difference in ACC losses."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from analyze_stages import matches_training_digest
from dec_background_study import DOMAINS, GROUPS, SOURCES
from idec_final.utils import ROOT, save_json
from photometric_analysis import digit_mapping


METHODS = ("dec", "idec", "gaussian_both", "gaussian_pretrain", "gaussian_finetune", "kl_from_scratch")
INITIAL = tuple("init_" + name for name in GROUPS)
NAMES = {"dec": "DEC", "idec": "IDEC", "gaussian_both": "Gaussian, both stages",
         "gaussian_pretrain": "Gaussian, pretraining only", "gaussian_finetune": "Gaussian, fine-tuning only",
         "kl_from_scratch": "KL only, random start", "init_uniform": "AE + K-means, uniform",
         "init_gaussian": "AE + K-means, Gaussian", "init_random": "K-means, random encoder"}


def percent(mask):
    return float(100 * np.mean(mask))


def stats(data):
    values = np.asarray(data, dtype=np.float64)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
            "positive": int((values > 1e-9).sum()), "negative": int((values < -1e-9).sum()),
            "differences": values.tolist()}


def verify_cell(record, cfg, seed, domain):
    assert (record["seed"], record["domain"]) == (seed, domain)
    assert set(record["initializations"]) == set(GROUPS) and set(record["methods"]) == set(METHODS)
    steps = list(range(0, cfg["updates"] + 1, cfg["target_interval"]))
    for init, methods in GROUPS.items():
        signature = record["initializations"][init]
        assert len(signature) == 64
        for name, loss in methods.items():
            branch = record["methods"][name]
            assert (branch["init"], branch["loss"], branch["start_sha256"]) == (init, loss, signature)
            assert [p["step"] for p in branch["trace"]] == steps


def scores(pred, name, domain, train_y, test_y, valid):
    train = pred[name + "_train"]
    clean = pred[name + "_clean"]
    views = pred[name + "_backgrounds"]
    assert train.shape == train_y.shape and clean.shape == test_y.shape
    assert views.shape == (3, len(test_y))
    for array in (train, clean, views):
        assert array.dtype == np.uint8 and array.max() < 10
    mapping = digit_mapping(train, train_y)
    clean_acc = percent(mapping[clean] == test_y)
    marked_acc = percent(mapping[views[0]] == test_y)
    observed = clean if domain == "clean" else views[0]
    result = {"domain_acc_pct": clean_acc if domain == "clean" else marked_acc,
              "clean_test_acc_pct": clean_acc, "marked_test_acc_pct": marked_acc,
              "frozen_test_drop_pp": clean_acc - marked_acc,
              "redraw_flip_pct": percent(views[1:] != views[0]),
              "occupied": len(np.unique(observed)), "largest_cluster_pct": 100 * np.bincount(observed).max() / len(observed),
              "decoder_response": None}
    if name + "_response" in pred:
        values = pred[name + "_response"]
        assert values.shape == test_y.shape and np.isfinite(values).all()
        result["decoder_response"] = float(values[valid].mean(dtype=np.float64))
    return result


def collect(out):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"] == cfg and manifest["groups"] == GROUPS and manifest["domains"] == list(DOMAINS)
    assert set(manifest["source_sha256"]) == set(SOURCES)
    for filename, digest in manifest["source_sha256"].items():
        assert matches_training_digest(ROOT / filename, digest), filename
    seeds = cfg["seeds"]
    assert {p.name for p in out.iterdir() if p.is_dir()} == {f"{domain}_{seed}" for domain in DOMAINS for seed in seeds}
    rows = []
    with np.load(ROOT / "varied/results/evaluation.npz") as data:
        train_y, test_y = data["train_digit"], data["test_digit"]
        assert (len(train_y), len(test_y)) == (12000, 10000)
        types, bank = data["test_types"], data["bank"]
        valid = np.any(bank[types[0]] != bank[types[1]], axis=(1, 2, 3))
        expected = {name + "_" + split for name in METHODS + INITIAL for split in ("train", "clean", "backgrounds")}
        expected.update(name + "_response" for name in ("idec", "gaussian_both", "gaussian_pretrain", "gaussian_finetune"))
        expected.add("valid_response")
        for domain in DOMAINS:
            for seed in seeds:
                cell = out / f"{domain}_{seed}"
                assert {p.name for p in cell.iterdir()} == {"record.json", "predictions.npz"}
                verify_cell(json.loads((cell / "record.json").read_text()), cfg, seed, domain)
                with np.load(cell / "predictions.npz") as pred:
                    assert set(pred.files) == expected and pred["valid_response"].dtype == np.bool_
                    np.testing.assert_array_equal(pred["valid_response"], valid)
                    for name in METHODS + INITIAL:
                        rows.append({"method": name, "domain": domain, "seed": seed,
                                     **scores(pred, name, domain, train_y, test_y, valid)})
    return rows, seeds


def analyze(out):
    rows, seeds = collect(out)
    csv_write(out / "runs.csv", rows)
    lookup = {(r["method"], r["domain"], r["seed"]): r for r in rows}

    def values(method, domain, metric):
        return np.array([lookup[method, domain, seed][metric] for seed in seeds])

    def acc(method, domain):
        return values(method, domain, "domain_acc_pct")

    drops = {m: acc(m, "clean") - acc(m, "varied") for m in METHODS + INITIAL}
    summary = []
    for name, drop in drops.items():
        summary.append({
            "method": name,
            "clean_acc_pct": acc(name, "clean").mean(), "clean_acc_sd": acc(name, "clean").std(ddof=1),
            "marked_acc_pct": acc(name, "varied").mean(), "marked_acc_sd": acc(name, "varied").std(ddof=1),
            "drop_pp": drop.mean(), "drop_sd": drop.std(ddof=1), "drop_positive_seeds": stats(drop)["positive"],
            "redraw_flip_pct": values(name, "varied", "redraw_flip_pct").mean(),
            "frozen_clean_trained_drop_pp": values(name, "clean", "frozen_test_drop_pp").mean(),
            "frozen_marked_trained_drop_pp": values(name, "varied", "frozen_test_drop_pp").mean(),
            "min_occupied": min(r["occupied"] for r in rows if r["method"] == name)})
    csv_write(out / "summary.csv", summary)
    effects = {"idec_minus_dec_drop_pp": drops["idec"] - drops["dec"],
               "idec_minus_gaussian_both_drop_pp": drops["idec"] - drops["gaussian_both"]}
    for name in ("gaussian_both", "gaussian_pretrain", "gaussian_finetune", "dec"):
        for domain in DOMAINS:
            effects[f"{name}_minus_idec_{domain}_acc_pp"] = acc(name, domain) - acc("idec", domain)
        effects[f"{name}_minus_idec_redraw_pp"] = (values(name, "varied", "redraw_flip_pct") -
                                                   values("idec", "varied", "redraw_flip_pct"))
        if name != "dec":
            effects[f"{name}_minus_idec_decoder_response"] = (values(name, "varied", "decoder_response") -
                                                               values("idec", "varied", "decoder_response"))
    np.testing.assert_allclose(effects["idec_minus_gaussian_both_drop_pp"],
                               effects["gaussian_both_minus_idec_varied_acc_pp"] -
                               effects["gaussian_both_minus_idec_clean_acc_pp"], atol=1e-12)
    paired = [{"seed": s, **{m + "_drop_pp": d[i] for m, d in drops.items()},
               **{k: d[i] for k, d in effects.items()}} for i, s in enumerate(seeds)]
    csv_write(out / "paired.csv", paired)
    save_json(out / "contrasts.json", {k: stats(d) for k, d in effects.items()})
    save_json(out / "verification.json", {
        "final_models": len(METHODS) * len(DOMAINS) * len(seeds),
        "initializations": len(INITIAL) * len(DOMAINS) * len(seeds), "seeds": seeds,
        "test_images_per_model": 10000, "background_views": 3,
        "training_digests_checked": True, "paired_fork_digests_checked": True,
        "min_occupied_primary": min(r["min_occupied"] for r in summary if r["method"] in ("dec","idec","gaussian_both"))})
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Clean ACC & Marked ACC & Drop (pp) & Redraw flips\\", r"\midrule"]
    for r in summary[:len(METHODS)]:
        drop = f'{r["drop_pp"]:.2f} $\\pm$ {r["drop_sd"]:.2f}'
        lines.append(f'{NAMES[r["method"]]} & {r["clean_acc_pct"]:.2f} & {r["marked_acc_pct"]:.2f} & '
                     + drop + f' & {r["redraw_flip_pct"]:.2f}' + r"\\")
    (out / "table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf8")
    print("\n".join(lines[4:]))
    for key in ("idec_minus_dec_drop_pp", "idec_minus_gaussian_both_drop_pp"):
        print(key, stats(effects[key]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "dec_background/results")
    analyze(parser.parse_args().input)
