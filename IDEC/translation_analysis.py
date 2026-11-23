"""Compare translated DEC/IDEC forks with the fixed v1.8 background controls."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from analyze_stages import matches_training_digest
from dec_background_analysis import METHODS, NAMES, collect, scores, stats
from dec_background_study import DOMAINS
from idec_final.utils import ROOT, save_json, sha256
from translation_control import CONTROLS, LOSSES, SOURCES, STUDY


NEW_METHODS = (*LOSSES, "init_translate")
TABLE_ORDER = ("dec", "idec", *LOSSES, *METHODS[2:])
LABELS = {**NAMES, "translate_dec": "DEC + translation", "translate_idec": "IDEC + translation"}


def translation_rows(out, seeds):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["config"] == cfg and manifest["domains"] == list(DOMAINS) and manifest["methods"] == LOSSES
    assert set(manifest["source_sha256"]) == set(SOURCES)
    for name, digest in manifest["source_sha256"].items():
        assert matches_training_digest(ROOT / name, digest), name
    cells = {f"{d}_{s}" for d in DOMAINS for s in seeds}
    assert {p.name for p in out.iterdir() if p.is_dir()} == cells
    assert set(manifest["control_predictions_sha256"]) == cells
    for cell, digest in manifest["control_predictions_sha256"].items():
        assert sha256(CONTROLS / cell / "predictions.npz") == digest, cell
    expected = {m + "_" + k for m in NEW_METHODS for k in ("train", "clean", "backgrounds")}
    expected.update(("translate_idec_response", "valid_response"))
    steps = list(range(0, cfg["updates"] + 1, cfg["target_interval"]))
    rows = []
    with np.load(ROOT / "varied/results/evaluation.npz") as data:
        train_y, test_y = data["train_digit"], data["test_digit"]
        bank, types = data["bank"], data["test_types"]
        valid = np.any(bank[types[0]] != bank[types[1]], axis=(1, 2, 3))
        for domain in DOMAINS:
            for seed in seeds:
                folder = out / f"{domain}_{seed}"
                assert {p.name for p in folder.iterdir()} == {"record.json", "predictions.npz"}
                record = json.loads((folder / "record.json").read_text())
                assert (record["seed"], record["domain"]) == (seed, domain)
                assert len(record["start_sha256"]) == 64 and set(record["methods"]) == set(LOSSES)
                for name, loss in LOSSES.items():
                    branch = record["methods"][name]
                    assert branch["start_sha256"] == record["start_sha256"] and branch["loss"] == loss
                    assert [t["step"] for t in branch["trace"]] == steps
                with np.load(folder / "predictions.npz") as pred:
                    assert set(pred.files) == expected and pred["valid_response"].dtype == np.bool_
                    np.testing.assert_array_equal(pred["valid_response"], valid)
                    for name in NEW_METHODS:
                        rows.append({"method": name, "domain": domain, "seed": seed,
                                     **scores(pred, name, domain, train_y, test_y, valid)})
    return rows


def analyze(out):
    controls, seeds = collect(CONTROLS)
    new = translation_rows(out, seeds)
    csv_write(out / "runs.csv", new)
    lookup = {(r["method"], r["domain"], r["seed"]): r for r in controls + new}

    def metric(method, domain, key="domain_acc_pct"):
        return np.array([lookup[method, domain, s][key] for s in seeds])

    drops = {}
    summary = []
    for name in (*TABLE_ORDER, "init_translate"):
        clean, marked = metric(name, "clean"), metric(name, "varied")
        drops[name] = clean - marked
        drop_stats = stats(drops[name])
        response = metric(name, "varied", "decoder_response")
        summary.append({"method": name, "clean_acc_pct": clean.mean(), "marked_acc_pct": marked.mean(),
                        "drop_pp": drop_stats["mean"], "drop_sd": drop_stats["sd"],
                        "drop_positive_seeds": drop_stats["positive"],
                        "redraw_flip_pct": metric(name, "varied", "redraw_flip_pct").mean(),
                        "frozen_clean_trained_drop_pp": metric(name, "clean", "frozen_test_drop_pp").mean(),
                        "frozen_marked_trained_drop_pp": metric(name, "varied", "frozen_test_drop_pp").mean(),
                        "marked_decoder_response": None if response[0] is None else float(response.mean()),
                        "min_occupied": min(lookup[name, domain, s]["occupied"] for domain in DOMAINS for s in seeds)})
    csv_write(out / "summary.csv", summary)
    effects = {"translated_idec_minus_dec_drop_pp": drops["translate_idec"] - drops["translate_dec"]}
    for base in ("dec", "idec"):
        name = "translate_" + base
        clean_gain = metric(name, "clean") - metric(base, "clean")
        marked_gain = metric(name, "varied") - metric(base, "varied")
        improvement = drops[base] - drops[name]
        np.testing.assert_allclose(improvement, marked_gain - clean_gain, atol=1e-12)
        effects[base + "_clean_gain_pp"] = clean_gain
        effects[base + "_marked_gain_pp"] = marked_gain
        effects[base + "_drop_reduction_pp"] = improvement
        for key, domain, label in (("redraw_flip_pct", "varied", "redraw_flip_reduction_pp"),
                                   ("frozen_test_drop_pp", "clean", "frozen_test_drop_reduction_pp")):
            effects[base + "_" + label] = metric(base, domain, key) - metric(name, domain, key)
    effects["idec_decoder_response_reduction"] = (metric("idec", "varied", "decoder_response") -
                                                 metric("translate_idec", "varied", "decoder_response"))
    for domain in DOMAINS:
        effects["translated_idec_minus_gaussian_pretrain_" + domain + "_pp"] = (
            metric("translate_idec", domain) - metric("gaussian_pretrain", domain))
    csv_write(out / "paired.csv", [{"seed": s, **{m + "_drop_pp": v[i] for m, v in drops.items()},
                                    **{k: v[i] for k, v in effects.items()}} for i, s in enumerate(seeds)])
    save_json(out / "contrasts.json", {k: stats(v) for k, v in effects.items()})
    save_json(out / "verification.json", {"translated_models": len(LOSSES) * len(DOMAINS) * len(seeds),
              "shared_initializations": len(DOMAINS) * len(seeds), "fixed_control_archive": "aede90d",
              "verified_control_prediction_files": len(DOMAINS) * len(seeds), "seeds": seeds,
              "all_start_digests_and_input_metadata_verified": True})
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Method & Clean ACC & Marked ACC & Drop (pp) & Redraw flips\\", r"\midrule"]
    for row in summary[:len(TABLE_ORDER)]:
        lines.append(f'{LABELS[row["method"]]} & {row["clean_acc_pct"]:.2f} & {row["marked_acc_pct"]:.2f} & '
                     + f'{row["drop_pp"]:.2f} $\\pm$ {row["drop_sd"]:.2f} & {row["redraw_flip_pct"]:.2f}' + r"\\")
    (out / "table.tex").write_text("\n".join(lines + [r"\bottomrule", r"\end{tabular}"]) + "\n", encoding="utf8")
    print("\n".join(lines[4:]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=STUDY / "results")
    analyze(parser.parse_args().input)
