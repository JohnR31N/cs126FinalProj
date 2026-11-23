"""Recompute randomized-background comparisons from all saved digit assignments."""
import argparse
import json
from pathlib import Path

import numpy as np

from analyze import csv_write
from idec_final.utils import ROOT, save_json, sha256
from photometric_analysis import digit_mapping
from varied_study import ARMS


NAMES = {
    "clean_full_full_none": "Clean / full target",
    "fixed_full_full_none": "Fixed form / full target",
    "fixed_clean_clean_none": "Fixed form / clean target",
    "varied_full_full_none": "Varied / full target",
    "varied_full_clean_none": "Varied / clean fine-tuning",
    "varied_clean_full_none": "Varied / clean pretraining",
    "varied_clean_clean_none": "Varied / clean target",
    "varied_full_full_translate": "Varied / translation",
    "varied_full_full_gaussian": "Varied / Gaussian",
}
METRICS = ("domain_acc", "clean_acc", "marked_acc", "mark_penalty", "redraw_flip",
           "pre_response", "final_response", "pre_background_error", "final_background_error",
           "pre_ink_error", "final_ink_error")
CONTRASTS = {
    "fixed_vs_clean": ("fixed_full_full_none", "clean_full_full_none"),
    "varied_vs_clean": ("varied_full_full_none", "clean_full_full_none"),
    "varied_vs_fixed": ("varied_full_full_none", "fixed_full_full_none"),
    "fixed_clean_target": ("fixed_clean_clean_none", "fixed_full_full_none"),
    "varied_clean_target": ("varied_clean_clean_none", "varied_full_full_none"),
    "clean_finetuning_after_full_pretrain": ("varied_full_clean_none", "varied_full_full_none"),
    "clean_pretraining_then_full_finetune": ("varied_clean_full_none", "varied_full_full_none"),
    "clean_pretraining_with_clean_finetune": ("varied_clean_clean_none", "varied_full_clean_none"),
    "clean_finetuning_after_clean_pretrain": ("varied_clean_clean_none", "varied_clean_full_none"),
    "translation_vs_standard": ("varied_full_full_translate", "varied_full_full_none"),
    "gaussian_vs_standard": ("varied_full_full_gaussian", "varied_full_full_none"),
    "translation_vs_gaussian": ("varied_full_full_translate", "varied_full_full_gaussian"),
}


def scores(pred, labels, arm):
    train_y, test_y = labels["train_digit"], labels["test_digit"]
    mapping = digit_mapping(pred["train"], train_y)
    clean_acc = (mapping[pred["clean"]] == test_y).mean()
    marked_acc = (mapping[pred["backgrounds"][0]] == test_y).mean()
    result = dict(domain_acc=clean_acc if arm.startswith("clean_") else marked_acc,
                  clean_acc=clean_acc, marked_acc=marked_acc, mark_penalty=clean_acc-marked_acc,
                  redraw_flip=(pred["backgrounds"][1:] != pred["backgrounds"][0]).mean(),
                  occupied=np.unique(pred["backgrounds"][0]).size)
    for stage in ("pre", "final"):
        valid = pred[stage+"_response_valid"]
        for metric in ("response", "background_error", "ink_error"):
            values = pred[stage+"_"+metric]
            assert values.shape == test_y.shape and np.isfinite(values).all()
            result[stage+"_"+metric] = values[valid].mean(dtype=np.float64) if metric == "response" else values.mean(dtype=np.float64)
        result[stage+"_valid_response_images"] = valid.sum()
    return result


def analyze(out):
    cfg = json.loads((ROOT/"protocol.json").read_text())
    manifest = json.loads((out/"run.json").read_text())
    assert manifest["config"] == cfg and manifest["arms"] == ARMS
    for name, digest in manifest["source_sha256"].items():
        assert sha256(ROOT/name) == digest, name
    expected = {f"{a}_{s}" for a in ARMS for s in cfg["seeds"]}
    assert {p.stem for p in out.glob("*.npz")} == expected | {"evaluation"}
    rows = []
    with np.load(out/"evaluation.npz") as labels, np.load(ROOT/"results/evaluation.npz") as original:
        for key in ("train_ids", "train_digit", "test_digit"):
            np.testing.assert_array_equal(labels[key], original[key])
        bank, fixed = labels["bank"], labels["fixed_bank"]
        np.testing.assert_allclose(np.square(bank).sum((1, 2)), np.square(fixed).sum((1, 2)), rtol=3e-7)
        for arm in ARMS:
            for seed in cfg["seeds"]:
                path = out/f"{arm}_{seed}.npz"
                history = json.loads(path.with_suffix(".json").read_text())
                assert [h["step"] for h in history] == list(range(0, cfg["updates"]+1, cfg["target_interval"]))
                with np.load(path) as predictions:
                    needed = {"train", "clean", "backgrounds"} | {
                        stage+"_"+metric for stage in ("pre","final")
                        for metric in ("response","response_valid","background_error","ink_error")}
                    assert set(predictions.files) == needed
                    assert predictions["backgrounds"].shape == (3, 10000)
                    for name in ("train", "clean", "backgrounds"):
                        assert predictions[name].dtype == np.uint8 and predictions[name].max() < 10
                    assert predictions["train"].shape == (12000,) and predictions["clean"].shape == (10000,)
                    assert predictions["pre_response_valid"].dtype == np.bool_
                    assert predictions["final_response_valid"].dtype == np.bool_
                    rows.append({"arm": arm, "seed": seed, **scores(predictions, labels, arm)})
        energies = np.square(bank).sum((1, 2))
        train_energy = energies[labels["train_types"][0]].sum(1)
    csv_write(out/"runs.csv", rows)
    values = {a: {m: np.asarray([r[m] for r in rows if r["arm"] == a]) for m in METRICS} for a in ARMS}
    summary = []
    for arm, measurements in values.items():
        row = {"arm": arm}
        for metric, numbers in measurements.items():
            row[metric+"_mean"], row[metric+"_sd"] = numbers.mean(), numbers.std(ddof=1)
        summary.append(row)
    csv_write(out/"summary.csv", summary)
    comparisons = []
    for comparison, (arm, base) in CONTRASTS.items():
        for metric in METRICS:
            differences = values[arm][metric] - values[base][metric]
            comparisons.append({"contrast": comparison, "metric": metric, "mean": float(differences.mean()),
                                "sd": float(differences.std(ddof=1)), "differences": differences.tolist(),
                                "positive": int((differences>1e-12).sum()), "negative": int((differences < -1e-12).sum())})
    save_json(out/"paired.json", comparisons)
    save_json(out/"verification.json", {"models": len(rows), "seeds": cfg["seeds"], "templates": 96,
              "symbols_per_image": 2, "mean_train_background_squared_energy": float(train_energy.mean()),
              "min_occupied_test_clusters": int(min(r["occupied"] for r in rows)),
              "saved_test_backgrounds_per_digit": 3, "train_images": 12000, "test_images": 10000,
              "all_source_digests_verified": True, "paired_template_energies_verified": True})
    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Training condition & In-domain ACC & Clean ACC & Redraw flips\\", r"\midrule"]
    for arm in ("clean_full_full_none", "fixed_full_full_none", "varied_full_full_none",
                "varied_clean_clean_none", "varied_full_full_translate", "varied_full_full_gaussian"):
        cells = [NAMES[arm], f"{100*values[arm]['domain_acc'].mean():.2f}", f"{100*values[arm]['clean_acc'].mean():.2f}",
                 "---" if arm.startswith("clean_") else f"{100*values[arm]['redraw_flip'].mean():.2f}"]
        lines.append(" & ".join(cells) + r"\\")
    (out/"table.tex").write_text("\n".join(lines+[r"\bottomrule", r"\end{tabular}"])+"\n", encoding="utf8")
    stage_lines = [r"\begin{tabular}{llrrr}", r"\toprule",
                   r"Pretrain target & Fine-tune target & Marked ACC & Redraw flips & Decoder response\\", r"\midrule"]
    for pre, fine in (("full","full"), ("full","clean"), ("clean","full"), ("clean","clean")):
        arm = f"varied_{pre}_{fine}_none"
        row = [pre.capitalize(), fine.capitalize(), f"{100*values[arm]['domain_acc'].mean():.2f}",
               f"{100*values[arm]['redraw_flip'].mean():.2f}", f"{values[arm]['final_response'].mean():.5f}"]
        stage_lines.append(" & ".join(row)+r"\\")
    (out/"stages.tex").write_text("\n".join(stage_lines+[r"\bottomrule", r"\end{tabular}"])+"\n", encoding="utf8")
    print("\n".join(lines[4:]))
    print("Verified", len(rows), "model records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT/"varied/results")
    analyze(parser.parse_args().input)
