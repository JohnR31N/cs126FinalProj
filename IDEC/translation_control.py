"""Fit the two whole-image translation rows for the matched background comparison."""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from dec_background_study import DOMAINS, fingerprint, fit_clusters, store_assignments
from geometry_study import SCALE
from idec_final.utils import ROOT, save_json, sha256
from spatial_augmentation import Translation
from varied_backgrounds import dataset
from varied_study import pretrain, readout


STUDY = ROOT / "dec_background/translation"
CONTROLS = ROOT / "dec_background/results"
LOSSES = {"translate_dec": "kl", "translate_idec": "uniform"}
SOURCES = (
    "dec_background/translation/design.md", "translation_control.py", "protocol.json",
    "dec_background_study.py", "dec_background/results/manifest.json", "varied/results/evaluation.npz",
    "varied_study.py", "varied_backgrounds.py", "spatial_augmentation.py", "geometry_study.py",
    "idec_final/model.py", "idec_final/utils.py", "idec_final/data.py", "idec_final/experiment.py",
)


def fit_cell(folder, cfg, seed, domain, images, clean, views):
    initial, centers = pretrain(images, images, "translate", cfg, seed)
    signature = fingerprint([*initial.parameters(), centers])
    decoder_start = fingerprint(initial.decoder.parameters())
    record = {"seed": seed, "domain": domain, "start_sha256": signature, "methods": {}}
    saved = {}
    store_assignments(saved, "init_translate", initial, centers, images, clean, views)
    for name, loss in LOSSES.items():
        arm = copy.deepcopy(initial)
        start = fingerprint([*arm.parameters(), centers])
        assert start == signature
        final_centers, trace = fit_clusters(arm, centers, images, loss, cfg)
        store_assignments(saved, name, arm, final_centers, images, clean, views)
        if loss == "kl":
            assert fingerprint(arm.decoder.parameters()) == decoder_start
        else:
            response = readout(arm, views, clean)
            saved[name + "_response"] = response["response"]
            saved["valid_response"] = response["response_valid"]
        record["methods"][name] = {"loss": loss, "start_sha256": start, "trace": trace}
    with TemporaryDirectory(dir=folder.parent) as tmp:
        cell = Path(tmp) / folder.name
        cell.mkdir()
        np.savez_compressed(cell / "predictions.npz", **saved)
        save_json(cell / "record.json", record)
        cell.rename(folder)
    print(f"Finished translated DEC and IDEC, {domain}, seed {seed}.", flush=True)


def run(out, device):
    cfg, data, metadata = dataset()
    with np.load(ROOT / "varied/results/evaluation.npz") as archived:
        assert set(metadata) == set(archived.files)
        for key, value in metadata.items():
            np.testing.assert_array_equal(value, archived[key])
    assert json.loads((CONTROLS / "manifest.json").read_text())["config"] == cfg
    raw = {"clean": data["train"]["clean"], "varied": data["train"]["varied"][0]}
    train = {name: torch.from_numpy(values).to(device) * SCALE for name, values in raw.items()}
    clean = torch.from_numpy(data["test"]["clean"]).to(device) * SCALE
    views = torch.from_numpy(data["test"]["varied"]).to(device) * SCALE
    cells = [f"{domain}_{seed}" for seed in cfg["seeds"] for domain in DOMAINS]
    manifest = {"design_commit": "c22ae3d", "control_commit": "aede90d", "config": cfg,
                "domains": list(DOMAINS), "methods": LOSSES, "torch": str(torch.__version__),
                "device": str(device), "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "source_sha256": {name: sha256(ROOT / name) for name in SOURCES},
                "control_predictions_sha256": {cell: sha256(CONTROLS / cell / "predictions.npz") for cell in cells},
                "offsets": {name: Translation(images).audit for name, images in train.items()}}
    out.mkdir(parents=True, exist_ok=True)
    path = out / "manifest.json"
    if any(out.iterdir()) and (not path.exists() or json.loads(path.read_text()) != manifest):
        raise ValueError("The output already contains another experiment. Choose a fresh directory.")
    save_json(path, manifest)
    for seed in cfg["seeds"]:
        for domain in DOMAINS:
            folder = out / f"{domain}_{seed}"
            if not folder.exists():
                fit_cell(folder, cfg, seed, domain, train[domain], clean, views)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/translation_control")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
