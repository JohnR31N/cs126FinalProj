"""Evaluate cluster-gradient saliency as a fixed pretraining weight on centered MNIST."""
import argparse
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
from torch.nn import functional as F

from clean_neighborhood_study import save_predictions
from dec_background_study import DOMAINS, store_assignments
from external_study import manifest
from gaussian_refinement_study import SOURCES as BASE_SOURCES, fit
from geometry_study import SCALE, gaussian_weights
from idec_final.utils import ROOT, save_json
from varied_backgrounds import dataset
from varied_study import pretrain


ARMS = ("combined", "shuffled", "saliency")
SOURCES = (*BASE_SOURCES, "gaussian_saliency/design.md", "gaussian_saliency_study.py", "varied_study.py")


def cluster_saliency(encoder, centers, images):
    maps = []
    for batch in images.split(256):
        x = batch.detach().requires_grad_(True)
        z = encoder(x)
        dist = (z[:, None] - centers.detach()[None]).square().sum(2)
        best = dist.topk(2, largest=False, sorted=True).values
        gradient, = torch.autograd.grad((best[:, 1] - best[:, 0]).sum(), x)
        a = (x.detach() * gradient).abs().reshape(-1, 1, 28, 28)
        a = F.avg_pool2d(a, 3, stride=1, padding=1, count_include_pad=False).flatten(1)
        s = a / (a + a.mean(1, keepdim=True)).clamp_min(1e-12)
        maps.append((255 * s).round().to(torch.uint8).cpu().numpy())
    return np.concatenate(maps)


def donor_order(n):
    cycle = np.random.default_rng(42126).permutation(n)
    donor = np.empty_like(cycle)
    donor[cycle] = np.roll(cycle, 1)
    return donor


def weights_for(arm, maps, donor):
    signal = maps[donor] if arm == "shuffled" else maps
    w = 0.5 + signal.float() / 255
    if arm in ("combined", "shuffled"):
        w *= gaussian_weights(w.device)
    elif arm != "saliency":
        raise ValueError(arm)
    return w / w.mean(1, keepdim=True)


def run(out, device):
    cfg, data, metadata = dataset()
    with np.load(ROOT / "varied/results/evaluation.npz") as archived:
        assert set(archived.files) == set(metadata)
        for name in metadata:
            np.testing.assert_array_equal(metadata[name], archived[name])
    manifest(out, cfg, device, SOURCES, dataset="marked_mnist", arms=list(ARMS), teacher_seed=126)
    test = torch.from_numpy(data["test"]["clean"]).to(device) * SCALE
    views = torch.from_numpy(data["test"]["varied"]).to(device) * SCALE
    clean = torch.from_numpy(data["train"]["clean"]).to(device) * SCALE
    marked = torch.from_numpy(data["train"]["varied"][0]).to(device) * SCALE
    audit = {}
    for domain in reversed(DOMAINS):
        train = marked if domain == "varied" else clean
        path = out / f"{domain}_maps.npz"
        if not path.exists():
            teacher, centers = pretrain(train, train, "none", cfg, 126)
            teacher.eval()
            maps = cluster_saliency(teacher.encoder, centers, train)
            save_predictions(path, saliency=maps, donor=donor_order(len(train)))
        with np.load(path) as saved:
            maps = torch.from_numpy(saved["saliency"]).to(device)
            donor = torch.from_numpy(saved["donor"]).to(device)
        for arm in ARMS:
            weights = weights_for(arm, maps, donor)
            torch.testing.assert_close(weights.mean(1), torch.ones_like(train[:, 0]), atol=1e-6, rtol=0)
            if domain == "varied":
                audit[arm] = {"ink": weights[clean > 0].mean().item(),
                              "symbol": weights[marked != clean].mean().item()}
                save_json(out / "weights.json", audit)
            for seed in cfg["seeds"]:
                path = out / f"{domain}_{arm}_{seed}.npz"
                if path.exists():
                    continue
                model, centers = fit(train, weights, cfg, seed)
                pred = {}
                store_assignments(pred, arm, model, centers, train, test, views)
                save_predictions(path, **pred)
                print(domain, arm, seed, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/gaussian_saliency")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
