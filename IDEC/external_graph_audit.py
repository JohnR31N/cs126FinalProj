"""Recover the graph-only predictions when the fixed nine-direction Fisher fit fails."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import normalized_digest, save_predictions
from cluster_feature_selection import graph_predictions, spectral_groups, stroke_features, vote
from external_data import load_external
from idec_final.utils import ROOT, save_json


def audit(folder, cache, device):
    config = json.loads((folder / "manifest.json").read_text())
    images, _ = load_external(config["dataset"], cache, config["config"])
    train = stroke_features(images["train"], device)
    half = len(train) // 2
    a, b = train[:half], train[half:]
    ga, _ = spectral_groups(a, device, 126)
    gb, _ = spectral_groups(b, device, 127)
    first = np.r_[ga, vote(a, ga, b, device).argmax(1)]
    second = np.r_[vote(b, gb, a, device).argmax(1), gb]
    overlap = np.bincount(10 * second + first, minlength=100).reshape(10, 10)
    _, mapping = linear_sum_assignment(overlap, maximize=True)
    first = first.astype(np.uint8)
    second = mapping[second].astype(np.uint8)
    agreed = np.bincount(first[first == second], minlength=10)
    save_predictions(folder / "graph_groups.npz", first=first, second=second, agreed_group_sizes=agreed)
    query = stroke_features(images["test"], device)
    test = graph_predictions(train, {"first": first, "second": second}, query, device)
    save_predictions(folder / "graph.npz", train=np.r_[second[:half], first[half:]], test=test)
    save_json(folder / "graph_audit.json", {"source_sha256_lf": normalized_digest(Path(__file__)),
                                            "agreed": int(agreed.sum()), "group_sizes": agreed.tolist()})
    print(f"{config['dataset']}: agreed group sizes {agreed.tolist()}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "scratch" / "external")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache" / "external")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        for failure in sorted(args.results.glob("*/unsupported.json")):
            audit(failure.parent, args.cache, torch.device(args.device))
