"""Fit adaptive Gaussian reconstruction with image-specific cluster saliency on clean MNIST."""
import argparse
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from adaptive_saliency_study import combined_weights
from clean_neighborhood_study import save_predictions
from dec_background_analysis import scores
from dec_background_study import store_assignments
from gaussian_refinement_study import fit, spatial_weights
from gaussian_saliency_study import cluster_saliency
from gaussian_shape_study import elliptical_gaussian
from geometry_study import SCALE
from idec_final.utils import ROOT, save_json
from varied_backgrounds import dataset
from varied_study import pretrain


def train_student(images, cfg, seed, shape="circle"):
    teacher, initial_centers = pretrain(images, images, "none", cfg, 126)
    teacher.eval()
    saliency = torch.from_numpy(cluster_saliency(teacher.encoder, initial_centers, images)).to(images.device)
    if shape == "circle":
        gaussian = spatial_weights(images, "centroid")
    elif shape == "ellipse":
        gaussian = elliptical_gaussian(images)
    else:
        raise ValueError(f"Unsupported Gaussian shape: {shape}")
    weights = combined_weights(gaussian, saliency)
    return fit(images, weights, cfg, seed)


def run(output, seed, device, shape="circle"):
    cfg, data, metadata = dataset()
    train = torch.from_numpy(data["train"]["clean"]).to(device) * SCALE
    model, centers = train_student(train, cfg, seed, shape)
    clean = torch.from_numpy(data["test"]["clean"]).to(device) * SCALE
    marked = torch.from_numpy(data["test"]["varied"]).to(device) * SCALE
    predictions = {}
    name = "adaptive_aligned" if shape == "circle" else "ellipse_aligned"
    store_assignments(predictions, name, model, centers, train, clean, marked)
    metrics = scores(predictions, name, "clean", metadata["train_digit"], metadata["test_digit"], None)
    output.mkdir(parents=True, exist_ok=True)
    save_predictions(output / f"clean_{name}_{seed}.npz", **predictions)
    suffix = "" if shape == "circle" else "_ellipse"
    metrics_path = output / f"metrics{suffix}_{seed}.json"
    save_json(metrics_path, {
        "student_seed": seed, "teacher_seed": 126, "shape": shape, "torch": str(torch.__version__),
        "device": str(device), "clean_acc_pct": metrics["clean_test_acc_pct"],
        "marked_acc_pct": metrics["marked_test_acc_pct"],
    })
    print(metrics_path.read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=range(5), default=0)
    parser.add_argument("--shape", choices=("circle", "ellipse"), default="circle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "scratch/final_fit")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, args.seed, torch.device(args.device), args.shape)
