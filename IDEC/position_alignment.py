"""Calibrate and apply label-free, lossless whole-image centering for MNIST."""
import torch

from geometry_study import gaussian_weights
from spatial_augmentation import Translation


def image_centers(images):
    axis = torch.arange(28, dtype=images.dtype, device=images.device)
    xy = torch.cartesian_prod(axis, axis)
    mass = images * gaussian_weights(images.device)
    total = mass.sum(1, keepdim=True)
    return torch.where(total > 0, (mass @ xy) / total.clamp_min(1e-12), 13.5)


def calibrate(images):
    centers = image_centers(images)
    reference = torch.quantile(centers, .5, dim=0)
    distances = torch.linalg.vector_norm(centers-reference, dim=1)
    return reference, torch.quantile(distances, .99)


def align(images, reference, threshold):
    """Return aligned observations and the signed integer offsets actually applied (dy, dx)."""
    delta = reference - image_centers(images)
    active = torch.linalg.vector_norm(delta, dim=1) > threshold
    requested = torch.where(active[:, None], delta.round().long(), 0)
    frame = Translation(images)
    offsets = torch.stack([requested[:, i].clamp(min=low, max=high)
                           for i, (low, high) in enumerate(frame.bounds)], dim=1)
    rows = (frame.rows - offsets[:, 0, None, None]) % 28
    cols = (frame.cols - offsets[:, 1, None, None]) % 28
    aligned = images.gather(1, (28*rows + cols).flatten(1))
    return aligned, offsets
