"""A controlled, label-independent orientation intervention."""
import numpy as np
import torch


def templates(device):
    wave = ((torch.arange(28, device=device) // 2) % 2).float() - 0.5
    horizontal = wave[:, None].expand(28, 28).reshape(-1)
    vertical = wave.repeat(28)
    patterns = torch.stack([horizontal, vertical])
    return patterns, torch.nn.functional.normalize(horizontal - vertical, dim=0)


def orientations(count, seed):
    return np.random.default_rng(seed).permutation(np.arange(count) % 2)


def observe(clean, orientation, strength, patterns):
    return (clean + strength * patterns[orientation]) * (255.0 / 50)


def projected_mse(residual, direction):
    remaining = residual - (residual @ direction)[:, None] * direction
    return remaining.square().mean()
