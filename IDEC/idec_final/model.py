"""IDEC network and Student-t objective from the project pilot (Guo et al., 2017)."""
import torch
from torch import nn


def mlp(dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layer = nn.Linear(a, b)
        nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encoder, self.decoder = mlp(dims), mlp(list(reversed(dims)))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def student_q(z, centers):
    q = 1 / (1 + (z[:, None, :] - centers[None, :, :]).square().sum(-1))
    return q / q.sum(1, keepdim=True)


def target_distribution(q):
    w = q.square() / q.sum(0, keepdim=True).clamp_min(1e-12)
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)
