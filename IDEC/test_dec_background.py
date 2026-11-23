import copy

import numpy as np
import pytest
import torch

from dec_background_study import fit_clusters
from idec_final.experiment import finetune
from idec_final.model import Autoencoder, student_q, target_distribution


@pytest.mark.parametrize(("kind", "archived"), [("kl", "cluster_only"), ("uniform", "idec")])
def test_forks_match_the_original_optimizer(kind, archived):
    torch.manual_seed(22)
    images = torch.rand(13, 6)
    model = Autoencoder([6, 5, 3])
    centers = torch.randn(3, 3)
    cfg = dict(lr=.002, gamma=.1, batch_size=5, target_interval=3, updates=8)
    reference = copy.deepcopy(model)
    expected, trace = finetune(reference, centers, images, None, archived, cfg)
    if kind == "kl":
        def forbid_reconstruction(*unused):
            raise AssertionError("DEC evaluated the decoder during clustering")
        model.decoder.register_forward_pre_hook(forbid_reconstruction)
    actual, actual_trace = fit_clusters(model, centers, images, kind, cfg)
    assert actual_trace == trace
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_gaussian_update_matches_explicit_spatial_loss():
    torch.manual_seed(18)
    images = torch.rand(9, 784, dtype=torch.float64)
    model = Autoencoder([784, 5, 2]).double()
    reference = copy.deepcopy(model)
    initial = torch.randn(3, 2, dtype=torch.float64)
    centers = torch.nn.Parameter(initial.clone())
    cfg = dict(lr=.002, gamma=.1, batch_size=9, target_interval=1, updates=2)
    optimizer = torch.optim.Adam([*reference.parameters(), centers], lr=cfg["lr"])
    rows, cols = np.ogrid[:28, :28]
    w = np.exp(-((rows - 13.5)**2 + (cols - 13.5)**2) / 98)
    # Independent spatial formula, unit mean, and sum over 28x28 pixels.
    weight = torch.from_numpy(w / w.mean())
    for _ in range(2):
        z = reference.encoder(images)
        q = student_q(z, centers)
        p = target_distribution(q.detach())
        kl = (p * (p.log() - q.log())).sum(1).mean()
        mse = ((reference.decoder(z) - images).reshape(-1, 28, 28).square() * weight).mean()
        optimizer.zero_grad(set_to_none=True)
        (mse + .1 * kl).backward()
        optimizer.step()
    result, _ = fit_clusters(model, initial, images, "gaussian", cfg)
    torch.testing.assert_close(centers.detach(), result, atol=1e-8, rtol=1e-7)
    for a, b in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=1e-6)
