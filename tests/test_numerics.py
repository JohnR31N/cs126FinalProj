import copy
import importlib.util
import json
import sys

import numpy as np
import torch

from src.data.mnist import gcn, load_mnist, split_indices
from src.eval.metrics import clustering
from src.method.cae import ConcreteAE, collision_penalty
from src.method.idec import Autoencoder, student_q, target_distribution
from src.method.svdd import initialize_center
from src.method.svdd_network import MNIST_LeNet, MNIST_LeNet_Autoencoder
from src.utils.common import ROOT


def test_student_t_and_target_independent_oracle():
    z = np.array([[0., 0.], [2., 1.], [-1., 3.]])
    centers = np.array([[0., 1.], [2., 2.]])
    inverse = np.array([[1 / (1 + np.dot(a-b, a-b)) for b in centers] for a in z])
    expected_q = inverse / inverse.sum(axis=1, keepdims=True)
    w = expected_q ** 2 / expected_q.sum(0)
    expected_p = w / w.sum(1, keepdims=True)
    q = student_q(torch.tensor(z), torch.tensor(centers))
    np.testing.assert_allclose(q.numpy(), expected_q, rtol=1e-12)
    np.testing.assert_allclose(target_distribution(q).numpy(), expected_p, rtol=1e-12)
    assert torch.autograd.gradcheck(lambda zz, cc: student_q(zz, cc),
        (torch.tensor(z, requires_grad=True), torch.tensor(centers, requires_grad=True)))


def test_collision_expected_probability_oracle():
    logits = torch.tensor([[1., 2., 3.], [-2., 1., 0.]], dtype=torch.float64, requires_grad=True)
    p = logits.detach().numpy()
    p = np.exp(p) / np.exp(p).sum(axis=1, keepdims=True)
    expected = sum(p[0, j]*p[1, j] for j in range(3))
    np.testing.assert_allclose(collision_penalty(logits).item(), expected)
    assert torch.autograd.gradcheck(collision_penalty, (logits,))


def test_cae_hard_selector_preserves_duplicate_slots():
    m = ConcreteAE(4, 3)
    with torch.no_grad():
        m.logits.copy_(torch.tensor([[0., 5., 0., 0.], [0., 0., 0., 5.], [0., 5., 0., 0.]]))
    m.eval()
    x = torch.tensor([[1., 2., 3., 4.]])
    torch.testing.assert_close(m(x), m.decoder(torch.tensor([[2., 4., 2.]])))


def test_unsupervised_losses_backpropagate_through_real_models():
    torch.manual_seed(126)
    x = torch.rand(9, 4)
    m = ConcreteAE(4, 2)
    loss = (m(x)-x).square().mean()
    loss.backward()
    assert torch.isfinite(m.logits.grad).all() and m.logits.grad.abs().sum() > 0
    ae = Autoencoder([4, 6, 2])
    centers = torch.nn.Parameter(torch.tensor([[0., 0.], [1., 1.]]))
    q = student_q(ae.encoder(x), centers)
    p = target_distribution(q).detach()
    loss = (ae(x)-x).square().mean() + .1*(p*(p.log()-q.log())).sum(1).mean()
    loss.backward()
    assert not p.requires_grad and centers.grad.abs().sum() > 0
    assert ae.encoder[0].weight.grad.abs().sum() > 0
    assert ae.decoder[0].weight.grad.abs().sum() > 0


def test_upstream_svdd_forward_parity():
    upstream = ROOT / "tests/fixtures/deep_svdd/src"
    if not upstream.exists():
        upstream = ROOT / "research/upstream/Deep-SVDD-PyTorch/src"
    if not upstream.exists():
        import pytest
        pytest.skip("Upstream checkout required for independent forward parity")
    sys.path.insert(0, str(upstream))
    spec = importlib.util.spec_from_file_location("upstream_mnist_network", upstream / "networks/mnist_LeNet.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    for own, original in [(MNIST_LeNet, mod.MNIST_LeNet), (MNIST_LeNet_Autoencoder, mod.MNIST_LeNet_Autoencoder)]:
        a, b = own().eval(), original().eval()
        b.load_state_dict(a.state_dict())
        x = torch.rand(3, 1, 28, 28)
        torch.testing.assert_close(a(x), b(x), rtol=0, atol=0)
    sys.path.pop(0)


def test_center_and_gcn_invariants():
    x = torch.tensor([[1., 2., 5., 4.], [10., 20., 50., 40.]])
    z = gcn(x)
    torch.testing.assert_close(z[0], z[1])
    torch.testing.assert_close(z.mean(1), torch.zeros(2))
    torch.testing.assert_close(z.abs().mean(1), torch.ones(2))
    c = initialize_center(torch.tensor([[-.01, .02, .3], [-.03, .04, .5]], requires_grad=True))
    torch.testing.assert_close(c, torch.tensor([-.1, .1, .4]))
    assert not c.requires_grad
    for layer in MNIST_LeNet().modules():
        if isinstance(layer, (torch.nn.Linear, torch.nn.Conv2d)):
            assert layer.bias is None


def test_real_data_split_integrity():
    cfg = json.loads((ROOT / "configs/pilot.json").read_text())
    train, valid = split_indices(cfg)
    assert len(train) == 12000 and len(valid) == 3000
    assert len(np.intersect1d(train, valid)) == 0
    x, y, xt, yt = load_mnist()
    assert x.shape == (60000, 784) and xt.shape == (10000, 784)
    assert y.shape == (60000,) and yt.shape == (10000,)
    assert x.min() == 0 and x.max() == 1


def test_clustering_metrics_permutation_invariant():
    r = clustering(np.array([0, 0, 1, 1, 2, 2]), np.array([2, 2, 0, 0, 1, 1]))
    assert r == dict(acc=1., nmi=1., ari=1.)
