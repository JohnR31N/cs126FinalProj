import numpy as np
import torch
from torch.nn import functional as F

from gaussian_saliency_study import cluster_saliency, donor_order, weights_for
from geometry_study import gaussian_weights


def test_margin_attribution_against_linear_encoder():
    enc = torch.nn.Linear(784, 2, bias=False)
    with torch.no_grad():
        enc.weight.zero_()
        enc.weight[0, 15] = 1
        enc.weight[1, 208] = 2
    centers = torch.tensor([[0., 0.], [2., 1.]])
    x = torch.ones(1, 784)
    got = cluster_saliency(enc, centers, x)
    # The absolute margin derivative is 2 times the absolute linear projection of the center difference.
    expected = torch.zeros_like(x)
    expected[0, [15, 208]] = 4
    expected = F.avg_pool2d(expected.reshape(1, 1, 28, 28), 3, 1, 1, count_include_pad=False).flatten(1)
    expected = torch.round(255 * expected / (expected + expected.mean()))
    np.testing.assert_array_equal(got, expected.numpy().astype(np.uint8))
    assert not cluster_saliency(enc, centers, x * 0).any()


def test_shuffling_moves_entire_maps_preserving_normalization():
    n = 7
    ids = torch.from_numpy(donor_order(n))
    assert (ids != torch.arange(n)).all() and len(ids.unique()) == n
    maps = torch.zeros(n, 784, dtype=torch.uint8)
    maps[torch.arange(n), 392 + torch.arange(n)] = 255
    aligned = weights_for("combined", maps, ids)
    shuffled = weights_for("shuffled", maps, ids)
    torch.testing.assert_close(shuffled, aligned[ids])
    torch.testing.assert_close(aligned.mean(1), torch.ones(n))
    blank = weights_for("combined", maps * 0, ids)
    torch.testing.assert_close(blank[0], gaussian_weights("cpu"))
