import numpy as np
import torch

from gaussian_refinement_study import spatial_weights
from gaussian_shape_study import elliptical_gaussian, read_pair, reconstruction_weights, shifted_pairs


def test_ellipse_and_direction_control_match_independent_scalar_formula():
    x = np.zeros((28, 28), dtype=np.float64)
    x[np.arange(5, 23), np.arange(7, 25)] = np.linspace(0.2, 0.9, 18)
    x[13, 9] = 0.4
    xy = np.indices(x.shape, dtype=float).reshape(2, -1).T
    m = x.ravel() * np.exp(-np.sum((xy-13.5)**2, axis=1)/98)
    center = np.average(xy, axis=0, weights=m)
    offsets = xy-center
    covariance = np.eye(2) + np.einsum("n,ni,nj->ij", m/m.sum(), offsets, offsets)
    eigenvalues, axes = np.linalg.eigh(covariance)
    ratio = min((eigenvalues[1]/eigenvalues[0])**0.25, 2.0)
    images = torch.tensor(x.reshape(1, -1))
    saliency = torch.arange(784).remainder(256).to(torch.uint8)[None]
    for arm, lengths in (("ellipse", [49/ratio, 49*ratio]), ("rotated", [49*ratio, 49/ratio])):
        precision = np.linalg.inv(axes @ np.diag(lengths) @ axes.T)
        assert np.isclose(np.linalg.det(precision), 1/49**2)
        weights = np.array([np.exp(-0.5 * pos @ precision @ pos) for pos in offsets])
        weights *= 0.5 + saliency[0].numpy().astype(float)/255
        expected = weights/weights.mean()
        np.testing.assert_allclose(reconstruction_weights(images, saliency, arm)[0].numpy(),
                                   expected, rtol=2e-6)


def test_isotropic_and_empty_inputs_reduce_to_prior_circle():
    x = torch.zeros(2, 784)
    x[1, 13*28+13] = 1
    expected = spatial_weights(x, "centroid")
    for rotated in (False, True):
        torch.testing.assert_close(elliptical_gaussian(x, rotated), expected, rtol=2e-6, atol=2e-6)


def test_joint_shift_preserves_both_views_and_rejects_marks_at_the_exit_edge():
    n = 40
    clean = np.zeros((n, 28, 28), np.float32)
    clean[:, 12, 11] = 1
    marked = clean.copy()
    marked[:20, 22, 16] = .75
    # A four-border frame makes every nonzero three-pixel shift illegal.
    marked[20:, [0, 27], :] = .5
    marked[20:, :, [0, 27]] = .5
    (shift_clean, shift_marked), ids, deltas = shifted_pairs(clean.reshape(n, -1), marked.reshape(n, -1), 101127)
    np.testing.assert_array_equal(ids, np.arange(20))
    for i, (dy, dx) in enumerate(deltas[ids]):
        for original, moved in ((clean, shift_clean), (marked, shift_marked)):
            target = np.zeros_like(original[i])
            ys, xs = np.nonzero(original[i])
            target[ys+dy, xs+dx] = original[i, ys, xs]
            np.testing.assert_array_equal(moved[i].reshape(28, 28), target)


def test_decoder_response_is_one_for_copying_and_zero_for_constant_output():
    x = torch.zeros(3, 784)
    marked = x.clone()
    marked[:, 0] = torch.tensor([.5, 1., 2.])
    model = torch.nn.Module()
    model.encoder = torch.nn.Identity()
    model.decoder = torch.nn.Identity()
    centers = torch.zeros(10, 784)
    np.testing.assert_allclose(read_pair(model, centers, x, marked)["response"], 1)
    model.decoder = torch.nn.Linear(784, 784)
    torch.nn.init.zeros_(model.decoder.weight)
    np.testing.assert_allclose(read_pair(model, centers, x, marked)["response"], 0)
