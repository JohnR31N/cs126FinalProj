import numpy as np
import torch

from gaussian_refinement_study import spatial_weights


def test_centroid_map_matches_an_independent_coordinate_calculation():
    row, col = np.mgrid[:28, :28]
    image = np.zeros((28, 28))
    image[8, 12] = 1
    image[5, 24] = 0.7
    spatial = np.exp(-((row-13.5)**2 + (col-13.5)**2) / 98)
    mass = image * spatial
    cy, cx = (mass * row).sum() / mass.sum(), (mass * col).sum() / mass.sum()
    expected = np.exp(-((row-cy)**2 + (col-cx)**2) / 98)
    inputs = torch.from_numpy(np.stack((image, image * 0)).reshape(2, 784)).float()
    actual = spatial_weights(inputs, "centroid").numpy().reshape(2, 28, 28)
    np.testing.assert_allclose(actual[0], expected / expected.mean(), rtol=2e-6)
    np.testing.assert_allclose(actual[1], spatial / spatial.mean(), rtol=2e-6)
    floor = spatial_weights(inputs, "floor").numpy().reshape(2, 28, 28)
    np.testing.assert_allclose(floor[0], (1 + spatial / spatial.mean()) / 2, rtol=2e-6)
    np.testing.assert_array_equal(floor[0], floor[1])
