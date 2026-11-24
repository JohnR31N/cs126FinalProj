import numpy as np
import torch

from adaptive_saliency_study import combined_weights
from gaussian_refinement_study import spatial_weights


def test_saliency_changes_pixel_weights_without_swapping_the_centers():
    x = torch.zeros(2, 784)
    x[0, 4*28+7], x[1, 23*28+20] = 1, 1
    base = spatial_weights(x, "centroid")
    maps = torch.zeros_like(x, dtype=torch.uint8)
    maps[0, 4*28+7], maps[1, 23*28+20] = 255, 255
    for signal in (maps, maps.flip(0)):
        actual = combined_weights(base, signal)
        expected = base.numpy().astype(float) * (0.5 + signal.numpy().astype(float) / 255)
        expected /= expected.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(actual.numpy(), expected, rtol=2e-6)
        np.testing.assert_array_equal(actual.argmax(1), [4*28+7, 23*28+20])
    torch.testing.assert_close(combined_weights(base, maps * 0), base)
