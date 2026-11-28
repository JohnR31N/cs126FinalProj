import numpy as np
import torch

from position_alignment import align, calibrate


def test_threshold_uses_only_training_geometry_and_identity_within_gate():
    x = torch.zeros(100, 784, dtype=torch.float64)
    x[:98, 14*28+14] = 1
    x[98, 16*28+14] = 1
    x[99, 20*28+14] = 1
    reference, threshold = calibrate(x)
    np.testing.assert_allclose(reference, [14, 14])
    np.testing.assert_allclose(threshold, np.quantile([0]*98+[2, 6], .99))
    near = torch.zeros(1, 784, dtype=x.dtype)
    near[0, 15*28+14] = 1
    corrected, offsets = align(near, reference, threshold)
    torch.testing.assert_close(corrected, near)
    assert not offsets.any()


def test_correction_moves_the_entire_observation_and_preserves_border_pixels():
    x = torch.zeros(3, 28, 28)
    x[:2, 18, 13] = 1
    x[:2, 20, 15] = 0.6
    x[1, 0, 0] = 0.001
    out, offsets = align(x.flatten(1), torch.tensor([14., 14.]), torch.tensor(.1))
    assert offsets[0, 0] == -3 and offsets[1, 0] == 0
    for i, (dy, dx) in enumerate(offsets.tolist()):
        rows, cols = torch.where(x[i] > 0)
        expected = torch.zeros(28, 28)
        assert ((rows+dy >= 0) & (rows+dy < 28) & (cols+dx >= 0) & (cols+dx < 28)).all()
        expected[rows+dy, cols+dx] = x[i, rows, cols]
        torch.testing.assert_close(out[i].reshape(28, 28), expected)
