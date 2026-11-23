import numpy as np
import torch

from geometry_study import SCALE
from photometric_study import jitter_images, shifted_test
from spatial_augmentation import Rotation, Translation


def test_jitter_per_image_order_clipping_and_grayscale_reference():
    image = np.array([0, 0.16, 0.5, 0.9, 1.0], dtype=np.float32)
    draws = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0.5, 0.5, 0.2]], dtype=np.float32)
    values = torch.tensor(np.repeat(image[None, :], len(draws), axis=0) * SCALE)
    result = jitter_images(values, torch.tensor(draws)).numpy() / SCALE
    references = []
    for db, dc, order in draws:
        x = image.copy()
        ops = ("brightness", "contrast") if order < .5 else ("contrast", "brightness")
        for op in ops:
            if op == "brightness":
                x = np.clip(x * (.7+.6*db), 0, 1)
            else:
                x = np.clip((.7+.6*dc)*x + (1-(.7+.6*dc))*x.mean(), 0, 1)
        references.append(x)
    np.testing.assert_allclose(result, references, atol=2e-7)
    assert not np.allclose(result[1], result[2])


def test_four_pixel_shift_excludes_cropped_faint_ink():
    source = np.zeros((4, 28, 28), dtype=np.float32)
    source[:, 12, 14] = 1
    source[2, 10, 0] = source[3, 10, -1] = 1/255
    shifted, indices = shifted_test(source.reshape(4, -1))
    np.testing.assert_array_equal(indices, [0, 1])
    _, row, col = np.nonzero(shifted.reshape(2, 28, 28))
    np.testing.assert_array_equal(row, [12, 12])
    np.testing.assert_array_equal(col, [10, 18])


def test_training_translations_move_all_ink_and_marks_together():
    images = torch.zeros(3, 28, 28)
    images[:, 11, 13] = .2
    images[0, 0, 0] = images[1, 27, 27] = images[2, 16, 17] = 1
    sampler = Translation(images.flatten(1))
    rng = torch.Generator().manual_seed(1)
    for _ in range(12):
        warped = sampler(rng).reshape_as(images)
        for before, after in zip(images, warped):
            delta = torch.nonzero(after == 1) - torch.nonzero(before == 1)
            torch.testing.assert_close(torch.nonzero(after == .2) - torch.nonzero(before == .2), delta)
            assert int(delta.abs().max()) <= 3 and int(after.count_nonzero()) == 2


def test_rotation_matches_bilinear_reference_and_excludes_clipped_corners():
    from scipy.ndimage import map_coordinates
    images = torch.zeros(2, 28, 28)
    images[0, 6:8, 15] = 1.
    images[1, 0, 0] = images[1, 27, 27] = 1.
    sampler = Rotation(images.flatten(1), 1.)
    rows, cols = np.indices((28, 28), dtype=np.float64) - 13.5
    assert sampler.legal[0].all()
    assert sampler.legal[1].sum() == 1
    for i, angle in enumerate(sampler.angles):
        r = np.deg2rad(angle)
        coords = [np.sin(r)*cols + np.cos(r)*rows + 13.5, np.cos(r)*cols - np.sin(r)*rows + 13.5]
        expected = map_coordinates(images[0].numpy(), coords, order=1, mode="grid-constant")
        np.testing.assert_allclose(sampler.views[i, 0].reshape(28, 28).numpy(), expected, atol=4e-6)
