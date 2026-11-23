import numpy as np
import torch

from geometry_study import SIZE, gaussian_weights, placements, render_views, stable_target


def test_two_disjoint_sites_preserve_faint_strokes_and_the_clearance_ring():
    digit = np.zeros((2, 28, 28), dtype=np.float32)
    digit[:, 9:21, 12] = 1
    digit[:, 11, 13:19] = 1 / 255
    clean = digit.reshape(2, -1)
    boxes = placements(clean, 2701)
    views = render_views(clean, boxes, np.array([1, 0]))
    for i, pair in enumerate(boxes):
        assert np.max(np.abs(pair[0] - pair[1])) >= SIZE
        for y, x in pair:
            region = digit[i, max(0,y-1):min(28,y+SIZE+1), max(0,x-1):min(28,x+SIZE+1)]
            assert not np.any(region)
    for image in views:
        np.testing.assert_array_equal(image[clean > 0], clean[clean > 0])
        np.testing.assert_array_equal(np.count_nonzero(image != clean, axis=1), [9,9])
    for idx in (0,1):
        for other in (2,3):
            np.testing.assert_array_equal(stable_target(views[idx], views[other]), clean)
    assert np.all(np.count_nonzero(views[0] != views[2], axis=1) == 18)
    assert np.all(np.count_nonzero(views[0] != views[1], axis=1) == 16)


def test_gaussian_prior_preserves_average_weight_and_is_centered():
    weight = gaussian_weights("cpu").reshape(28, 28)
    torch.testing.assert_close(weight.mean(), torch.ones(()))
    torch.testing.assert_close(weight, weight.flip([0,1]))
    assert weight[13,13] == weight.max() and weight[0,0] < weight[7,7] < weight.max()
