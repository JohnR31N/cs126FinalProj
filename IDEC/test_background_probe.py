import numpy as np

from background_probe_analysis import ranking_pct, readouts
from background_probe_study import brightness_pair, eligible


def test_brightness_changes_only_empty_background_at_fixed_sites():
    digits = np.zeros((9, 28, 28), dtype=np.float32)
    digits[:, 9:20, 12:14] = 0.7
    image = digits.reshape(len(digits), -1)
    dim, bright, _, _, gap = brightness_pair(image)
    np.testing.assert_array_equal(dim > image, bright > image)
    np.testing.assert_allclose((dim - image) / .6, bright - image, atol=2e-7)
    assert (gap > 0).all() and (bright > dim).any(axis=1).all()
    assert eligible(image).all() and not eligible(np.ones((1, 784), dtype=np.float32))[0]


def test_paired_readout_direction_ties_and_calibration_only():
    rng = np.random.default_rng(33)
    n = 100
    clean = rng.normal(size=(2 * n, 3))
    baseline = clean.copy()
    offset = np.array([.1, -.6, .3])
    result = readouts(baseline, baseline + offset, clean, n)
    assert result["paired_pct"] == 100. and result["identical_pct"] == 50.
    assert result["code_distance"] > 0
    flipped = baseline + offset
    flipped[n:] -= 2 * offset
    reversed_test = readouts(baseline, flipped, clean, n)
    assert reversed_test["paired_pct"] == 0.
    np.testing.assert_allclose(result["code_distance"], reversed_test["code_distance"])
    assert ranking_pct(np.array([-2., 0., 0., 2.])) == 50.
