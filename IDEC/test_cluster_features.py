import numpy as np
from threadpoolctl import threadpool_limits

from cluster_coordinate_study import select_bins
from cluster_feature_probes import assignment_style_scores, style_values
from cluster_feature_selection import (coordinates, fisher_basis, nearest, reconstruction_target,
                                       stroke_features, target_embedding)


def test_orientation_bank_has_known_bins_and_preserves_input():
    row, col = np.mgrid[:28, :28].astype(np.float32)
    images = np.stack((col/27, 1-col/27, .4*row/27, np.zeros_like(row))).reshape(-1, 784)
    original = images.copy()
    f = stroke_features(images, "cpu").reshape(-1, 7, 7, 8)
    np.testing.assert_allclose(np.square(f).sum(axis=(1, 2, 3)), [1, 1, 1, 0], atol=2e-6)
    np.testing.assert_allclose(f[0], f[1], atol=1e-6)
    assert np.count_nonzero(f[0, :, :, 1:]) == 0
    np.testing.assert_allclose(f[2, :, :, 4], f[0, :, :, 0].T, atol=1e-6)
    np.testing.assert_array_equal(images, original)


def test_neighbor_ranks_match_direct_distance_without_self():
    ref = np.random.default_rng(42).normal(size=(27, 10)).astype(np.float32)
    ids, dist = nearest(ref, ref, "cpu", same=True)
    exact = np.square(ref[:, None] - ref[None]).sum(2)
    np.fill_diagonal(exact, np.inf)
    np.testing.assert_array_equal(ids, np.argsort(exact, axis=1)[:, :15])
    np.testing.assert_allclose(dist, np.take_along_axis(exact, ids, axis=1), rtol=2e-6, atol=3e-6)


def test_fisher_span_recovers_group_signal_and_excludes_dominant_within_group_noise():
    dim = 14
    means = np.zeros((10, dim))
    means[:, :10] = np.eye(10) - .1
    jitter = np.diag([.1]*10 + [50.]*4)
    offsets = np.vstack((jitter, -jitter))
    points = (means[:, None] + offsets[None]).reshape(-1, dim)
    groups = np.repeat(np.arange(10), len(offsets))
    basis, _ = fisher_basis(points, groups)
    selector = {"mean": points.mean(0), "selected": basis}
    retained = coordinates(points, selector, "selected") @ basis.T
    remainder = coordinates(points, selector, "complement")
    np.testing.assert_allclose(retained + remainder, points - points.mean(0), atol=1e-7)
    np.testing.assert_allclose(remainder.reshape(10, len(offsets), dim).mean(1), 0, atol=1e-6)
    np.testing.assert_allclose(retained[:, 10:], 0, atol=1e-6)
    permuted, _ = fisher_basis(points, 9-groups)
    np.testing.assert_allclose(permuted @ permuted.T, basis @ basis.T, atol=1e-6)


def test_dense_decoder_targets_preserve_distances_and_match_pixel_scale():
    raw = np.random.default_rng(23).normal(size=(17, 9)).astype(np.float32)
    with threadpool_limits(1):
        mapped = reconstruction_target(raw, target_embedding(), 1.3)
    assert mapped.shape == (17, 784)
    np.testing.assert_allclose(mapped.mean(0), 0, atol=1e-6)
    np.testing.assert_allclose(np.square(mapped).mean(), 1.3**2, rtol=1e-6)
    original_gram = (raw - raw.mean(0)) @ (raw - raw.mean(0)).T
    target_gram = mapped @ mapped.T
    np.testing.assert_allclose(target_gram/np.trace(target_gram), original_gram/np.trace(original_gram), atol=2e-7)


def test_handwriting_style_measurements_are_in_original_pixel_coordinates():
    images = np.zeros((2, 28, 28), dtype=np.float32)
    images[0, 1, 2] = images[0, 5, 6] = 1
    images[1, 1, 9] = images[1, 5, 5] = .5
    np.testing.assert_allclose(style_values(images.reshape(2, -1)), [[2, 4, 3, 1], [1, 7, 3, -1]])


def test_literal_selection_requires_separation_in_both_halves_and_matches_locations():
    labels = np.tile(np.arange(10), 20)
    x = np.zeros((len(labels), 392))
    x[:, :98] = labels[:, None]
    x[:100, 98] = labels[:100]
    selector = {"agreed": np.ones(len(x), bool), "first": labels, "second": labels}
    chosen = select_bins(x, selector)
    np.testing.assert_array_equal(chosen["top"], np.arange(98))
    np.testing.assert_allclose(chosen["score"][:98], 1)
    assert chosen["half_a"][98] == 1 and chosen["score"][98] == 0
    np.testing.assert_array_equal(np.bincount(chosen["top"]//8, minlength=49),
                                  np.bincount(chosen["location_matched"]//8, minlength=49))


def test_cluster_style_readout_uses_training_cells_and_falls_back_to_digit_mean():
    y_train = np.repeat([0, 1], 40)
    cluster = np.tile(np.repeat([0, 1], 20), 2)
    y_test = np.array([0, 0, 1, 1])
    test_cluster = np.array([0, 1, 0, 1])
    labels = {"train": y_train, "test": y_test, "extra": y_test}
    residual = np.tile((2*test_cluster-1)[:, None], (1, 4))
    styles = {"train": np.tile((2*cluster-1)[:, None], (1, 4)), "test": residual, "extra": residual}
    pred = {"idec_train": cluster, "idec_test": test_cluster, "idec_extra": np.full(4, 9)}
    actual = assignment_style_scores(pred, labels, styles)
    np.testing.assert_allclose(actual["test"], 8/9)
    np.testing.assert_allclose(actual["extra"], 0)
