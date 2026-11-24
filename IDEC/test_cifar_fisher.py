import numpy as np
import pytest

from cifar_fisher_study import supported_bases


def test_fisher_uses_the_span_of_the_occupied_groups():
    rng = np.random.default_rng(126)
    groups = np.repeat([0, 4, 9], [20, 25, 30])
    x = rng.normal(size=(len(groups), 5))
    x[groups == 4, 0] += 3
    x[groups == 9, 1] += 3
    result = supported_bases(x, groups)
    np.testing.assert_array_equal(result["group_ids"], [0, 4, 9])
    assert result["fisher"].shape == result["pca"].shape == (5, 2)
    means = [x[groups == g].mean(0) for g in (0, 4, 9)]
    within = sum((x[groups == g] - mu).T @ (x[groups == g] - mu) for g, mu in zip((0, 4, 9), means)) / len(x)
    regularized = within + 0.1 * within.diagonal().mean() * np.eye(5)
    contrasts = np.stack((means[1] - means[0], means[2] - means[0])).T
    oracle, _ = np.linalg.qr(np.linalg.solve(regularized, contrasts))
    np.testing.assert_allclose(result["fisher"] @ result["fisher"].T, oracle @ oracle.T, atol=1e-6)
    _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
    np.testing.assert_allclose(result["pca"] @ result["pca"].T, vt[:2].T @ vt[:2], atol=1e-6)


def test_single_consensus_group_and_singleton_are_reported():
    x = np.arange(20, dtype=float).reshape(5, 4)
    for groups in ([1, 1, 1, 1, 1], [0, 0, 0, 0, 9]):
        with pytest.raises(ValueError, match="occupied consensus groups"):
            supported_bases(x, groups)
