import numpy as np
import torch
from threadpoolctl import threadpool_limits

from clean_neighborhood_study import fit
from dec_background_study import fit_clusters
from neighborhood_weights import reconstruction_weights
from varied_study import pretrain


def test_context_measurement_matches_a_direct_distance_calculation():
    rng = np.random.default_rng(67)
    reference = rng.random((37, 28, 28))
    query = rng.random((1, 28, 28))
    original = query.copy()
    actual = reconstruction_weights(reference, query, "cpu")
    pooled = reference.reshape(-1, 7, 4, 7, 4).mean((2, 4))
    query_pooled = query[0].reshape(7, 4, 7, 4).mean((1, 3))
    expected = np.empty((28, 28))
    for top in range(0, 28, 7):
        for left in range(0, 28, 7):
            visible = np.ones((28, 28), dtype=bool)
            visible[max(0, top-4):top+11, max(0, left-4):left+11] = False
            keep = visible.reshape(7, 4, 7, 4).all((1, 3))
            distances = np.square(pooled[:, keep] - query_pooled[keep]).sum(1)
            nearest = np.argsort(distances)[:20]
            region = np.s_[top:top+7, left:left+7]
            values = reference[:, region[0], region[1]]
            obs = original[0][region]
            gain = np.square(obs - values.mean(0)) - np.square(obs - values[nearest].mean(0))
            expected[region] = 0.05 + np.maximum(0, gain)
    np.testing.assert_allclose(actual[0], (expected / expected.mean()).ravel(), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(query, original)


def test_uniform_and_unit_weight_objectives_match_the_existing_clean_fit():
    rng = np.random.default_rng(13)
    images = torch.from_numpy(rng.random((17, 8), dtype=np.float32))
    cfg = dict(dims=[8, 5, 2], clusters=2, pretrain_epochs=2, batch_size=5, updates=8,
               kmeans_restarts=3, target_interval=3, gamma=.1, lr=.001)
    with threadpool_limits(1):
        previous, initial = pretrain(images, images, "none", cfg, 22)
        centers, _ = fit_clusters(previous, initial, images, "uniform", cfg)
        for weights in (None, torch.ones_like(images)):
            model, result = fit(images, weights, cfg, 22)
            torch.testing.assert_close(result, centers, rtol=0, atol=0)
            for parameter, expected in zip(model.parameters(), previous.parameters()):
                torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
