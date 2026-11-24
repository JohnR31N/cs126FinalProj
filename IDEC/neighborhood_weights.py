"""Content-dependent reconstruction weights computed from unlabelled, unmodified images."""
import numpy as np
import torch
from torch.nn.functional import avg_pool2d


@torch.no_grad()
def reconstruction_weights(reference, query, device):
    """Score query images against a disjoint set of reference images scaled to [0, 1]."""
    ref = torch.as_tensor(reference, device=device).reshape(-1, 28, 28)
    qry = torch.as_tensor(query, device=device).reshape(-1, 28, 28)
    ref_features = avg_pool2d(ref[:, None], 4).flatten(1)
    qry_features = avg_pool2d(qry[:, None], 4).flatten(1)
    axis = torch.arange(0, 28, 4, device=device)
    row, col = torch.meshgrid(axis, axis, indexing="ij")
    weights = torch.empty_like(qry)
    block, margin, neighbors = 7, 4, 20
    for top in range(0, 28, block):
        for left in range(0, 28, block):
            rows = slice(top, top + block)
            cols = slice(left, left + block)
            # Exclude cells touching the scored region or its guard band from the distance.
            close_row = (row < rows.stop + margin) & (row + 4 > rows.start - margin)
            close_col = (col < cols.stop + margin) & (col + 4 > cols.start - margin)
            visible = ~(close_row & close_col).flatten()
            ref_context = ref_features[:, visible].contiguous()
            qry_context = qry_features[:, visible].contiguous()
            ref_norm = ref_context.square().sum(1)
            ref_block = ref[:, rows, cols].contiguous()
            marginal = ref_block.mean(0)
            for start in range(0, len(qry), 256):
                batch = slice(start, start + 256)
                # The squared query norm is common to every candidate in the same row.
                distances = torch.addmm(ref_norm[None], qry_context[batch], ref_context.T, alpha=-2)
                ids = distances.topk(neighbors, dim=1, largest=False).indices
                prediction = ref_block[ids].mean(1)
                observed = qry[batch, rows, cols]
                gain = (observed - marginal).square() - (observed - prediction).square()
                weights[batch, rows, cols] = 0.05 + gain.clamp_min(0)
    weights /= weights.mean((1, 2), keepdim=True)
    return weights.flatten(1).cpu().numpy()


def training_weights(images, device):
    """Compute one fixed map per training image using the opposite fold as its reference."""
    folds = np.array_split(np.arange(len(images)), 2)
    weights = np.empty_like(images)
    for fold in (0, 1):
        query, reference = folds[fold], folds[1 - fold]
        weights[query] = reconstruction_weights(images[reference], images[query], device)
    assert np.isfinite(weights).all() and weights.min() > 0
    np.testing.assert_allclose(weights.mean(1), 1, rtol=5e-6)
    return weights
