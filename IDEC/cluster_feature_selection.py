"""Select continuous stroke features using agreement between two unlabeled partitions."""
import numpy as np
import torch
from scipy import linalg, sparse
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from torch.nn import functional as F


FEATURE_NAMES = ("all_strokes", "selected", "complement", "pca", "permuted")


@torch.no_grad()
def stroke_features(images, device):
    rows = []
    bins = torch.arange(8, device=device).reshape(1, 8, 1, 1)
    for start in range(0, len(images), 512):
        image = torch.as_tensor(images[start:start+512], device=device).reshape(-1, 1, 28, 28)
        padded = F.pad(image, (1, 1, 1, 1), mode="replicate")
        dx = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
        dy = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
        orientation = torch.atan2(dy, dx).remainder(torch.pi) * (8 / torch.pi)
        distance = (orientation - bins).abs()
        votes = (1 - torch.minimum(distance, 8 - distance)).clamp_min(0)
        hist = F.avg_pool2d(votes * torch.hypot(dx, dy), 4)
        features = hist.sqrt().permute(0, 2, 3, 1).reshape(len(image), -1)
        rows.append(F.normalize(features, dim=1).cpu().numpy())
    return np.concatenate(rows)


@torch.no_grad()
def nearest(reference, query, device, same=False):
    ref = torch.as_tensor(reference, device=device)
    ref_norm = ref.square().sum(1)
    all_ids, all_dist = [], []
    for start in range(0, len(query), 256):
        batch = torch.as_tensor(query[start:start+256], device=device)
        dist = torch.addmm(ref_norm[None], batch, ref.T, alpha=-2) + batch.square().sum(1)[:, None]
        if same:
            index = torch.arange(len(batch), device=device)
            dist[index, start + index] = torch.inf
        values, ids = dist.topk(15, largest=False, sorted=True)
        all_ids.append(ids.cpu().numpy())
        all_dist.append(values.clamp_min(0).cpu().numpy())
    return np.concatenate(all_ids), np.concatenate(all_dist)


def spectral_groups(features, device, seed):
    ids, squared = nearest(features, features, device, same=True)
    widths = np.maximum(squared[:, -1], 1e-8)
    affinity = np.exp(-squared / np.sqrt(widths[:, None] * widths[ids]))
    rows = np.repeat(np.arange(len(ids)), ids.shape[1])
    graph = sparse.csr_matrix((affinity.ravel(), (rows, ids.ravel())), shape=(len(ids), len(ids)))
    graph = graph.maximum(graph.T).astype(np.float64)
    degree = np.asarray(graph.sum(1)).ravel()
    inv_sqrt = sparse.diags(1 / np.sqrt(degree))
    eig, vectors = eigsh(inv_sqrt @ graph @ inv_sqrt, k=10, which="LA",
                         v0=np.random.default_rng(seed).normal(size=len(ids)), tol=1e-8)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    labels = KMeans(n_clusters=10, n_init=20, random_state=seed).fit_predict(unit)
    return labels, eig


def vote(reference, groups, query, device):
    ids, _ = nearest(reference, query, device)
    return np.eye(10, dtype=np.uint8)[groups[ids]].sum(axis=1, dtype=np.int32)


def fisher_basis(features, groups):
    """Return an orthonormal basis for the top between/within-group discriminant span."""
    x = features.astype(np.float64)
    sizes = np.bincount(groups, minlength=10)
    if np.any(sizes < 2):
        raise ValueError("Every consensus group needs at least two training examples.")
    means = np.stack([x[groups == group].mean(0) for group in range(10)])
    diff = means - x.mean(0)
    between = (diff.T * (sizes / len(x))) @ diff
    centered = x - means[groups]
    within = centered.T @ centered / len(x)
    within.flat[::within.shape[0]+1] += 0.1 * np.trace(within) / within.shape[0]
    eig, directions = linalg.eigh(between, within, subset_by_index=(x.shape[1]-9, x.shape[1]-1))
    if eig[0] <= 0:
        raise ValueError("The consensus partition does not support nine discriminant directions.")
    basis, _ = np.linalg.qr(directions[:, ::-1])
    return basis.astype(np.float32), eig[::-1]


def fit_selector(train, device):
    half = len(train) // 2
    a, b = train[:half], train[half:]
    ga, ea = spectral_groups(a, device, 126)
    gb, eb = spectral_groups(b, device, 127)
    first = np.concatenate((ga, vote(a, ga, b, device).argmax(1)))
    second = np.concatenate((vote(b, gb, a, device).argmax(1), gb))
    overlap = np.bincount(second * 10 + first, minlength=100).reshape(10, 10)
    _, mapping = linear_sum_assignment(overlap, maximize=True)
    second = mapping[second]
    agreed = first == second
    accepted = train[agreed]
    groups = first[agreed]
    selected, eigenvalues = fisher_basis(accepted, groups)
    permuted, _ = fisher_basis(accepted, np.random.default_rng(8126).permutation(groups))
    cent = accepted.astype(np.float64) - accepted.mean(0, dtype=np.float64)
    _, pca = linalg.eigh(cent.T @ cent, subset_by_index=(cent.shape[1]-9, cent.shape[1]-1))
    return {
        "mean": train.mean(0, dtype=np.float64).astype(np.float32), "selected": selected,
        "pca": pca[:, ::-1].astype(np.float32), "permuted": permuted,
        "eigenvalues": eigenvalues, "graph_eigenvalues": np.stack((ea, eb)),
        "agreed": agreed, "first": first.astype(np.uint8), "second": second.astype(np.uint8),
    }


def graph_predictions(train, selector, query, device):
    half = len(train) // 2
    first = vote(train[:half], selector["first"][:half], query, device)
    second = vote(train[half:], selector["second"][half:], query, device)
    return (first + second).argmax(1).astype(np.uint8)


def coordinates(features, selector, name):
    x = features - selector["mean"]
    if name == "all_strokes":
        return x
    if name == "complement":
        basis = selector["selected"]
        return x - (x @ basis) @ basis.T
    if name not in ("selected", "pca", "permuted"):
        raise ValueError(f"Unknown feature choice {name}")
    return x @ selector[name]


def target_embedding():
    # Dense orthonormal coordinates give every arm the same 784-output decoder.
    q, _ = np.linalg.qr(np.random.default_rng(9126).normal(size=(784, 392)))
    return q.astype(np.float32)


def reconstruction_target(features, embedding, pixel_rms):
    centered = features - features.mean(0, dtype=np.float64).astype(np.float32)
    energy = np.square(centered).sum(dtype=np.float64) / len(centered)
    scale = pixel_rms * np.sqrt(embedding.shape[0] / energy)
    return ((centered @ embedding[:, :features.shape[1]].T) * np.float32(scale)).astype(np.float32)
