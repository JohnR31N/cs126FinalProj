"""Post-fit interpretation of retained and remaining features. Digit labels are evaluation data."""
import json

import numpy as np
from scipy.linalg import solve

from clean_neighborhood_study import save_predictions
from cluster_feature_selection import FEATURE_NAMES, coordinates, stroke_features
from cluster_feature_study import SPLITS, original_splits
from idec_final.utils import ROOT


STYLE_NAMES = ("ink_mass", "center_x", "center_y", "slant")


def style_values(images):
    y, x = np.mgrid[:28, :28]
    x, y = x.ravel(), y.ravel()
    moments = np.stack((np.ones(784), x, y, x*y, y*y)).astype(np.float32)
    raw = (images @ moments.T).astype(np.float64)
    mass = raw[:, 0]
    mx, my, mxy, myy = (raw[:, 1:] / mass[:, None]).T
    slant = (mxy - mx*my) / np.maximum(myy - my*my, 1e-8)
    return np.column_stack((mass, mx, my, slant))


def ridge(x, y):
    """Scale the ridge to .01 times the mean diagonal of the training Gram matrix."""
    gram = x.T @ x
    gram.flat[::len(gram)+1] += 0.01 * np.trace(gram) / len(gram)
    return solve(gram, x.T @ y, assume_a="pos")


def fit_probes(out, device):
    cfg = json.loads((ROOT / "protocol.json").read_text())
    images, labels = original_splits(cfg)
    with np.load(out / "evaluation.npz") as archived:
        for key, array in labels.items():
            np.testing.assert_array_equal(array, archived[key])
    with np.load(out / "selector.npz") as stored:
        selector = {name: stored[name] for name in stored.files}
    feats = {s: stroke_features(images[s], device) for s in SPLITS}
    y = {s: labels[s+"_digit"] for s in SPLITS}
    style = {s: style_values(images[s]) for s in SPLITS}
    style_means = np.stack([style["train"][y["train"] == k].mean(0) for k in range(10)])
    style = {s: style[s] - style_means[y[s]] for s in SPLITS}
    saved = {s+"_style": style[s].astype(np.float32) for s in ("test", "extra")}
    one_hot = np.eye(10)[y["train"]]
    digit_mean = one_hot.mean(0)
    for name in FEATURE_NAMES:
        train = coordinates(feats["train"], selector, name).astype(np.float64)
        overall = train.mean(0)
        digit_fit = ridge(train - overall, one_hot - digit_mean)
        means = np.stack([train[y["train"] == k].mean(0) for k in range(10)])
        style_fit = ridge(train - means[y["train"]], style["train"])
        for s in ("test", "extra"):
            query = coordinates(feats[s], selector, name).astype(np.float64)
            digit_scores = (query - overall) @ digit_fit + digit_mean
            saved[f"{name}_{s}_digit"] = digit_scores.argmax(1).astype(np.uint8)
            saved[f"{name}_{s}_style"] = ((query - means[y[s]]) @ style_fit).astype(np.float32)
    # Show what the tails of the first three retained coordinates contain on training data.
    selected = coordinates(feats["train"], selector, "selected")
    n = int(.05 * len(selected))
    tails = np.argsort(selected[:, :3], axis=0)
    ids = np.stack((tails[:n].T, tails[-n:].T), axis=1)
    saved["tail_ids"] = ids
    saved["tail_images"] = images["train"][ids].mean(2).reshape(3, 2, 28, 28)
    save_predictions(out / "probes.npz", **saved)


def score_probes(out, labels):
    results = {}
    with np.load(out / "probes.npz") as p:
        expected = {f"{n}_{s}_{kind}" for n in FEATURE_NAMES for s in ("test", "extra")
                    for kind in ("digit", "style")} | {"test_style", "extra_style", "tail_ids", "tail_images"}
        assert set(p.files) == expected
        for s in ("test", "extra"):
            actual = p[s+"_style"].astype(np.float64)
            assert actual.shape == (len(labels[s]), 4) and np.isfinite(actual).all()
            denom = np.square(actual).sum(0)
            results[s] = {}
            for name in FEATURE_NAMES:
                pred = p[f"{name}_{s}_style"].astype(np.float64)
                digit = p[f"{name}_{s}_digit"]
                assert pred.shape == actual.shape and np.isfinite(pred).all()
                assert digit.shape == labels[s].shape and digit.dtype == np.uint8 and digit.max() < 10
                r2 = 1 - np.square(actual - pred).sum(0) / denom
                results[s][name] = {"linear_digit_acc_pct": float(100*np.mean(digit == labels[s])),
                                    "within_digit_style_r2": dict(zip(STYLE_NAMES, r2.tolist()))}
        ids = p["tail_ids"]
        assert ids.shape == (3, 2, 600) and ids.min() >= 0 and ids.max() < 12000
        assert all(len(np.unique(pair.ravel())) == 1200 for pair in ids)
        assert p["tail_images"].shape == (3, 2, 28, 28) and np.isfinite(p["tail_images"]).all()
        tails = [{end: np.bincount(labels["train"][ix], minlength=10).tolist()
                  for end, ix in zip(("low", "high"), pair)} for pair in ids]
    return results, tails
