"""Render random symbol attributes without modifying any MNIST handwriting."""
import json
from itertools import product

import numpy as np
from scipy.ndimage import maximum_filter

from geometry_study import placements
from idec_final.data import load_mnist
from idec_final.utils import ROOT


def template_bank():
    axis = (np.arange(40) + .5) / 8 - 2.5
    x, y = np.meshgrid(axis, axis)
    templates, attributes = [], []
    for shape, radius, angle, intensity in product(range(4), (2., 2.4), (0, 20, 40, 60), (.6, .8, 1.)):
        radians = np.deg2rad(angle)
        u = (x * np.cos(radians) + y * np.sin(radians)) / radius
        v = (-x * np.sin(radians) + y * np.cos(radians)) / radius
        square = (np.abs(u) <= np.sqrt(.5)) & (np.abs(v) <= np.sqrt(.5))
        masks = (square, np.abs(u)+np.abs(v) <= 1,
                 (v <= .5) & (v >= -1 + np.sqrt(3)*np.abs(u)),
                 square & ((u <= -.08) | (v >= .08)))
        pixels = masks[shape].reshape(5, 8, 5, 8).mean(axis=(1, 3)) * intensity
        templates.append(pixels)
        attributes.append((shape, radius, angle, intensity))
    bank = np.asarray(templates, dtype=np.float32)
    circle = (x*x+y*y <= 2.5**2).reshape(5, 8, 5, 8).mean(axis=(1, 3)).astype(np.float32)
    energy = np.square(bank).sum(axis=(1, 2))
    fixed = circle[None] * np.sqrt(energy / np.square(circle).sum())[:, None, None]
    np.testing.assert_allclose(np.square(fixed).sum(axis=(1, 2)), energy, rtol=3e-7)
    assert len(bank) == 96 and 0 < bank.max() <= 1 and fixed.max() <= 1
    return bank, fixed, np.asarray(attributes, dtype=np.float32)


def render(clean, boxes, types, bank):
    n = len(clean)
    image = clean.reshape(n, 28, 28).copy()
    rows, cols = np.indices((5, 5))
    for slot in range(2):
        top, left = boxes[:, slot].T
        image[np.arange(n)[:, None, None], top[:, None, None]+rows, left[:, None, None]+cols] = bank[types[:, slot]]
    return image.reshape(n, -1)


def dataset():
    train, train_y, test, test_y = load_mnist()
    cfg = json.loads((ROOT/"protocol.json").read_text())
    ids = np.random.default_rng(cfg["split_seed"]).permutation(len(train))[:cfg["train_size"]]
    bank, fixed, attributes = template_bank()
    result = {}
    metadata = {"train_ids": ids, "bank": bank, "fixed_bank": fixed, "attributes": attributes}
    for split, clean, labels, site_seed, attribute_seed, count in (
        ("train", train[ids], train_y[ids], 2701, 5701, 1),
        ("test", test, test_y, 2702, 5702, 3),
    ):
        boxes = placements(clean, site_seed)
        rng = np.random.default_rng(attribute_seed)
        types = np.stack([rng.permutation(np.arange(len(clean)*2) % len(bank)).reshape(-1, 2) for _ in range(count)])
        varied_views = np.stack([render(clean, boxes, t, bank) for t in types])
        fixed_views = np.stack([render(clean, boxes, t, fixed) for t in types])
        guard = maximum_filter(clean.reshape(-1, 28, 28)>0, size=(1, 3, 3), mode="constant").reshape(-1, 784)
        for variants in (varied_views, fixed_views):
            for view in variants:
                assert not ((view != clean) & guard).any()
        np.testing.assert_allclose(np.square(varied_views-clean).sum(2), np.square(fixed_views-clean).sum(2), rtol=4e-7)
        result[split] = {"clean": clean, "fixed": fixed_views, "varied": varied_views}
        metadata[split+"_digit"], metadata[split+"_boxes"], metadata[split+"_types"] = labels, boxes, types
    return cfg, result, metadata
