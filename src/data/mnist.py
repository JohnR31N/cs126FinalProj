"""Original MNIST IDX files from the mirror used by torchvision, with MD5 checks."""
import gzip
import hashlib
import struct
from pathlib import Path

import numpy as np
import requests
import torch

from src.utils.common import ROOT, save_json, sha256

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def load_mnist():
    root = ROOT / "data" / "mnist"
    root.mkdir(parents=True, exist_ok=True)
    arrays, manifest = [], []
    for name, expected in FILES.items():
        path = root / name
        if not path.exists():
            r = requests.get(BASE + name, timeout=90)
            r.raise_for_status()
            assert hashlib.md5(r.content).hexdigest() == expected, name
            path.write_bytes(r.content)
        assert hashlib.md5(path.read_bytes()).hexdigest() == expected, name
        data = gzip.decompress(path.read_bytes())
        magic, n = struct.unpack(">II", data[:8])
        if magic == 2051:
            rows, cols = struct.unpack(">II", data[8:16])
            a = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows * cols).astype(np.float32) / 255
        else:
            assert magic == 2049
            a = np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)
        arrays.append(a)
        manifest.append(dict(file=name, url=BASE + name, md5=expected, sha256=sha256(path)))
    save_json(root / "manifest.json", manifest)
    return arrays


def split_indices(cfg):
    ids = np.random.default_rng(cfg["split_seed"]).permutation(60000)
    n, v = cfg["train_size"], cfg["validation_size"]
    return ids[:n], ids[n:n + v]


def gcn(x):
    """Per-image L1 global contrast normalization; no cross-sample statistics."""
    z = x - x.mean(dim=1, keepdim=True)
    return z / z.abs().mean(dim=1, keepdim=True).clamp_min(1e-12)


MIN_MAX = [(-0.8826567065619495, 9.001545489292527),
           (-0.6661464580883915, 20.108062262467364),
           (-0.7820454743183202, 11.665100841080346),
           (-0.7645772083211267, 12.895051191467457),
           (-0.7253923114302238, 12.683235701611533),
           (-0.7698501867861425, 13.103278415430502),
           (-0.778418217980696, 10.457837397569108),
           (-0.7129780970522351, 12.057777597673047),
           (-0.8280402650205075, 10.581538445782988),
           (-0.7369959242164307, 10.697039838804978)]


def svdd_scale(x, normal):
    lo, hi = MIN_MAX[normal]
    return ((gcn(x) - lo) / (hi - lo)).reshape(-1, 1, 28, 28)
