"""Original MNIST IDX files from the mirror used by torchvision, with MD5 checks."""
import gzip
import hashlib
import struct
from pathlib import Path

import numpy as np
import requests
import os

from .utils import ROOT, save_json, sha256

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def load_mnist():
    root = Path(os.environ.get("MNIST_CACHE", ROOT / ".cache" / "mnist"))
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


