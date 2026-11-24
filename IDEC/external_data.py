"""Download verified external benchmarks and build the common 28-by-28 inputs."""
import gzip
import hashlib
import struct
import tarfile
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import requests
import torch
from torch.nn.functional import interpolate


IDX = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
       "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")
SIZES = {"usps": (7291, 2007), "kmnist": (60000, 10000),
         "fashion": (60000, 10000), "cifar10_gray": (50000, 10000)}
BASE = {"usps": "https://hastie.su.domains/ElemStatLearn/datasets/",
        "kmnist": "https://codh.rois.ac.jp/kmnist/dataset/kmnist/",
        "fashion": "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
        "cifar10_gray": "https://cave.cs.toronto.edu/kriz/"}
FILES = {
    "usps": {"zip.train.gz": "206835541ea704bcfb4b6eea872678bf", "zip.test.gz": "f9350e940232b89f1dc8f8eacac88625"},
    "kmnist": dict(zip(IDX, ("bdb82020997e1d708af4cf47b453dcf7", "e144d726b3acfaa3e44228e80efcd344",
                             "5c965bf0a639b31b8f53240b1b52f4d7", "7320c461ea6c1c855c0b718fb2a4b134"), strict=True)),
    "fashion": dict(zip(IDX, ("8d4fb7e6c68d591d4c3dfef9ec88bf0d", "25c81989df183df01b3e8a0aad5dffbe",
                             "bef4ecab320f06d8554ea6380940ec79", "bb300cfdad3c16e7a12a480ee83cd310"), strict=True)),
    "cifar10_gray": {"cifar-10-binary.tar.gz": "c32a1d4ab5d03f1284b67883e8d87530"},
}


def verify_files(cache, dataset):
    folder = cache / dataset
    folder.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES[dataset].items():
        path = folder / name
        if path.exists():
            with path.open("rb") as src:
                if hashlib.file_digest(src, "md5").hexdigest() == expected:
                    continue
        with NamedTemporaryFile(dir=folder, delete=False) as dst:
            pending = Path(dst.name)
        try:
            with requests.get(BASE[dataset] + name, stream=True, timeout=(15, 120)) as response:
                response.raise_for_status()
                digest = hashlib.md5()
                with pending.open("wb") as dst:
                    for chunk in response.iter_content(262144):
                        digest.update(chunk)
                        dst.write(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Unexpected digest for {dataset}/{name}")
            pending.replace(path)
        finally:
            pending.unlink(missing_ok=True)
    return folder


def read_idx(path, expected_magic, count):
    with gzip.open(path, "rb") as f:
        if struct.unpack(">II", f.read(8)) != (expected_magic, count):
            raise ValueError(f"Invalid IDX header in {path}")
        if expected_magic == 2051 and struct.unpack(">II", f.read(8)) != (28, 28):
            raise ValueError(f"Unexpected image size in {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape((count, 784) if expected_magic == 2051 else (count,))


def to_gray28(images):
    converted = []
    for start in range(0, len(images), 512):
        x = torch.from_numpy(images[start:start + 512]).float()
        if x.shape[1] == 3:
            x = ((0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]) / 255).clamp(0, 1)
        converted.append(interpolate(x, (28, 28), mode="bilinear", align_corners=False,
                                     antialias=True).clamp(0, 1).flatten(1).numpy())
    return np.concatenate(converted)


def load_external(dataset, cache, cfg):
    folder = verify_files(cache, dataset)
    ntrain, ntest = SIZES[dataset]
    ids = np.random.default_rng(cfg["split_seed"]).permutation(ntrain)[:cfg["train_size"]]
    selection = (ids, slice(None))
    images, labels = {}, {}
    if dataset == "cifar10_gray":
        with tarfile.open(folder / "cifar-10-binary.tar.gz", "r:gz") as tar:
            def read_member(name):
                with tar.extractfile("cifar-10-batches-bin/" + name + ".bin") as f:
                    return np.frombuffer(f.read(), dtype=np.uint8).reshape(10000, 3073)
            train = np.concatenate([read_member(f"data_batch_{i}") for i in range(1, 6)])
            for split, data in (("train", train[ids]), ("test", read_member("test_batch"))):
                labels[split] = data[:, 0].copy()
                images[split] = to_gray28(data[:, 1:].copy().reshape(-1, 3, 32, 32))
    else:
        for i, (split, count) in enumerate((("train", ntrain), ("test", ntest))):
            sel = selection[i]
            if dataset == "usps":
                data = np.loadtxt(folder / f"zip.{split}.gz", dtype=np.float32)
                if data.shape != (count, 257):
                    raise ValueError(f"Unexpected USPS shape {data.shape}")
                data = data[sel]
                labels[split] = data[:, 0].astype(np.uint8)
                images[split] = to_gray28(((data[:, 1:] + 1) / 2).reshape(-1, 1, 16, 16))
            else:
                images[split] = read_idx(folder / IDX[2*i], 2051, count)[sel].astype(np.float32) / 255
                labels[split] = read_idx(folder / IDX[2*i+1], 2049, count)[sel]
    for split, size in (("train", len(ids)), ("test", ntest)):
        x = images[split]
        if x.shape != (size, 784) or x.dtype != np.float32 or not (0 <= x.min() <= x.max() <= 1):
            raise ValueError(f"Invalid {dataset}/{split} image array")
    return images, {"train_ids": ids, "train_class": labels["train"], "test_class": labels["test"]}
