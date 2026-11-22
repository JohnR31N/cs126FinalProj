import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batches(n, size, device, shuffle=True):
    order = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
    yield from order.split(size)


@torch.no_grad()
def predict(model, x, size=1024):
    model.eval()
    return torch.cat([model(b) for b in x.split(size)])


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
