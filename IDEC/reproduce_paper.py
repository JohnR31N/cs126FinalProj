"""MNIST transductive reproduction using Guo et al.'s published initialization."""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from threadpoolctl import threadpool_limits

from idec_final.data import load_mnist
from idec_final.model import Autoencoder, student_q, target_distribution
from idec_final.utils import ROOT, predict, save_json, seed_all, sha256, sync

URL = ("https://raw.githubusercontent.com/XifengGuo/data-and-models/"
       "d361bfe06686c0762df58624ef95a8c0fc9162be/ae_weights/mnist_ae_weights.h5")
DIGEST = "cfbe90aaf332c63ed0eeb1fbd1441d3d089249900ce0c9abf94169e25dcc8f6e"
LAYERS = ["encoder_1", "encoder_2", "encoder_3", "hidden",
          "decoder_3", "decoder_2", "decoder_1", "output"]


def initialized_model(device, probe):
    source = ROOT / ".cache/mnist_ae_weights.h5"
    if not source.exists():
        response = requests.get(URL, timeout=90)
        response.raise_for_status()
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(response.content)
    if sha256(source) != DIGEST:
        raise ValueError(f"Unexpected author weight file: {source}")
    model = Autoencoder([784, 500, 500, 2000, 10])
    linear = [layer for layer in model.modules() if isinstance(layer, torch.nn.Linear)]
    oracle = probe.copy()
    with h5py.File(source, "r") as weights, torch.no_grad():
        for i, (layer, name) in enumerate(zip(linear, LAYERS, strict=True)):
            group = weights[name][name]
            kernel, bias = group["kernel:0"][:], group["bias:0"][:]
            layer.weight.copy_(torch.from_numpy(kernel.T))
            layer.bias.copy_(torch.from_numpy(bias))
            oracle = oracle @ kernel + bias
            if i not in (3, 7):
                oracle = np.maximum(oracle, 0)
            if i == 3:
                embedding = oracle.copy()
        probe_t = torch.from_numpy(probe)
        z_error = float(np.abs(model.encoder(probe_t).numpy() - embedding).max())
        x_error = float(np.abs(model(probe_t).numpy() - oracle).max())
        np.testing.assert_allclose(model.encoder(probe_t).numpy(), embedding, atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(model(probe_t).numpy(), oracle, atol=1e-4, rtol=1e-4)
    return model.to(device), {"embedding_max_abs":z_error, "reconstruction_max_abs":x_error}


def metrics(y, pred):
    counts = np.zeros((10, 10), dtype=int)
    np.add.at(counts, (pred, y), 1)
    idx = linear_sum_assignment(-counts)
    return {"acc": float(counts[idx].sum()/len(y)),
            "nmi": float(normalized_mutual_info_score(y, pred))}


def run(out, device):
    seed = 0
    seed_all(seed)
    train, yt, test, yv = load_mnist()
    images = np.concatenate([train, test]) * (255.0/50)
    y = np.concatenate([yt, yv])
    model, parity = initialized_model(device, images[:8])
    x = torch.from_numpy(images).to(device)
    start = time.perf_counter()
    z = predict(model.encoder, x).cpu().numpy()
    km = KMeans(n_clusters=10, n_init=20, random_state=seed).fit(z)
    centers = torch.nn.Parameter(torch.from_numpy(km.cluster_centers_).to(device))
    previous = km.labels_
    opt = torch.optim.Adam([*model.parameters(), centers], lr=0.001)
    trace = []
    for step in range(20001):
        if step % 140 == 0 or step == 20000:
            with torch.no_grad():
                q = student_q(predict(model.encoder, x), centers)
                p = target_distribution(q)
                labels = q.argmax(1).cpu().numpy()
                delta = float(np.mean(labels != previous))
                trace.append(dict(step=step, change=delta))
                previous = labels
            if (step > 0 and delta < 0.001) or step == 20000:
                break
        offset = step % ((len(x)+255)//256) * 256
        batch = x[offset:offset+256]
        model.train()
        embed = model.encoder(batch)
        q = student_q(embed, centers)
        kl = torch.nn.functional.kl_div(q.log(), p[offset:offset+256], reduction="batchmean")
        loss = (model.decoder(embed)-batch).square().mean() + 0.1*kl
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sync()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/"predictions.npz", initial=km.labels_.astype("uint8"),
                        final=labels.astype("uint8"), digits=y.astype("uint8"))
    save_json(out/"trace.json",trace)
    result = dict(n=len(y), seed=seed, updates=step, last_change=delta,
                  initial=metrics(y,km.labels_), final=metrics(y,labels), seconds=time.perf_counter()-start,
                  converted_weight_parity=parity, weight_url=URL, weight_sha256=DIGEST,
                  torch=str(torch.__version__), device=str(device))
    save_json(out/"result.json",result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"reference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    with threadpool_limits(4):
        run(args.output, torch.device(args.device))
