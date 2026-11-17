"""IDEC core objective and paired no-reconstruction ablation; Guo et al. 2017."""
import copy
import time

import numpy as np
import torch
from sklearn.cluster import KMeans
from torch import nn

from src.eval.metrics import clustering
from src.utils.common import ROOT, batches, event, predict, save_json, seed_all, sha256, sync


def mlp(dims):
    layers = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layer = nn.Linear(a, b)
        nn.init.xavier_uniform_(layer.weight); nn.init.zeros_(layer.bias)
        layers.append(layer)
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.encoder, self.decoder = mlp(dims), mlp(list(reversed(dims)))

    def forward(self, x):
        return self.decoder(self.encoder(x))


def student_q(z, centers):
    q = 1 / (1 + (z[:, None, :] - centers[None, :, :]).square().sum(-1))
    return q / q.sum(1, keepdim=True)


def target_distribution(q):
    w = q.square() / q.sum(0, keepdim=True).clamp_min(1e-12)
    return w / w.sum(1, keepdim=True).clamp_min(1e-12)


def run(cfg, arrays, train_ids, val_ids, seed, out):
    c, device = cfg["idec"], cfg["device"]
    x, y, xt, yt = arrays
    # Upstream IDEC uses uint8 / 50, unlike CAE's uint8 / 255.
    tx = torch.tensor(x[train_ids] * (255.0 / 50.0), device=device)
    test = torch.tensor(xt * (255.0 / 50.0), device=device)
    save_json(out / "implementation.json", {str(f.relative_to(ROOT)): sha256(f)
              for f in (ROOT / "src").rglob("*.py")})
    seed_all(seed)
    prepath = out / f"pretrain_s{seed}"
    prepath.mkdir(parents=True, exist_ok=True)
    ae = Autoencoder(c["dims"]).to(device)
    if (prepath / "model.pt").exists():
        ae.load_state_dict(torch.load(prepath / "model.pt", map_location=device, weights_only=True))
    else:
        opt = torch.optim.Adam(ae.parameters(), lr=c["lr"])
        hist = []
        sync(); start = time.perf_counter()
        for epoch in range(c["pretrain_epochs"]):
            ae.train(); losses = []
            for ids in batches(len(tx), c["batch_size"], device):
                loss = (ae(tx[ids]) - tx[ids]).square().mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(loss.detach())
            hist.append(dict(epoch=epoch+1, mse=torch.stack(losses).mean().item()))
            if (epoch+1) % 25 == 0:
                event(f"IDEC pretrain seed={seed} epoch={epoch+1} mse={hist[-1]['mse']:.5f}")
        sync()
        torch.save(ae.state_dict(), prepath / "model.pt")
        save_json(prepath / "history.json", hist)
        save_json(prepath / "timing.json", dict(seconds=time.perf_counter()-start))
    z = predict(ae.encoder, tx).cpu().numpy()
    km = KMeans(n_clusters=10, n_init=20, random_state=seed).fit(z)
    initial = clustering(y[train_ids], km.labels_)
    save_json(prepath / "result.json", dict(method="ae_kmeans", seed=seed, **initial))
    for name, reconstruction_weight in [("idec", 1.0), ("dec_ablation", 0.0)]:
        path = out / f"{name}_s{seed}"
        if (path / "result.json").exists():
            continue
        path.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        model = copy.deepcopy(ae)
        centers = nn.Parameter(torch.tensor(km.cluster_centers_, device=device))
        opt = torch.optim.Adam(list(model.parameters()) + [centers], lr=c["lr"])
        last = km.labels_.copy()
        history, converged = [], False
        sync(); start = time.perf_counter()
        for step in range(c["max_steps"]):
            if step % c["update_interval"] == 0:
                with torch.no_grad():
                    q = student_q(predict(model.encoder, tx), centers)
                    p = target_distribution(q).detach()
                    pred = q.argmax(1).cpu().numpy()
                    delta = float(np.mean(pred != last)); last = pred.copy()
                    h = dict(step=step, delta=delta, **clustering(y[train_ids], pred))
                    history.append(h)
                if step > 0 and delta < c["tol"]:
                    converged = True; break
            # Match upstream fixed sequential minibatches and global target refresh.
            offset = (step % int(np.ceil(len(tx)/c["batch_size"]))) * c["batch_size"]
            b = tx[offset:offset+c["batch_size"]]
            pb = p[offset:offset+c["batch_size"]]
            model.train()
            zb = model.encoder(b)
            qb = student_q(zb, centers)
            kl = (pb * (pb.clamp_min(1e-12).log() - qb.clamp_min(1e-12).log())).sum(1).mean()
            rec = (model.decoder(zb) - b).square().mean()
            loss = c["gamma"] * kl + reconstruction_weight * rec
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sync(); seconds = time.perf_counter() - start
        with torch.no_grad():
            q = student_q(predict(model.encoder, tx), centers)
            pred = q.argmax(1).cpu().numpy()
            tp = student_q(predict(model.encoder, test), centers).argmax(1).cpu().numpy()
            reconstruction_mse = (predict(model, tx)-tx).square().mean().item()
        result = dict(method=name, seed=seed, dataset="MNIST", train_n=len(tx), config=c,
                      train_seconds=seconds, steps=step+1, converged=converged,
                      reconstruction_mse=reconstruction_mse, initial=initial,
                      heldout=clustering(yt, tp), **clustering(y[train_ids], pred))
        torch.save(dict(ae=model.state_dict(), centers=centers.detach()), path / "model.pt")
        np.savez_compressed(path / "predictions.npz", train_labels=pred, test_labels=tp)
        save_json(path / "history.json", history); save_json(path / "result.json", result)
        event(f"DONE {name} seed={seed} ACC={result['acc']:.4f} NMI={result['nmi']:.4f} {seconds:.1f}s")
