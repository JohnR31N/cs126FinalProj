"""Ruff et al. one-class Deep SVDD; upstream CNN reused, tensor-based trainer."""
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data.mnist import svdd_scale
from src.method.svdd_network import MNIST_LeNet, MNIST_LeNet_Autoencoder
from src.utils.common import batches, event, predict, save_json, seed_all, sync


def initialize_center(z, eps=.1):
    c = z.mean(0).detach()
    c[(c.abs() < eps) & (c < 0)] = -eps
    c[(c.abs() < eps) & (c > 0)] = eps
    return c


def fit(model, x, epochs, c, cfg, pretrain=False):
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           weight_decay=cfg["ae_weight_decay"] if pretrain else cfg["weight_decay"])
    schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[cfg["lr_milestone"]], gamma=.1)
    history = []
    for epoch in range(epochs):
        model.train(); losses = []
        for ids in batches(len(x), cfg["batch_size"], x.device):
            b = x[ids]
            outputs = model(b)
            loss = (outputs-b).square().flatten(1).sum(1).mean() if pretrain else (outputs-c).square().sum(1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.detach())
        schedule.step()
        history.append(dict(epoch=epoch+1, loss=torch.stack(losses).mean().item(), lr=opt.param_groups[0]["lr"]))
    return history


def run(cfg, arrays, train_ids, val_ids, seed, out):
    c, device = cfg["svdd"], cfg["device"]
    x, y, xt, yt = arrays
    for normal in c["normal_classes"]:
        path = out / f"svdd_c{normal}_s{seed}"
        if (path / "result.json").exists():
            continue
        path.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        tx = svdd_scale(torch.tensor(x[y == normal], device=device), normal)
        test = svdd_scale(torch.tensor(xt, device=device), normal)
        ae = MNIST_LeNet_Autoencoder().to(device)
        sync(); start = time.perf_counter()
        ah = fit(ae, tx, c["ae_epochs"], None, c, pretrain=True)
        ae_scores = (predict(ae, test)-test).square().flatten(1).sum(1).cpu().numpy()
        event(f"SVDD seed={seed} class={normal} pretrain complete")
        net = MNIST_LeNet().to(device)
        net.load_state_dict({k: v for k, v in ae.state_dict().items() if k in net.state_dict()})
        center = initialize_center(predict(net, tx))
        sh = fit(net, tx, c["epochs"], center, c)
        sync(); seconds = time.perf_counter() - start
        scores = (predict(net, test)-center).square().sum(1).cpu().numpy()
        label = (yt != normal).astype(np.int64)
        result = dict(method="deep_svdd", dataset="MNIST", seed=seed, normal_class=normal,
                      train_n=len(tx), test_n=len(test), config=c, train_seconds=seconds,
                      auc=float(roc_auc_score(label, scores)), ae_auc=float(roc_auc_score(label, ae_scores)))
        torch.save(dict(net=net.state_dict(), ae=ae.state_dict(), center=center), path / "model.pt")
        np.savez_compressed(path / "scores.npz", scores=scores, ae_scores=ae_scores, labels=label)
        save_json(path / "history.json", dict(autoencoder=ah, svdd=sh))
        save_json(path / "result.json", result)
        event(f"DONE SVDD seed={seed} class={normal} AUC={result['auc']:.4f} AE={result['ae_auc']:.4f} {seconds:.1f}s")
