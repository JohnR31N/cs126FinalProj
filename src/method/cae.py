"""Independent PyTorch implementation of Balin et al. (2019), linear CAE variant."""
import time

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from torch import nn

from src.utils.common import batches, event, predict, save_json, seed_all, sync


class ConcreteAE(nn.Module):
    def __init__(self, d, k):
        super().__init__()
        self.logits = nn.Parameter(torch.empty(k, d))
        nn.init.xavier_normal_(self.logits)
        self.decoder = nn.Linear(k, d)
        nn.init.xavier_uniform_(self.decoder.weight)
        nn.init.zeros_(self.decoder.bias)
        self.temperature = 10.0

    def probabilities(self):
        return self.logits.softmax(-1)

    def forward(self, x):
        if self.training:
            u = torch.rand_like(self.logits).clamp(1e-7, 1 - 1e-7)
            g = -torch.log(-torch.log(u))
            selection = ((self.logits + g) / self.temperature).softmax(-1)
            z = x @ selection.T
        else:
            z = x[:, self.logits.argmax(-1)]
        return self.decoder(z)


def collision_penalty(logits):
    """Expected pairwise selector collision, per selector, with ordered pairs."""
    p = logits.softmax(-1)
    gram = p @ p.T
    return (gram.sum() - gram.diagonal().sum()) / len(p)


def evaluate_selection(indices, train, val, test, ytrain, ytest, seed):
    # OLS refit is the evaluation protocol of the paper's linear-CAE experiment.
    reg = LinearRegression().fit(train[:, indices], train)
    vp, tp = reg.predict(val[:, indices]), reg.predict(test[:, indices])
    clf = ExtraTreesClassifier(n_estimators=50, random_state=seed, n_jobs=4)
    clf.fit(train[:, indices], ytrain)
    return dict(val_mse=float(mean_squared_error(val, vp)),
                test_mse=float(mean_squared_error(test, tp)),
                classification_accuracy=float(accuracy_score(ytest, clf.predict(test[:, indices]))),
                unique_features=int(len(np.unique(indices))), indices=indices.tolist()), tp[:20]


def run(cfg, arrays, train_ids, val_ids, seed, out):
    c = cfg["cae"]
    x, y, xt, yt = arrays
    train, val = x[train_ids], x[val_ids]
    device = cfg["device"]
    tx, vx = torch.tensor(train, device=device), torch.tensor(val, device=device)
    rows = []
    for weight in c["collision_weights"]:
        name = "cae" if weight == 0 else "cae_collision"
        path = out / f"{name}_s{seed}"
        if (path / "result.json").exists():
            continue
        path.mkdir(parents=True, exist_ok=True)
        seed_all(seed)
        model = ConcreteAE(train.shape[1], c["features"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
        steps = c["epochs"] * int(np.ceil(len(train) / c["batch_size"]))
        history, step = [], 0
        sync(); start = time.perf_counter()
        for epoch in range(c["epochs"]):
            model.train()
            losses = []
            for ids in batches(len(tx), c["batch_size"], device):
                step += 1
                model.temperature = c["start_temp"] * (c["end_temp"] / c["start_temp"]) ** (step / steps)
                pred = model(tx[ids])
                mse = (pred - tx[ids]).square().mean()
                loss = mse + weight * collision_penalty(model.logits)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                losses.append(mse.detach())
            if epoch == 0 or (epoch + 1) % 20 == 0 or epoch + 1 == c["epochs"]:
                with torch.no_grad():
                    confidence = model.probabilities().max(-1).values.mean().item()
                    valid_mse = (predict(model, vx) - vx).square().mean().item()
                    unique = model.logits.argmax(-1).unique().numel()
                h = dict(epoch=epoch+1, train_stochastic_mse=torch.stack(losses).mean().item(),
                         val_joint_hard_mse=valid_mse, mean_max_probability=confidence,
                         unique_features=unique, temperature=model.temperature)
                history.append(h)
                if (epoch + 1) % 100 == 0:
                    event(f"{name} seed={seed} epoch={epoch+1} unique={unique}/50 confidence={confidence:.4f}")
        sync(); seconds = time.perf_counter() - start
        indices = model.logits.detach().argmax(-1).cpu().numpy()
        metrics, recon = evaluate_selection(indices, train, val, xt, y[train_ids], yt, seed)
        result = dict(method=name, seed=seed, dataset="MNIST", config=c, collision_weight=weight,
                      train_seconds=seconds, train_n=len(train), test_n=len(xt),
                      mean_max_probability=confidence, converged_099=confidence >= .99,
                      **metrics)
        torch.save(model.state_dict(), path / "model.pt")
        np.savez_compressed(path / "examples.npz", original=xt[:20], reconstructed=recon,
                            logits=model.logits.detach().cpu().numpy())
        save_json(path / "history.json", history)
        save_json(path / "result.json", result)
        event(f"DONE {name} seed={seed} MSE={metrics['test_mse']:.5f} acc={metrics['classification_accuracy']:.4f} {seconds:.1f}s")
    for name, indices in [("random", np.random.default_rng(seed).choice(784, c["features"], replace=False)),
                          ("variance", np.argsort(train.var(0))[-c["features"]:])]:
        path = out / f"{name}_s{seed}"
        if (path / "result.json").exists():
            continue
        metrics, recon = evaluate_selection(indices, train, val, xt, y[train_ids], yt, seed)
        save_json(path / "result.json", dict(method=name, seed=seed, dataset="MNIST", **metrics))
