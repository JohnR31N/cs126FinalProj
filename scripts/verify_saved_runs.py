"""Reload delivered checkpoints and reproduce predictions, not just metric arithmetic."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from sklearn.cluster import KMeans

from src.data.mnist import load_mnist, svdd_scale
from src.method.cae import ConcreteAE
from src.method.idec import Autoencoder, student_q
from src.method.svdd_network import MNIST_LeNet
from src.eval.metrics import clustering
from src.utils.common import ROOT, predict, save_json, seed_all


def main():
    seed_all(0)
    x,y,xt,yt=load_mnist()
    pilot=ROOT/"results/pilot"
    ids=np.load(pilot/"split.npz")["train"]
    checked=[]
    kmeans_refit_diagnostics=[]
    with threadpool_limits(4), torch.no_grad():
        for path in sorted(pilot.glob("*/*/result.json")):
            r=json.loads(path.read_text())
            model_path=path.parent/"model.pt"
            if r["method"] == "deep_svdd":
                state=torch.load(model_path,map_location="cuda",weights_only=True)
                net=MNIST_LeNet().cuda(); net.load_state_dict(state["net"])
                test=svdd_scale(torch.tensor(xt,device="cuda"),r["normal_class"])
                scores=(predict(net,test)-state["center"]).square().sum(1).cpu().numpy()
                saved=np.load(path.parent/"scores.npz")["scores"]
                np.testing.assert_allclose(scores,saved,rtol=1e-5,atol=1e-7)
                checked.append(dict(run=str(path.parent.relative_to(ROOT)),max_score_difference=float(abs(scores-saved).max())))
            elif r["method"] in ["idec","dec_ablation"]:
                state=torch.load(model_path,map_location="cuda",weights_only=True)
                net=Autoencoder(r["config"]["dims"]).cuda(); net.load_state_dict(state["ae"])
                tx=torch.tensor(x[ids]*(255/50),device="cuda")
                test=torch.tensor(xt*(255/50),device="cuda")
                labels=student_q(predict(net.encoder,tx),state["centers"]).argmax(1).cpu().numpy()
                test_labels=student_q(predict(net.encoder,test),state["centers"]).argmax(1).cpu().numpy()
                saved=np.load(path.parent/"predictions.npz")
                np.testing.assert_array_equal(labels,saved["train_labels"])
                np.testing.assert_array_equal(test_labels,saved["test_labels"])
                checked.append(dict(run=str(path.parent.relative_to(ROOT)),predictions_identical=True))
            elif r["method"] == "ae_kmeans":
                cfg=json.loads((pilot/"config.json").read_text())
                net=Autoencoder(cfg["idec"]["dims"]).cuda()
                net.load_state_dict(torch.load(model_path,map_location="cuda",weights_only=True))
                tx=torch.tensor(x[ids]*(255/50),device="cuda")
                labels=KMeans(n_clusters=10,n_init=20,random_state=r["seed"]).fit_predict(predict(net.encoder,tx).cpu().numpy())
                metrics=clustering(y[ids],labels)
                diagnostic=dict(run=str(path.parent.relative_to(ROOT)),original={k:r[k] for k in metrics},
                                refit=metrics,difference={k:metrics[k]-r[k] for k in metrics})
                kmeans_refit_diagnostics.append(diagnostic)
                checked.append(dict(run=str(path.parent.relative_to(ROOT)),autoencoder_loaded=True,
                                    kmeans_refit_is_diagnostic=True))
        for root in [pilot,ROOT/"results/budget1600"]:
            for path in sorted(root.glob("cae/cae*/result.json")):
                r=json.loads(path.read_text())
                state=torch.load(path.parent/"model.pt",map_location="cpu",weights_only=True)
                net=ConcreteAE(784,50); net.load_state_dict(state)
                indices=net.logits.argmax(1).numpy()
                np.testing.assert_array_equal(indices,r["indices"])
                confidence=net.probabilities().max(1).values.mean().item()
                assert abs(confidence-r["mean_max_probability"])<1e-6
                checked.append(dict(run=str(path.parent.relative_to(ROOT)),indices_identical=True))
    assert len(checked)==30, len(checked)
    save_json(ROOT/"results/checkpoint_verification.json",dict(checkpoints_loaded=len(checked),
              final_checkpoints_verified=27,checks=checked,kmeans_refit_diagnostics=kmeans_refit_diagnostics,
              passed=True,note="KMeans refitting can vary in borderline assignments; original results are retained."))
    print("PASS: 27 final checkpoints reproduce saved scores, labels or feature indices; 3 AE checkpoints also loaded.")
    print("KMeans refit differences are recorded as diagnostics, not claimed to be bitwise identical.")


if __name__=="__main__":
    main()
