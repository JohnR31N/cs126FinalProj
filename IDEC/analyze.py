"""Verify all predictions, regenerate tables and figures, and summarize paired effects."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, confusion_matrix, normalized_mutual_info_score

from idec_final.data import load_mnist
from idec_final.utils import ROOT, save_json

NAMES = {"initial":"AE + K-means", "idec":"IDEC", "projected":"Projected loss", "cluster_only":"Clustering only"}
STYLE = {"initial":("#818992","D",":"), "idec":("#204990","o","-"),
         "projected":("#c15122","s","-"), "cluster_only":("#17826d","^","--")}
METRICS = ["digit_acc","digit_nmi","digit_ari","stripe_nmi","flip_rate","train_acc","seconds"]


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def verify(train_y, test_y, orient, a, b, flipped, expected):
    matrix = confusion_matrix(train_y, a, labels=np.arange(10))
    digit, group = linear_sum_assignment(matrix, maximize=True)
    independent_optimum = matrix[digit,group].sum()
    # The estimator fixed cluster IDs as rows. Preserve that deterministic tie
    # convention and verify optimality separately with a transposed maximization.
    clusters, lookup = linear_sum_assignment(-matrix.T)
    assert matrix.T[clusters,lookup].sum() == independent_optimum
    actual = dict(train_acc=np.mean(train_y==lookup[a]), digit_acc=np.mean(test_y==lookup[b]),
                  digit_nmi=normalized_mutual_info_score(test_y,b),
                  digit_ari=adjusted_rand_score(test_y,b),
                  stripe_nmi=normalized_mutual_info_score(orient,b), flip_rate=np.mean(b!=flipped))
    np.testing.assert_allclose([expected[k] for k in actual], list(actual.values()), atol=1e-12, rtol=0)
    # Sensitivity only: range of test ACC among mappings equally optimal on
    # training. Neither extreme is chosen for the headline score.
    test_counts = confusion_matrix(test_y,b,labels=np.arange(10))
    bounds = []
    for sign in (-1,1):
        pair = linear_sum_assignment(matrix*(len(b)+1)+sign*test_counts,maximize=True)
        assert matrix[pair].sum() == independent_optimum
        bounds.append(float(test_counts[pair].sum()/len(b)))
    return bounds


def collect(out, cfg, metadata):
    rows = []
    for strength in cfg["strengths"]:
        for seed in cfg["seeds"]:
            folder = out / f"strength_{strength:.1f}" / f"seed_{seed}"
            scores = json.loads((folder/"scores.json").read_text())
            history = json.loads((folder/"training.json").read_text())
            assert set(scores) == set(NAMES)
            with np.load(folder/"predictions.npz", allow_pickle=False) as arrays:
                for method in NAMES:
                    a, b, f = [arrays[f"{method}_{k}"] for k in ("train","test","flipped")]
                    assert a.shape==(cfg["train_size"],) and b.shape==f.shape==(10000,)
                    assert max(a.max(),b.max(),f.max())<10
                    if not strength:
                        np.testing.assert_array_equal(b,f)
                    trace = history[method]
                    if method != "initial":
                        assert [h["step"] for h in trace] == list(range(0,cfg["updates"]+1,cfg["target_interval"]))
                    bounds = verify(metadata["train_digit"], metadata["test_digit"], metadata["test_orientation"],
                                    a,b,f,scores[method])
                    rows.append(dict(strength=strength, seed=seed, method=method,
                                     **scores[method], digit_acc_tie_min=bounds[0],digit_acc_tie_max=bounds[1],
                                     last_change=trace[-1]["assignment_change"] if trace else None))
    return rows


def summary(rows, cfg):
    groups, contrasts = [], []
    for strength in cfg["strengths"]:
        for method in NAMES:
            selected = [r for r in rows if r["strength"]==strength and r["method"]==method]
            row = dict(strength=strength, method=method, n=len(selected))
            for k in METRICS:
                vals=[r[k] for r in selected]
                row[k+"_mean"], row[k+"_sd"] = mean(vals), stdev(vals)
            groups.append(row)
        for other in ["projected","cluster_only"]:
            paired = {r["seed"]: {} for r in rows}
            for r in rows:
                if r["strength"]==strength:
                    paired[r["seed"]][r["method"]] = r
            for k in ("digit_acc","stripe_nmi","flip_rate"):
                diff = [v[other][k]-v["idec"][k] for v in paired.values()]
                contrasts.append(dict(strength=strength, comparison=other+" minus idec", metric=k,
                                      mean=mean(diff), sd=stdev(diff), differences=diff))
    return groups, contrasts


def style():
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "axes.labelcolor":"#28384a","axes.edgecolor":"#80909a","axes.axisbelow":True,
                         "figure.facecolor":"white","savefig.facecolor":"white"})


def write_latex_table(out, groups):
    lines = [r"\begin{tabular}{llrrrrr}",r"\toprule",
             r"$a$ & Method & Digit ACC (\%) & Digit NMI & Digit ARI & Stripe NMI & Flip (\%)\\",
             r"\midrule"]
    for row in groups:
        values = []
        for key,scale,precision in (("digit_acc",100,1),("digit_nmi",1,3),
                                    ("digit_ari",1,3),("stripe_nmi",1,3),("flip_rate",100,1)):
            avg,sd = row[key+"_mean"]*scale,row[key+"_sd"]*scale
            values.append(f"{avg:.{precision}f} $\\pm$ {sd:.{precision}f}")
        if row["method"]=="initial" and row["strength"]>0:
            lines.append(r"\addlinespace")
        lines.append(" & ".join([f"{row['strength']:.1f}",NAMES[row["method"]],*values])+r"\\")
    lines.extend([r"\bottomrule",r"\end{tabular}"])
    (out/"table.tex").write_text("\n".join(lines)+"\n",encoding="utf-8")


def flip_cases(out, figdir, test_x):
    """Descriptive cases at the predeclared strongest contrast, using seed zero."""
    with np.load(out/"evaluation.npz") as meta, np.load(out/"strength_0.8/seed_0/predictions.npz") as pred:
        flags = {m: pred[m+"_test"] != pred[m+"_flipped"] for m in ("idec","projected")}
        strata = [(False,False),(True,False),(False,True),(True,True)]
        names = ["Both stable","Only IDEC flips","Only projected flips","Both flip"]
        fig,axes=plt.subplots(2,4,figsize=(8.7,3.7),layout="constrained")
        stats = []
        wave=np.arange(28)//2%2-.5
        for j,((a,b),name) in enumerate(zip(strata,names)):
            indices=np.flatnonzero((flags["idec"]==a)&(flags["projected"]==b))
            axes[0,j].set_title(f"{name}\n{len(indices):,} images",fontsize=10)
            row={"case":name,"count":int(len(indices))}
            if len(indices):
                idx=int(indices[0]);row.update(index=idx,digit=int(meta["test_digit"][idx]))
                for i,pattern in enumerate([wave[:,None],wave[None,:]]):
                    axes[i,j].imshow(test_x[idx].reshape(28,28)+.8*pattern,
                                     cmap="gray",vmin=-.4,vmax=1.4,interpolation="nearest")
                axes[1,j].set_xlabel(f"Test #{idx}, digit {row['digit']}",fontsize=9)
            else:
                axes[0,j].text(.5,.5,"No cases",ha="center",va="center")
            stats.append(row)
            for ax in axes[:,j]:
                ax.set(xticks=[],yticks=[])
                for border in ax.spines.values(): border.set_visible(False)
        axes[0,0].set_ylabel("Horizontal");axes[1,0].set_ylabel("Vertical")
        assert sum(r["count"] for r in stats)==len(test_x)
        save_json(out/"flip_cases.json",stats)
        fig.savefig(figdir/"flip_cases.png",dpi=180);plt.close(fig)


def plots(figdir, rows, groups, cfg):
    figdir.mkdir(parents=True,exist_ok=True)
    style()
    fig, axes = plt.subplots(1,3,figsize=(11.7,3.55),layout="constrained")
    for ax,metric,title,factor in zip(axes,["digit_acc","stripe_nmi","flip_rate"],
                                     ["Digit accuracy (%)","Cluster–orientation NMI","Flip rate (%)"],[100,1,100]):
        for j,method in enumerate(NAMES):
            g=[r for r in groups if r["method"]==method]
            color,mark,line=STYLE[method]
            x=np.arange(3)+(j-1.5)*.058
            ax.errorbar(x,[r[metric+"_mean"]*factor for r in g],
                        [r[metric+"_sd"]*factor for r in g],color=color,marker=mark,linestyle=line,
                        linewidth=1.7,markersize=4,capsize=2.5,label=NAMES[method])
        ax.set(xticks=range(3),xticklabels=cfg["strengths"],xlabel="Stripe amplitude",title=title)
        ax.grid(axis="y",alpha=.18)
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=4,frameon=False)
    fig.savefig(figdir/"outcomes.png",dpi=185); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(9.8,3.1),sharey=True,layout="constrained")
    for strength,ax in zip(cfg["strengths"],axes):
        for seed in cfg["seeds"]:
            samples={r["method"]:r for r in rows if r["strength"]==strength and r["seed"]==seed}
            ax.plot([0,1], [100*samples[k]["digit_acc"] for k in ("idec","projected")],
                    marker="o",color=plt.cm.viridis(.12+.16*seed),alpha=.85,lw=1.4,label=f"Seed {seed}")
        ax.set(xlim=(-.23,1.23),xticks=[0,1],xticklabels=["IDEC","Projected"],title=f"Amplitude {strength:g}")
        ax.grid(axis="y",alpha=.18)
    axes[0].set_ylabel("Test digit accuracy (%)")
    fig.legend(*axes[0].get_legend_handles_labels(),loc="outside lower center",ncol=5,frameon=False,fontsize=9)
    fig.savefig(figdir/"paired.png",dpi=180);plt.close(fig)



def intervention_figure(figdir, cfg, test_x):
    figdir.mkdir(parents=True,exist_ok=True)
    base=test_x[0].reshape(28,28)
    wave=((np.arange(28)//2)%2)-.5
    fig,axes=plt.subplots(2,3,figsize=(6.8,3.5),layout="constrained")
    for i,(orientation,pattern) in enumerate([("Horizontal",wave[:,None]),("Vertical",wave[None,:])]):
        for j,strength in enumerate(cfg["strengths"]):
            ax=axes[i,j];ax.imshow(base+strength*pattern,cmap="gray",vmin=-.4,vmax=1.4,interpolation="nearest")
            if i==0: ax.set_title(f"Amplitude {strength:g}",fontsize=10)
            if j==0: ax.set_ylabel(orientation)
            ax.set(xticks=[],yticks=[])
            for border in ax.spines.values(): border.set_visible(False)
    fig.savefig(figdir/"intervention.png",dpi=165);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,default=ROOT/"results")
    parser.add_argument("--figures",type=Path,default=ROOT/"figures")
    parser.add_argument("--reference",type=Path,default=ROOT/"reference")
    args=parser.parse_args()
    out=args.input
    cfg=json.loads((ROOT/"protocol.json").read_text())
    assert json.loads((out/"manifest.json").read_text())["protocol"]==cfg
    train_x,train_y,test_x,test_y=load_mnist()
    del train_x
    with np.load(out/"evaluation.npz") as m:
        ids=np.random.default_rng(cfg["split_seed"]).permutation(len(train_y))[:cfg["train_size"]]
        np.testing.assert_array_equal(ids,m["train_ids"])
        np.testing.assert_array_equal(train_y[ids],m["train_digit"])
        np.testing.assert_array_equal(test_y,m["test_digit"])
        for split in ("train","test"):
            orient=m[f"{split}_orientation"]
            expected=np.random.default_rng(cfg[f"{split}_orientation_seed"]).permutation(np.arange(len(orient))%2)
            np.testing.assert_array_equal(orient,expected)
        rows=collect(out,cfg,m)
    ref=json.loads((args.reference/"result.json").read_text())
    with np.load(args.reference/"predictions.npz") as data:
        yy=np.concatenate([train_y,test_y]); np.testing.assert_array_equal(yy,data["digits"])
        for k in ("initial","final"):
            pp=data[k]
            mat=confusion_matrix(yy,pp)
            rr,cc=linear_sum_assignment(mat,maximize=True)
            np.testing.assert_allclose([mat[rr,cc].sum()/len(yy),normalized_mutual_info_score(yy,pp)],
                                       [ref[k]["acc"],ref[k]["nmi"]],atol=1e-12,rtol=0)
    groups, contrasts=summary(rows,cfg)
    csv_write(out/"runs.csv",rows);csv_write(out/"summary.csv",groups)
    write_latex_table(out,groups)
    save_json(out/"paired.json",contrasts)
    plots(args.figures,rows,groups,cfg)
    intervention_figure(args.figures,cfg,test_x)
    flip_cases(out,args.figures,test_x)
    save_json(out/"verification.json",dict(scored_models=len(rows),reference_images=ref["n"],
              seeds=cfg["seeds"],strengths=cfg["strengths"],metric_tolerance=1e-12,
              verified=["MNIST labels, partitions, orientation assignments","complete fine-tuning traces",
                        "six saved metrics per model","paired counterfactual predictions at zero amplitude",
                        "reference initial and final ACC and NMI"]))


if __name__=="__main__":
    main()
