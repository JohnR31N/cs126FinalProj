"""Regenerate tables and static research figures exclusively from saved runs."""
import csv
import html
import itertools
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score

from src.data.mnist import load_mnist
from src.eval.metrics import clustering
from src.utils.common import ROOT, save_json


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v)>1 else 0., n=len(v))


def text_stat(stat, scale=1, digits=4):
    return f"{stat['mean']*scale:.{digits}f} ± {stat['std']*scale:.{digits}f}"


def build_report(root):
    records = []
    for path in sorted(root.glob("*/*/result.json")):
        r = read(path); r["path"] = str(path.relative_to(ROOT)); records.append(r)
    assert len(records) == 30, f"Expected 30 result records, found {len(records)}"
    groups = defaultdict(list)
    for r in records:
        key = r["method"] + (f"_c{r['normal_class']}" if "normal_class" in r else "")
        groups[key].append(r)
    for key, group in groups.items():
        assert sorted(r["seed"] for r in group) == [0,1,2], key
    summary = {}
    metrics = ["test_mse", "val_mse", "classification_accuracy", "unique_features", "mean_max_probability",
               "acc", "nmi", "ari", "auc", "ae_auc", "train_seconds", "reconstruction_mse"]
    for key, group in groups.items():
        summary[key] = {m: mean_std([r[m] for r in group]) for m in metrics if all(m in r for r in group)}
    paired = []
    for seed in [0,1,2]:
        a = next(r for r in groups["cae"] if r["seed"]==seed)
        b = next(r for r in groups["cae_collision"] if r["seed"]==seed)
        paired.append(dict(seed=seed, mse_difference=b["test_mse"]-a["test_mse"],
                           unique_difference=b["unique_features"]-a["unique_features"]))
    summary["cae_paired"] = dict(differences=paired, mse=mean_std([r["mse_difference"] for r in paired]))
    for key in ["cae", "cae_collision"]:
        sets = [set(r["indices"]) for r in groups[key]]
        summary[key]["pairwise_jaccard"] = mean_std([len(a & b)/len(a | b) for a,b in itertools.combinations(sets,2)])
        summary[key]["confidence_converged_seeds"] = sum(r["converged_099"] for r in groups[key])
    save_json(root / "summary.json", summary)
    cols = ["method", "seed", "normal_class"] + metrics + ["path"]
    with (root/"runs.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); writer.writeheader(); writer.writerows(records)

    # Recompute headline metrics independently from persisted raw predictions/scores.
    x, y, xt, yt = load_mnist()
    split = np.load(root/"split.npz")
    train_ids, val_ids = split["train"], split["validation"]
    # Exploratory, label-free diagnostic: different pixels may be correlated substitutes.
    for key in ["cae", "cae_collision"]:
        correlations=[]
        for left,right in itertools.combinations(sorted(groups[key],key=lambda r:r["seed"]),2):
            a=x[train_ids][:,left["indices"]].astype(np.float64)
            b=x[train_ids][:,right["indices"]].astype(np.float64)
            a-=a.mean(0); b-=b.mean(0)
            a/=np.maximum(np.linalg.norm(a,axis=0),1e-12)
            b/=np.maximum(np.linalg.norm(b,axis=0),1e-12)
            corr=np.abs(a.T@b)
            row,col=linear_sum_assignment(-corr)
            correlations.append(float(corr[row,col].mean()))
        summary[key]["matched_abs_train_correlation"] = mean_std(correlations)
    save_json(root/"summary.json",summary)
    from src.method.cae import evaluate_selection
    ols_max_absolute_difference = 0.0
    for r in records:
        path = ROOT/r["path"]
        if r["method"] == "deep_svdd":
            a = np.load(path.parent/"scores.npz")
            np.testing.assert_array_equal(a["labels"], yt != r["normal_class"])
            assert abs(roc_auc_score(a["labels"],a["scores"])-r["auc"])<1e-12
        if r["method"] in ["idec", "dec_ablation"]:
            a=np.load(path.parent/"predictions.npz")
            m=clustering(y[train_ids],a["train_labels"])
            for k,v in m.items(): assert abs(v-r[k])<1e-12
        if r["method"] in ["cae", "cae_collision", "random", "variance"]:
            computed, _ = evaluate_selection(np.asarray(r["indices"]),x[train_ids],x[val_ids],xt,y[train_ids],yt,r["seed"])
            for k in ["test_mse","val_mse","classification_accuracy"]:
                # Float32 OLS on sparse/collinear pixel columns is not bitwise invariant.
                # Keep a tight relative tolerance and record the actual discrepancy.
                np.testing.assert_allclose(computed[k],r[k],rtol=1e-5,atol=1e-8,
                                           err_msg=str((r["method"],r["seed"],k)))
                ols_max_absolute_difference=max(ols_max_absolute_difference,abs(computed[k]-r[k]))
    save_json(root/"validation.json", dict(records=len(records), groups=len(groups), seeds=[0,1,2],
              svdd_scores_recomputed=9, clustering_predictions_recomputed=6,
              feature_selection_evaluations_recomputed=12,ols_max_absolute_difference=ols_max_absolute_difference,
              ols_rtol=1e-5,ols_atol=1e-8,passed=True))

    figdir = ROOT / "figures"
    figdir.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,
                         "figure.dpi":140,"savefig.dpi":180})
    fig, axes = plt.subplots(1,3,figsize=(15.3,4.6),layout="constrained")
    names=["random","variance","cae","cae_collision"]
    labels=["Random","Variance","CAE","CAE + penalty"]
    values=[summary[k]["test_mse"] for k in names]
    axes[0].bar(labels,[v["mean"] for v in values],yerr=[v["std"] for v in values],capsize=4,
                color=["#afb7c3","#afb7c3","#176b91","#db8d32"])
    axes[0].set(title="Feature selection: test reconstruction",ylabel="MSE (lower is better)")
    axes[0].tick_params(axis="x",labelrotation=25)
    names=["ae_kmeans","dec_ablation","idec"]
    for i,k in enumerate(names):
        v=summary[k]["acc"]
        axes[1].bar(i,100*v["mean"],yerr=100*v["std"],capsize=4,color=["#afb7c3","#db8d32","#176b91"][i])
    axes[1].set(xticks=range(3),xticklabels=["AE + KMeans","No recon.","IDEC"],
                title="Clustering: 12,000 training images",ylabel="Clustering ACC (%)",ylim=(0,100))
    for i,k in enumerate([0,3,8]):
        v=summary[f"deep_svdd_c{k}"]
        for j,(metric,color) in enumerate([("ae_auc","#afb7c3"),("auc","#176b91")]):
            axes[2].bar(i+(j-.5)*.32,100*v[metric]["mean"],.30,yerr=100*v[metric]["std"],capsize=3,
                        color=color,label=["Pretrained AE","Deep SVDD"][j] if i==0 else None)
    axes[2].set(xticks=range(3),xticklabels=["Normal 0","Normal 3","Normal 8"],
                title="Anomaly detection: original test set",ylabel="AUROC (%)",ylim=(50,100))
    axes[2].legend(fontsize=9)
    fig.suptitle("MNIST core reproductions | 3 seeds | error bars: sample SD",fontsize=14)
    fig.savefig(figdir/"pilot_summary.png"); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(14,4),layout="constrained")
    for name,color in [("cae","#176b91"),("cae_collision","#db8d32")]:
        for seed in [0,1,2]:
            h=read(root/f"cae/{name}_s{seed}/history.json")
            epochs=[r["epoch"] for r in h]
            for ax,metric in zip(axes,["unique_features","mean_max_probability","val_joint_hard_mse"]):
                ax.plot(epochs,[r[metric] for r in h],color=color,alpha=.65,label=name if seed==0 else None)
    for ax,title in zip(axes,["Unique selected pixels (out of 50)","Mean maximum probability","Joint hard-decoder validation MSE"]):
        ax.set(title=title,xlabel="Epoch"); ax.legend(fontsize=8)
    axes[1].axhline(.99,color="black",ls=":",lw=1)
    axes[2].set_yscale("log")
    fig.savefig(figdir/"cae_diagnostics.png"); plt.close(fig)

    fig,axes=plt.subplots(2,6,figsize=(10,4),layout="constrained")
    for row,name in enumerate(["cae","cae_collision"]):
        for seed in [0,1,2]:
            r=read(root/f"cae/{name}_s{seed}/result.json")
            m=np.zeros(784); m[r["indices"]]=1
            axes[row,seed].imshow(m.reshape(28,28),cmap="gray",vmin=0,vmax=1)
            axes[row,seed].set_title(f"seed {seed}: {r['unique_features']}/50",fontsize=10)
        a=np.load(root/f"cae/{name}_s0/examples.npz")
        for j in range(3):
            axes[row,j+3].imshow(a["reconstructed"][j].reshape(28,28),cmap="gray",vmin=0,vmax=1)
            axes[row,j+3].set_title(f"Test example {j}",fontsize=10)
        axes[row,0].set_ylabel(["CAE","CAE + penalty"][row])
    for ax in axes.flat: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Selected pixels and OLS reconstruction (fixed examples, no cherry-picking)")
    fig.savefig(figdir/"cae_selections.png"); plt.close(fig)

    def table(headers, rows):
        return '<table><thead><tr>'+''.join('<th>'+html.escape(v)+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'
    cae_table=table(["方法","测试 MSE ↓","分类准确率 ↑","不同像素数 /50"],[
        [k,text_stat(summary[k]["test_mse"]),text_stat(summary[k]["classification_accuracy"],100,2)+"%",
         text_stat(summary[k]["unique_features"],1,2)] for k in ["random","variance","cae","cae_collision"]])
    idec_table=table(["方法","聚类 ACC ↑","NMI ↑","ARI ↑"],[
        [k,text_stat(summary[k]["acc"],100,2)+"%",text_stat(summary[k]["nmi"]),text_stat(summary[k]["ari"])]
        for k in ["ae_kmeans","dec_ablation","idec"]])
    svdd_table=table(["正常类别","Deep SVDD AUROC","预训练 AE AUROC","原论文 One-Class AUROC"],[
        [str(k),text_stat(summary[f"deep_svdd_c{k}"]["auc"],100,2)+"%",text_stat(summary[f"deep_svdd_c{k}"]["ae_auc"],100,2)+"%",ref]
        for k,ref in [(0,"98.0 ± 0.7%"),(3,"91.9 ± 1.5%"),(8,"93.9 ± 1.6%")]])
    budget_html = ""
    budget_root = root.parent / "budget1600"
    budget_paths = sorted(budget_root.glob("cae/*/result.json"))
    if len(budget_paths) == 12:
        extended = [read(p) for p in budget_paths]
        ext_groups = {m: sorted([r for r in extended if r["method"] == m],key=lambda r:r["seed"])
                      for m in ["cae","cae_collision","random","variance"]}
        ext_summary = {}
        for method, rs in ext_groups.items():
            assert [r["seed"] for r in rs] == [0,1,2]
            ext_summary[method] = {m: mean_std([r[m] for r in rs]) for m in metrics if all(m in r for r in rs)}
            for r in rs:
                computed,_ = evaluate_selection(np.asarray(r["indices"]),x[train_ids],x[val_ids],xt,y[train_ids],yt,r["seed"])
                for metric in ["test_mse","val_mse","classification_accuracy"]:
                    np.testing.assert_allclose(computed[metric],r[metric],rtol=1e-5,atol=1e-8)
            if method.startswith("cae"):
                assert all(r["config"]["epochs"]==1600 for r in rs)
                ext_summary[method]["confidence_converged_seeds"] = sum(r["converged_099"] for r in rs)
        ext_split=np.load(budget_root/"split.npz")
        np.testing.assert_array_equal(train_ids,ext_split["train"])
        np.testing.assert_array_equal(val_ids,ext_split["validation"])
        ext_summary["paired_mse_differences"] = [b["test_mse"]-a["test_mse"] for a,b in zip(ext_groups["cae"],ext_groups["cae_collision"])]
        save_json(budget_root/"summary.json", ext_summary)
        save_json(budget_root/"validation.json",dict(records=12,evaluations_recomputed=12,split_identical=True,passed=True))
        with (budget_root/"runs.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=cols,extrasaction="ignore"); w.writeheader(); w.writerows(extended)
        rows=[]
        for epochs,s in [(800,summary),(1600,ext_summary)]:
            for m in ["cae","cae_collision"]:
                rows.append([str(epochs),m,text_stat(s[m]["test_mse"]),text_stat(s[m]["mean_max_probability"]),
                             str(s[m]["confidence_converged_seeds"])+"/3"])
        budget_html = '<h3>追加实验：双方同时增加训练预算</h3>'+table(["Epochs","方法","测试 MSE ↓","平均最大选择概率","达到 0.99 的 seeds"],rows)
        budget_html += '<p>追加实验在观察到 800 epoch 未收敛后提出，1600 epoch 从头训练，所有其余条件固定。'+ \
                       '1600 epoch 的逐种子惩罚减基线 MSE 差：'+', '.join(f'{v:+.6f}' for v in ext_summary["paired_mse_differences"])+ \
                       '。两种预算均完整报告，未按测试结果挑选终点。此对照仍不能区分所有正则强度、困难重复特征条件或更长预算。</p>'
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
        for m,color in [("cae","#176b91"),("cae_collision","#db8d32")]:
            for ax,metric in zip(axes,["test_mse","mean_max_probability"]):
                vs=[summary[m][metric],ext_summary[m][metric]]
                ax.errorbar([800,1600],[v["mean"] for v in vs],yerr=[v["std"] for v in vs],marker="o",capsize=4,color=color,label=m)
        axes[0].set(ylabel="Test OLS MSE",xlabel="Training epochs",title="Does more budget change the comparison?")
        axes[1].set(ylabel="Mean maximum selection probability",xlabel="Training epochs",title="Convergence diagnostic")
        axes[1].axhline(.99,color="black",ls=":",lw=1)
        for ax in axes: ax.legend(); ax.set_xticks([800,1600])
        fig.savefig(figdir/"cae_budget.png"); plt.close(fig)
        budget_html += '<img src="../../figures/cae_budget.png" alt="800 versus 1600 epochs">'
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>COSI 126A 论文选题与复现实验</title>
<style>body{{font:16px/1.75 system-ui,"Microsoft YaHei",sans-serif;color:#223044;background:#f4f6f8;margin:0}}main{{max-width:1080px;margin:32px auto;padding:42px;background:white;border-radius:12px}}h1{{font-size:30px}}h2{{margin-top:38px;color:#176b91}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{text-align:left;padding:10px;border-bottom:1px solid #dce2e8}}th{{background:#eef3f6}}img{{max-width:100%;height:auto}}a{{color:#176b91}}.note{{border-left:4px solid #db8d32;padding:12px 18px;background:#fff8ed}}code{{background:#edf1f4;padding:2px 5px}}li{{margin-bottom:8px}}</style>
<main><h1>COSI 126A 论文选题与复现实验</h1><p>2026-11-15 · 五篇论文调研，三篇核心方法实跑 · 本地 RTX 5080 · seeds 0, 1, 2</p>
<p class="note">这是选题阶段的复现与研究判断。作业最终需要选一篇作为主线，完成假设、修改、多条件实验、报告和讲稿。当前结果使用明示的缩减协议，不等于重现原论文全部表格。</p>
<h2>三组实验的实际结果</h2><img src="../../figures/pilot_summary.png" alt="Three experiment summaries">
<h3>Concrete Autoencoders</h3>{cae_table}
<p>50 个选择槽位，12,000 训练样本，3,000 验证样本；原始 10,000 测试样本保持独立。训练阶段不使用标签。MSE 来自选择特征后重新拟合的 OLS；分类来自固定 50 棵树的 ExtraTrees 下游评估。所有选择器采用同一评估协议。</p>
<p>去重惩罚相对基线的逐种子 MSE 差为 {', '.join(f"{r['mse_difference']:+.6f}" for r in paired)}；正数表示更差。CAE 达到论文平均最大选择概率 0.99 标准的运行数为 {summary['cae']['confidence_converged_seeds']}/3，修改版为 {summary['cae_collision']['confidence_converged_seeds']}/3。固定 800 epoch 的结果须结合这一收敛诊断解释。</p>
<img src="../../figures/cae_diagnostics.png" alt="Selection, confidence and validation diagnostics"><img src="../../figures/cae_selections.png" alt="Pixel selections and fixed reconstructed examples">
<p>跨种子所选像素集合的平均 Jaccard：CAE {text_stat(summary['cae']['pairwise_jaccard'])}；修改版 {text_stat(summary['cae_collision']['pairwise_jaccard'])}。三个两两比较并非独立重复，不能作为显著性检验。</p>
<p>探索性诊断：用训练数据的绝对 Pearson 相关系数，对两次运行的 50 个像素做最优一对一匹配，平均相关为 CAE {text_stat(summary['cae']['matched_abs_train_correlation'])}、修改版 {text_stat(summary['cae_collision']['matched_abs_train_correlation'])}。低坐标重合率不自动等于低信息稳定性；相关像素可能相互替代。这不是预注册的主要指标，也不构成因果证明。</p>
{budget_html}
<h3>IDEC</h3>{idec_table}<p>此表为 12,000 个训练点上的聚类指标，按 Hungarian 匹配计算 ACC。采用作者网络和 uint8/50 缩放，但使用 100 epoch 普通 AE 预训练，取代论文逐层去噪预训练。原论文使用 70,000 点。三个 IDEC 运行全部达到 4,000 步上限，未满足标签变化停止条件；不能据此否定原论文结论。No-reconstruction 是同初始化、同损失系数的受控消融，不代表完整 DEC 官方训练配置。独立测试点的描述性聚类结果保存在各 run 的 <code>heldout</code> 字段，未用于调参。</p>
<h3>Deep SVDD</h3>{svdd_table}<p>每个正常类别使用完整原始训练子集，每次测试全部 10,000 样本。作者 PyTorch 网络逐值校验通过。150 epoch AE 预训练，再做 150 epoch one-class 优化；原论文使用不同学习率阶段长度、激活斜率与 10 seeds。原论文数字用于参照，不作完全相同条件下的比较。AE 列来自预训练终点，训练预算少于 SVDD 全流程，属于流程诊断。</p>
<h2>如何据此选择</h2><p>建议优先选择 Concrete Autoencoders：低成本的多种子实验、明确的离散选择行为和直观像素图，适合把研究问题聚焦到“预算、重复/相关特征与泛化”的关系。Deep SVDD 适合异常检测方向；IDEC 则适合在既有聚类知识上继续深入。</p>
<p>详细的五篇论文比较、候选假设、实现差异与下一阶段实验设计见仓库 <a href="../../README.md">README</a>。原始均值、样本标准差、逐种子记录和独立复算结果分别见 <a href="summary.json">summary.json</a>、<a href="runs.csv">runs.csv</a> 和 <a href="validation.json">validation.json</a>。</p>
<h2>主要来源</h2><ul>
<li><a href="https://proceedings.mlr.press/v97/balin19a.html">Balin et al. 2019: Concrete Autoencoders</a>，线性解码器/OLS 协议见正文 4.2 与补充 D。</li>
<li><a href="https://proceedings.mlr.press/v235/nilsson24b.html">Nilsson et al. 2024: Indirectly Parameterized Concrete Autoencoders</a>。重复选择已有相关研究，本次惩罚不是新颖性声明，也不是 IP-CAE 复现。</li>
<li><a href="https://www.ijcai.org/proceedings/2017/243">Guo et al. 2017: IDEC</a>，作者代码用于核对损失、网络、缩放和更新周期。</li>
<li><a href="https://proceedings.mlr.press/v80/ruff18a.html">Ruff et al. 2018: Deep One-Class Classification</a>；复用 MIT 许可作者网络，训练器与数据路径在本地实现。</li></ul>
</main></html>'''
    (root/"review.html").write_text(body,encoding="utf-8")
    lines=["MNIST core reproductions, 2026-11-15", "Mean +/- sample SD over seeds 0,1,2.",
           "This is a research pilot, not exact reproduction of all paper tables.", "",
           "CAE: 800 epochs, 50 features, 12k train/3k validation/10k test, OLS refit"]
    for key in ["random","variance","cae","cae_collision"]:
        lines.append(key+": MSE "+text_stat(summary[key]["test_mse"],1,6)+
                     "; downstream accuracy "+text_stat(summary[key]["classification_accuracy"],100,2)+"%")
    lines += ["", "IDEC: 12k training-point clustering; shortened AE pretraining; all3 IDEC runs hit4000-step cap"]
    for key in ["ae_kmeans","dec_ablation","idec"]:
        lines.append(key+": ACC "+text_stat(summary[key]["acc"],100,2)+"%; NMI "+text_stat(summary[key]["nmi"]))
    lines += ["", "Deep SVDD: full normal-class training subsets, 150+150 epochs, original10k test"]
    for cls in [0,3,8]:
        lines.append("Normal class "+str(cls)+": AUROC "+text_stat(summary[f"deep_svdd_c{cls}"]["auc"],100,2)+"%")
    if budget_html:
        lines += ["", "Exploratory follow-up: 1600 epochs from scratch, same seeds/split/other hyperparameters"]
        for key in ["cae","cae_collision"]:
            lines.append(key+": MSE "+text_stat(ext_summary[key]["test_mse"],1,6)+
                         "; confidence "+text_stat(ext_summary[key]["mean_max_probability"])+
                         "; converged seeds "+str(ext_summary[key]["confidence_converged_seeds"])+"/3")
    lines += ["", "See README.md and research/experiment_plan.txt for protocol differences and attribution."]
    (root/"RESULTS.txt").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2,ensure_ascii=False))
