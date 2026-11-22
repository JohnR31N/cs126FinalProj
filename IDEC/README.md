# IDEC final project

**What does IDEC preserve?** This COSI 126A project reproduces the MNIST result of Guo et al. and studies whether reconstructing a label-independent background makes digit clusters sensitive to stripe orientation. The controlled modification removes one known orientation direction from the decoder's reconstruction penalty. A clustering-only arm measures the effect of removing the entire reconstruction term.

- [Research report (PDF)](report/report.pdf) and [editable LaTeX](report/report.tex)
- [Presentation slides (PDF)](slides/slides.pdf) and [editable Beamer source](slides/slides.tex)
- [Prespecified design](design.md), [hyperparameters](protocol.json), [paper reproduction](reference/result.json), [all model results](results/runs.csv), [paired seed effects](results/paired.json)
- [Analysis and verification](analyze.py), [result checks](results/verification.json)

At strong stripe amplitude, switching the orientation of an otherwise identical test digit changed its cluster **41.72%** of the time under standard IDEC and **39.50%** with projected reconstruction. The reduction appeared in all five paired seeds. Digit accuracy rose from **58.79% to 59.43%** on average (four of five seeds). The shared initial AE + K-means already flipped on **42.79%** of images. The full ablation reached **60.54%** digit ACC. These results attribute a modest part of orientation sensitivity to the final reconstruction loss while leaving the origin of the larger initialization effect open.

![Digit accuracy, stripe NMI and paired flip rate across amplitudes](figures/outcomes.png)

## The two MNIST protocols

| | Reference reproduction | Controlled stripe study |
|---|---|---|
| Training data | All 70,000 MNIST images, as in the paper | Fixed 12,000-image subset of the original training split |
| Initialization | Author-released AE weights, K-means with 20 restarts, seed 0 | Plain AE pretrained for 100 epochs independently in each of 15 seed/amplitude pairs |
| Evaluation | Hungarian ACC and NMI on the same 70,000 images | Fit a cluster-to-digit mapping on training labels after optimization; evaluate on the original 10,000 test images |
| Optimization | Up to 20,000 updates; check 0.1% change every 140 steps | Exactly 5,600 updates per arm, no early stopping, five model seeds |
| Evidence | Paper 88.06% ACC and 0.8672 NMI; this run 88.10% and 0.8650 | 3 amplitudes × 5 seeds × (one shared AE + K-means initialization and three fine-tuning arms) |

The author-weight run reached the update cap. Its published numerical target is transductive. All stripe results use held-out evaluation and fresh initialization. Ordinary and projected reconstruction use identical input images. The projected objective needs the two stripe templates; individual orientation labels are used only to construct and evaluate the data.

## Reproduction

Run from **this folder**. For the tested Windows setup use Python 3.12, PyTorch 2.11.0 with CUDA 12.8, and the pinned scientific packages in `requirements.txt`. The CPU path uses the same protocol and can take substantially longer. Hardware-dependent variation is possible. The recorded runs used a local RTX 5080.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pytest -q test_objectives.py
```

For Linux/macOS use `.venv/bin/python`. A CPU-only PyTorch installation works with `--device cpu`. When working inside the original parent repository, its configured Python can also be used. MNIST's four original IDX gzip files are downloaded from the torchvision mirror, verified against their expected MD5 digests, and recorded with SHA-256 in the cache manifest. Set `MNIST_CACHE` to an existing folder of verified IDX files to reuse it. Author AE weights (about 13 MB) are fetched from the pinned revision below and checked before conversion.

**Audit the shipped results and regenerate the tables and figures** without training or downloading the author weights:

```powershell
.venv/Scripts/python.exe analyze.py
```

**Rerun training** into new directories, then rebuild that run's tables and figures:

```powershell
.venv/Scripts/python.exe reproduce_paper.py --output scratch/reference
.venv/Scripts/python.exe -m idec_final.experiment --output scratch/study
.venv/Scripts/python.exe analyze.py --reference scratch/reference --input scratch/study --figures scratch/figures
```

The study checks the source/environment manifest before resuming an output folder and marks a seed/amplitude pair complete only after writing all predictions and traces. Published metrics, predictions and trace files are small enough to ship with the project. The original input images and author weights remain in the local cache. The reference script checks its HDF5 conversion against forward passes in NumPy. The analysis independently rebuilds the Hungarian mapping, digit ACC/NMI/ARI, orientation NMI and flip rate from the stored predictions and checks the full grid against the locked protocol.

**Recompile documents.** The PDFs and editable sources are included. With Tectonic installed, run the following from `IDEC/`. The report reads its detailed table from `results/table.tex`; both documents use the included PNGs in `figures/`.

```powershell
tectonic report/report.tex
tectonic slides/slides.tex
```

## Where to look in the code

| File | Responsibility |
|---|---|
| `reproduce_paper.py` | Checked conversion of the published HDF5 weights, full transductive fitting, stop rule and paper metrics |
| `idec_final/data.py`, `stripes.py` | Verified MNIST loader, orientation assignment, image intervention and projected reconstruction loss |
| `idec_final/model.py`, `experiment.py` | Mirrored autoencoder, Student-t assignments, target refresh, paired training and saved predictions |
| `idec_final/metrics.py`, `analyze.py` | Frozen digit mapping, evaluation, cross-checks, paired contrasts and figures |
| `test_objectives.py` | Student-t probabilities, target weighting, projector gradients, counterfactual direction and test-label leakage check |
| `results/strength_*/seed_*/` | Scores, assignments for observed and flipped test images, and target-refresh traces |

## Attribution

Guo, Gao, Liu and Yin introduced IDEC in [IJCAI 2017](https://ijcai.org/proceedings/2017/243), extending the clustering objective of [DEC](https://proceedings.mlr.press/v48/xieb16.html). We consulted [the official implementation at revision 4a661c2](https://github.com/XifengGuo/IDEC/tree/4a661c211f082022249758ab3312f7db21e2aa99). The reference weight file is [Guo's pretrained MNIST autoencoder at revision d361bfe](https://github.com/XifengGuo/data-and-models/blob/d361bfe06686c0762df58624ef95a8c0fc9162be/ae_weights/mnist_ae_weights.h5), SHA-256 `cfbe90aaf332c63ed0eeb1fbd1441d3d089249900ce0c9abf94169e25dcc8f6e`. The PyTorch network, target equations, MNIST loader and helpers reuse the independent implementation in this repository's earlier paper-selection pilot. The stripe generator, projected objective, paired evaluation, figures and report were built for this final. [DynAE](https://arxiv.org/abs/1901.07752) discusses the broader reconstruction–clustering conflict. MNIST is by LeCun, Cortes and Burges.
