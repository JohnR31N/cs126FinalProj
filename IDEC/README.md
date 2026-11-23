# IDEC final project

IDEC reconstructs every pixel while learning a representation for clustering. We study background features that matter for reconstruction but carry no intended digit information. The geometric experiment places a plus or diagonal cross in blank space and checks that every stroke pixel and a one-pixel clearance ring are preserved. We compare ordinary full-image reconstruction, background augmentation, reconstruction of stable content from paired views, and a fixed Gaussian prior on the reconstruction loss.

- [Research report (PDF)](report/report.pdf) and [editable LaTeX](report/report.tex)
- [Presentation slides (PDF)](slides/slides.pdf) and [editable Beamer source](slides/slides.tex)
- [Geometric experiment design](background/design.md), [paired digit examples](figures/geometry.png), [four-arm results](background/results/summary.csv), [paired differences](background/results/paired.json)
- [Historical stripe report source](report/stripe_study.tex) and [the previous published PDF](https://github.com/JohnR31N/cs126FinalProj/blob/idec-final-v1.4/IDEC/report/report.pdf)
- [Prespecified design](design.md), [hyperparameters](protocol.json), [paper reproduction](reference/result.json), [all model results](results/runs.csv), [paired seed effects](results/paired.json)
- [Analysis and verification](analyze.py), [result checks](results/verification.json)
- [Training-stage follow-up design](stage/design.md), [the four final conditions and two initializations](stage/results/summary.csv), [within-seed effects](stage/results/paired.json)

## Detached geometric backgrounds

Moving a nine-pixel mark changed 6.20% of cluster assignments under ordinary IDEC, even though all handwriting stayed identical. Background augmentation reduced this to 4.34%, with a reduction in each of the five seeds. The paired reconstruction target reduced the decoder's mean response to the movement from 0.00313 under augmentation alone to 0.00085 (one represents perfect tracking). It brought movement flips to 4.18% but increased shape flips from 3.43% to 3.59%. The Gaussian-weighted arm yielded the highest mean marked-image accuracy, 72.91% compared with 71.02% under augmentation alone. All four arms retained ten occupied test clusters.

## Earlier additive-overlay studies

At strong stripe amplitude, switching the orientation of an otherwise identical test digit changed its cluster **41.72%** of the time under standard IDEC and **39.50%** with projected fine-tuning. The reduction appeared in all five paired seeds. Digit accuracy rose from **58.79% to 59.43%** on average (four of five seeds). The shared initial AE + K-means already flipped on **42.79%** of images. The clustering-only ablation reached **60.54%** digit ACC.

In the follow-up, applying the same projection earlier had a larger effect. With projected fine-tuning fixed, changing pretraining lowered flip rate from **39.53% to 2.61%** (paired reduction **36.92 ± 12.54 percentage points**, five of five seeds). Mean digit ACC rose from **59.96% to 69.61%**, with one seed losing 1.88 points. All ten clusters were occupied. Full-pixel fine-tuning after projected pretraining gave **4.73%** flips. These are within-follow-up comparisons at amplitude 0.8 on the same split and known stripe templates.

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

The follow-up used the strong-stripe condition and the same five seeds. It crossed ordinary or projected pretraining with ordinary or projected fine-tuning. We reran all four cells and compared models trained within this follow-up. Each pair of fine-tuning runs starts from the same weights and K-means centers. Changing pretraining can change both the embedding and its K-means centers.

## Reproduction

Run from **this folder**. For the tested Windows setup use Python 3.12, PyTorch 2.11.0 with CUDA 12.8, and the pinned scientific packages in `requirements.txt`. The CPU path uses the same protocol and can take substantially longer. Hardware-dependent variation is possible. The recorded runs used a local RTX 5080.

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
uv pip install --python .venv/Scripts/python.exe torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pytest -q test_objectives.py test_geometry.py
```

For Linux/macOS use `.venv/bin/python`. A CPU-only PyTorch installation works with `--device cpu`. When working inside the original parent repository, its configured Python can also be used. MNIST's four original IDX gzip files are downloaded from the torchvision mirror, verified against their expected MD5 digests, and recorded with SHA-256 in the cache manifest. Set `MNIST_CACHE` to an existing folder of verified IDX files to reuse it. Author AE weights (about 13 MB) are fetched from the pinned revision below and checked before conversion.

**Audit the shipped results and regenerate the tables and figures** without training or downloading the author weights.

```powershell
.venv/Scripts/python.exe analyze.py
.venv/Scripts/python.exe analyze_stages.py
.venv/Scripts/python.exe geometry_analysis.py
```

**Rerun training** into new directories, then rebuild that run's tables and figures:

```powershell
.venv/Scripts/python.exe reproduce_paper.py --output scratch/reference
.venv/Scripts/python.exe -m idec_final.experiment --output scratch/study
.venv/Scripts/python.exe analyze.py --reference scratch/reference --input scratch/study --figures scratch/figures
.venv/Scripts/python.exe train_stages.py --output scratch/stages
.venv/Scripts/python.exe analyze_stages.py --input scratch/stages
.venv/Scripts/python.exe geometry_study.py --output scratch/geometry
.venv/Scripts/python.exe geometry_readout.py --output scratch/geometry/decoder.npz
.venv/Scripts/python.exe geometry_analysis.py --input scratch/geometry --figure scratch/figures/geometry.png
```

The study checks the source/environment manifest before resuming an output folder and marks a seed/amplitude pair complete only after writing all predictions and traces. Published metrics, predictions and trace files are small enough to ship with the project. The original input images and author weights remain in the local cache. The reference script checks its HDF5 conversion against forward passes in NumPy. The analysis independently rebuilds the Hungarian mapping, digit ACC/NMI/ARI, orientation NMI and flip rate from the stored predictions and checks the full grid against the locked protocol.

The stage audit recomputes each model's six metrics from its stored assignments, verifies the data and training manifests, and counts the test clusters occupied. The four-cell comparisons in `stage/results/paired.json` are signed differences in fractions. Multiply flip rates and accuracy by 100 to express their differences in percentage points. The interaction is the effect of projecting during pretraining with projected fine-tuning minus the same effect with full-pixel fine-tuning.

The geometry analysis regenerates the marks, verifies exact pixel preservation, fits the post-training digit map and checks all 20 model records. Site availability depends on the digit silhouette. The three augmented arms share the four-view schedule. The paired target uses aligned same-digit pairs during training. The Gaussian weights are fixed with sigma 7 pixels. Decoding the signed motion of a mark is a separate exploratory readout from replayed pretraining. In the result CSVs, ACC and flips are fractions. In the report table, those measures are percentages. The decoder response is dimensionless. Standard deviations and within-seed differences are in the CSV and JSON outputs.

**Recompile documents.** The PDFs and editable sources are included. With Tectonic installed, run the following from `IDEC/`. The documents use the figures and the tables generated in `background/results/` (the historical stripe study also uses `results/` and `stage/results/`).

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
| `train_stages.py`, `analyze_stages.py` | Train the four stage combinations, rescore all predictions, and calculate within-seed differences |
| `test_objectives.py` | Student-t probabilities, target weighting, projector gradients, counterfactual direction and test-label leakage check |
| `results/strength_*/seed_*/` | Scores, assignments for observed and flipped test images, and target-refresh traces |

## Attribution

Guo, Gao, Liu and Yin introduced IDEC in [IJCAI 2017](https://ijcai.org/proceedings/2017/243), extending the clustering objective of [DEC](https://proceedings.mlr.press/v48/xieb16.html). We consulted [the official implementation at revision 4a661c2](https://github.com/XifengGuo/IDEC/tree/4a661c211f082022249758ab3312f7db21e2aa99). The reference weight file is [Guo's pretrained MNIST autoencoder at revision d361bfe](https://github.com/XifengGuo/data-and-models/blob/d361bfe06686c0762df58624ef95a8c0fc9162be/ae_weights/mnist_ae_weights.h5), SHA-256 `cfbe90aaf332c63ed0eeb1fbd1441d3d089249900ce0c9abf94169e25dcc8f6e`. The PyTorch network, target equations, MNIST loader and helpers reuse the independent implementation in this repository's earlier paper-selection pilot. The stripe generator, projected objective, paired evaluation, figures and report were built for this final. [DynAE](https://arxiv.org/abs/1901.07752) discusses the broader reconstruction–clustering conflict. MNIST is by LeCun, Cortes and Burges.
