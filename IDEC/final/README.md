# Reproducing the final IDEC project

**John Zhang and Ryan Jian · COSI 126A**

[Report](../report/report.pdf) · [Slides](../slides/slides.pdf) · [Per-seed accuracy](per_seed.csv) · [Ablation means](ablation.csv) · [Paired contrasts](paired.csv)

## Setup

Use Python 3.12. From the repository root, enter `IDEC` and install its pinned dependencies. The recorded runs used PyTorch 2.11.0 with CUDA 12.8. The scoring path also runs on CPU. In PowerShell:

```powershell
cd IDEC
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/Activate.ps1
```

For a CPU installation, replace `cu128` with `cpu`. On Linux use `source .venv/bin/activate` and `.venv/bin/python` for the installation commands. All remaining commands run from `IDEC`. The fit and image-figure scripts fetch the original four MNIST IDX gzip files and verify their checksums. Set `MNIST_CACHE` to reuse an existing data directory. The original reference reproduction additionally downloads the authors' verified weights.

## Rescore all five seeds without training

```shell
python final_evidence.py --output scratch/final
python -m pytest -q
```

This reads saved per-image cluster assignments and evaluation labels, validates the archived source manifests, compares shared controls across the two saliency studies, and rebuilds the 35 method/seed rows, means, paired contrasts, LaTeX table and paired plot. No MNIST download is needed for rescoring. To regenerate the two-row weight-map figure as well:

```shell
python final_evidence.py --output scratch/final --weight-figure
```

The [weight maps](weights.png) use the first training digit under each domain's teacher. The [paired plot](paired.png) shows differences in percentage points. The table and both figures used in the report and presentation are in this folder. The [full source data](../adaptive_saliency/results) and the [saliency-only arm and archived maps](../gaussian_saliency/results) are included in the release.

## Fit the final method from raw MNIST

```shell
python fit_final.py --seed 0 --output scratch/final_fit
```

This refits the teacher and student, computes saliency from scratch, then writes train/test cluster assignments and held-out clean and marked accuracy. Repeat with seeds 1 through 4 to run the entire primary arm. Use `--device cpu` to force CPU training. A fresh seed-0 GPU fit scored **78.35% clean and 72.86% marked**, versus **78.53% and 73.07%** in the archived primary seed-0 fit. The independent refit has its own [metrics](refit/metrics_0.json) and [predictions](refit/clean_adaptive_aligned_0.npz). Only the original five paired runs contribute to the ablation table.

- **Adaptive center.** Begin with a width-seven Gaussian at (13.5, 13.5). Compute the centroid of the observed intensity multiplied by this Gaussian. Recenter the width-seven Gaussian at the result, with the image midpoint as the blank-image fallback.
- **Aligned saliency.** Fit one ordinary pixel autoencoder and K-means teacher per training domain (seed 126). Smooth the absolute input-times-gradient of the squared-distance margin between the two nearest centers over 3 × 3 pixels. Divide each value by itself plus the image-wide mean, then cache it in eight bits.
- **Reconstruction.** Multiply the adaptive Gaussian by 0.5 plus the input's own saliency. Normalize the product to mean one per image. Use it for 100 epochs of student autoencoder pretraining, initialize ten centers with K-means, then run 5,600 ordinary IDEC updates. Deployment uses the student encoder and centers.
- **Ablation definitions.** Fixed Gaussian uses the midpoint. Saliency only uses 0.5 plus the map. Adaptive Gaussian uses only the recentered weights. The fixed product holds the center fixed and uses aligned saliency. The shuffled product keeps each adaptive center and changes which training image supplied the saliency. Every weighted arm normalizes to mean one.

Architecture, optimizer and seeds are in [protocol.json](../protocol.json). Input intensity is scaled from byte values by 1/50. The primary fit uses 12,000 unlabeled clean images (split seed 126). After fitting, training labels determine a one-to-one cluster-to-digit mapping, frozen for all 10,000 held-out images and their marked views. No clean counterpart or foreground annotation enters the final training loss. The separate matched-domain branch fits on marked inputs and has its own teacher.

## Refit the controls and original baseline

The [fixed-site brightness experiment](../background_probe/README.md) refits the final encoder and ordinary IDEC in both training domains and provides 20 sets of raw embeddings for CPU readout analysis. The [full clean MNIST comparison](../full_gaussian/README.md) evaluates both students after matched fresh pretraining on all 70,000 digits. Both guides include training and rescoring commands.

The [Gaussian shape follow-up](../gaussian_shape/README.md) adds a covariance-based ellipse, a perpendicular-ellipse control, five-seed refits of the circular controls and the ordinary baseline, plus a second held-out set. Run the candidate from raw MNIST with `python fit_final.py --shape ellipse --seed 0 --output scratch/ellipse_fit`. The archived ablation above uses the adaptive circle.

| Experiment | Command from IDEC |
|---|---|
| Author-checkpoint, 70,000-image reproduction | `python reproduce_paper.py --output scratch/reference` |
| DEC, ordinary IDEC and fixed Gaussian | `python dec_background_study.py --output scratch/dec_background` |
| Marked-input reconstruction-target diagnostic | `python varied_study.py --output scratch/varied` |
| Adaptive Gaussian alone | `python gaussian_refinement_study.py --output scratch/gaussian_refinement` |
| Saliency only, fixed products, teacher maps | `python gaussian_saliency_study.py --output scratch/gaussian_saliency` |
| Final and shuffled adaptive products in both domains, using released teacher maps | `python adaptive_saliency_study.py --output scratch/adaptive_saliency` |

These fit commands write separate scratch results. For a new two-domain adaptive fit, run `python gaussian_refinement_analysis.py --study adaptive_saliency --results scratch/adaptive_saliency --output scratch/adaptive_scores`. It compares the new arms to the released controls. The data generator and [target diagnostic](../varied/results/stages.tex) establish the limitation. Our [adaptive-saliency protocol](../adaptive_saliency/design.md) records the known earlier results and the choice of the primary endpoint.

## Build the documents

With Tectonic installed, run `tectonic report.tex` inside `IDEC/report` and `tectonic slides.tex` inside `IDEC/slides`. Both inputs use the same generated [ablation table](ablation.tex). The release includes both compiled PDFs and their editable sources.

Guo et al. introduced the original IDEC objective. Our project implements it in PyTorch, uses the authors' checkpoint for the paper comparison, and adds the background generator, diagnostic targets, weighted pretraining and ablation analysis. [Original IDEC paper](https://www.ijcai.org/proceedings/2017/243) · [Official IDEC code](https://github.com/XifengGuo/IDEC/tree/4a661c211f082022249758ab3312f7db21e2aa99) · [DEC paper](https://proceedings.mlr.press/v48/xieb16.html)
