# What Does IDEC Preserve?

**John Zhang and Ryan Jian** · COSI 126A final project

We reproduced IDEC on MNIST at **88.10%** clustering accuracy (paper **88.06%**). Our project asks how reconstructing task-irrelevant background affects digit clustering. A 45-run follow-up tests two symbols drawn from 96 combinations of shape, size, angle and brightness, a fixed-form control with matched energy, and an oracle clean-digit reconstruction target. The preceding 60-run study compares whole-image augmentation and a Gaussian reconstruction prior.

![Identical handwriting with fixed-form and varied background symbols](IDEC/figures/varied.png)

In the random-background study, standard IDEC fell from **73.96%** clean accuracy to **65.40%** on varied marked digits. Keeping the same marked inputs and reconstructing the known clean digit recovered accuracy to **71.44%**. The decline and recovery each occurred in all five seeds. Fixed-form backgrounds of matched energy yielded **65.74%**, so the additional effect of geometric diversity was inconsistent. Whole-image translation reached **77.63%** on the varied backgrounds.

In the preceding single-mark study, **random translation** raised mean marked-digit accuracy from **67.51% to 80.63%** and reduced movement-induced cluster switches from **6.23% to 3.41%**. Gaussian center weighting alone reached **71.42%** accuracy and **5.19%** switches. All five paired seeds favored translation on both outcomes.

[Read the report](IDEC/report/report.pdf) · [View the slides](IDEC/slides/slides.pdf) · [Download the submission package](https://github.com/JohnR31N/cs126FinalProj/releases/download/idec-final-v1.7/IDEC_final.zip)

The [project README](IDEC/README.md) gives setup and reproduction commands. The [random-background design](IDEC/varied/design.md), [results](IDEC/varied/results/summary.csv) and [paired differences](IDEC/varied/results/paired.json) cover the target ablation. The [augmentation protocol](IDEC/photometric/design.md), [spatial extension](IDEC/photometric/spatial_design.md) and [60-model results](IDEC/photometric/analysis/summary.csv) document the preceding comparison. Earlier studies retain their own protocols and outputs.
