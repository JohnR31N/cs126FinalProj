# What Does IDEC Preserve?

**John Zhang and Ryan Jian** · COSI 126A final project

We reproduced IDEC on MNIST at **88.10%** clustering accuracy (paper **88.06%**). Our project tests whether rewarding the reconstruction of irrelevant background symbols harms digit clustering. We measure how much accuracy DEC and IDEC lose when the same digits receive two random geometric symbols, then test a fixed Gaussian prior that gives central pixels more weight in IDEC's reconstruction loss.

![Identical handwriting with fixed-form and varied background symbols](IDEC/figures/varied.png)

In the reconstruction-target diagnostic, changing the target from the complete marked image to the known clean digit raised marked accuracy from **65.40% to 71.44%**. All five seeds improved. The clean target uses privileged generator information. The Gaussian method needs only the observed image and its pixel coordinates. On the training data, it assigned mean weight **1.78** to digit ink and **0.66** to the added symbols, against a uniform weight of one.

In the matched comparison, both methods lost accuracy in every seed. **DEC fell from 77.63% to 70.86%, a 6.77-point drop. IDEC fell from 73.23% to 65.14%, an 8.09-point drop.** IDEC lost more in four of five seeds (paired excess loss **1.32 ± 3.35** points). They started from identical weights and centers within each domain.

Applying the Gaussian in both stages reduced the drop to **4.44 points**. Its marked accuracy rose by only **0.45** points and its clean accuracy fell by **3.20**. Applying it only in pretraining produced the best marked accuracy among the Gaussian variants, **68.80%**, with a **3.66**-point gain over ordinary IDEC. The per-image predictions, shared initializations and separate KL-only random-start control are included.

Whole-image translation during pretraining gave **83.88% clean and 77.61% marked ACC for IDEC**, versus **81.36% and 77.18% for DEC**. IDEC gained 12.47 points on marked digits and 10.65 on clean digits. Its background penalty narrowed by 1.82 points on average (three of five seeds), and its background-redraw flip rate fell from 2.99% to 1.78% (all five seeds). Translation still requires the decoder to reproduce the entire shifted image. The Gaussian prior changes the reconstruction cost of a peripheral error.

[Read the report](IDEC/report/report.pdf) · [View the slides](IDEC/slides/slides.pdf) · [Download the submission package](https://github.com/JohnR31N/cs126FinalProj/releases/download/idec-final-v1.9/IDEC_final.zip)

The [project README](IDEC/README.md) gives setup and reproduction commands. The [DEC comparison design](IDEC/dec_background/design.md), [translation addendum](IDEC/dec_background/translation/design.md), [combined method table](IDEC/dec_background/translation/results/summary.csv) and [per-seed translation effects](IDEC/dec_background/translation/results/paired.csv) document the comparison. The [target ablation](IDEC/varied/results/paired.json) and [single-symbol augmentation controls](IDEC/photometric/analysis/summary.csv) address the reconstruction mechanism and alternative explanations.
