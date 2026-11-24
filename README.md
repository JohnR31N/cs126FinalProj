# What Does IDEC Preserve?

**John Zhang and Ryan Jian** · COSI 126A final project

We reproduced IDEC on MNIST at **88.10%** clustering accuracy (paper **88.06%**). Two random background symbols provide a controlled way to test whether rewarding the reconstruction of irrelevant features harms digit clustering. We compare DEC and IDEC under that intervention and test a fixed Gaussian prior. A separate study learns reconstruction weights from the original unlabeled MNIST images and evaluates them on the original test digits.

![Identical handwriting with fixed-form and varied background symbols](IDEC/figures/varied.png)

In the reconstruction-target diagnostic, changing the target from the complete marked image to the known clean digit raised marked accuracy from **65.40% to 71.44%**. All five seeds improved. The clean target uses privileged generator information. The Gaussian method needs only the observed image and its pixel coordinates. On the training data, it assigned mean weight **1.78** to digit ink and **0.66** to the added symbols, against a uniform weight of one.

In the matched comparison, both methods lost accuracy in every seed. **DEC fell from 77.63% to 70.86%, a 6.77-point drop. IDEC fell from 73.23% to 65.14%, an 8.09-point drop.** IDEC lost more in four of five seeds (paired excess loss **1.32 ± 3.35** points). They started from identical weights and centers within each domain.

Applying the Gaussian in both stages reduced the drop to **4.44 points**. Its marked accuracy rose by only **0.45** points and its clean accuracy fell by **3.20**. Applying it only in pretraining produced the best marked accuracy among the Gaussian variants, **68.80%**, with a **3.66**-point gain over ordinary IDEC. The per-image predictions, shared initializations and separate KL-only random-start control are included.

Whole-image translation during pretraining gave **83.88% clean and 77.61% marked ACC for IDEC**, versus **81.36% and 77.18% for DEC**. IDEC gained 12.47 points on marked digits and 10.65 on clean digits. Its background penalty narrowed by 1.82 points on average (three of five seeds), and its background-redraw flip rate fell from 2.99% to 1.78% (all five seeds). Translation still requires the decoder to reproduce the entire shifted image. The Gaussian prior changes the reconstruction cost of a peripheral error.

On original MNIST, learning which reconstruction errors to emphasize from neighboring images lowered accuracy. The ordinary IDEC refit reached **73.70%**, the content-dependent weights reached **68.59%**, and the same weight maps assigned to other images reached **65.85%**. Ordinary IDEC led the content-dependent method in all five paired seeds. [Original-image experiment and results](IDEC/clean_neighborhood/design.md)

A follow-up used two independent unlabeled graph partitions to select stroke measurements that separate reproducible groups. Reconstructing nine selected directions raised accuracy from **73.63% to 76.84%** in a new matched comparison on the 10,000 original test digits. On 48,000 further original digits withheld from fitting, the change was **72.80% to 76.71%**. Keeping the 98 highest-ranked individual measurements reached **75.82%** on the primary test, compared with **17.87%** for the lowest-ranked 98 and **72.43%** for random orientations selected in the same locations. Ordinary DEC reached **77.64%**, and the discarded feature span retained substantial digit information in a labeled evaluation probe. [Feature selection methods, controls and visual examples](IDEC/cluster_features/README.md)

[Read the report](IDEC/report/report.pdf) · [View the slides](IDEC/slides/slides.pdf) · [Download the submission package](https://github.com/JohnR31N/cs126FinalProj/releases/download/idec-final-v1.9/IDEC_final.zip)

The [project README](IDEC/README.md) gives setup and reproduction commands. The [DEC comparison design](IDEC/dec_background/design.md), [translation addendum](IDEC/dec_background/translation/design.md), [combined method table](IDEC/dec_background/translation/results/summary.csv) and [per-seed translation effects](IDEC/dec_background/translation/results/paired.csv) document the comparison. The [target ablation](IDEC/varied/results/paired.json) and [single-symbol augmentation controls](IDEC/photometric/analysis/summary.csv) address the reconstruction mechanism and alternative explanations.
