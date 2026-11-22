# What Does IDEC Preserve?

**John Zhang and Ryan Jian** · COSI 126A final project

We reproduce Improved Deep Embedded Clustering (IDEC) on MNIST and test whether changing a digit’s background changes its cluster. With the original authors’ pretrained weights, our implementation achieves **88.10%** clustering accuracy compared with the paper’s **88.06%**. In a separate held-out experiment with strong stripes, removing the orientation contrast from the reconstruction loss reduces the mean cluster flip rate from **41.72% to 39.50%** across five paired seeds.

[Read the report](IDEC/report/report.pdf) · [View the slides](IDEC/slides/slides.pdf) · [Download the submission package](https://github.com/JohnR31N/cs126FinalProj/releases/download/idec-final-v1.1/IDEC_final.zip)

The [project README](IDEC/README.md) gives setup and reproduction commands. The [experiment design](IDEC/design.md), [per-run results](IDEC/results/runs.csv), and [verification record](IDEC/results/verification.json) are included alongside the code.
