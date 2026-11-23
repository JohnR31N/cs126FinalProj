# What Does IDEC Preserve?

**John Zhang and Ryan Jian** · COSI 126A final project

We reproduce Improved Deep Embedded Clustering (IDEC) on MNIST at **88.10%** clustering accuracy (paper **88.06%**). We then ask whether the same held-out digit changes cluster when horizontal background stripes become vertical. Removing that orientation difference from the fine-tuning loss reduced flips from **41.72% to 39.50%**. A follow-up tested the same change during autoencoder pretraining. Holding projected fine-tuning fixed, earlier intervention reduced flips from **39.53% to 2.61%** across five paired seeds and raised mean digit accuracy from **59.96% to 69.61%**.

[Read the report](IDEC/report/report.pdf) · [View the slides](IDEC/slides/slides.pdf) · [Download the submission package](https://github.com/JohnR31N/cs126FinalProj/releases/download/idec-final-v1.4/IDEC_final.zip)

The [project README](IDEC/README.md) gives setup and reproduction commands. Both the [main design](IDEC/design.md) and [follow-up design](IDEC/stage/design.md) were recorded before their respective runs. The repository includes [main results](IDEC/results/runs.csv), [follow-up results](IDEC/stage/results/runs.csv), and [paired training-stage comparisons](IDEC/stage/results/paired.json).
