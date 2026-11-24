# What Fisher reconstruction preserves on grayscale CIFAR-10

Each image first supplies 392 local edge measurements. A fixed matrix learned from the training images maps them to reconstruction coordinates. A matrix column contains one coefficient per spatial cell and gradient orientation. Multiplying the centered measurements by that column gives one scalar for the image. The original ten-group MNIST fit uses nine columns. On the archived CIFAR-10 partition, 5,881 images had agreeing assignments across two splits and only nine groups were represented. This exploratory fit uses all eight between-group directions available from those nine groups.

The encoder sees the original grayscale image. During pretraining we ask the decoder to recover the selected coordinates from the embedding. During clustering we keep this reconstruction loss and the original IDEC clustering loss. The eight coordinates are embedded by an orthonormal map and scaled to match the pixel target energy, so the decoder architecture stays the same. The student still assigns each image to one of ten clusters. Selecting eight directions imposes a reconstruction objective; other information can remain in the embedding.

The pixel control uses the archived external benchmark predictions from the same images and seeds. The PCA control reconstructs eight linear combinations of the very same 392 measurements, fitted on the same 5,881 consensus examples. All students train on the full 12,000-image fitting subset. Class labels enter only after fitting to match clusters to classes on the training split and score the official 10,000 test images. [Protocol fixed before scoring these fits](design.md) · [Original fixed-nine benchmark](../README.md)

## Held-out results

| Reconstruction target | ACC after pretraining and K-means | Final IDEC ACC | Final NMI | Final ARI |
|---|---:|---:|---:|---:|
| Pixels | 21.58% | **22.25%** | 0.0775 | 0.0432 |
| Eight Fisher directions | 18.90% | **18.50%** | 0.0807 | 0.0214 |
| Eight PCA directions | 19.05% | **18.18%** | 0.0784 | 0.0214 |

Each value is a five-seed mean. Fisher reconstruction lowered clustering accuracy relative to pixels in all five seeds, by 3.75 percentage points on average (paired sample SD 1.38). The gap was already present at the K-means initialization. At the same target width Fisher gained 0.31 points over PCA, with a paired SD of 1.37 points and three wins out of five. All students occupied ten test clusters. Using a supported rank made the selector trainable, and pixel reconstruction still yielded the best mean accuracy.

One example from the original unlabeled graph illustrates the distinction between repeatable image groupings and benchmark classes. The graph cluster matched to airplanes using the training labels contained 650 official test images, including 266 airplanes and 234 ships. These are graph assignments before student training. The counts establish class mixing. Determining whether the mix arose from background, pose, color removal or the choice of descriptor would require a controlled intervention. [Official CIFAR-10 class definitions and source](https://www.cs.toronto.edu/~kriz/cifar.html)

## Reproduce

From the IDEC directory, run the following commands. The fitting script uses the archived unlabeled partitions and verified CIFAR-10 archive.

```powershell
python download_cifar.py
python cifar_fisher_study.py --output scratch/cifar_fisher
python cifar_fisher_analysis.py --results scratch/cifar_fisher --output scratch/cifar_fisher_tables
```

To recompute the published scores from saved per-image predictions on CPU, use the following command.

```powershell
python cifar_fisher_analysis.py --output scratch/cifar_fisher_verified
```

[All five seed scores](runs.csv) · [Paired differences, secondary metrics and graph diagnostic](summary.json) · [Fitted bases and per-image predictions](results)
