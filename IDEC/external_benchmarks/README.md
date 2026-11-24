# IDEC+ on full MNIST and four external datasets

Here **IDEC+** means our modification that reconstructs nine selected combinations of 392 local gradient measurements. Two independent graph partitions of the unlabeled fitting set identify groups that agree across halves. Fisher directions emphasize differences between those groups relative to variation within them. The student encoder still receives the complete image. The full modification uses the selected target throughout pretraining and clustering. [Feature definition and original held-out results](../cluster_features/README.md) · [Design recorded before scoring](design.md)

## The 88% MNIST setting

All four runs below fit and score the same 70,000 original images, at seed 0 with 20,000 clustering updates. Hungarian matching uses this full set, as in the reference reproduction.

| Starting recipe | Pixel target, initial ACC | Selected target, initial ACC | Pixel target, final ACC | Selected target, final ACC | Paired difference |
|---|---:|---:|---:|---:|---:|
| Authors' pixel-pretrained checkpoint, change only fine-tuning | 81.85% | 81.85% | 88.03% | 86.08% | −1.95 pp |
| From scratch, 100 pretraining epochs on each arm's target | 83.40% | 80.33% | 87.78% | **82.10%** | −5.68 pp |

The authors' two forks assigned every image identically before fine-tuning. Switching only the target left 1,364 fewer correctly matched digits at the end. In the from-scratch comparison, some of the gap was already present after pretraining and K-means. The two-stage method fell below pixel IDEC in this full-data run. The single seed matches the archived reproduction's seed.

The archived reproduction reached **88.10%** (paper **88.06%**). Rescoring its predictions on the 10,000 official test indices, using a mapping estimated on the original 60,000 training indices, gives **88.22%**. All 70,000 had already participated in fitting this model. The author-initialized pixel rerun above reached 88.03%, a 0.07-point difference. [Archived assignments and results](../reference/result.json) · [Original paper, Table 2](https://www.ijcai.org/proceedings/2017/0243.pdf)

The background experiment instead started from scratch on 12,000 training digits, used 5,600 clustering updates and evaluated 10,000 unseen digits with a mapping frozen on training assignments. Across five seeds it measured **73.23% clean and 65.14% marked**. The independent feature experiment on the identical split measured **73.63%** for pixels and **76.84%** for nine features. That 3.21-point gain uses its own paired control. Sample size, initialization, training budget and evaluation all change between the held-out and paper settings. [Background comparison](../dec_background/results/summary.csv) · [Held-out feature comparison](../cluster_features/results/summary.json)

## Independent image datasets

The original training and test splits stay separate. All arms share the same images, resizing and intensity scale within a dataset. We refit the fixed selection rule on each dataset's unlabeled training images, without transferring the MNIST directions or tuning any feature parameter. Student fits use five paired seeds. Each accuracy uses a cluster-to-class mapping estimated on the training data after optimization and frozen for the official test set. No synthetic geometry or augmentation is added.

| Dataset (fit / test) | DEC | Pixel IDEC | Full IDEC+ | IDEC+ minus pixel, pp (mean ± seed SD) | Positive seeds |
|---|---:|---:|---:|---:|---:|
| USPS digits (7,291 / 2,007) | 71.26% | 69.11% | **75.78%** | +6.68 ± 0.98 | 5/5 |
| Kuzushiji-MNIST (12,000 / 10,000) | 33.22% | 28.04% | Fit failed | — | — |
| Fashion-MNIST (12,000 / 10,000) | 55.48% | 54.03% | **59.12%** | +5.08 ± 1.54 | 5/5 |
| CIFAR-10, grayscale 28×28 (12,000 / 10,000) | 20.74% | 22.25% | Fit failed | — | — |

The differences use unrounded values within each seed. On USPS and Fashion-MNIST the selected target beat ordinary pixel IDEC and ordinary DEC. The graph used for selection also reached 78.33% and 59.40%, respectively. Keeping the selected-feature pretraining and switching to KL-only DEC fine-tuning reached 76.64% and 60.25%. These controls show that the image groups and pretraining already contributed much of the class information. Each reported seed uses the same fitting sample and selector.

On Kuzushiji-MNIST, 10,163 of 12,000 images received agreeing group assignments, but one of the ten aligned groups contained zero agreed examples. The same happened on grayscale CIFAR-10, with 5,881 agreements in total. Nine occupied consensus groups support at most eight between-group directions. The fixed nine-direction selector stopped before student training, so there is no IDEC+ accuracy on either dataset. This concerns missing agreement within the learned partitions. Both evaluation datasets contain all ten true classes. The graph-only test accuracies were 45.86% and 21.51%, respectively.

USPS inputs are resized from 16×16. For CIFAR-10, all arms receive the same grayscale 28×28 adaptation of the original color images. [USPS ZIP digits](https://hastie.su.domains/ElemStatLearn/data.html) · [Kuzushiji-MNIST source](https://github.com/rois-codh/kmnist) · [Fashion-MNIST source](https://github.com/zalandoresearch/fashion-mnist) · [CIFAR-10 source](https://www.cs.toronto.edu/~kriz/cifar.html)

## Reproduce or rescore

Run from the IDEC directory with the project's pinned environment. The data loader downloads and checks the original files. The external score uses the entire official test split. The full MNIST experiment also downloads and verifies the authors' checkpoint.

```powershell
python download_cifar.py
python external_study.py --output scratch/external
python mnist_full_study.py --output scratch/external/mnist_full
python external_graph_audit.py --results scratch/external
python external_analysis.py --results scratch/external --output scratch/external_tables
```

To audit the published outcomes, run the following command. It reads the stored assignments, original project records and source hashes, checks the expected seeds and shared initial assignments, then recomputes accuracy, NMI, ARI and paired differences. This step runs on CPU and uses no raw-image downloads.

```powershell
python external_analysis.py --output scratch/verified_external
```

[All seed and initialization scores](runs.csv) · [Summary, paired effects and consensus-group counts](summary.json) · [Per-image predictions and training manifests](results)
