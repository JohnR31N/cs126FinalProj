# Final adaptive Gaussian on full, clean MNIST

Guo et al. reported 88.06% digit clustering accuracy on all 70,000 MNIST images. Our original reproduction with their published autoencoder reaches 88.10%. To test the final method on the same sample and 20,000-update budget, we fitted two new students from identical random initialization at seed 0. The proposed change acts during pretraining, so a matched control needs the opportunity to pretrain on the same data. [Prespecified comparison](design.md) · [Original paper](https://www.ijcai.org/proceedings/2017/243) · [Author-checkpoint reproduction](../reference/result.json)

| Initialization and training | Final digit ACC | Final NMI |
|---|---:|---:|
| Authors' pretrained checkpoint, ordinary IDEC reproduction | 88.10% | 0.8650 |
| Fresh student, uniform reconstruction | 87.68% | 0.8585 |
| Fresh student, adaptive circular Gaussian × aligned saliency | 87.36% | 0.8486 |

Adaptive minus ordinary was **−0.32 percentage points**, or 226 fewer correctly matched digits. This single run shows no gain on the full clean sample. All 70,000 images took part in fitting and scoring. The five-seed clean-fit/marked-test result of 71.59% versus 64.62% answers a different question about new digits with distractors. [Matched results](summary.csv) · [Saved assignments](results) · [Held-out study](../final/README.md)

The two fresh students use 100 pretraining epochs, K-means with 20 restarts, then uniform IDEC fine-tuning. The adaptive student uses the same released weighting rule and one additional ordinary autoencoder teacher trained on these 70,000 unlabeled images at seed 126. The student, teacher and clustering pipeline receive clean observed pixels. Digit labels enter only after fitting, for Hungarian accuracy and normalized mutual information.

## Reproduce

From the project’s `IDEC` directory, the first command rescores the archived assignments on CPU. The second command fits both students, saves all assignments and scores them. [Environment setup](../final/README.md)

```shell
python full_gaussian_study.py --output scratch/full_gaussian_scores
python full_gaussian_study.py --fit --device cuda --results scratch/full_gaussian_refit --output scratch/full_gaussian_refit_scores
```
