# Adaptive-center Gaussian with image-specific saliency

Estimate the Gaussian center separately for every image from its intensity centroid under the original fixed Gaussian. Keep the width at seven pixels. Multiply that **adaptive** Gaussian by the stored saliency multiplier for the same image and normalize the product to mean one. The pretraining loss is the mean weighted squared reconstruction error. Fine-tuning and inference follow ordinary IDEC. In the shuffled arm the image keeps its own estimated center and only the saliency multiplier comes from another training image.

Clean and marked teachers, maps, and donor permutations are shared with the earlier fixed-center experiment. The primary comparison fits on 12,000 clean digits, introduces symbols only at evaluation and scores the official 10,000 test digits. The clean-to-clean and marked-to-marked comparisons use the same five student seeds. The earlier component scores and the test split were known in advance. [Protocol](design.md) · [Original saliency study](../gaussian_saliency/README.md) · [Adaptive center without saliency](../gaussian_refinement/README.md)

## Paired results

| Gaussian center and saliency | Clean train / clean test | Clean train / marked test | Marked train / marked test | Redraw flips, marked-trained |
|---|---:|---:|---:|---:|
| Fixed center | 74.07% | 68.92% | 68.80% | 2.62% |
| Adaptive center | 76.20% | 70.28% | 69.71% | 2.58% |
| Fixed center × aligned saliency | 75.95% | 71.51% | 70.52% | 2.38% |
| Fixed center × shuffled saliency | 72.96% | 68.27% | 71.05% | 2.41% |
| **Adaptive center × aligned saliency** | **76.10%** | **71.59%** | **71.06%** | **2.47%** |
| Adaptive center × shuffled saliency | 73.80% | 68.47% | 68.63% | 2.44% |

The new combination reached 76.10% on clean test digits and 71.59% when symbols were introduced only at test time. On the latter comparison it gained 2.67 percentage points over the fixed Gaussian in all five seeds. Its mean gain against adaptive centering alone was 1.31 points (paired SD 3.03; two of five positive). Against the **fixed center with the same image-specific saliency**, the mean change was only +0.09 points (paired SD 2.54; two of five positive). Deltas use unrounded accuracy.

With the image-specific center held fixed, aligned saliency outperformed shuffled saliency by 3.12 points on the primary comparison (four of five seeds) and 2.43 points under marked training (five of five). The primary center-by-saliency interaction was −1.28 points (paired SD 3.09). The average saliency gain was smaller with an adaptive center. Under marked training the combined method assigned mean weight 2.62 to digit ink and 0.91 to added symbols. With a fixed center and aligned saliency those weights were 2.61 and 0.87. The combined method improved mean accuracy without reducing the absolute weight on symbols.

## Reproduce

Run from the IDEC directory in the project environment. The fitting command uses the verified MNIST loader and the two archived unlabeled saliency maps.

```powershell
python adaptive_saliency_study.py --output scratch/adaptive_saliency
python gaussian_refinement_analysis.py --study adaptive_saliency --results scratch/adaptive_saliency --output scratch/adaptive_saliency_tables
```

To rescore all paired comparisons from the released predictions on CPU, run `python gaussian_refinement_analysis.py --study adaptive_saliency --output scratch/adaptive_verified`. [Per-seed results](runs.csv) · [Means, paired differences and interaction](summary.json) · [New predictions](results)
