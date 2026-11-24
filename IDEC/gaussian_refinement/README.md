# Refining the Gaussian reconstruction prior

The existing pretraining-only Gaussian has a fixed center at (13.5, 13.5), a width of seven pixels and mean pixel weight one. We change one property at a time. The **weight floor** averages the fixed Gaussian with uniform weights. A peripheral pixel then receives at least half the ordinary pixel weight. The **adaptive center** uses the intensity centroid of the observed image after weighting its intensities with the original Gaussian. This gives nearby bright pixels more influence over the location estimate. It requires no foreground mask. Both variants keep the width, mean weight and ordinary IDEC fine-tuning unchanged. [Experiment design](design.md)

The fitting set contains 12,000 digits and the official test set contains 10,000. The primary accuracy measures train and test each method in matching domains. We also apply the clean-trained student to the marked test view as an exploratory check. Test redraws keep the handwriting and symbol positions fixed and change the symbol attributes. Flips are measured on marked-trained models. The pixel and fixed-Gaussian controls reuse the matched experiment's predictions. The same five seeds and split are used throughout. These test images were examined in earlier experiments. In the paired JSON output, a positive difference counts toward `wins` for each metric; a lower flip rate or penalty is preferable.

## Results

| Pretraining weights | Clean ACC | Marked ACC | Clean-trained ACC on marked test | Redraw flips |
|---|---:|---:|---:|---:|
| Uniform pixel IDEC | 73.23% | 65.14% | 64.62% | 2.99% |
| Original fixed Gaussian | 74.07% | 68.80% | 68.92% | 2.62% |
| Gaussian with uniform floor | 73.39% | 68.96% | 67.27% | 2.75% |
| Gaussian with adaptive center | 76.20% | 69.71% | 70.28% | 2.58% |

The adaptive center gained 0.91 points on marked accuracy against the fixed Gaussian (3 of 5 seeds, paired SD 4.52). It gained 2.13 points on clean digits. The clean-to-marked gap therefore widened from 5.27 to 6.49 points. Its average weight on handwriting moved from 1.78 to 1.79, and its weight on added symbols rose from 0.66 to 0.69. The result favors the adaptive center on mean accuracy, with little change in redraw sensitivity and substantial seed variation. It provides no evidence of stronger downweighting of the added symbols.

The uniform floor gained 0.16 points on marked accuracy (2 of 5 seeds, paired SD 3.57) and lost 0.68 points on clean digits. Its mean weight on added symbols rose to 0.83. This intermediate study used the fixed prior as its reference. The [final project](../final/README.md) combines the image-dependent center with aligned cluster saliency on the same previously studied split.

## Reproduce

Run from the IDEC directory using the project environment.

```powershell
python gaussian_refinement_study.py --output scratch/gaussian_refinement
python gaussian_refinement_analysis.py --results scratch/gaussian_refinement --output scratch/gaussian_tables
```

To recompute the published scores on CPU, run `python gaussian_refinement_analysis.py --output scratch/verified_gaussian_tables`. [Per-seed scores](runs.csv) · [Paired comparisons and measured weights](summary.json) · [Predictions and manifest](results)
