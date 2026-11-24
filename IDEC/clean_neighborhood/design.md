# Content-dependent reconstruction on original MNIST

The geometric backgrounds in the earlier DEC and IDEC comparison provide a controlled test of sensitivity to irrelevant features. The experiment here evaluates the proposed weighting rule on the original MNIST images. Training and evaluation use the same unmodified inputs and targets as the project's clean IDEC control.

## Weight estimation

Use the project's 12,000 training examples in their fixed seeded order. Divide them into two disjoint folds and score each image against the other fold. For each 7-by-7 region, find 20 neighbors using squared Euclidean distance on 4-by-4 average-pooled intensities. The distance uses pooling cells outside the region and its four-pixel margin. This exclusion affects the neighbor calculation only. The clustering model sees the complete image.

For each pixel, compare the squared error of the mean reference image with the squared error of the mean neighbor image, both at that coordinate. Its raw weight is 0.05 plus the positive part of the reduction in error. Normalize to mean one per image. This estimates how much knowing the surrounding image improves prediction. A stable background correlated with the rest of an image can also earn a high score.

## Controlled comparison

In each of five paired seeds, fit ordinary IDEC, IDEC with its own per-image weights, and IDEC with the same weights assigned to different images. The control permutes whole weight maps without changing their coordinates. Both weighted arms use the maps for autoencoder pretraining and clustering fine-tuning. They reconstruct the original input. Hold the initial random seed, minibatch order, network, input scaling and training budget fixed across arms. Use 100 pretraining epochs and 5,600 clustering updates.

Evaluate each model on all 10,000 original test images. Assign each test image to a cluster directly from the encoder and cluster centers. Fit the cluster-to-digit mapping from training labels after fitting. Report held-out accuracy, NMI, ARI, occupied clusters and the paired accuracy differences. The comparison with uniform weights tests the proposed objective. The comparison with shuffled weights tests whether matching the map to its own image adds value beyond the common spatial weighting. The previous synthetic results serve as the motivation for this study.

## Results on original images

| Reconstruction objective | Accuracy | NMI | ARI |
|---|---:|---:|---:|
| Ordinary IDEC | 73.70% | 0.673 | 0.568 |
| Neighborhood weights | 68.59% | 0.636 | 0.523 |
| Same weight maps assigned to other images | 65.85% | 0.593 | 0.451 |

Each entry is a five-seed mean. Each fit occupied ten clusters. Neighborhood weighting lost 5.11 ± 3.49 accuracy points against the ordinary IDEC refit and scored lower in all five pairs. It gained 2.73 ± 6.83 points against shuffled maps, winning three pairs. Both standard deviations describe the paired differences. The earlier clean baseline belongs to its own training run.

On original MNIST, weighting pixel errors by how well neighbors predict them lowered digit clustering accuracy. We applied the weights during pretraining and fine-tuning, so this experiment measures their combined effect.

The per-seed assignments can be rescored with `python clean_neighborhood_study.py --output clean_neighborhood/results --evaluate-only` from the IDEC directory. The training revision determines the source digests in the manifest. [Per-seed results](results/summary.json) · [Training source](https://github.com/JohnR31N/cs126FinalProj/blob/9191cee6f4390d2983591a617f3b73aad9857421/IDEC/clean_neighborhood_study.py)
