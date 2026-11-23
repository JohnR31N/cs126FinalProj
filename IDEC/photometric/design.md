# Whole-image jitter versus a Gaussian reconstruction prior

This protocol tests two standalone pretraining interventions on the same centered MNIST split as the detached-mark study. The previous augmentation changed the shape and site of a decoration. In this experiment the assigned mark stays fixed throughout model training.

## Matched comparisons

Train four arms on marked images using every combination of whole-image jitter on/off and Gaussian reconstruction weights on/off. Jitter uses independent per-image brightness and contrast factors uniform on [0.7, 1.3], in a random order, with clipping to the valid intensity range after each operation. Contrast blends around the image mean. Apply the same transformed image as encoder input and reconstruction target. Grayscale MNIST supplies one intensity channel. The Gaussian uses the existing center (13.5, 13.5), width seven pixels and mean-one normalization. Both interventions operate during the 100 pretraining epochs. All four arms run ordinary IDEC fine-tuning on their original unjittered marked images. Fix the original five seeds, minibatch schedule, split, optimizer and 5,600-update budget. No test outcomes will be used to choose the augmentation range or Gaussian width.

Separately train two controls on clean MNIST throughout, one with jitter and one without. Compare these controls on clean test digits to see whether jitter can help digit clustering in the complete absence of the synthetic mark. Compare gains in the marked and clean domains descriptively. A treatment can improve digit clustering and background invariance simultaneously.

## Outcomes and choice of method

Use a training-only digit mapping and freeze each model. Primary background metric: assignment changes on the same digit when its mark moves. Secondary: switch its shape at the original location, and record the marked-to-clean accuracy gap. Also record marked and clean ACC and occupied clusters. Fit an independent, standardized logistic regression to decode plus versus cross from the frozen final embedding. Use both shapes for every train and test digit at its first site, so digit identity and site are exactly balanced across shapes. Fit scaling and probe only on training embeddings, evaluate on unseen digits, chance 50%. This measures linearly accessible shape information, conditional on this probe. The assignment comparison directly tests whether background changes affect a clustering decision. Neither readout identifies a mediation fraction for ACC.

For a spatial-prior stress test, shift the entire clean digit four pixels horizontally (direction fixed by alternating test index). Score only pairs for which the shift preserves every nonzero ink pixel. Compare accuracy before and after shifting the same eligible digits, with the same frozen mapping. This tests reliance on centering under one controlled translation.

Compare jitter alone against Gaussian weighting alone for the primary background measure, corroborating shape-switch sensitivity, and check for a digit-accuracy or cluster-collapse tradeoff. Use the full factorial to measure paired marginal effects and interaction. Report seed differences, their sample SD and direction counts. Treat unstable tradeoffs as inconclusive. The joint arm is additional evidence on whether the interventions complement each other.
