# Selecting named location and orientation bins

This additional test was specified while the main training jobs were running, before scoring either holdout. It addresses the distinction between projecting onto nine combinations of stroke features and keeping a literal subset of the 392 measurements.

Score each orientation bin separately on each training half using the examples where the two graph partitions agree. The pseudo-group IDs for a half come from the opposite half's neighbor vote. For each bin, calculate the proportion of its variance explained by differences between group means. A constant bin scores zero. Use the lower of its two half-scores and keep the 98 highest-scoring bins. This asks whether the separation repeats.

Reconstruct only the continuous responses in those 98 bins. Compare the 98 lowest-scoring bins and an equal-size control that randomly selects exactly the same number of orientations in each spatial cell as the retained set. A fixed seed determines this control before scoring. Some selected and location-matched bins can coincide. Also evaluate a uniformly random 98-bin baseline with frozen K-means. Use the same feature targets, decoder readout, training split, five seeds, and held-out metrics as the main study. The per-bin table records the two half-scores, their minimum, and the selected coordinates.
