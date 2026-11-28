# Uniform reference

Before scoring any follow-up predictions, add five ordinary IDEC fits so the new 10,000-image endpoint also has an original-method reference. Fit the same clean training images under the same seeds, architecture, optimizer, update schedule and evaluation code. Set all pretraining pixel weights to one and use ordinary IDEC fine-tuning. No teacher is needed. Score the identical held-out and official pairs, plus both views of the shift-eligible subset. The original four-arm shape comparison and its selection rule remain as registered.
