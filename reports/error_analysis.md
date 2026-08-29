**Error Analysis**

On a 600-image held-out sample (300 REAL, 300 FAKE) from CIFAKE, our adaptive model misclassified 64 images (89.3% accuracy on this subsample, consistent with our full clean-accuracy result of 90.6%). Examining these errors, we found two consistent patterns rather than a single dominant failure mode:

1. **Errors cluster near the blur-gate decision boundary.** Misclassified images predominantly had blur scores in the low-to-mid range relative to our threshold (1197.32), suggesting these are cases where the gating heuristic is making a genuinely uncertain call rather than a clearly wrong one.
2. **Low prediction confidence, not confident wrongness.** Predicted probabilities for misclassified images clustered close to 0.5 (ranging roughly 0.19–0.64) rather than near 0 or 1, indicating the model is genuinely uncertain on these cases rather than being confidently fooled.

We also note a dataset-specific limitation: CIFAKE images are natively 32×32 pixels, which compresses the achievable range of blur scores compared to full-resolution photos. This means our blur threshold, tuned on CIFAKE, may not transfer directly to higher-resolution datasets (e.g. SID\_Set) without recalibration — a limitation we plan to investigate as we extend evaluation to additional datasets.
