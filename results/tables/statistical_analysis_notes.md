
## Statistical Analysis Summary — Reconciling the Results

Four statistical tests were run on matched per-image results (n=528 test images
with ground truth) comparing YOLOv8n and Faster R-CNN.

**Results from this run:**

| Test | Statistic | p-value | Significant (α=0.05) |
|---|---|---|---|
| McNemar's Test | 5.2632 | 0.0218 | Yes |
| Wilcoxon signed-rank | 55177.0 | 0.0000 | Yes |
| Paired Permutation | -0.0097 | 0.1504 | No |
| Bootstrap 95% CI | [-0.0230, 0.0027] | N/A | No (CI includes 0) |

Effect size: Cohen's d (paired) = -0.0638 (very small).

**Why McNemar's and Wilcoxon disagree with the permutation test and bootstrap CI:**

McNemar's test, restricted to the 19 images where the two detectors'
correctness disagreed, found Faster R-CNN correct in 15 of those cases versus
YOLOv8n in only 4 — a significant asymmetry linked to Faster R-CNN's higher
recall. Wilcoxon (rank-based) detected a consistent directional shift across
the full paired IoU distribution. However, the paired permutation test and
bootstrap confidence interval, which are more sensitive to the magnitude and
variance of the difference, did not find the mean IoU difference significant
(the 95% CI spans zero), and the effect size is very small.

This divergence traces to a small number of complete-miss images (IoU = 0):
in this run, YOLOv8n produced 13 such images versus only 1 for Faster R-CNN.
These few cases inflate variance (shrinking Cohen's d and widening the
bootstrap CI) while still registering as a consistent directional signal
under rank-based tests. The exact count of complete misses fluctuates
slightly between runs due to minor GPU/cuDNN non-determinism, which is why
results can shift somewhat run-to-run — see the notebook's discussion for
details.

**Conclusion:** The two detectors are statistically distinguishable under
some framings (McNemar, Wilcoxon) but not others (permutation, bootstrap),
and the effect size is consistently small across runs. This instability
itself is informative: it confirms that the practical difference between the
two detectors is small and sensitive to a handful of edge cases, rather than
reflecting a robust, large effect — consistent with the recall/precision
trade-off already observed in the Phase 5 comparison.
