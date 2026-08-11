
## Statistical Analysis Summary — Reconciling Apparently Conflicting Results

Four statistical tests were run on matched per-image results (n=527 test images
with ground truth) comparing YOLOv8n and Faster R-CNN.

**Key finding — the tests measure different things, and both are valid:**

1. **McNemar's Test (p=0.0002, highly significant):** Tests binary correctness
   (IoU ≥ 0.5 threshold) on *discordant pairs only* — images where the two models
   disagreed. Faster R-CNN was correct in 23 of these 26 discordant cases, versus
   YOLO in only 3. This shows Faster R-CNN more reliably clears the 0.5 IoU bar
   in the specific images where the models disagree.

2. **Wilcoxon signed-rank test (p=0.068, not significant at α=0.05):** Tests the
   full paired distribution of continuous IoU values. Borderline, suggesting the
   overall IoU distributions are similar between models across most images.

3. **Paired permutation test (p=0.0135, significant) and Bootstrap 95% CI
   ([-0.0321, -0.0041], excludes 0):** Both indicate Faster R-CNN's mean IoU is
   statistically higher than YOLO's, but the **effect size is small**
   (Cohen's d = -0.106), meaning the practical magnitude of this difference is
   minor.

**Why this doesn't contradict Phase 5's finding that YOLO scored higher on IoU:**
Phase 5's IoU (0.8865 for YOLO vs 0.8755 for Faster R-CNN) was computed *only
over correctly matched true-positive detections* — i.e., conditioned on YOLO
already having found the object. Phase 8's per-image IoU includes a score of 0
for any image where a model produced no detection at all, penalizing missed
detections. Since Faster R-CNN's recall is much higher (0.9865 vs 0.9426), it
records far fewer zero-IoU images, which raises its per-image average despite
slightly lower quality on the detections it does make.

**Conclusion:** The two models are statistically distinguishable (McNemar,
permutation, bootstrap all significant), but the practical difference is small
(Cohen's d = -0.11). Faster R-CNN's advantage stems primarily from higher
recall (fewer complete misses) rather than sharper localization on successful
detections — consistent with the recall-vs-precision trade-off already observed
in Phase 5.
