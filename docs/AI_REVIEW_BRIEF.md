# AI Review Brief: HODC26

## Goal

Propose experiments that can improve Kaggle Public/Private performance for a 16-band hyperspectral object detection task. Prefer methods that are testable locally before spending a Kaggle submission.

## Current best facts

- Current best Kaggle Public LB: **0.60617**, from the same RF-DETR Small-v2 checkpoint using legal single-model multi-scale TTA at short sides `384/448/512` plus class-wise WBF (`IoU=0.7`).
- TTA local evidence: **val 0.69284 / independent holdout 0.69007**, both improving by >0.01 over fixed 384 inference.
- The non-TTA RF-DETR Small-v2 checkpoint scored Public **0.59025**, so inference scale/fusion is a real externally verified factor.
- YOLO11s native 16-band remains a strong non-TTA baseline at Public **0.59086**.
- Training data: 3,000 images, test: 1,000, classes: 18.
- Final local split: 2,400 train / 300 validation / 300 holdout.
- Annotation audit removed 41 zero-area boxes and clips boundary violations.

## Strongest RF-DETR recipe

- RF-DETR Small.
- 16 native hyperspectral channels.
- Warm-start via pseudo-RGB `(5,8,13)`, then 16-band fine-tuning.
- Final stage: 20 epochs at fixed short side 384, aspect ratio preserved.
- LR: `3e-5`; encoder LR: `1.5e-5`; batch 16.
- Weak spectral augmentation: gain 0.95–1.05. A matched ablation found that removing tilt/noise slightly improved val/holdout, but only by ~0.002 mAP, below the submission threshold.
- 300 DETR queries.
- Test submission: threshold 0.001, no extra NMS, class IDs 0–17, ~231 detections/image.

## Updated validation audit

Conservative content-based grouping found 47 non-singleton near-duplicate/scene candidate groups (151 images total; largest group 10) among 3,000 training images. Two group-aware `2400/300/300` splits were trained independently with zero group overlap:

- YOLO11s group seed42: **val 0.70230 / holdout 0.69142**;
- YOLO11s group seed2026: **val 0.67923 / holdout 0.68557**.

Thus obvious near-duplicate image leakage is **not currently supported as the main explanation** for high local scores. Acquisition-level leakage remains possible because official time/sequence/session metadata are unavailable.

RF-DETR was also retrained independently on both group-aware folds. At short side 672, group42 Phase B reached **val 0.60818 / holdout 0.59214**, while group2026 Phase B reached **val 0.55723 / holdout 0.57764**. This confirms that RF-DETR is much more split-sensitive than YOLO and that the later fixed-384 stage is essential to its strong local performance.

The exact submission-CSV path has also been checked against RF-DETR framework evaluation on the same checkpoint/data and differs by only about `0.0005` mAP, so the basic CSV conversion path is not the source of the ~0.09 gap.

## Most important anomaly

RF-DETR local performance improved dramatically:

`Phase A 0.615 -> Phase B 0.623 -> Small-v2 0.686`

but fixed-scale Kaggle Public LB remained around `0.590`; multi-scale TTA improved it to **0.60617**.

YOLO native HSI has a smaller local-to-Public gap, but the current Public-LB winner is the single-model RF-DETR three-scale TTA at **0.60617**.

## Completed negative / low-yield experiments

Do not repeat these as first-line suggestions unless the method is materially different:

- confidence/top-K sweeps: only ~0.001 local mAP effect;
- horizontal-flip six-view TTA: no consistent >=0.005 gain over three-scale TTA;
- non-uniform 384/448/512 WBF weights: worse than equal weights;
- float32 clip(raw/320) + cyclic ImageNet normalization: ~0.022 worse than uint8 baseline on val and holdout;
- float32 + train-fold per-band statistics: better matched than cyclic ImageNet stats but still below uint8 baseline;
- simple identity-initialized full-rank `1x1 Conv(16->16)` adapter: **val 0.68361 / holdout 0.66916**, worse than no-adapter gain-only control **0.69044 / 0.67466**;
- gain-only vs gain+tilt+noise: gain-only is slightly better on both sets, but only by ~0.002 and therefore not a confirmed leaderboard-worthy gain.

Detailed error analysis shows small-object AP around **0.65**, with recurrent weak classes including `stone_block`, `people`, `car`, and unstable `e-bike`.

## Please review these questions first

1. **Small-object localization**: propose training or architectural changes that specifically improve small AP and `stone_block/people/car/e-bike` without sacrificing other classes.
2. **YOLO + RF-DETR complementarity**: design a holdout experiment for class-aware fusion/rank fusion and identify which classes should prefer which detector.
3. **Structured spectral modeling**: propose a spectral stem that is materially different from the rejected plain 1x1 adapter, e.g. low-rank, gated, separable, wavelength-aware, or attention-based.
4. **Residual domain grouping**: given that conservative image-content grouping did not collapse YOLO performance, what stronger acquisition/session grouping test can be constructed without explicit metadata?
5. **Final single-model training recipe**: how should the verified fixed-384 stage and three-scale TTA be used when retraining on all available labeled data while preserving a trustworthy final audit set?
6. **Annotation quality**: propose additional automated checks for mislabeled class, truncated objects, duplicate boxes, and scene-level duplicate/near-duplicate images.

## Constraints

- Prefer one-model solutions first; TTA/multi-scale inference is acceptable if competition rules allow it.
- Avoid methods that require hidden test labels or leaderboard probing as the primary optimizer.
- GPU environment available during experimentation: RTX 4090 class GPUs, 24 GB each.
- Do not assume the published 2,997-image split is exactly reproducible; the three omitted IDs were not publicly specified in our source.

## Desired reviewer output

Please return:

1. the top 5 experiments in priority order;
2. expected mechanism for each;
3. exact training/inference changes;
4. what local evidence would justify a Kaggle submission;
5. likely failure modes;
6. estimated implementation complexity and GPU cost;
7. any flaws you detect in the current methodology or code assumptions.

For full details, read `docs/EXPERIMENT_REPORT.md` and `results/experiment_summary.*`.
