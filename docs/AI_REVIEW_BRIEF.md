# AI Review Brief: HODC26

## Goal

Propose experiments that can improve Kaggle Public/Private performance for a 16-band hyperspectral object detection task. Prefer methods that are testable locally before spending a Kaggle submission.

## Current best facts

- Current best Kaggle Public LB: **0.59086**, from YOLO11s native 16-band HSI.
- Best local model: RF-DETR Small-v2, **val 0.68629 / independent holdout 0.68030**.
- RF-DETR Small-v2 Kaggle Public LB: **0.59025**.
- Therefore, improving local mAP alone is not currently translating into leaderboard improvement.
- Training data: 3,000 images, test: 1,000, classes: 18.
- Final local split: 2,400 train / 300 validation / 300 holdout.
- Annotation audit removed 41 zero-area boxes and clips boundary violations.

## Strongest RF-DETR recipe

- RF-DETR Small.
- 16 native hyperspectral channels.
- Warm-start via pseudo-RGB `(5,8,13)`, then 16-band fine-tuning.
- Final stage: 20 epochs at fixed short side 384, aspect ratio preserved.
- LR: `3e-5`; encoder LR: `1.5e-5`; batch 16.
- Weak spectral augmentation: gain 0.95–1.05, tilt ±0.03, Gaussian noise sigma 0.0002.
- 300 DETR queries.
- Test submission: threshold 0.001, no extra NMS, class IDs 0–17, ~231 detections/image.

## Most important anomaly

RF-DETR local performance improved dramatically:

`Phase A 0.615 -> Phase B 0.623 -> Small-v2 0.686`

but Kaggle Public LB remains around `0.590`.

YOLO native HSI has a smaller local-to-Public gap and currently wins Public LB despite lower local validation score.

## Please review these questions first

1. **Validation leakage/domain grouping**: could class-stratified random splitting place near-duplicate scenes/acquisition sequences into train, validation, and holdout? Propose a concrete grouped-CV construction using only images/annotations if explicit group metadata is absent.
2. **Metric mismatch**: what exact aspects of DETR output (`~231 boxes/image`, 0.001 threshold, 300 queries, no NMS) could hurt the Kaggle evaluator? Propose local sweeps that do not require Kaggle submissions.
3. **Train/test spectral shift**: propose quantitative tests for per-band intensity/domain shift and robust normalization strategies.
4. **YOLO + RF-DETR ensemble**: design a holdout experiment for class-aware WBF or rank/score fusion and specify parameters to sweep.
5. **Spectral modeling**: propose a minimal spectral front-end that can exploit 16 bands while retaining RGB-pretrained spatial weights.
6. **Resolution/localization**: propose fixed 448/512 or multi-scale TTA experiments that preserve aspect ratio and avoid reintroducing the invalid square-resize behavior.
7. **Annotation quality**: propose additional automated checks for mislabeled class, truncated objects, duplicate boxes, and scene-level duplicate/near-duplicate images.

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
