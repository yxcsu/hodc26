# HODC26

Code for the Kaggle **Hyperspectral Object Detection Challenge 2026** experiments.

## Current RF-DETR pipeline

The strongest local pipeline currently uses RF-DETR Small with 16-band hyperspectral input:

- annotation audit and invalid-box cleanup;
- stratified `2400 / 300 / 300` train/validation/holdout split;
- Phase A pseudo-RGB warm start;
- Phase B 16-band fine-tuning;
- Small-v2 fine-tuning with fixed short-side resolution `384`, aspect-ratio preservation, low learning rates, and weak spectral augmentation.

Frozen reload-and-score results for the Small-v2 checkpoint are:

- validation mAP50-95: `0.679889`;
- holdout mAP50-95: `0.679205`.

The earlier `0.68629` value was a training-time peak and is not used as the
reproducible checkpoint score. With legal same-model `384/448/512` TTA and
class-wise WBF, the same checkpoint reaches `0.69284 / 0.69007`
(validation/holdout) and **Public LB `0.60617`**, which is the current
repository best.

The frozen checkpoint/split/script hashes and environment are recorded under
`results/repro_384/`.

## Current best: YOLO + RF-DETR class-wise WBF ensemble

The strongest pipeline fuses YOLO11s HSI16 and RF-DETR Small-v2 (group-2026
split, 3-scale TTA) predictions with per-class WBF parameters:

- validation mAP50-95: `0.705525`; holdout mAP50-95: `0.709547`
  (class-wise selection, tolerance 0.003);
- strict variant (both splits non-decreasing per class): `0.704382 / 0.709693`;
- **Public LB `0.64630`** (`submissions/yolo_rfdetr_classwise_tol003.csv`,
  validator-clean), the current repository best and `+0.04013` over the
  single-model TTA submission `0.60617`.

Per-class parameters are selected from 8 pre-scored stable fusion
configurations (`results/yolo_rfdetr_class_candidates/`) with a per-split
tolerance cap of 0.003 AP; widening the tolerance to 0.005/0.01 no longer
changes the selection (parameter plateau). Confidence truncation and top-K
sweeps confirmed no extra post-processing is needed (top-K=300 best).

## Experiment documentation

For external/AI review, start with:

- [`docs/EXPERIMENT_REPORT.md`](docs/EXPERIMENT_REPORT.md) — dataset, preprocessing, model variants, exact local/Kaggle results, failures, and reproducibility notes.
- [`docs/AI_REVIEW_BRIEF.md`](docs/AI_REVIEW_BRIEF.md) — compact problem statement and the highest-priority questions for proposing new methods.
- [`results/experiment_summary.csv`](results/experiment_summary.csv) — machine-readable experiment table.
- [`results/experiment_summary.json`](results/experiment_summary.json) — machine-readable dataset/method/result metadata.

## Main scripts

- `scripts/audit_annotations.py` — annotation audit.
- `scripts/prepare_stratified_hsi_split.py` — audited stratified split generation.
- `scripts/prepare_multispectral_yolo.py` — 16-band TIFF preparation.
- `scripts/prepare_rfdetr_coco.py` — RF-DETR COCO dataset conversion.
- `scripts/train_rfdetr_multispectral.py` — multispectral RF-DETR training and weak spectral augmentation.
- `scripts/make_rfdetr_submission.py` — RF-DETR Kaggle submission generation.
- `scripts/make_rfdetr_crop_tta.py` — legal same-model 2x2 crop-TTA inference.
- `scripts/fuse_detection_csv.py` — class-wise same-model WBF/NMS fusion.
- `scripts/fuse_detection_csv_classwise.py` — cross-model (YOLO+RF-DETR) fusion with per-class WBF parameters.
- `scripts/select_classwise_fusion.py` — conservative per-class fusion parameter selection from pre-scored candidates.
- `scripts/score_detection_csv.py` — COCO-style scoring of prediction CSVs.
- `scripts/eval_yolo_scales.py` — matched YOLO inference-scale audit.
- `scripts/submit_kaggle.py` — Kaggle submission helper using a local access token.

## Local-only files

Datasets, prepared data, model weights, run outputs, submissions, and credentials are intentionally excluded from Git by `.gitignore`.
