# HODC26

Code for the Kaggle **Hyperspectral Object Detection Challenge 2026** experiments.

## Current RF-DETR pipeline

The strongest local pipeline currently uses RF-DETR Small with 16-band hyperspectral input:

- annotation audit and invalid-box cleanup;
- stratified `2400 / 300 / 300` train/validation/holdout split;
- Phase A pseudo-RGB warm start;
- Phase B 16-band fine-tuning;
- Small-v2 fine-tuning with fixed short-side resolution `384`, aspect-ratio preservation, low learning rates, and weak spectral augmentation.

Best Small-v2 local results from the current run:

- validation mAP50-95: `0.68629`;
- holdout mAP50-95: `0.68030`.

The corresponding Kaggle submission is generated from the RF-DETR checkpoint with 300 DETR queries, confidence threshold `0.001`, and no extra NMS. Its Public LB score is `0.59025`. The current best Public LB in this repository remains `0.59086` from YOLO11s native 16-band HSI.

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
- `scripts/submit_kaggle.py` — Kaggle submission helper using a local access token.

## Local-only files

Datasets, prepared data, model weights, run outputs, submissions, and credentials are intentionally excluded from Git by `.gitignore`.
