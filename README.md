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
(validation/holdout) and Public LB `0.60617`.

The frozen checkpoint/split/script hashes and environment are recorded under
`results/repro_384/`.

## Current best: YOLO + RF-DETR + DINO class-gated WBF

The strongest Public-LB pipeline adds a DINO-R50 4-scale branch to the
YOLO11s HSI16 + RF-DETR Small-v2 fusion and enables DINO only on selected
classes:

- DINO gate (10 classes): validation `0.709772`, holdout `0.708759`;
- **Public LB `0.64913`** (`submissions/yolo_rfdetr_dino_gate.csv`),
  the current repository best;
- conservative 8-class DINO gate: `0.709693 / 0.709400`, Public
  `0.64912`;
- YOLO + RF-DETR classwise-v2 without DINO: `0.706265 / 0.709939`,
  Public `0.64685`.

The best single-model HSI-DINO experiment uses a 19-channel
(`16 HSI + pseudo-RGB 5/8/13`) residual stem with spectral residual scale
`alpha=0.23`. It reaches `0.657806 / 0.657169` locally and Public
`0.58096`, versus pseudo-RGB DINO Public `0.57900`. The independent
seed-9173 audit preserves the HSI gain (`+0.000519 / +0.000915`).
See `docs/DINO_EXPERIMENT_20260920.md` for the complete audit and rejected
adaptation variants.

The competition's final no-ensemble/code-review requirements must be checked
before treating the cross-model WBF score as a final eligible solution.

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
