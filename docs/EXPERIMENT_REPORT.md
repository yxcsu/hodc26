# HODC26 Experiment Report

Last updated: 2026-09-16

This document summarizes the experiments performed for the Kaggle **Hyperspectral Object Detection Challenge 2026**. It is intended to be sufficiently explicit for another researcher or AI system to audit the current approach, identify failure modes, and propose stronger methods.

## 1. Problem and data

- Task: multi-class object detection from hyperspectral imagery.
- Training images: 3,000.
- Test images: 1,000.
- Classes: 18.
- Raw annotation objects before cleanup: 10,107.
- The raw hyperspectral representation is reconstructed into 16 spectral bands.
- Original image aspect ratio is approximately 2:1 for many samples, although dimensions vary.

Class names used by the code:

`apple`, `apple_plastic`, `badminton`, `banana`, `banana_plastic`, `car`, `car_toy`, `charger_head`, `e-bike`, `egg`, `egg_plastic`, `egg_wood`, `orange`, `orange_plastic`, `people`, `rubik`, `stone_block`, `table_tennis`.

## 2. Annotation audit

The original XML annotations were audited before the final RF-DETR experiments.

Observed severe annotation issues included:

- zero-area / degenerate boxes in image IDs `298`, `423`, `953`, `1590`, `1974`, `3294`, and `4763`;
- out-of-image boxes in image IDs `1989` and `3093`;
- repeated zero-area boxes in several of the affected images, especially `1974`, `423`, and `3294`.

The audited conversion removed **41 degenerate boxes** and clips valid boxes to image boundaries. We do not delete entire images when valid objects remain.

Important caution: an earlier public solution description used a total of 2,997 samples (`2400/298/299`), but the public description did not expose the three exact sample IDs or prove that three images should simply be deleted. We therefore did not force our data to 2,997 merely to match that count.

Relevant scripts:

- `scripts/audit_annotations.py`
- `scripts/prepare_stratified_hsi_split.py`

## 3. Data split

Current audited split:

- train: 2,400 images;
- validation: 300 images;
- independent holdout: 300 images.

The split is class-aware/stratified and uses seed 42 for the final RF-DETR experiments. Validation and holdout are never used as training images.

The important open question is whether random/class-aware image splitting is optimistic because nearby IDs may come from the same acquisition sequence, scene, or object session. This is now one of the highest-priority items to audit because local scores substantially exceed Public LB performance.

## 4. Spectral representations

### 4.1 Pseudo-RGB

Three selected spectral bands `(5, 8, 13)` are converted to an RGB-like input. Each selected band is stretched per image for the pseudo-RGB representation.

This is primarily used for RGB-pretrained warm starts.

### 4.2 Native 16-band HSI

The packed source image is reshaped into a 16-channel cube. A fixed global intensity scale is used for the main HSI pipeline so cross-band intensity relationships are not destroyed by independent per-band stretching.

The current preparation uses a clip/scale convention based on observed sensor values concentrated roughly in `0..320`.

## 5. YOLO baselines

The strongest YOLO model is currently more competitive on Public LB than RF-DETR despite lower local validation mAP.

| Model | Representation | Main training setup | Local mAP50-95 | Kaggle Public LB |
|---|---|---|---:|---:|
| YOLO11n | pseudo-RGB 5/8/13 | 25 ep | 0.62767 | 0.53102 |
| YOLO11s | pseudo-RGB 5/8/13 | 25 ep, 640 px, batch 32 | 0.64575 | 0.53704 |
| YOLO11s | native 16-band | 22 ep after smoke/warm start, 640 px, batch 32 | 0.64114 | **0.59086** |
| YOLO11s | native 16-band continuous | 25 ep, 640 px, batch 32 | 0.64525 | 0.58259 |

The two strongest native-HSI YOLO runs have a much smaller local-to-Public gap than pseudo-RGB YOLO or RF-DETR.

Representative YOLO settings include standard Ultralytics detection augmentation (mosaic, horizontal flip, HSV parameters), `max_det=300`, and image size 640. See each run's `args.yaml` locally and `scripts/train_yolo.py` in Git.

## 6. RF-DETR pipeline

### 6.1 Phase A: pseudo-RGB warm start

Model: RF-DETR Small.

Input:

- 3 channels from pseudo-RGB bands `(5,8,13)`.

Training:

- 50 epochs;
- batch 16;
- nominal resolution 512;
- aspect-ratio preserving resize (not forced square);
- RF-DETR multi-scale enabled;
- learning rate `1e-4`;
- encoder learning rate `1.5e-4`;
- 300 queries;
- no optional image augmentation (`aug_config={}`), no scale jitter;
- BF16 AMP.

Best local result:

- zero-based epoch index 25 (26th epoch);
- validation mAP50-95: **0.61537**;
- validation mAP50: 0.94296;
- validation mAP75: 0.72261;
- independent holdout mAP50-95: **0.60153**.

Longer training was useful, but the last epoch was worse (`0.58867`), so best-checkpoint selection matters.

### 6.2 Phase B: native 16-band fine-tuning

Initialization: Phase A best checkpoint.

Training:

- 16 input channels;
- 50 epochs;
- batch 16;
- nominal resolution 512;
- aspect-ratio preserving resize;
- multi-scale enabled;
- learning rate `5e-5`;
- encoder learning rate `7.5e-5`;
- 300 queries.

Best local result:

- zero-based epoch index 7 (8th epoch);
- validation mAP50-95: **0.62264**;
- validation mAP50: 0.94474;
- validation mAP75: 0.74299;
- independent holdout mAP50-95: **0.61244**.

Again, later epochs degraded: epoch 50 ended near `0.58357`.

### 6.3 Small-v2: 384 short-side + low LR + weak spectral augmentation

Initialization: Phase B best checkpoint.

Key changes:

- 16 bands;
- fixed short-side resolution 384, preserving aspect ratio;
- typical tensor shape is around `16 x 384 x 779` for ~2:1 images;
- RF-DETR multi-scale disabled;
- 20 epochs;
- batch 16;
- head/main LR `3e-5`;
- encoder/backbone LR `1.5e-5`;
- 300 queries;
- weak train-only spectral augmentation:
  - global gain: `0.95..1.05`;
  - linear spectral tilt: `±0.03` over band index;
  - Gaussian noise: `sigma=0.0002` in the normalized pre-ImageNet-normalization intensity domain;
- no extra spatial augmentation in `aug_config`;
- positional embeddings are bicubically interpolated when transferring 512-resolution checkpoints to 384.

Best local result:

- zero-based epoch index 13 (14th epoch);
- validation mAP50-95: **0.68629**;
- validation mAP50: 0.95171;
- validation mAP75: 0.82218;
- validation mAR: 0.74249;
- independent holdout mAP50-95: **0.68030**;
- holdout mAP50: 0.94904;
- holdout mAP75: 0.81680;
- holdout mAR: 0.74353.

This is by far the best local model.

## 7. RF-DETR inference and Kaggle submission

For the current Small-v2 submission:

- fixed short-side 384, aspect ratio preserved;
- 300 DETR queries;
- confidence threshold `0.001`;
- no additional NMS;
- no box scaling;
- background slot is removed, keeping class IDs `0..17` only;
- 1,000 test images are all represented;
- submission contains 231,237 detections, approximately 231 predictions/image on average.

Kaggle submission:

- submission ref: `56279353`;
- Public LB: **0.59025**.

This did **not** beat the current repository best Public LB `0.59086` from YOLO11s native 16-band HSI.

## 8. Local score versus Kaggle Public LB

The central unresolved issue is the widening gap between local mAP and Public LB.

| Experiment | Local validation | Holdout | Public LB | Validation → Public gap |
|---|---:|---:|---:|---:|
| YOLO11n pseudo 5/8/13 | 0.62767 | n/a | 0.53102 | 0.09665 |
| YOLO11s pseudo 5/8/13 | 0.64575 | n/a | 0.53704 | 0.10871 |
| YOLO11s native HSI16 | 0.64114 | n/a | **0.59086** | 0.05028 |
| YOLO11s native HSI16 continuous | 0.64525 | n/a | 0.58259 | 0.06266 |
| RF-DETR Small-v2 HSI16 | **0.68629** | **0.68030** | 0.59025 | **0.09604** |

The RF-DETR holdout agrees very closely with validation, but both overestimate the Public LB by about 0.09. This makes simple validation overfitting less likely than if only one local split were high, but it does not exclude correlated train/val/holdout sampling from the same acquisition groups.

## 9. Known implementation pitfalls already fixed

The following issues were found during development and should not be reintroduced:

1. RF-DETR's standard loader initially assumes 3 input channels. Custom loading logic is required for already-16-channel checkpoints.
2. Transferring a 512-resolution checkpoint to 384 requires positional-embedding interpolation; `strict=False` alone does not bypass shape mismatches.
3. RF-DETR evaluation must use the same aspect-ratio-preserving resize policy as training. An accidental `384x384` square + multi-scale evaluation produced an invalid holdout score around 0.286 and was discarded.
4. RF-DETR output includes an extra background class slot; submission generation must filter class ID 18 and keep only 0..17.
5. A nominal RF-DETR `resolution=512` with multi-scale enabled did not mean a fixed 512 short side; logs showed a training scale of 672. Small-v2 explicitly disables multi-scale to obtain the intended fixed 384 short side.

## 10. High-priority hypotheses for improvement

These are hypotheses, not established conclusions.

### A. Group-aware validation may be required

If image IDs or adjacent samples correspond to the same acquisition session/scene/object arrangement, random stratification can leak scene-level information across train/val/holdout. A grouped split should be constructed using any available acquisition metadata or image-similarity clustering.

### B. Submission density / score calibration may be suboptimal

RF-DETR submits ~231 boxes per test image at `conf=0.001`. Even when AP theoretically ranks predictions by confidence, the competition's exact evaluator/max-detection behavior may differ from local COCO evaluation. Threshold sweeps, top-K per image, class-specific thresholds, and calibration should be evaluated locally using the exact competition metric implementation if available.

### C. RF-DETR spatial resolution may still be limiting

384 short-side improves local localization substantially, but controlled inference at 448/512/576 or aspect-ratio-aware multi-scale TTA may improve small-object localization if memory permits. This must be evaluated without changing preprocessing semantics.

### D. Spectral front-end is primitive

The current model expands a standard image patch embedding to 16 channels. Stronger alternatives include:

- spectral stem / 1D spectral mixer before the 2D backbone;
- learned 16→3 or 16→C projection initialized by PCA/linear regression from pretrained RGB features;
- low-rank spectral adapters;
- band attention/gating;
- spectral-spatial separable convolutions;
- wavelength-aware positional/channel embeddings if wavelengths are available.

### E. Ensembling may help if errors are complementary

YOLO11s HSI16 has the best Public LB while RF-DETR has much stronger local localization. Their predictions may be complementary. Weighted Boxes Fusion, class-aware box matching, or score-rank fusion should be tested on validation/holdout before any Kaggle submission.

### F. Test-domain normalization may differ

Because the model uses a fixed clip/scale convention, inspect per-band train/test distributions. If test acquisition has intensity drift, robust percentile calibration, per-cube gain correction, or train-derived per-band normalization may improve transfer.

## 11. Suggested experiment order

To avoid spending Kaggle submissions inefficiently, a reviewer should prioritize experiments that can be rejected locally:

1. build group-aware or similarity-clustered CV and quantify whether the current 0.68 drops;
2. reproduce the exact competition metric locally and sweep DETR top-K/confidence strategies;
3. analyze train-vs-test per-band distribution shift;
4. evaluate YOLO/RF-DETR error complementarity and WBF/rank fusion on holdout;
5. test fixed-resolution 448/512 variants from the 384 checkpoint;
6. test a learnable spectral projection/stem with conservative initialization;
7. only then consume new Kaggle submissions.

## 12. Reproducibility map

- Annotation audit: `scripts/audit_annotations.py`
- Audited split: `scripts/prepare_stratified_hsi_split.py`
- Hyperspectral conversion: `scripts/prepare_multispectral_yolo.py`
- RF-DETR COCO export: `scripts/prepare_rfdetr_coco.py`
- RF-DETR multispectral training: `scripts/train_rfdetr_multispectral.py`
- RF-DETR submission generation: `scripts/make_rfdetr_submission.py`
- Kaggle submission helper: `scripts/submit_kaggle.py`
- YOLO training: `scripts/train_yolo.py`
- Machine-readable results: `results/experiment_summary.csv` and `results/experiment_summary.json`

Large datasets, weights, run logs, and submission CSVs remain intentionally excluded from Git.
