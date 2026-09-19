# HODC26 Experiment Report

Last updated: 2026-09-19

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

### 7.1 Single-model multi-scale TTA

The same Small-v2 checkpoint was evaluated at three aspect-ratio-preserving short-side resolutions and fused as a **single-model TTA** experiment:

- 384;
- 448;
- 512;
- class-wise Weighted Boxes Fusion (WBF), IoU threshold `0.7`.

Single-scale validation mAP50-95:

- 384: `0.67990`;
- 448: `0.68272`;
- 512: `0.68121`.

Three-scale WBF results:

- validation mAP50-95: **0.69284**;
- validation mAP75: **0.83884**;
- holdout mAP50-95: **0.69007**;
- holdout mAP75: **0.83079**.

Kaggle submission ref `56281009` achieved **Public LB 0.60617**, improving the previous repository best `0.59086` by `+0.01531`. This is currently the strongest externally verified improvement in the project.

## 8. Local score versus Kaggle Public LB

The central unresolved issue is the widening gap between local mAP and Public LB.

| Experiment | Local validation | Holdout | Public LB | Validation → Public gap |
|---|---:|---:|---:|---:|
| YOLO11n pseudo 5/8/13 | 0.62767 | n/a | 0.53102 | 0.09665 |
| YOLO11s pseudo 5/8/13 | 0.64575 | n/a | 0.53704 | 0.10871 |
| YOLO11s native HSI16 | 0.64114 | n/a | **0.59086** | 0.05028 |
| YOLO11s native HSI16 continuous | 0.64525 | n/a | 0.58259 | 0.06266 |
| RF-DETR Small-v2 HSI16, frozen reload | **0.67989** | **0.67921** | 0.59025 | **0.08964** |
| RF-DETR Small-v2 HSI16, 384+448+512 TTA | 0.69284 | 0.69007 | **0.60617** | 0.08667 |

The previously quoted `0.68629` validation value was the training-time peak,
not the score reproduced after reloading the saved checkpoint. The frozen
checkpoint reload is `0.679889 / 0.679205` on validation/holdout. Multi-scale
TTA improves both local sets and the Kaggle Public LB in the same direction,
which supports the conclusion that localization scale is a real factor rather
than a validation-only artifact. A substantial local-to-Public gap remains.

## 8.1 Group-aware validation audit

To test whether near-duplicate scenes were leaking across random splits, conservative content-based scene groups were built using low-resolution image features plus perceptual-hash constraints.

Grouping summary:

- 3,000 training images;
- 2,896 groups;
- 47 non-singleton candidate groups;
- 151 images in non-singleton groups;
- largest group: 10 images;
- train/validation/holdout group overlap: zero.

Two independent `2400/300/300` group-aware splits were trained from the original `yolo11s.pt` initialization without cross-fold checkpoint reuse:

| Group split | YOLO11s HSI16 val mAP50-95 | Holdout mAP50-95 |
|---|---:|---:|
| seed 42 | **0.70230** | **0.69142** |
| seed 2026 | **0.67923** | **0.68557** |

These results do **not** support the hypothesis that the earlier local scores were primarily caused by obvious near-duplicate scene leakage. This is not proof that no acquisition-level leakage exists, because official sequence/time/session metadata are unavailable and the grouping is image-content-derived rather than sensor-metadata-derived.

### 8.2 Group-aware RF-DETR audit

RF-DETR was also retrained independently on both group-aware splits. Each fold used its own pseudo-RGB Phase A initialization and did not reuse checkpoints from another fold.

Representative reload-and-evaluate results:

| Group split | Stage | Inference short side | Val mAP50-95 | Holdout mAP50-95 |
|---|---|---:|---:|---:|
| seed 42 | Phase A pseudo-RGB | 672 | 0.59529 | 0.57995 |
| seed 42 | Phase B HSI16 | 672 | **0.60818** | **0.59214** |
| seed 2026 | Phase A pseudo-RGB | 672 | 0.57332 | 0.59106 |
| seed 2026 | Phase B HSI16 | 672 | **0.55723** | **0.57764** |

The two folds confirm two points. First, 16-band Phase B can improve over Phase A within a fold, but the magnitude is fold-dependent. Second, RF-DETR remains much weaker than YOLO11s under the conservative group-aware splits unless the later fixed-384 Small-v2 stage and TTA are used. The model is also highly sensitive to inference scale; 672 generally outperformed 512 on these group-aware checkpoints.

### 8.3 Photometric / spectral preprocessing ablations

The group-42 Phase B setup was used for controlled ablations.

At 672 inference:

| Input / normalization | Val mAP50-95 | Holdout mAP50-95 | Conclusion |
|---|---:|---:|---|
| uint8 + ImageNet-cyclic | **0.60818** | **0.59214** | baseline |
| float32 + ImageNet-cyclic | 0.58659 | 0.57043 | worse by ~0.022 on both sets |
| float32 + train-fold band statistics | 0.58634 | 0.57871 | recovers part of holdout loss but still below uint8 baseline |

Therefore, simply removing uint8 quantization is not beneficial in the current training recipe. Train-fold normalization is better matched to float32 HSI than cyclic RGB ImageNet statistics, but it still does not recover the original baseline.

### 8.4 Weak spectral augmentation ablation

Starting from the same group-42 Phase B checkpoint, both variants used fixed short-side 384, identical low learning rates, and a 10-epoch budget.

Reloaded fixed-384 results:

| Augmentation | Val mAP50-95 | Holdout mAP50-95 |
|---|---:|---:|
| gain 0.95-1.05 + tilt ±0.03 + noise 0.0002 | 0.68885 | 0.67284 |
| gain 0.95-1.05 only | **0.69044** | **0.67466** |

Gain-only is slightly better on both sets, but the improvement is only about +0.0016 to +0.0018, below the predeclared +0.005 submission threshold. It is therefore not considered a confirmed performance gain. It is nevertheless the cleaner default because it removes the physically unverified channel-index tilt without reducing performance.

### 8.5 Identity spectral-adapter ablation

A `1x1 Conv(16->16)` spectral adapter was inserted before the patch projection and initialized to the identity matrix with zero bias. Initialization was verified exactly (`maxdiff=0`). The adapter has only 272 parameters.

Training protocol:

- 2 epochs adapter-only (all detector parameters frozen);
- then 8 epochs joint fine-tuning at reduced LR;
- gain-only augmentation;
- same group-42 split and fixed-384 geometry.

Final reload-and-evaluate results:

- validation mAP50-95: **0.68361**;
- holdout mAP50-95: **0.66916**;
- validation AP75: 0.81974;
- holdout AP75: 0.79971.

This is lower than the no-adapter gain-only control (`0.69044 / 0.67466`), so this simple full-rank identity 1x1 spectral adapter is rejected for the current recipe.

### 8.6 TTA and error-analysis conclusions

Horizontal-flip TTA and non-uniform scale weights were also tested and rejected locally:

- six-view `384/448/512 x {original,hflip}` WBF: val `0.69266`, holdout `0.69084`;
- three-scale original-view WBF: val `0.69279`, holdout `0.69007`;
- the extra hflip views do not provide a consistent >=0.005 gain;
- weighting 448 or 512 more heavily reduced mean val/holdout mAP relative to equal `1:1:1` weights.

Detailed class/size analysis of the strongest three-scale TTA shows:

- validation small-object AP: ~0.657; holdout: ~0.653;
- medium-object AP is materially higher (~0.746 val, ~0.722 holdout);
- recurrent weak classes include `stone_block`, `people`, and `car`, with `e-bike` also unstable across splits.

Future work should therefore prioritize small-object localization and these weak classes rather than additional confidence-threshold, hflip, or WBF-weight sweeps.

### 8.7 Group-2026 final fixed-384 and crop-TTA audit

The group-2026 Phase-B checkpoint was followed by the planned fixed-384,
gain-only 10-epoch stage. Reloaded results for
`runs/rfdetr_group2026_v2_gain_only_10ep/checkpoint_best_total.pth` are:

| Inference | Val mAP50-95 | Holdout mAP50-95 | Val small AP | Holdout small AP |
|---|---:|---:|---:|---:|
| fixed 384 | 0.678855 | 0.672859 | 0.648785 | 0.647123 |
| 384/448/512 WBF | 0.686540 | 0.681814 | 0.660571 | 0.655599 |
| + 2x2 crop view | **0.688383** | **0.685149** | **0.663995** | **0.659502** |

The crop view uses four 60%-by-60% corner crops, 20% overlap in original-image
coordinates, box-center ownership by image quadrant, remapping to original
coordinates, then class-wise WBF with the three full-image scales. Its net gain
over three-scale TTA is only `+0.00184 / +0.00334` overall and
`+0.00342 / +0.00390` small AP (val/holdout), below the predeclared
`+0.005` overall and `+0.01` small-AP thresholds. This crop-TTA route is
therefore rejected as a submission candidate.

## 9. Cross-model YOLO + RF-DETR WBF ensemble (2026-09-18/19)

The two strongest models were fused: YOLO11s native HSI16 (group-2026
training) and RF-DETR Small-v2 HSI16 (group-2026 split, 3-scale TTA
384/448/512). Both produce prediction CSVs on the same validation/holdout
images; fusion is class-wise Weighted Boxes Fusion with a YOLO weight and an
IoU matching threshold, RF-DETR weight fixed at 1.0.

Baselines (best single model, group-2026):

- YOLO val `0.684876` / holdout `0.686274`;
- RF-DETR TTA val `0.686540` / holdout `0.681814`.

### 9.1 Global weight/IoU sweep

A global sweep over YOLO weight `1.0..2.0` x IoU `0.60..0.70`
(`results/yolo_rfdetr_wbf_refine/summary.csv`, 72 configs passing a
`+0.005 min-gain` gate on both splits) found a broad plateau near
`val ~0.703-0.705 / holdout ~0.704-0.709`, i.e. a `+0.017..+0.022` gain over
the best single model on both splits. Representative points:

| Config (YOLO:w, IoU) | Val mAP50-95 | Holdout mAP50-95 |
|---|---:|---:|
| 1.5, 0.65 (submitted earlier, LB 0.64458) | 0.703514 | 0.707621 |
| 1.3, 0.64 | 0.703585 | 0.708775 |
| 1.4, 0.66 | 0.703858 | 0.708169 |
| 1.9, 0.66 | 0.704027 | 0.704715 |

### 9.2 Class-wise parameter selection

Because the plateau is broad, per-class parameters were selected from 8
pre-scored stable candidates
(`results/yolo_rfdetr_class_candidates/`: YOLO weight 1.3/1.4/1.5/1.9 x IoU
0.63/0.64/0.65/0.66 variants) using
`scripts/select_classwise_fusion.py`. Selection requires, per class, that
both validation and holdout AP be non-decreasing versus the `13b` baseline
(strict), or allowed to regress by at most a tolerance (0.003/0.005/0.01/0.02
variants).

| Variant | Val mAP50-95 | Holdout mAP50-95 |
|---|---:|---:|
| strict (tol 0) | 0.704382 | **0.709693** |
| tol 0.001 | 0.704906 | 0.709797 |
| tol 0.003 (submitted) | **0.705525** | 0.709547 |
| tol 0.005 / 0.01 / 0.02 | 0.705525 | 0.709547 |

Widening the tolerance beyond 0.003 changes nothing: the selection has
reached a plateau, so further relaxation does not overfit. The tol-0.003
per-class selection picks `19/0.66` for `rubik`, `apple_plastic`,
`badminton`, `egg` (+0.0081 val for `rubik`, +0.0056 holdout for
`apple_plastic`), and `14b/0.66` for `stone_block` (+0.0070 val) among
others; the largest single regressions allowed are within 0.003 AP on one
split (`egg`, `table_tennis`).

### 9.3 Confidence / top-K post-processing sweep

Additional confidence truncation and top-K limits were scanned on the fused
output: no truncation with `top-K=300` (the validator limit) is best; extra
trimming only removes useful low-confidence boxes. No additional
post-processing was adopted.

### 9.4 Kaggle verification

| Submission | Local val/holdout | Public LB |
|---|---:|---:|
| global 1.5:1 IoU 0.65 | 0.703514 / 0.707621 | 0.64458 |
| class-wise tol 0.003 | 0.705525 / 0.709547 | **0.64630** |

The class-wise selection improved Public LB by `+0.00172` over the global
fusion and is `+0.04013` over the single-model TTA submission `0.60617`.
Local-to-Public gap remains ~0.06, consistent with earlier observations.

The class-wise test submission `yolo_rfdetr_classwise_tol003.csv` passed the
full local validator (1000/1000 test images, 18 legal classes, 0 NaN/Inf,
0 invalid/out-of-bounds boxes, 0 duplicate row ids, <=300 detections/image).

### 9.5 Expanded 16-candidate pool (2026-09-19, not submitted)

The candidate pool was extended with 8 finer configurations
(`results/yolo_rfdetr_class_candidates2/`, generator
`scripts/gen_classwise_candidates2.sh`): `13e` (1.3, 0.60), `13f` (1.3, 0.65),
`13g` (1.3, 0.66), `14c` (1.4, 0.63), `14d` (1.4, 0.65), `12_66` (1.2, 0.66),
`19b` (1.9, 0.65), `22_066` (2.2, 0.66).

| Variant | Val mAP50-95 | Holdout mAP50-95 |
|---|---:|---:|
| previous tol 0.003 (8 candidates, submitted, LB 0.64630) | 0.705525 | 0.709547 |
| strict, 16 candidates | 0.705329 | **0.710004** |
| tol 0.003, 16 candidates | **0.706265** | 0.709939 |

The expansion lifts val by `+0.00074` and holdout by `+0.00039` over the
submitted config, mainly from `egg` (`22_066`, +0.0123/+0.0036),
`badminton` (`22_066`), and `egg_wood`/`orange_plastic` (`12_66`). This is
roughly 7x below the predeclared `+0.005` consistent val+holdout
submission threshold, so **no submission was spent**; the configs and
scores are stored under `results/yolo_rfdetr_classwise_v2/` for a future
free-submission opportunity.

## 10. Known implementation pitfalls already fixed

The following issues were found during development and should not be reintroduced:

1. RF-DETR's standard loader initially assumes 3 input channels. Custom loading logic is required for already-16-channel checkpoints.
2. Transferring a 512-resolution checkpoint to 384 requires positional-embedding interpolation; `strict=False` alone does not bypass shape mismatches.
3. RF-DETR evaluation must use the same aspect-ratio-preserving resize policy as training. An accidental `384x384` square + multi-scale evaluation produced an invalid holdout score around 0.286 and was discarded.
4. RF-DETR output includes an extra background class slot; submission generation must filter class ID 18 and keep only 0..17.
5. A nominal RF-DETR `resolution=512` with multi-scale enabled did not mean a fixed 512 short side; logs showed a training scale of 672. Small-v2 explicitly disables multi-scale to obtain the intended fixed 384 short side.

## 11. High-priority hypotheses for improvement

These are hypotheses, not established conclusions.

### A. Acquisition-level metadata remains the best validation improvement

Content-based group-aware validation has already been implemented and did not cause YOLO performance to collapse. If official acquisition time/sequence/session metadata becomes available, it would still be preferable to the current image-similarity grouping.

### B. Submission density is no longer a leading hypothesis

Confidence/top-K sweeps changed local mAP only at roughly the 0.001 level, and the formal CSV submission path was shown to reproduce RF-DETR evaluation within about 0.0005. Submission density should not be a primary optimization target unless the exact official metric reveals a different truncation rule.

### C. Small-object / high-IoU localization is the clearest remaining weakness

Scale sensitivity is experimentally established, and three-scale TTA already improved Kaggle Public LB from 0.59086 to 0.60617. The next spatial experiments should target small objects and weak classes rather than generic additional scale sweeps.

### D. More structured spectral front-ends remain open

The current model expands a standard image patch embedding to 16 channels. Stronger alternatives include:

- spectral stem / 1D spectral mixer before the 2D backbone;
- learned 16→3 or 16→C projection initialized by PCA/linear regression from pretrained RGB features;
- low-rank or gated spectral adapters (the simple identity full-rank 1x1 adapter was tested and was worse);
- band attention/gating;
- spectral-spatial separable convolutions;
- wavelength-aware positional/channel embeddings if wavelengths are available.

### E. Ensembling may help if errors are complementary

YOLO11s HSI16 has the best Public LB while RF-DETR has much stronger local localization. Their predictions may be complementary. Weighted Boxes Fusion, class-aware box matching, or score-rank fusion should be tested on validation/holdout before any Kaggle submission.

### F. Large global test-domain intensity drift is not strongly supported

Per-band diagnostics found small train/test standardized mean differences and very low saturation under clip=320, except modest concentration in one band. Float32 plus train-fold normalization did not improve the group-aware baseline. More local or scene-conditional normalization could still be explored, but global train-fold statistics are not currently promising.

## 12. Suggested experiment order

To avoid spending Kaggle submissions inefficiently, a reviewer should prioritize experiments that can be rejected locally:

1. target small-object recall/localization and the recurrent weak classes (`stone_block`, `people`, `car`, unstable `e-bike`);
2. evaluate YOLO/RF-DETR error complementarity and class-aware single-submission fusion locally;
3. test a more structured spectral stem (low-rank/gated/wavelength-aware) rather than the rejected plain 1x1 adapter;
4. if available, rebuild validation using official acquisition/session metadata;
5. consider a final single-model retraining recipe that preserves the verified 384-stage behavior and three-scale TTA;
6. only consume a Kaggle submission after >=0.005 consistent val+holdout improvement.

## 13. Reproducibility map

- Annotation audit: `scripts/audit_annotations.py`
- Audited split: `scripts/prepare_stratified_hsi_split.py`
- Hyperspectral conversion: `scripts/prepare_multispectral_yolo.py`
- RF-DETR COCO export: `scripts/prepare_rfdetr_coco.py`
- RF-DETR multispectral training: `scripts/train_rfdetr_multispectral.py`
- RF-DETR submission generation: `scripts/make_rfdetr_submission.py`
- RF-DETR crop-TTA generation: `scripts/make_rfdetr_crop_tta.py`
- Same-model WBF/NMS fusion: `scripts/fuse_detection_csv.py`
- Cross-model class-wise WBF fusion: `scripts/fuse_detection_csv_classwise.py`
- Class-wise fusion parameter selection: `scripts/select_classwise_fusion.py`
- Class-wise fusion results: `results/yolo_rfdetr_wbf_refine/`,
  `results/yolo_rfdetr_class_candidates/`, `results/yolo_rfdetr_classwise/`
- Prediction CSV scoring: `scripts/score_detection_csv.py`
- Kaggle submission helper: `scripts/submit_kaggle.py`
- YOLO training: `scripts/train_yolo.py`
- YOLO scale audit: `scripts/eval_yolo_scales.py`
- Frozen RF-DETR hashes/environment: `results/repro_384/`
- Machine-readable results: `results/experiment_summary.csv` and `results/experiment_summary.json`

Large datasets, weights, run logs, and submission CSVs remain intentionally excluded from Git.
