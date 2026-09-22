# DINO-R50 4-scale experiment (2026-09-20)

## Setup

- Official IDEA-Research DINO, R50 4-scale.
- COCO 12-epoch checkpoint initialization.
- Pseudo-RGB hyperspectral bands: (5, 8, 13).
- Group-2026 split: 2400 train / 300 validation / 300 holdout.
- 18 classes, 300 queries.
- Multi-scale short side: 480 / 544 / 608 / 672, max size 1344.
- Physical batch 2 with gradient accumulation 8 (effective batch 16).
- Head LR 1e-4, backbone LR 1e-5.

## DINO baseline

Validation mAP@0.5:0.95 by epoch:

| Epoch | Val mAP |
|---:|---:|
| 0 | 0.502817 |
| 1 | 0.587612 |
| 2 | 0.609080 |
| 3 | 0.611780 |
| 4 | 0.635530 |
| 5 | 0.630326 |
| 6 | 0.629611 |
| 7 | 0.642185 |
| 8 | 0.633526 |
| 9 | 0.637843 |
| 10 | 0.650260 |
| 11 | **0.653625** |

Using the repository scorer with maxDets=300, epoch 11 gives:

- Validation: **0.653968**
- Holdout: **0.654564**

No confidence threshold improved validation mAP; conf=0 and top-k=300
were best. An epoch-10/epoch-11 checkpoint WBF improved validation but not
holdout, so it was rejected for the final test prediction.

## Cross-model fusion

Adding DINO to the existing YOLO + RF-DETR predictions improved the locked
three-model WBF to:

- Weights YOLO:RF-DETR:DINO = 1.6:1.0:0.4
- WBF IoU = 0.66
- Validation: **0.707403**
- Holdout: **0.708141**

DINO was most useful for apple, badminton, banana, car, charger_head,
egg_wood, orange, and people, while several other classes remained stronger
with the existing class-wise YOLO + RF-DETR fusion.

The best validation-selected class gate used DINO on ten classes:
apple, apple_plastic, badminton, banana, car, charger_head, e-bike,
egg_wood, orange, and people.

- Validation: **0.709772**
- Holdout: **0.708759**
- Kaggle Public LB: **0.64913**

A more conservative eight-class gate produced:

- Validation: **0.709693**
- Holdout: **0.709400**
- Kaggle Public LB: **0.64912**

For comparison, classwise-v2 YOLO + RF-DETR without DINO produced:

- Validation: **0.706265**
- Holdout: **0.709939**
- Kaggle Public LB: **0.64685**

The previous best Public LB was 0.64630, so the DINO gate improved the
current best Public LB by **+0.00283**, to **0.64913**.

## 19-channel HSI residual DINO (2026-09-22)

The pseudo-RGB DINO was extended to a behavior-preserving 19-channel stem:

- channels 0--15: globally scaled HSI bands;
- channels 16--18: pseudo-RGB bands (5, 8, 13);
- the original pretrained RGB `conv1` is retained unchanged;
- a zero-initialized 16-to-64 spectral residual convolution is added to the
  RGB stem output.

Only the spectral convolution was trained during the first HSI adaptation
stage. Full-strength residual training was too aggressive. Scaling the
learned spectral residual after training gave:

| Residual scale | Validation mAP |
|---:|---:|
| 0.125 | 0.655585 |
| 0.20 | 0.655274 |
| 0.225 | 0.65513 |
| **0.23** | **0.657806** |
| 0.235 | 0.65574 |
| 0.24 | 0.65541 |
| 0.25 | 0.657286 |
| 0.30 | 0.656884 |
| 0.375 | 0.653916 |
| 0.50 | 0.651405 |
| 1.00 | about 0.648 |

The locked `alpha=0.23` checkpoint obtained:

- group-2026 validation: **0.657806**;
- group-2026 holdout: **0.657169**;
- Kaggle Public LB: **0.58096**.

The matched pseudo-RGB DINO obtained Public **0.57900**, so HSI improved the
same DINO family by **+0.00196 Public**, even though DINO itself remained
weaker on the hidden test distribution than the YOLO/RF-DETR branches.

Larger inference scales did not help the HSI model: 672 / 800 / 960 gave
approximately 0.65730 / 0.65044 / 0.64026 validation mAP. Same-model WBF also
failed to beat the 672 single-scale result.

Rejected HSI adaptation variants included:

- full-model low-LR phase-B: 0.655446 validation;
- lower-LR spectral-only training: about 0.6522 validation;
- learned 64-channel residual gate after folding: 0.656835 validation;
- RGB-teacher / HSI-student self-distillation from the alpha=0.23 checkpoint:
  0.651011 after one epoch;
- interpolating only 5% / 10% of that distillation update back into
  alpha=0.23: 0.656913 / 0.656370.

These failures indicate that the HSI contribution is useful but small, and
aggressive adaptation easily destroys the pretrained RGB detector behavior.

## Independent audit split

Group seed 9173 was prepared as an independent audit split:

- 2400 train / 300 validation / 300 holdout.
- No group overlap between train, validation, and holdout.
- Split balance score: 0.04245.
- No model score was inspected on seed 9173 while the alpha, scale, or HSI
  architecture was selected.

After `alpha=0.23` and scale 672 were locked, seed 9173 was opened as an
out-of-sample audit:

| Model | Seed-9173 val | Seed-9173 holdout |
|---|---:|---:|
| pseudo-RGB DINO | 0.682228 | 0.701828 |
| HSI DINO alpha=0.23 | **0.682747** | **0.702743** |
| HSI gain | **+0.000519** | **+0.000915** |

Thus the global HSI improvement replicated on both independent partitions.
Per-class gains were not uniformly stable, however; for example badminton
decreased on all four audited local partitions, which helps explain why
replacing individual DINO classes inside the ensemble did not reliably
improve Public LB.

## Kaggle submissions

| Submission | Public LB |
|---|---:|
| YOLO + RF-DETR classwise v1 | 0.64630 |
| YOLO + RF-DETR classwise v2 | 0.64685 |
| YOLO + RF-DETR + DINO gate (8 classes) | 0.64912 |
| **YOLO + RF-DETR + DINO gate (10 classes)** | **0.64913** |
| HSI-DINO alpha=0.23 single model | 0.58096 |
| pseudo-RGB DINO single model | 0.57900 |
| classwise-v2 + broad HSI-DINO gate | 0.64720 |
| best gate + HSI swap (apple, badminton) | 0.64907 |
| best gate + HSI swap (apple, egg_wood) | 0.64905 |

The competition's final no-ensemble/code-review requirements must be checked
before treating cross-model WBF as a final eligible solution. The DINO
single-model baseline and the prepared seed-9173 audit split remain useful
for developing a compliant single-model teacher/student route.
