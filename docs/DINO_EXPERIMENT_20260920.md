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

## Independent audit split

Group seed 9173 was prepared as a future independent audit split:

- 2400 train / 300 validation / 300 holdout.
- No group overlap between train, validation, and holdout.
- Split balance score: 0.04245.
- No model score was inspected on seed 9173 during this experiment.

It should remain untouched until the next model design is locked.

## Kaggle submissions

| Submission | Public LB |
|---|---:|
| YOLO + RF-DETR classwise v1 | 0.64630 |
| YOLO + RF-DETR classwise v2 | 0.64685 |
| YOLO + RF-DETR + DINO gate (8 classes) | 0.64912 |
| **YOLO + RF-DETR + DINO gate (10 classes)** | **0.64913** |

The competition's final no-ensemble/code-review requirements must be checked
before treating cross-model WBF as a final eligible solution. The DINO
single-model baseline and the prepared seed-9173 audit split remain useful
for developing a compliant single-model teacher/student route.
