#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINO_ROOT="${DINO_ROOT:-/home/yangxiao/.cache/hodc26/DINO}"
PYTHON="${PYTHON:-/home/yangxiao/hotc-venv/bin/python}"
COCO_ROOT="${COCO_ROOT:-$ROOT/prepared/dino_coco_pseudo5813_group2026}"
PRETRAIN="${PRETRAIN:-$ROOT/weights/dino/checkpoint0011_4scale.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/dino_r50_4scale_pseudo5813_group2026_12ep}"
GPU="${GPU:-1}"
BATCH_SIZE="${BATCH_SIZE:-2}"
# Official DINO uses an effective batch of 16. Batch 2 is already verified
# on a 24 GB RTX 4090, so accumulate eight micro-batches by default.
ACCUM_STEPS="${ACCUM_STEPS:-8}"
EPOCHS="${EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-8}"

if [[ ! -d "$DINO_ROOT" ]]; then
  echo "DINO source not found: $DINO_ROOT" >&2
  exit 1
fi
if [[ ! -f "$PRETRAIN" ]]; then
  echo "DINO pretrained checkpoint not found: $PRETRAIN" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$DINO_ROOT"
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" main.py \
  -c config/DINO/DINO_4scale.py \
  --coco_path "$COCO_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --pretrain_model_path "$PRETRAIN" \
  --finetune_ignore label_enc.weight class_embed tgt_embed \
  --num_workers "$NUM_WORKERS" \
  --amp \
  --options \
    num_classes=18 \
    dn_labelbook_size=19 \
    num_queries=300 \
    num_select=300 \
    batch_size="$BATCH_SIZE" \
    accum_iter="$ACCUM_STEPS" \
    epochs="$EPOCHS" \
    lr=0.0001 \
    lr_backbone=0.00001 \
    lr_drop=11 \
    data_aug_scales=480,544,608,672 \
    data_aug_max_size=1344
