#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DINO_ROOT="${DINO_ROOT:-/home/yangxiao/.cache/hodc26/DINO}"
PYTHON="${PYTHON:-/home/yangxiao/hotc-venv/bin/python}"
COCO_ROOT="${COCO_ROOT:-$ROOT/prepared/dino_coco_pseudo5813_group2026}"
PRETRAIN="${PRETRAIN:-$ROOT/weights/dino/checkpoint0011_5scale.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/runs/dino_r50_5scale_pseudo5813_group2026_12ep}"
GPU="${GPU:-0}"
BATCH_SIZE="${BATCH_SIZE:-1}"
ACCUM_STEPS="${ACCUM_STEPS:-16}"
EPOCHS="${EPOCHS:-12}"
NUM_WORKERS="${NUM_WORKERS:-6}"
LR="${LR:-0.0001}"
BACKBONE_LR="${BACKBONE_LR:-0.00001}"
LR_DROP="${LR_DROP:-11}"

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
  -c config/DINO/DINO_5scale.py \
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
    lr="$LR" \
    lr_backbone="$BACKBONE_LR" \
    lr_drop="$LR_DROP" \
    data_aug_scales=480,544,608,672 \
    data_aug_max_size=1344
