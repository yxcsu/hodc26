#!/bin/bash
# Fine-grained candidate generation v4 — SAME input scheme as the original 8 candidates:
#   yolo   = results/yolo_group2026_csv/{split}_768_mapped.csv
#   rfdetr = results/group2026_final_crop_tta/{split}_full3_wbf07.csv
set -e
cd /home/yangxiao/hyperspectral-object-detection-challenge-2026
PY=/home/yangxiao/.conda/envs/q2l/bin/python
OUT=${1:-results/yolo_rfdetr_class_candidates2}
GT_VAL=prepared/rfdetr_hsi16_group2026/valid/_annotations.coco.json
GT_HO=prepared/rfdetr_hsi16_group2026_holdout/valid/_annotations.coco.json
YOLO_VAL=results/yolo_group2026_csv/val_768_mapped.csv
YOLO_HO=results/yolo_group2026_csv/holdout_768_mapped.csv
RF_VAL=results/group2026_final_crop_tta/holdout_full3_wbf07.csv
RF_VAL=results/group2026_final_crop_tta/val_full3_wbf07.csv
RF_HO=results/group2026_final_crop_tta/holdout_full3_wbf07.csv

ls results/group2026_final_crop_tta/

mkdir -p "$OUT"

mk_cfg() {
  local w="$1" iou="$2"
  local tmp
  tmp=$(mktemp /home/yangxiao/.claude/jobs/963331b7/tmp/cfg_XXXXXX.json)
  python3 - "$w" "$iou" "$tmp" <<'PYEOF'
import json, sys
w, iou, path = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
cfg = {"classes": {str(i): {"yolo_weight": w, "rfdetr_weight": 1.0, "iou": iou} for i in range(18)}}
open(path, "w").write(json.dumps(cfg))
PYEOF
  echo "$tmp"
}

TAGS="13e:1.3:0.60 13f:1.3:0.65 13g:1.3:0.66 14c:1.4:0.63 14d:1.4:0.65 12_66:1.2:0.66 19b:1.9:0.65 22_066:2.2:0.66"

for ent in $TAGS; do
  tag="${ent%%:*}"
  rest="${ent#*:}"
  w="${rest%%:*}"
  iou="${rest#*:}"
  if [ -f "$OUT/val_${tag}_score.json" ]; then
    echo "skip $tag (done)"
    continue
  fi
  echo "== $tag (w=$w iou=$iou) val =="
  cfg=$(mk_cfg "$w" "$iou")
  $PY scripts/fuse_detection_csv_classwise.py \
    --inputs "$YOLO_VAL" "$RF_VAL" \
    --class-config "$cfg" \
    --out "$OUT/val_${tag}.csv" \
    --coco-gt "$GT_VAL" > /dev/null
  $PY scripts/score_detection_csv.py --gt "$GT_VAL" --pred "$OUT/val_${tag}.csv" --json-out "$OUT/val_${tag}_score.json" > /dev/null
  rm -f "$cfg"
  echo "== $tag holdout =="
  cfg=$(mk_cfg "$w" "$iou")
  $PY scripts/fuse_detection_csv_classwise.py \
    --inputs "$YOLO_HO" "$RF_HO" \
    --class-config "$cfg" \
    --out "$OUT/holdout_${tag}.csv" \
    --coco-gt "$GT_HO" > /dev/null
  $PY scripts/score_detection_csv.py --gt "$GT_HO" --pred "$OUT/holdout_${tag}.csv" --json-out "$OUT/holdout_${tag}_score.json" > /dev/null
  rm -f "$cfg"
  python3 - "$OUT" "$tag" <<'PYEOF'
import json, sys
out, tag = sys.argv[1], sys.argv[2]
v = json.load(open(f'{out}/val_{tag}_score.json')); h = json.load(open(f'{out}/holdout_{tag}_score.json'))
print(f'{tag} val={v["mAP_50_95"]:.6f} holdout={h["mAP_50_95"]:.6f}')
PYEOF
done
echo ALL_DONE
