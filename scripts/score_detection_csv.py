from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_predictions(path: Path, conf: float, top_k: int | None) -> list[dict]:
    by_image: dict[int, list[dict]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            score = float(row["confidence"])
            if score <= conf:
                continue
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])
            if x2 <= x1 or y2 <= y1:
                continue
            image_id = int(row["image_id"])
            by_image[image_id].append(
                {
                    "image_id": image_id,
                    "category_id": int(row["class_id"]),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )

    predictions: list[dict] = []
    for image_id, rows in by_image.items():
        rows.sort(key=lambda item: item["score"], reverse=True)
        if top_k is not None:
            rows = rows[:top_k]
        predictions.extend(rows)
    return predictions


def evaluate(gt_path: Path, predictions: list[dict], max_dets: int) -> dict[str, object]:
    coco_gt = COCO(str(gt_path))
    if predictions:
        coco_dt = coco_gt.loadRes(predictions)
    else:
        # COCO.loadRes cannot construct a result object from an empty list in
        # some pycocotools releases. Keep a valid empty result set instead.
        empty = {
            "images": coco_gt.dataset["images"],
            "categories": coco_gt.dataset["categories"],
            "annotations": [],
        }
        coco_dt = COCO()
        coco_dt.dataset = empty
        coco_dt.createIndex()

    evaluator = COCOeval(coco_gt, coco_dt, "bbox")
    evaluator.params.maxDets = [1, 10, max_dets]
    evaluator.evaluate()
    evaluator.accumulate()
    # COCOeval.summarize() hard-codes maxDets=100 for its first AP line in
    # several pycocotools releases. Once maxDets is changed (e.g. 300 RF-DETR
    # queries), evaluator.stats[0] can therefore be -1 even though evaluation is
    # valid. Read the accumulated tensors directly instead.
    precision = evaluator.eval["precision"]  # [T, R, K, A, M]
    recall = evaluator.eval["recall"]  # [T, K, A, M]
    area_idx = evaluator.params.areaRngLbl.index("all")
    max_det_idx = evaluator.params.maxDets.index(max_dets)

    def mean_valid(values: np.ndarray) -> float:
        valid = values[values > -1]
        return float(np.mean(valid)) if valid.size else 0.0

    all_precision = precision[:, :, :, area_idx, max_det_idx]
    all_recall = recall[:, :, area_idx, max_det_idx]
    iou_thrs = evaluator.params.iouThrs
    iou50_idx = int(np.argmin(np.abs(iou_thrs - 0.50)))
    iou75_idx = int(np.argmin(np.abs(iou_thrs - 0.75)))
    per_class: dict[str, float] = {}
    for k, category_id in enumerate(evaluator.params.catIds):
        category = coco_gt.cats.get(category_id, {})
        name = str(category.get("name", category_id))
        per_class[name] = mean_valid(all_precision[:, :, k])

    area_ap: dict[str, float] = {}
    area_ar: dict[str, float] = {}
    for label in evaluator.params.areaRngLbl:
        idx = evaluator.params.areaRngLbl.index(label)
        area_ap[label] = mean_valid(precision[:, :, :, idx, max_det_idx])
        area_ar[label] = mean_valid(recall[:, :, idx, max_det_idx])

    return {
        "mAP_50_95": mean_valid(all_precision),
        "mAP_50": mean_valid(all_precision[iou50_idx]),
        "mAP_75": mean_valid(all_precision[iou75_idx]),
        "mAR": mean_valid(all_recall),
        "detections": float(len(predictions)),
        "per_class_AP_50_95": per_class,
        "area_AP_50_95": area_ap,
        "area_AR": area_ar,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    predictions = load_predictions(args.pred, args.conf, args.top_k)
    metrics = evaluate(args.gt, predictions, args.max_dets)
    print(json.dumps(metrics, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
