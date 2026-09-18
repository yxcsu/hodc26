from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def iou_xyxy(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def load_gt(path: Path) -> tuple[dict[int, str], dict[int, list[dict[str, Any]]], dict[int, dict[str, Any]]]:
    data = json.loads(path.read_text())
    categories = {int(x["id"]): str(x["name"]) for x in data["categories"]}
    images = {int(x["id"]): x for x in data["images"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        if int(ann.get("iscrowd", 0)):
            continue
        x, y, w, h = map(float, ann["bbox"])
        by_image[int(ann["image_id"])].append(
            {
                "ann_id": int(ann["id"]),
                "class_id": int(ann["category_id"]),
                "box": [x, y, x + w, y + h],
                "area": float(ann.get("area", w * h)),
            }
        )
    return categories, by_image, images


def load_predictions(path: Path, min_conf: float) -> dict[int, list[dict[str, Any]]]:
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            score = float(row["confidence"])
            if score < min_conf:
                continue
            by_image[int(row["image_id"])].append(
                {
                    "class_id": int(row["class_id"]),
                    "score": score,
                    "box": [float(row[k]) for k in ("x1", "y1", "x2", "y2")],
                }
            )
    for preds in by_image.values():
        preds.sort(key=lambda x: x["score"], reverse=True)
    return by_image


def classify_gt_error(
    gt: dict[str, Any],
    preds: list[dict[str, Any]],
    match_iou: float,
    loc_iou: float,
) -> tuple[str, dict[str, Any]]:
    same = [p for p in preds if p["class_id"] == gt["class_id"]]
    best_same = max(((iou_xyxy(gt["box"], p["box"]), p) for p in same), default=(0.0, None), key=lambda x: x[0])
    best_any = max(((iou_xyxy(gt["box"], p["box"]), p) for p in preds), default=(0.0, None), key=lambda x: x[0])

    detail = {
        "best_same_iou": best_same[0],
        "best_same_score": None if best_same[1] is None else best_same[1]["score"],
        "best_any_iou": best_any[0],
        "best_any_class": None if best_any[1] is None else best_any[1]["class_id"],
        "best_any_score": None if best_any[1] is None else best_any[1]["score"],
    }
    if best_same[0] >= match_iou:
        return "detected", detail
    if best_any[0] >= match_iou and best_any[1] is not None and best_any[1]["class_id"] != gt["class_id"]:
        return "classification", detail
    if best_same[0] >= loc_iou:
        return "localization", detail
    return "miss", detail


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gt", type=Path, required=True)
    p.add_argument("--pred", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--min-conf", type=float, default=0.01)
    p.add_argument("--match-iou", type=float, default=0.5)
    p.add_argument("--loc-iou", type=float, default=0.1)
    p.add_argument("--top-images", type=int, default=100)
    args = p.parse_args()

    categories, gt_by_image, images = load_gt(args.gt)
    name_to_id = {v: k for k, v in categories.items()}
    missing = [x for x in args.classes if x not in name_to_id]
    if missing:
        raise ValueError(f"Unknown classes: {missing}; available={sorted(name_to_id)}")
    target_ids = {name_to_id[x] for x in args.classes}
    preds_by_image = load_predictions(args.pred, args.min_conf)

    summary: dict[str, dict[str, Any]] = {
        name: {
            "class_id": name_to_id[name],
            "gt": 0,
            "detected": 0,
            "localization": 0,
            "classification": 0,
            "miss": 0,
        }
        for name in args.classes
    }
    error_rows: list[dict[str, Any]] = []
    image_scores: dict[int, float] = defaultdict(float)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    severity = {"detected": 0.0, "localization": 1.0, "classification": 1.5, "miss": 2.0}
    for image_id, gts in gt_by_image.items():
        preds = preds_by_image.get(image_id, [])
        for gt in gts:
            if gt["class_id"] not in target_ids:
                continue
            class_name = categories[gt["class_id"]]
            kind, detail = classify_gt_error(gt, preds, args.match_iou, args.loc_iou)
            summary[class_name]["gt"] += 1
            summary[class_name][kind] += 1
            image_scores[image_id] += severity[kind]

            if kind != "detected":
                wrong_class = detail["best_any_class"]
                if kind == "classification" and wrong_class is not None:
                    confusion[class_name][categories.get(int(wrong_class), str(wrong_class))] += 1
                error_rows.append(
                    {
                        "image_id": image_id,
                        "file_name": images.get(image_id, {}).get("file_name"),
                        "ann_id": gt["ann_id"],
                        "class": class_name,
                        "class_id": gt["class_id"],
                        "error": kind,
                        "area": gt["area"],
                        "bbox_xyxy": gt["box"],
                        **detail,
                    }
                )

    for stats in summary.values():
        gt = max(1, stats["gt"])
        for key in ("detected", "localization", "classification", "miss"):
            stats[f"{key}_rate"] = stats[key] / gt

    ranked_images = []
    for image_id, score in sorted(image_scores.items(), key=lambda x: (-x[1], x[0]))[: args.top_images]:
        per_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in error_rows:
            if row["image_id"] == image_id:
                per_class[row["class"]][row["error"]] += 1
        ranked_images.append(
            {
                "image_id": image_id,
                "file_name": images.get(image_id, {}).get("file_name"),
                "hardness": score,
                "errors": {k: dict(v) for k, v in per_class.items()},
            }
        )

    payload = {
        "gt": str(args.gt),
        "pred": str(args.pred),
        "min_conf": args.min_conf,
        "match_iou": args.match_iou,
        "loc_iou": args.loc_iou,
        "classes": args.classes,
        "summary": summary,
        "classification_confusion": {k: dict(v) for k, v in confusion.items()},
        "hard_images": ranked_images,
        "errors": error_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))

    print(json.dumps({"summary": summary, "classification_confusion": payload["classification_confusion"]}, indent=2))
    print(f"hard_images={len(ranked_images)} errors={len(error_rows)} out={args.out}")


if __name__ == "__main__":
    main()
