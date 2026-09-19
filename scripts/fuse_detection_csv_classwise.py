from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from fuse_detection_csv import wbf


def load_rows(
    paths: list[Path], conf: float
) -> dict[tuple[int, int], list[dict[str, object]]]:
    grouped: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for source_idx, path in enumerate(paths):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
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
                class_id = int(row["class_id"])
                grouped[(image_id, class_id)].append(
                    {
                        "source": source_idx,
                        "score": score,
                        "box": [x1, y1, x2, y2],
                    }
                )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fuse YOLO/RF-DETR CSVs using per-class WBF parameters."
    )
    parser.add_argument("--inputs", type=Path, nargs=2, required=True)
    parser.add_argument("--class-config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--coco-gt",
        type=Path,
        default=None,
        help="Optional COCO JSON used to clip output boxes to image bounds.",
    )
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=300)
    args = parser.parse_args()

    config = json.loads(args.class_config.read_text())
    class_cfg = config["classes"]
    image_sizes: dict[int, tuple[float, float]] = {}
    if args.coco_gt is not None:
        coco = json.loads(args.coco_gt.read_text())
        image_sizes = {
            int(image["id"]): (float(image["width"]), float(image["height"]))
            for image in coco["images"]
        }
    grouped = load_rows(args.inputs, args.conf)

    by_image: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    for (image_id, class_id), base_items in grouped.items():
        if str(class_id) not in class_cfg:
            raise KeyError(f"Missing class config for class_id={class_id}")
        spec = class_cfg[str(class_id)]
        weights = [
            float(spec["yolo_weight"]),
            float(spec.get("rfdetr_weight", 1.0)),
        ]
        items = [
            {
                **item,
                "source_weight": weights[int(item["source"])],
            }
            for item in base_items
        ]
        fused = wbf(items, float(spec["iou"]), weights)
        for item in fused:
            box = list(item["box"])
            if image_sizes:
                width, height = image_sizes[image_id]
                box = [
                    max(0.0, min(float(box[0]), width)),
                    max(0.0, min(float(box[1]), height)),
                    max(0.0, min(float(box[2]), width)),
                    max(0.0, min(float(box[3]), height)),
                ]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
            by_image[image_id].append(
                (class_id, float(item["score"]), box)
            )

    rows: list[list[object]] = []
    row_id = 0
    for image_id in sorted(by_image):
        predictions = sorted(
            by_image[image_id], key=lambda item: item[1], reverse=True
        )[: args.top_k]
        for class_id, score, box in predictions:
            rows.append([row_id, image_id, class_id, score, *box])
            row_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer.writerows(rows)
    print(f"rows={len(rows)} out={args.out}")


if __name__ == "__main__":
    main()
