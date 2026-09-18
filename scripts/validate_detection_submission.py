from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


EXPECTED_HEADER = [
    "id",
    "image_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a detection CSV against a COCO test annotation file."
    )
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--max-per-image", type=int, default=300)
    args = parser.parse_args()

    gt = json.loads(args.gt.read_text())
    images = {int(item["id"]): item for item in gt["images"]}
    category_ids = {int(item["id"]) for item in gt["categories"]}

    seen_row_ids: set[int] = set()
    seen_image_ids: set[int] = set()
    per_image: Counter[int] = Counter()
    row_ids: list[int] = []

    invalid_image_ids = 0
    invalid_class_ids = 0
    nonfinite_values = 0
    invalid_boxes = 0
    out_of_bounds_boxes = 0
    duplicate_row_ids = 0
    rows = 0
    min_confidence = math.inf
    max_confidence = -math.inf

    with args.csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        header_ok = reader.fieldnames == EXPECTED_HEADER
        for row in reader:
            rows += 1
            row_id = int(row["id"])
            image_id = int(row["image_id"])
            class_id = int(row["class_id"])
            confidence = float(row["confidence"])
            x1 = float(row["x1"])
            y1 = float(row["y1"])
            x2 = float(row["x2"])
            y2 = float(row["y2"])

            row_ids.append(row_id)
            if row_id in seen_row_ids:
                duplicate_row_ids += 1
            seen_row_ids.add(row_id)

            seen_image_ids.add(image_id)
            per_image[image_id] += 1
            if image_id not in images:
                invalid_image_ids += 1
            if class_id not in category_ids:
                invalid_class_ids += 1

            values = [confidence, x1, y1, x2, y2]
            if not all(math.isfinite(value) for value in values):
                nonfinite_values += 1
                continue

            min_confidence = min(min_confidence, confidence)
            max_confidence = max(max_confidence, confidence)
            if x2 <= x1 or y2 <= y1:
                invalid_boxes += 1
                continue

            image = images.get(image_id)
            if image is not None:
                width = float(image["width"])
                height = float(image["height"])
                eps = 1e-5
                if (
                    x1 < -eps
                    or y1 < -eps
                    or x2 > width + eps
                    or y2 > height + eps
                ):
                    out_of_bounds_boxes += 1

    gt_image_ids = set(images)
    missing_image_ids = sorted(gt_image_ids - seen_image_ids)
    extra_image_ids = sorted(seen_image_ids - gt_image_ids)
    max_predictions_per_image = max(per_image.values(), default=0)
    row_ids_contiguous = row_ids == list(range(rows))

    result = {
        "valid": all(
            [
                header_ok,
                not missing_image_ids,
                not extra_image_ids,
                invalid_image_ids == 0,
                invalid_class_ids == 0,
                nonfinite_values == 0,
                invalid_boxes == 0,
                out_of_bounds_boxes == 0,
                duplicate_row_ids == 0,
                row_ids_contiguous,
                max_predictions_per_image <= args.max_per_image,
            ]
        ),
        "header_ok": header_ok,
        "rows": rows,
        "gt_images": len(gt_image_ids),
        "prediction_images": len(seen_image_ids),
        "missing_images": len(missing_image_ids),
        "missing_image_sample": missing_image_ids[:20],
        "extra_images": len(extra_image_ids),
        "extra_image_sample": extra_image_ids[:20],
        "invalid_image_ids": invalid_image_ids,
        "invalid_class_ids": invalid_class_ids,
        "nonfinite_values": nonfinite_values,
        "invalid_boxes": invalid_boxes,
        "out_of_bounds_boxes": out_of_bounds_boxes,
        "duplicate_row_ids": duplicate_row_ids,
        "row_ids_contiguous": row_ids_contiguous,
        "max_predictions_per_image": max_predictions_per_image,
        "min_predictions_per_image": min(per_image.values(), default=0),
        "confidence_min": None if rows == 0 else min_confidence,
        "confidence_max": None if rows == 0 else max_confidence,
    }
    print(json.dumps(result, indent=2))

    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
