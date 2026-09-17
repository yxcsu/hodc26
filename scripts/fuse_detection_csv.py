from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


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
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_rows(
    paths: list[Path], conf: float, source_weights: list[float]
) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for source_idx, path in enumerate(paths):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                score = float(row["confidence"])
                if score <= conf:
                    continue
                item = {
                    "source": source_idx,
                    "source_weight": float(source_weights[source_idx]),
                    "score": score,
                    "box": [float(row["x1"]), float(row["y1"]), float(row["x2"]), float(row["y2"])],
                }
                grouped[(int(row["image_id"]), int(row["class_id"]))].append(item)
    return grouped


def nms(items: list[dict], iou_threshold: float) -> list[dict]:
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    kept: list[dict] = []
    for item in items:
        if any(iou_xyxy(item["box"], prev["box"]) >= iou_threshold for prev in kept):
            continue
        kept.append(item)
    return kept


def wbf(items: list[dict], iou_threshold: float, source_weights: list[float]) -> list[dict]:
    # Same-model WBF. A cluster's score is divided by the number of views, so a
    # box supported by multiple scales keeps its confidence while a single-view
    # outlier is down-weighted.
    items = sorted(items, key=lambda x: x["score"], reverse=True)
    total_source_weight = max(sum(source_weights), 1e-12)
    clusters: list[dict] = []
    for item in items:
        best_idx = None
        best_iou = 0.0
        for idx, cluster in enumerate(clusters):
            overlap = iou_xyxy(item["box"], cluster["box"])
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is None:
            clusters.append(
                {
                    "members": [item],
                    "box": item["box"][:],
                    "score": item["score"] * item["source_weight"] / total_source_weight,
                }
            )
            continue

        cluster = clusters[best_idx]
        cluster["members"].append(item)
        members = cluster["members"]
        weight_sum = sum(m["score"] * m["source_weight"] for m in members)
        cluster["box"] = [
            sum(m["box"][d] * m["score"] * m["source_weight"] for m in members)
            / max(weight_sum, 1e-12)
            for d in range(4)
        ]
        # At most one strong prediction per view should contribute fully. If a
        # view contributes duplicate queries to the same cluster, cap its score
        # contribution at that view's maximum confidence.
        per_source: dict[int, float] = {}
        for m in members:
            per_source[m["source"]] = max(per_source.get(m["source"], 0.0), m["score"])
        cluster["score"] = sum(
            score * source_weights[source] for source, score in per_source.items()
        ) / total_source_weight
    return clusters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", choices=("wbf", "nms"), default="wbf")
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional non-negative per-input weights for WBF. Defaults to equal weights.",
    )
    args = parser.parse_args()

    source_weights = args.weights or [1.0] * len(args.inputs)
    if len(source_weights) != len(args.inputs):
        raise ValueError("--weights must contain exactly one value per --inputs file")
    if any(weight < 0 for weight in source_weights) or sum(source_weights) <= 0:
        raise ValueError("--weights must be non-negative and sum to a positive value")

    grouped = read_rows(args.inputs, args.conf, source_weights)
    by_image: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    for (image_id, class_id), items in grouped.items():
        if args.method == "wbf":
            fused = wbf(items, args.iou, source_weights)
        else:
            fused = nms(items, args.iou)
        for item in fused:
            by_image[image_id].append((class_id, float(item["score"]), list(item["box"])))

    rows: list[list[object]] = []
    row_id = 0
    for image_id in sorted(by_image):
        preds = sorted(by_image[image_id], key=lambda x: x[1], reverse=True)[: args.top_k]
        for class_id, score, box in preds:
            rows.append([row_id, image_id, class_id, score, *box])
            row_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)
    print(
        f"inputs={len(args.inputs)} method={args.method} iou={args.iou} "
        f"weights={source_weights} rows={len(rows)} out={args.out}"
    )


if __name__ == "__main__":
    main()
