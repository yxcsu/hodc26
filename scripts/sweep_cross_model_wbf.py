from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
from collections import defaultdict
from pathlib import Path

from fuse_detection_csv import wbf
from score_detection_csv import evaluate, load_predictions


def load_grouped(paths: list[Path], conf: float) -> dict[tuple[int, int], list[dict]]:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
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
                grouped[(int(row["image_id"]), int(row["class_id"]))].append(
                    {
                        "source": source_idx,
                        "score": score,
                        "box": [x1, y1, x2, y2],
                    }
                )
    return grouped


def fuse_grouped(
    grouped: dict[tuple[int, int], list[dict]],
    weights: list[float],
    iou: float,
    top_k: int,
) -> tuple[list[dict], list[list[object]]]:
    by_image: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    for (image_id, class_id), base_items in grouped.items():
        items = [
            {
                **item,
                "source_weight": float(weights[int(item["source"])]),
            }
            for item in base_items
        ]
        fused = wbf(items, iou, weights)
        for item in fused:
            by_image[image_id].append((class_id, float(item["score"]), list(item["box"])))

    predictions: list[dict] = []
    rows: list[list[object]] = []
    row_id = 0
    for image_id in sorted(by_image):
        items = sorted(by_image[image_id], key=lambda item: item[1], reverse=True)[:top_k]
        for class_id, score, box in items:
            x1, y1, x2, y2 = box
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": score,
                }
            )
            rows.append([row_id, image_id, class_id, score, x1, y1, x2, y2])
            row_id += 1
    return predictions, rows


def quiet_evaluate(gt: Path, predictions: list[dict], max_dets: int) -> dict[str, object]:
    with contextlib.redirect_stdout(io.StringIO()):
        return evaluate(gt, predictions, max_dets)


def score_csv(gt: Path, pred: Path, top_k: int, max_dets: int) -> float:
    predictions = load_predictions(pred, conf=0.0, top_k=top_k)
    return float(quiet_evaluate(gt, predictions, max_dets)["mAP_50_95"])


def write_rows(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)


def parse_weight_pair(value: str) -> tuple[float, float]:
    left, right = value.split(":", maxsplit=1)
    pair = (float(left), float(right))
    if min(pair) < 0 or sum(pair) <= 0:
        raise argparse.ArgumentTypeError("weights must be non-negative and sum to > 0")
    return pair


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep YOLO:RF-DETR cross-model WBF on val and holdout."
    )
    parser.add_argument("--yolo-val", type=Path, required=True)
    parser.add_argument("--yolo-holdout", type=Path, required=True)
    parser.add_argument("--rfdetr-val", type=Path, required=True)
    parser.add_argument("--rfdetr-holdout", type=Path, required=True)
    parser.add_argument("--gt-val", type=Path, required=True)
    parser.add_argument("--gt-holdout", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=parse_weight_pair,
        nargs="+",
        default=[
            (1.0, 1.0),
            (2.0, 1.0),
            (1.0, 2.0),
            (3.0, 2.0),
            (2.0, 3.0),
        ],
        metavar="YOLO:RFDETR",
    )
    parser.add_argument(
        "--ious",
        type=float,
        nargs="+",
        default=[0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    )
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--max-dets", type=int, default=300)
    parser.add_argument(
        "--required-gain",
        type=float,
        default=0.005,
        help="Required gain over the better single model on both val and holdout.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val_grouped = load_grouped([args.yolo_val, args.rfdetr_val], args.conf)
    holdout_grouped = load_grouped([args.yolo_holdout, args.rfdetr_holdout], args.conf)

    baselines = {
        "yolo_val": score_csv(args.gt_val, args.yolo_val, args.top_k, args.max_dets),
        "yolo_holdout": score_csv(
            args.gt_holdout, args.yolo_holdout, args.top_k, args.max_dets
        ),
        "rfdetr_val": score_csv(args.gt_val, args.rfdetr_val, args.top_k, args.max_dets),
        "rfdetr_holdout": score_csv(
            args.gt_holdout, args.rfdetr_holdout, args.top_k, args.max_dets
        ),
    }
    best_single_val = max(baselines["yolo_val"], baselines["rfdetr_val"])
    best_single_holdout = max(baselines["yolo_holdout"], baselines["rfdetr_holdout"])

    records: list[dict[str, object]] = []
    best_payload: tuple[dict[str, object], list[list[object]], list[list[object]]] | None = None
    for yolo_weight, rfdetr_weight in args.weights:
        weights = [yolo_weight, rfdetr_weight]
        for iou in args.ious:
            val_predictions, val_rows = fuse_grouped(
                val_grouped, weights, iou, args.top_k
            )
            holdout_predictions, holdout_rows = fuse_grouped(
                holdout_grouped, weights, iou, args.top_k
            )
            val_metrics = quiet_evaluate(args.gt_val, val_predictions, args.max_dets)
            holdout_metrics = quiet_evaluate(
                args.gt_holdout, holdout_predictions, args.max_dets
            )
            val_map = float(val_metrics["mAP_50_95"])
            holdout_map = float(holdout_metrics["mAP_50_95"])
            val_gain = val_map - best_single_val
            holdout_gain = holdout_map - best_single_holdout
            record = {
                "yolo_weight": yolo_weight,
                "rfdetr_weight": rfdetr_weight,
                "iou": iou,
                "val_map": val_map,
                "holdout_map": holdout_map,
                "val_gain_vs_best_single": val_gain,
                "holdout_gain_vs_best_single": holdout_gain,
                "min_gain": min(val_gain, holdout_gain),
                "mean_map": (val_map + holdout_map) / 2.0,
                "passes_required_gain": (
                    val_gain >= args.required_gain and holdout_gain >= args.required_gain
                ),
                "val_detections": int(val_metrics["detections"]),
                "holdout_detections": int(holdout_metrics["detections"]),
            }
            records.append(record)
            if best_payload is None or (
                float(record["min_gain"]),
                float(record["mean_map"]),
            ) > (
                float(best_payload[0]["min_gain"]),
                float(best_payload[0]["mean_map"]),
            ):
                best_payload = (record, val_rows, holdout_rows)
            print(json.dumps(record, sort_keys=True))

    records.sort(
        key=lambda row: (float(row["min_gain"]), float(row["mean_map"])), reverse=True
    )
    summary = {
        "baselines": baselines,
        "best_single_val": best_single_val,
        "best_single_holdout": best_single_holdout,
        "required_gain": args.required_gain,
        "target_val": best_single_val + args.required_gain,
        "target_holdout": best_single_holdout + args.required_gain,
        "best": records[0],
        "num_passing": sum(bool(row["passes_required_gain"]) for row in records),
        "records": records,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.out_dir / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    assert best_payload is not None
    write_rows(args.out_dir / "best_val.csv", best_payload[1])
    write_rows(args.out_dir / "best_holdout.csv", best_payload[2])
    print("SUMMARY")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
