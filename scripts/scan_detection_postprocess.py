from __future__ import annotations

import argparse
import csv
from pathlib import Path

from score_detection_csv import evaluate, load_predictions


def parse_float_list(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--confs", default="0.001,0.005,0.01,0.03")
    parser.add_argument("--top-ks", default="30,100,300")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    for conf in parse_float_list(args.confs):
        for top_k in parse_int_list(args.top_ks):
            predictions = load_predictions(args.pred, conf=conf, top_k=top_k)
            metrics = evaluate(args.gt, predictions, max_dets=top_k)
            row: dict[str, float | int] = {
                "conf": conf,
                "top_k": top_k,
                "detections": int(metrics["detections"]),
                "mAP_50_95": metrics["mAP_50_95"],
                "mAP_50": metrics["mAP_50"],
                "mAP_75": metrics["mAP_75"],
                "mAR": metrics["mAR"],
            }
            rows.append(row)
            print(row, flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
