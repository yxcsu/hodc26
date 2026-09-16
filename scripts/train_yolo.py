from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO
from ultralytics.engine.trainer import BaseTrainer


def _read_results_csv_without_polars(self: BaseTrainer) -> dict[str, list]:
    path = Path(self.csv)
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {}
    result = {key: [] for key in rows[0]}
    for row in rows:
        for key, value in row.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                pass
            result[key].append(value)
    return result


BaseTrainer.read_results_csv = _read_results_csv_without_polars


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/home/yangxiao/PLNet/yolo11n.pt")
    parser.add_argument("--data", default="prepared/pseudo_5_8_13/data.yaml")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--name", default="y11n_pseudo5813_smoke3")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rect", action="store_true")
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=3.0)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--close-mosaic", type=int, default=10)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--translate", type=float, default=0.1)
    args = parser.parse_args()

    YOLO(args.model).train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        rect=args.rect,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        warmup_epochs=args.warmup_epochs,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        scale=args.scale,
        translate=args.translate,
        deterministic=True,
        plots=False,
        save=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
