from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--scales", type=int, nargs="+", default=[640, 768, 896])
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    model = YOLO(str(args.weights))
    output: dict[str, dict[str, dict[str, float]]] = {}
    for split in args.splits:
        output[split] = {}
        for scale in args.scales:
            result = model.val(
                data=str(args.data),
                split=split,
                imgsz=scale,
                batch=args.batch,
                device=args.device,
                workers=args.workers,
                plots=False,
                verbose=False,
                rect=True,
                augment=args.augment,
            )
            metrics = {
                "mAP_50_95": float(result.box.map),
                "mAP_50": float(result.box.map50),
                "mAP_75": float(result.box.map75),
            }
            output[split][str(scale)] = metrics
            print(f"split={split} scale={scale} metrics={metrics}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "weights": str(args.weights),
                "data": str(args.data),
                "augment": args.augment,
                "rect": True,
                "metrics": output,
            },
            indent=2,
        )
    )
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
