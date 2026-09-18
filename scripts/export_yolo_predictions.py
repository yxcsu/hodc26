from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO


def load_yolo_names(data_path: Path) -> dict[int, str]:
    payload = yaml.safe_load(data_path.read_text())
    names = payload["names"]
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {int(idx): str(name) for idx, name in names.items()}


def load_coco_maps(coco_path: Path) -> tuple[dict[str, int], dict[str, int]]:
    payload = json.loads(coco_path.read_text())
    image_id_by_file_name: dict[str, int] = {}
    for image in payload["images"]:
        file_name = Path(str(image["file_name"])).name
        image_id = int(image["id"])
        if file_name in image_id_by_file_name and image_id_by_file_name[file_name] != image_id:
            raise ValueError(f"Duplicate COCO file_name with different ids: {file_name}")
        image_id_by_file_name[file_name] = image_id

    category_id_by_name = {
        str(category["name"]): int(category["id"]) for category in payload["categories"]
    }
    return image_id_by_file_name, category_id_by_name


def convert_predictions(
    predictions: list[dict],
    *,
    yolo_names: dict[int, str],
    image_id_by_file_name: dict[str, int],
    category_id_by_name: dict[str, int],
    category_offset: int,
) -> list[list[object]]:
    rows: list[list[object]] = []
    for pred in predictions:
        if "file_name" not in pred:
            raise KeyError(
                "Ultralytics predictions.json is missing file_name; cannot safely map image ids."
            )
        file_name = Path(str(pred["file_name"])).name
        if file_name not in image_id_by_file_name:
            raise KeyError(f"{file_name!r} is not present in the target COCO ground truth")

        yolo_class_id = int(pred["category_id"]) - category_offset
        if yolo_class_id not in yolo_names:
            raise KeyError(
                f"Ultralytics category_id={pred['category_id']} with offset={category_offset} "
                f"does not map to a dataset class"
            )
        class_name = yolo_names[yolo_class_id]
        if class_name not in category_id_by_name:
            raise KeyError(f"Class {class_name!r} is not present in the target COCO categories")

        x, y, w, h = map(float, pred["bbox"])
        rows.append(
            [
                len(rows),
                image_id_by_file_name[file_name],
                category_id_by_name[class_name],
                float(pred["score"]),
                x,
                y,
                x + w,
                y + h,
            ]
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run YOLO validation inference and export detections in the shared CSV format."
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--coco-gt",
        type=Path,
        required=True,
        help="Target COCO annotation JSON used to map file_name and class name to shared ids.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("results/yolo_prediction_exports"),
        help="Directory used for Ultralytics' intermediate predictions.json.",
    )
    parser.add_argument(
        "--predictions-json",
        type=Path,
        default=None,
        help="Reuse an existing Ultralytics predictions.json instead of running inference again.",
    )
    parser.add_argument(
        "--category-offset",
        type=int,
        default=1,
        help=(
            "Subtract this value from Ultralytics JSON category_id before looking up the "
            "YOLO class name. Current custom-dataset Ultralytics JSON output is 1-based."
        ),
    )
    args = parser.parse_args()

    metrics = None
    if args.predictions_json is not None:
        json_path = args.predictions_json
    else:
        if args.weights is None:
            parser.error("--weights is required unless --predictions-json is provided")
        run_name = f"{args.out.stem}_imgsz{args.imgsz}"
        work_dir = args.work_dir.resolve()
        model = YOLO(str(args.weights))
        metrics = model.val(
            data=str(args.data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            rect=True,
            augment=args.augment,
            plots=False,
            verbose=False,
            save_json=True,
            conf=args.conf,
            max_det=args.max_det,
            project=str(work_dir),
            name=run_name,
            exist_ok=True,
        )
        json_path = Path(metrics.save_dir) / "predictions.json"

    predictions = json.loads(json_path.read_text())
    yolo_names = load_yolo_names(args.data)
    image_id_by_file_name, category_id_by_name = load_coco_maps(args.coco_gt)
    rows = convert_predictions(
        predictions,
        yolo_names=yolo_names,
        image_id_by_file_name=image_id_by_file_name,
        category_id_by_name=category_id_by_name,
        category_offset=args.category_offset,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    meta_path = args.out.with_suffix(".meta.json")
    metadata = {
        "weights": str(args.weights) if args.weights is not None else None,
        "data": str(args.data),
        "coco_gt": str(args.coco_gt),
        "split": args.split,
        "imgsz": args.imgsz,
        "augment": args.augment,
        "confidence_threshold": args.conf,
        "max_det": args.max_det,
        "detections": len(rows),
        "category_offset": args.category_offset,
        "intermediate_json": str(json_path),
    }
    if metrics is not None:
        metadata.update(
            {
                "ultralytics_map_50_95": float(metrics.box.map),
                "ultralytics_map_50": float(metrics.box.map50),
                "ultralytics_map_75": float(metrics.box.map75),
            }
        )
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"detections={len(rows)}")
    if metrics is not None:
        print(f"mAP50-95={float(metrics.box.map):.9f}")
    print(f"csv={args.out}")
    print(f"meta={meta_path}")


if __name__ == "__main__":
    main()
