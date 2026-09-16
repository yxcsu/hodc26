from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--fallback-conf", type=float, default=0.0001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--augment", action="store_true")
    args = parser.parse_args()

    image_paths = sorted(
        [
            *args.images.glob("*.png"),
            *args.images.glob("*.tif"),
            *args.images.glob("*.tiff"),
        ],
        key=lambda p: int(p.stem),
    )
    if not image_paths:
        raise SystemExit(f"No supported image files found under {args.images}")

    model = YOLO(str(args.model))
    rows: list[list[object]] = []
    row_id = 0
    # Pass the directory itself instead of a Python list of all images.
    # Ultralytics treats a large in-memory list as a single source batch during
    # warmup, which can allocate several GiB even when --batch is small.
    results = model.predict(
        source=str(args.images),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        augment=args.augment,
        verbose=False,
        stream=True,
    )

    image_count = 0
    det_count = 0
    detected_ids: set[int] = set()
    for result in results:
        image_count += 1
        image_id = int(Path(result.path).stem)
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for box, score, class_id in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = map(float, box)
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append([
                row_id,
                image_id,
                int(class_id),
                float(score),
                x1,
                y1,
                x2,
                y2,
            ])
            row_id += 1
            det_count += 1
            detected_ids.add(image_id)

    # Some detectors can emit no boxes for a test image at the main threshold.
    # Keep every test image represented by rerunning only those missing images
    # with a much lower threshold and retaining the single top-scoring valid box.
    missing_paths = [p for p in image_paths if int(p.stem) not in detected_ids]
    for path in missing_paths:
        fallback = model.predict(
            source=str(path),
            imgsz=args.imgsz,
            batch=1,
            device=args.device,
            conf=args.fallback_conf,
            iou=args.iou,
            max_det=args.max_det,
            augment=args.augment,
            verbose=False,
        )[0]
        boxes = fallback.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        scores = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        order = scores.argsort()[::-1]
        for idx in order:
            x1, y1, x2, y2 = map(float, xyxy[idx])
            if x2 <= x1 or y2 <= y1:
                continue
            image_id = int(path.stem)
            rows.append([
                row_id,
                image_id,
                int(classes[idx]),
                float(scores[idx]),
                x1,
                y1,
                x2,
                y2,
            ])
            row_id += 1
            det_count += 1
            detected_ids.add(image_id)
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"images={image_count}")
    print(f"detections={det_count}")
    print(f"submission={args.out}")


if __name__ == "__main__":
    main()
