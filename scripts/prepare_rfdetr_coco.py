from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2


CLASS_NAMES = [
    "apple",
    "apple_plastic",
    "badminton",
    "banana",
    "banana_plastic",
    "car",
    "car_toy",
    "charger_head",
    "e-bike",
    "egg",
    "egg_plastic",
    "egg_wood",
    "orange",
    "orange_plastic",
    "people",
    "rubik",
    "stone_block",
    "table_tennis",
]


def image_hw(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    ok, frames = cv2.imdecodemulti(
        __import__("numpy").frombuffer(data, dtype="uint8"), cv2.IMREAD_UNCHANGED
    )
    if not ok or not frames:
        raise RuntimeError(f"Failed to decode {path}")
    h, w = frames[0].shape[:2]
    return h, w


def safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def build_split(source_root: Path, out_root: Path, source_split: str, rf_split: str) -> dict:
    image_dir = source_root / "images" / source_split
    label_dir = source_root / "labels" / source_split
    out_dir = out_root / rf_split
    out_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [*image_dir.glob("*.tiff"), *image_dir.glob("*.tif"), *image_dir.glob("*.png")],
        key=lambda p: int(p.stem),
    )
    if not image_paths:
        raise RuntimeError(f"No supported images found in {image_dir}")

    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1
    for index, src in enumerate(image_paths, start=1):
        image_id = int(src.stem)
        h, w = image_hw(src)
        safe_link(src, out_dir / src.name)
        images.append(
            {
                "id": image_id,
                "file_name": src.name,
                "width": w,
                "height": h,
            }
        )

        label_path = label_dir / f"{src.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                class_id_s, cx_s, cy_s, bw_s, bh_s = line.split()[:5]
                class_id = int(class_id_s)
                cx, cy, bw, bh = map(float, (cx_s, cy_s, bw_s, bh_s))
                box_w = bw * w
                box_h = bh * h
                x = (cx - bw / 2.0) * w
                y = (cy - bh / 2.0) * h
                x = max(0.0, min(float(w), x))
                y = max(0.0, min(float(h), y))
                box_w = max(0.0, min(float(w) - x, box_w))
                box_h = max(0.0, min(float(h) - y, box_h))
                if box_w <= 0 or box_h <= 0:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": [x, y, box_w, box_h],
                        "area": box_w * box_h,
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
        if index % 250 == 0 or index == len(image_paths):
            print(f"{rf_split}: {index}/{len(image_paths)} images, {len(annotations)} boxes", flush=True)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": name} for i, name in enumerate(CLASS_NAMES)],
    }
    (out_dir / "_annotations.coco.json").write_text(json.dumps(coco, separators=(",", ":")))
    return {"images": len(images), "annotations": len(annotations)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("prepared/hsi16_clip320"))
    parser.add_argument("--out", type=Path, default=Path("prepared/rfdetr_hsi16"))
    parser.add_argument(
        "--valid-source-split",
        default="val",
        choices=("val", "holdout"),
        help="Source split to expose as RF-DETR's valid/ directory.",
    )
    args = parser.parse_args()

    stats = {
        "train": build_split(args.source, args.out, "train", "train"),
        "valid": build_split(args.source, args.out, args.valid_source_split, "valid"),
        "test": build_split(args.source, args.out, "test", "test"),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
