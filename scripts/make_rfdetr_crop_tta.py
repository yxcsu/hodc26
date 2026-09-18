from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

import rfdetr.datasets.coco as coco_mod
from rfdetr import RFDETRSmall
from rfdetr.models.weights import interpolate_position_embeddings
from rfdetr.utilities.tensors import make_collate_fn

from train_rfdetr_multispectral import (
    configure_normalization,
    install_identity_spectral_adapter,
    install_multispectral_patches,
)


VIEW_NAMES = ("tl", "tr", "bl", "br")


def crop_geometry(width: int, height: int, fraction: float, view_idx: int) -> tuple[int, int, int, int]:
    crop_w = max(1, min(width, int(round(width * fraction))))
    crop_h = max(1, min(height, int(round(height * fraction))))
    right = view_idx in (1, 3)
    bottom = view_idx in (2, 3)
    x0 = width - crop_w if right else 0
    y0 = height - crop_h if bottom else 0
    return x0, y0, crop_w, crop_h


def owns_center(view_idx: int, cx: float, cy: float, width: int, height: int) -> bool:
    """Assign each global box center to exactly one crop responsibility quadrant."""
    right = view_idx in (1, 3)
    bottom = view_idx in (2, 3)
    owns_x = cx > width / 2.0 if right else cx <= width / 2.0
    owns_y = cy > height / 2.0 if bottom else cy <= height / 2.0
    return owns_x and owns_y


def decode_hsi_tiff(path: Path) -> tv_tensors.Image:
    file_bytes = np.fromfile(path, dtype=np.uint8)
    ok, frames = cv2.imdecodemulti(file_bytes, cv2.IMREAD_UNCHANGED)
    if not ok or not frames:
        raise RuntimeError(f"Failed to decode multispectral TIFF: {path}")
    image = np.stack(frames, axis=0)
    if image.ndim != 3:
        raise RuntimeError(f"Expected CxHxW TIFF stack, got {image.shape} for {path}")
    return tv_tensors.Image(torch.from_numpy(np.ascontiguousarray(image)))


class CropTTADataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        split: str,
        resolution: int,
        crop_fraction: float,
        max_images: int | None = None,
    ) -> None:
        split_dir = dataset_dir / ("valid" if split == "valid" else "test")
        annotation_path = split_dir / "_annotations.coco.json"
        annotation = json.loads(annotation_path.read_text())
        images = sorted(annotation["images"], key=lambda item: int(item["id"]))
        if max_images is not None:
            images = images[:max_images]
        self.images = images
        self.split_dir = split_dir
        self.crop_fraction = crop_fraction
        self.transforms = coco_mod.make_coco_transforms(
            "val" if split == "valid" else "test",
            resolution,
            multi_scale=False,
            expanded_scales=False,
            skip_random_resize=True,
            patch_size=16,
            num_windows=2,
            aug_config={},
            scale_jitter=False,
            gpu_postprocess=False,
            keypoint_flip_pairs=[],
        )

    def __len__(self) -> int:
        return len(self.images) * len(VIEW_NAMES)

    def __getitem__(self, index: int):
        image_idx, view_idx = divmod(index, len(VIEW_NAMES))
        info = self.images[image_idx]
        image_id = int(info["id"])
        full_w = int(info["width"])
        full_h = int(info["height"])
        image = decode_hsi_tiff(self.split_dir / info["file_name"])
        if tuple(map(int, image.shape[-2:])) != (full_h, full_w):
            raise RuntimeError(
                f"Annotation/image size mismatch for image_id={image_id}: "
                f"annotation={(full_h, full_w)} decoded={tuple(map(int, image.shape[-2:]))}"
            )

        x0, y0, crop_w, crop_h = crop_geometry(full_w, full_h, self.crop_fraction, view_idx)
        crop = tv_tensors.Image(image[:, y0 : y0 + crop_h, x0 : x0 + crop_w].contiguous())
        empty_boxes = torch.zeros((0, 4), dtype=torch.float32)
        target = {
            "boxes": empty_boxes,
            "labels": torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.as_tensor([image_id], dtype=torch.int64),
            "area": torch.zeros((0,), dtype=torch.float32),
            "iscrowd": torch.zeros((0,), dtype=torch.int64),
            "orig_size": torch.as_tensor([crop_h, crop_w], dtype=torch.int64),
            "size": torch.as_tensor([crop_h, crop_w], dtype=torch.int64),
        }
        crop, target = self.transforms(crop, target)
        target["full_size"] = torch.as_tensor([full_h, full_w], dtype=torch.int64)
        target["crop_offset"] = torch.as_tensor([y0, x0], dtype=torch.int64)
        target["crop_view_idx"] = torch.as_tensor([view_idx], dtype=torch.int64)
        return crop, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--crop-fraction", type=float, default=0.60)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--classes", type=int, default=18)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--top-k", type=int, default=300)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--meta-out", type=Path, default=None)
    parser.add_argument(
        "--normalization",
        choices=("imagenet-cyclic", "train-stats"),
        default="imagenet-cyclic",
    )
    parser.add_argument("--normalization-stats", type=Path, default=None)
    args = parser.parse_args()

    if not (0.5 < args.crop_fraction <= 1.0):
        raise ValueError("--crop-fraction must be in (0.5, 1.0] so the four crops cover the full image")

    configure_normalization(args.normalization, args.normalization_stats)
    if args.channels != 3:
        install_multispectral_patches()

    model = RFDETRSmall(
        num_channels=args.channels,
        num_classes=args.classes,
        resolution=args.resolution,
        pretrain_weights=None,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"].copy()
    interpolate_position_embeddings(state, int(model.model_config.positional_encoding_size))
    adapter_keys = [key for key in state if "spectral_adapter." in key]
    if adapter_keys:
        patch_embeddings = model.model.model.backbone[0].encoder.encoder.embeddings.patch_embeddings
        install_identity_spectral_adapter(patch_embeddings, args.channels)
    incompatible = model.model.model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load incomplete: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    device = torch.device(args.device)
    model.model.model.to(device).eval()

    dataset = CropTTADataset(
        args.dataset,
        args.split,
        args.resolution,
        args.crop_fraction,
        max_images=args.max_images,
    )
    block_size = int(model.model_config.patch_size) * int(model.model_config.num_windows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=make_collate_fn(block_size=block_size, pack=False),
    )

    by_image: dict[int, list[tuple[int, float, list[float]]]] = defaultdict(list)
    raw_kept = 0
    ownership_rejected = 0
    processed_crops = 0

    with torch.inference_mode():
        for batch_idx, (samples, targets) in enumerate(loader, start=1):
            samples = samples.to(device)
            predictions = model.model.model(samples)
            target_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            results = model.model.postprocess(predictions, target_sizes=target_sizes, score_threshold=0.0)

            for result, target in zip(results, targets):
                processed_crops += 1
                image_id = int(target["image_id"].reshape(-1)[0].item())
                crop_h, crop_w = map(int, target["orig_size"].tolist())
                full_h, full_w = map(int, target["full_size"].tolist())
                y0, x0 = map(int, target["crop_offset"].tolist())
                view_idx = int(target["crop_view_idx"].reshape(-1)[0].item())

                scores = result["scores"].detach().cpu()
                labels = result["labels"].detach().cpu()
                boxes = result["boxes"].detach().cpu()
                valid_class = (labels >= 0) & (labels < args.classes)
                keep = torch.nonzero((scores > args.conf) & valid_class, as_tuple=False).flatten()
                if keep.numel() > args.top_k:
                    order = torch.argsort(scores[keep], descending=True)[: args.top_k]
                    keep = keep[order]

                for idx in keep.tolist():
                    raw_kept += 1
                    x1, y1, x2, y2 = map(float, boxes[idx].tolist())
                    # Clamp in crop coordinates before mapping back to the full image.
                    x1 = min(max(x1, 0.0), float(crop_w)) + x0
                    x2 = min(max(x2, 0.0), float(crop_w)) + x0
                    y1 = min(max(y1, 0.0), float(crop_h)) + y0
                    y2 = min(max(y2, 0.0), float(crop_h)) + y0
                    if x2 <= x1 or y2 <= y1:
                        continue
                    cx = (x1 + x2) * 0.5
                    cy = (y1 + y2) * 0.5
                    if not owns_center(view_idx, cx, cy, full_w, full_h):
                        ownership_rejected += 1
                        continue
                    by_image[image_id].append(
                        (int(labels[idx].item()), float(scores[idx].item()), [x1, y1, x2, y2])
                    )

            if batch_idx % 20 == 0 or processed_crops == len(dataset):
                print(
                    f"processed_crops={processed_crops}/{len(dataset)} "
                    f"owned_predictions={sum(len(v) for v in by_image.values())}",
                    flush=True,
                )

    rows: list[list[object]] = []
    row_id = 0
    for image_id in sorted(by_image):
        predictions = sorted(by_image[image_id], key=lambda item: item[1], reverse=True)[: args.top_k]
        for class_id, score, box in predictions:
            rows.append([row_id, image_id, class_id, score, *box])
            row_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    image_count = len(dataset.images)
    print(f"images={image_count}")
    print(f"crop_views={len(VIEW_NAMES)}")
    print(f"raw_kept={raw_kept}")
    print(f"ownership_rejected={ownership_rejected}")
    print(f"detections={len(rows)}")
    print(f"submission={args.out}")

    if args.meta_out is not None:
        args.meta_out.parent.mkdir(parents=True, exist_ok=True)
        args.meta_out.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "dataset": str(args.dataset),
                    "split": args.split,
                    "resolution": args.resolution,
                    "crop_fraction": args.crop_fraction,
                    "nominal_overlap_fraction_of_full_image": max(0.0, 2.0 * args.crop_fraction - 1.0),
                    "responsibility": "box center assigned to TL/TR/BL/BR by full-image W/2,H/2",
                    "channels": args.channels,
                    "classes": args.classes,
                    "confidence_threshold": args.conf,
                    "top_k_per_crop_before_ownership": args.top_k,
                    "top_k_per_image_after_ownership": args.top_k,
                    "normalization": args.normalization,
                    "normalization_stats": str(args.normalization_stats) if args.normalization_stats else None,
                    "spectral_adapter": bool(adapter_keys),
                    "images": image_count,
                    "crop_views": list(VIEW_NAMES),
                    "raw_kept": raw_kept,
                    "ownership_rejected": ownership_rejected,
                    "detections": len(rows),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
