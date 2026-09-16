from __future__ import annotations

import argparse
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from rfdetr import RFDETRSmall
from rfdetr.datasets.coco import build_roboflow_from_coco
from rfdetr.utilities.tensors import make_collate_fn

from train_rfdetr_multispectral import install_multispectral_patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=384)
    parser.add_argument("--channels", type=int, default=16)
    parser.add_argument("--classes", type=int, default=18)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    install_multispectral_patches()

    model = RFDETRSmall(
        num_channels=args.channels,
        num_classes=args.classes,
        resolution=args.resolution,
        pretrain_weights=None,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    incompatible = model.model.model.load_state_dict(state, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint load incomplete: missing={incompatible.missing_keys[:10]} "
            f"unexpected={incompatible.unexpected_keys[:10]}"
        )

    device = torch.device(args.device)
    model.model.model.to(device).eval()

    cfg = SimpleNamespace(
        dataset_dir=str(args.dataset),
        square_resize_div_64=False,
        segmentation_head=False,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        patch_size=model.model_config.patch_size,
        num_windows=model.model_config.num_windows,
        use_grouppose_keypoints=False,
        aug_config={},
        scale_jitter=False,
        augmentation_backend="cpu",
    )
    dataset = build_roboflow_from_coco("test", cfg, args.resolution)
    block_size = int(model.model_config.patch_size) * int(model.model_config.num_windows)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=make_collate_fn(block_size=block_size, pack=False),
    )

    rows: list[list[object]] = []
    row_id = 0
    image_count = 0
    missing_count = 0

    with torch.inference_mode():
        for batch_idx, (samples, targets) in enumerate(loader, start=1):
            samples = samples.to(device)
            predictions = model.model.model(samples)
            target_sizes = torch.stack([target["orig_size"] for target in targets]).to(device)
            # Keep raw DETR predictions. We intentionally do not apply an extra NMS.
            results = model.model.postprocess(predictions, target_sizes=target_sizes, score_threshold=0.0)

            for result, target in zip(results, targets):
                image_count += 1
                image_id = int(target["image_id"].reshape(-1)[0].item())
                scores = result["scores"].detach().cpu()
                labels = result["labels"].detach().cpu()
                boxes = result["boxes"].detach().cpu()

                valid_class = (labels >= 0) & (labels < args.classes)
                keep = torch.nonzero((scores > args.conf) & valid_class, as_tuple=False).flatten()
                if keep.numel() == 0 and scores.numel() > 0:
                    # Match the previous submission pipeline's guarantee that every
                    # image has at least one prediction, but keep it at its native
                    # low confidence so it cannot outrank stronger detections.
                    valid_idx = torch.nonzero(valid_class, as_tuple=False).flatten()
                    if valid_idx.numel() > 0:
                        best_local = int(torch.argmax(scores[valid_idx]).item())
                        keep = valid_idx[best_local : best_local + 1]
                        missing_count += 1

                for idx in keep.tolist():
                    x1, y1, x2, y2 = map(float, boxes[idx].tolist())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    rows.append(
                        [
                            row_id,
                            image_id,
                            int(labels[idx].item()),
                            float(scores[idx].item()),
                            x1,
                            y1,
                            x2,
                            y2,
                        ]
                    )
                    row_id += 1

            if batch_idx % 20 == 0 or image_count == len(dataset):
                print(
                    f"processed={image_count}/{len(dataset)} detections={len(rows)}",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
        writer.writerows(rows)

    print(f"images={image_count}")
    print(f"detections={len(rows)}")
    print(f"fallback_images={missing_count}")
    print(f"submission={args.out}")


if __name__ == "__main__":
    main()
