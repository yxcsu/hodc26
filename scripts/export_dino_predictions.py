from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, SequentialSampler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official DINO predictions to the repository detection CSV format."
    )
    parser.add_argument(
        "--dino-root",
        type=Path,
        default=Path("/home/yangxiao/.cache/hodc26/DINO"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--coco-root",
        type=Path,
        required=True,
        help="COCO2017-style root produced by prepare_dino_coco_layout.py.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--conf", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dino_root = args.dino_root.resolve()
    sys.path.insert(0, str(dino_root))

    from datasets import build_dataset  # type: ignore
    from main import build_model_main  # type: ignore
    from util import misc as utils  # type: ignore
    from util.slconfig import SLConfig  # type: ignore

    cfg = SLConfig.fromfile(str(dino_root / "config/DINO/DINO_4scale.py"))
    model_args = argparse.Namespace(**cfg._cfg_dict.to_dict())
    model_args.dataset_file = "coco"
    model_args.coco_path = str(args.coco_root.resolve())
    model_args.fix_size = False
    model_args.strong_aug = False
    model_args.num_classes = 18
    model_args.dn_labelbook_size = 19
    model_args.num_queries = 300
    model_args.num_select = args.top_k
    model_args.data_aug_scales = [480, 544, 608, 672]
    model_args.data_aug_max_size = 1344
    model_args.data_aug_scales2_resize = [400, 500, 600]
    model_args.data_aug_scales2_crop = [384, 600]

    device = torch.device(args.device)
    model, _, postprocessors = build_model_main(model_args)

    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = utils.clean_state_dict(checkpoint["model"])
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    dataset = build_dataset("val", model_args)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=SequentialSampler(dataset),
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    rows: list[list[object]] = []
    row_id = 0
    with torch.inference_mode():
        for samples, targets in loader:
            samples = samples.to(device)
            outputs = model(samples)
            orig_sizes = torch.stack([target["orig_size"] for target in targets]).to(
                device
            )
            results = postprocessors["bbox"](outputs, orig_sizes)
            for target, result in zip(targets, results):
                image_id = int(target["image_id"])
                height, width = (int(v) for v in target["orig_size"])
                scores = result["scores"].detach().cpu()
                labels = result["labels"].detach().cpu()
                boxes = result["boxes"].detach().cpu()
                keep = scores > args.conf
                scores = scores[keep][: args.top_k]
                labels = labels[keep][: args.top_k]
                boxes = boxes[keep][: args.top_k]
                for score, label, box in zip(scores, labels, boxes):
                    x1, y1, x2, y2 = (float(v) for v in box)
                    x1 = max(0.0, min(x1, float(width)))
                    y1 = max(0.0, min(y1, float(height)))
                    x2 = max(0.0, min(x2, float(width)))
                    y2 = max(0.0, min(y2, float(height)))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    rows.append(
                        [
                            row_id,
                            image_id,
                            int(label),
                            float(score),
                            x1,
                            y1,
                            x2,
                            y2,
                        ]
                    )
                    row_id += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["id", "image_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]
        )
        writer.writerows(rows)
    print(f"images={len(dataset)} rows={len(rows)} out={args.out}")


if __name__ == "__main__":
    main()
