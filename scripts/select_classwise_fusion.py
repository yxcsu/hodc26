from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_candidate(value: str) -> tuple[str, float, float]:
    tag, weight, iou = value.split(":", maxsplit=2)
    return tag, float(weight), float(iou)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select conservative per-class YOLO/RF-DETR fusion parameters from "
            "pre-scored candidate configurations."
        )
    )
    parser.add_argument("--scores-dir", type=Path, required=True)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline-tag", required=True)
    parser.add_argument(
        "--candidate",
        type=parse_candidate,
        action="append",
        required=True,
        metavar="TAG:YOLO_WEIGHT:IOU",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed per-split AP regression versus the baseline candidate. "
            "Default 0 requires both validation and holdout AP to be non-decreasing."
        ),
    )
    args = parser.parse_args()

    candidate_specs = {
        tag: {"yolo_weight": weight, "rfdetr_weight": 1.0, "iou": iou}
        for tag, weight, iou in args.candidate
    }
    if args.baseline_tag not in candidate_specs:
        raise ValueError("--baseline-tag must also be listed as a --candidate")

    gt = json.loads(args.gt.read_text())
    categories = sorted(gt["categories"], key=lambda item: int(item["id"]))

    scores: dict[str, dict[str, dict[str, float]]] = {}
    for tag in candidate_specs:
        scores[tag] = {}
        for split in ("val", "holdout"):
            path = args.scores_dir / f"{split}_{tag}_score.json"
            payload = json.loads(path.read_text())
            scores[tag][split] = {
                str(name): float(ap)
                for name, ap in payload["per_class_AP_50_95"].items()
            }

    baseline = args.baseline_tag
    selections: dict[str, dict[str, object]] = {}
    for category in categories:
        class_id = int(category["id"])
        class_name = str(category["name"])
        base_val = scores[baseline]["val"][class_name]
        base_holdout = scores[baseline]["holdout"][class_name]

        feasible: list[tuple[float, float, str, float, float]] = []
        for tag in candidate_specs:
            val_ap = scores[tag]["val"][class_name]
            holdout_ap = scores[tag]["holdout"][class_name]
            val_delta = val_ap - base_val
            holdout_delta = holdout_ap - base_holdout
            if (
                val_delta >= -args.tolerance
                and holdout_delta >= -args.tolerance
            ):
                feasible.append(
                    (
                        (val_ap + holdout_ap) / 2.0,
                        min(val_delta, holdout_delta),
                        tag,
                        val_ap,
                        holdout_ap,
                    )
                )

        if not feasible:
            chosen_tag = baseline
            chosen_val = base_val
            chosen_holdout = base_holdout
        else:
            _, _, chosen_tag, chosen_val, chosen_holdout = max(feasible)

        spec = candidate_specs[chosen_tag]
        selections[str(class_id)] = {
            "class_id": class_id,
            "class_name": class_name,
            "tag": chosen_tag,
            **spec,
            "val_ap": chosen_val,
            "holdout_ap": chosen_holdout,
            "baseline_val_ap": base_val,
            "baseline_holdout_ap": base_holdout,
            "val_delta": chosen_val - base_val,
            "holdout_delta": chosen_holdout - base_holdout,
        }

    output = {
        "baseline_tag": baseline,
        "tolerance": args.tolerance,
        "candidate_specs": candidate_specs,
        "classes": selections,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))

    print(
        "class_id class_name           tag    yolo_w iou   "
        "val_delta holdout_delta"
    )
    for class_id in sorted(selections, key=int):
        item = selections[class_id]
        print(
            f"{int(item['class_id']):8d} "
            f"{str(item['class_name']):20s} "
            f"{str(item['tag']):6s} "
            f"{float(item['yolo_weight']):6.2f} "
            f"{float(item['iou']):.2f} "
            f"{float(item['val_delta']):+9.6f} "
            f"{float(item['holdout_delta']):+13.6f}"
        )
    print(f"config={args.out}")


if __name__ == "__main__":
    main()
