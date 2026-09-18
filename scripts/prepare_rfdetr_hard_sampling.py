from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    os.symlink(src.resolve(), dst)


def clone_dataset(
    source: Path,
    out: Path,
    repeats: dict[int, int],
    manifest: dict[str, Any],
) -> None:
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)

    train_src = source / "train"
    train_out = out / "train"
    train_out.mkdir(parents=True, exist_ok=True)
    for path in train_src.iterdir():
        if path.name == "_annotations.coco.json":
            continue
        if path.is_file():
            link_file(path, train_out / path.name)

    for split in ("valid", "test"):
        src = source / split
        if src.exists():
            os.symlink(src.resolve(), out / split)

    ann_path = train_src / "_annotations.coco.json"
    data = json.loads(ann_path.read_text())
    images = list(data["images"])
    annotations = list(data["annotations"])
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        anns_by_image[int(ann["image_id"])].append(ann)

    max_image_id = max(int(x["id"]) for x in images)
    max_ann_id = max(int(x["id"]) for x in annotations)
    next_image_id = max_image_id + 1
    next_ann_id = max_ann_id + 1
    image_lookup = {int(x["id"]): x for x in images}

    clone_records: list[dict[str, Any]] = []
    for image_id in sorted(repeats):
        repeat = int(repeats[image_id])
        if repeat <= 0:
            continue
        original = image_lookup[image_id]
        for copy_index in range(repeat):
            new_image = dict(original)
            new_image["id"] = next_image_id
            images.append(new_image)
            cloned_anns = 0
            for ann in anns_by_image.get(image_id, []):
                new_ann = dict(ann)
                new_ann["id"] = next_ann_id
                new_ann["image_id"] = next_image_id
                annotations.append(new_ann)
                next_ann_id += 1
                cloned_anns += 1
            clone_records.append(
                {
                    "source_image_id": image_id,
                    "clone_image_id": next_image_id,
                    "copy_index": copy_index,
                    "annotations": cloned_anns,
                }
            )
            next_image_id += 1

    data["images"] = images
    data["annotations"] = annotations
    (train_out / "_annotations.coco.json").write_text(json.dumps(data))
    manifest = {
        **manifest,
        "source": str(source),
        "output": str(out),
        "original_train_images": len(image_lookup),
        "cloned_images": len(clone_records),
        "final_train_images": len(images),
        "original_annotations": len(data["annotations"]) - sum(x["annotations"] for x in clone_records),
        "final_annotations": len(annotations),
        "clone_records": clone_records,
    }
    (out / "sampling_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "output": str(out),
                "cloned_images": len(clone_records),
                "final_train_images": len(images),
                "final_annotations": len(annotations),
            },
            indent=2,
        )
    )


def build_hard_repeats(source: Path) -> tuple[dict[int, int], dict[str, Any]]:
    data = json.loads((source / "train" / "_annotations.coco.json").read_text())
    categories = {int(x["id"]): str(x["name"]) for x in data["categories"]}
    anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in data["annotations"]:
        anns_by_image[int(ann["image_id"])].append(ann)

    repeats: dict[int, int] = {}
    reasons: dict[int, list[str]] = defaultdict(list)
    for image in data["images"]:
        image_id = int(image["id"])
        anns = anns_by_image.get(image_id, [])
        names = {categories[int(a["category_id"])] for a in anns}

        stone = [a for a in anns if categories[int(a["category_id"])] == "stone_block"]
        people = [a for a in anns if categories[int(a["category_id"])] == "people"]
        car = [a for a in anns if categories[int(a["category_id"])] == "car"]
        ebike = [a for a in anns if categories[int(a["category_id"])] == "e-bike"]

        repeat = 0
        if stone:
            repeat = max(repeat, 2)
            reasons[image_id].append("stone_block_any_x2")
        if any(float(a.get("area", a["bbox"][2] * a["bbox"][3])) <= 500 for a in people):
            repeat = max(repeat, 1)
            reasons[image_id].append("people_small_area<=500")
        if any(float(a.get("area", a["bbox"][2] * a["bbox"][3])) <= 600 for a in car):
            repeat = max(repeat, 1)
            reasons[image_id].append("car_small_area<=600")
        if ebike and (
            "people" in names
            or any(float(a.get("area", a["bbox"][2] * a["bbox"][3])) <= 1200 for a in ebike)
        ):
            repeat = max(repeat, 1)
            reasons[image_id].append("ebike_people_cooccur_or_area<=1200")
        if repeat:
            repeats[image_id] = repeat

    repeat_hist: dict[str, int] = defaultdict(int)
    for repeat in repeats.values():
        repeat_hist[str(repeat)] += 1
    manifest = {
        "mode": "targeted_hard_sampling",
        "rules": {
            "stone_block": "duplicate each containing image twice",
            "people": "duplicate once if any people box area <= 500 px^2",
            "car": "duplicate once if any car box area <= 600 px^2",
            "e-bike": "duplicate once if co-occurs with people or any e-bike area <= 1200 px^2",
            "combine": "maximum repeat count across matched rules",
        },
        "repeat_histogram_images": dict(repeat_hist),
        "selected_images": len(repeats),
        "clone_total": sum(repeats.values()),
        "reasons": {str(k): v for k, v in reasons.items()},
    }
    return repeats, manifest


def build_random_repeats(
    source: Path,
    hard_repeats: dict[int, int],
    seed: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    data = json.loads((source / "train" / "_annotations.coco.json").read_text())
    image_ids = [int(x["id"]) for x in data["images"]]
    rng = random.Random(seed)

    n_repeat2 = sum(1 for x in hard_repeats.values() if x == 2)
    n_repeat1 = sum(1 for x in hard_repeats.values() if x == 1)
    chosen = rng.sample(image_ids, n_repeat2 + n_repeat1)
    repeats = {x: 2 for x in chosen[:n_repeat2]}
    repeats.update({x: 1 for x in chosen[n_repeat2:]})
    manifest = {
        "mode": "random_duplication_control",
        "seed": seed,
        "matched_to_hard_sampling": {
            "repeat2_images": n_repeat2,
            "repeat1_images": n_repeat1,
            "clone_total": sum(hard_repeats.values()),
        },
        "selected_image_ids": chosen,
    }
    return repeats, manifest


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--hard-out", type=Path, required=True)
    p.add_argument("--random-out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=2026)
    args = p.parse_args()

    hard_repeats, hard_manifest = build_hard_repeats(args.source)
    random_repeats, random_manifest = build_random_repeats(args.source, hard_repeats, args.seed)
    clone_dataset(args.source, args.hard_out, hard_repeats, hard_manifest)
    clone_dataset(args.source, args.random_out, random_repeats, random_manifest)


if __name__ == "__main__":
    main()
