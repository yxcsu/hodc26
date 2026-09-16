from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from prepare_stratified_hsi_split import collect, sanitize_label_file, safe_link, split_score


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


def choose_group_split(
    obj: np.ndarray,
    presence: np.ndarray,
    group_indices: list[np.ndarray],
    seed: int,
    val_size: int,
    holdout_size: int,
    trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(obj)
    group_sizes = np.asarray([len(indices) for indices in group_indices], dtype=np.int32)
    frac_val = val_size / n
    frac_hold = holdout_size / n

    def consume(order: np.ndarray, start: int, target: int) -> tuple[list[int], int]:
        selected: list[int] = []
        total = 0
        pos = start
        while pos < len(order):
            g = int(order[pos])
            size = int(group_sizes[g])
            before = abs(total - target)
            after = abs(total + size - target)
            if total < target or after < before:
                selected.append(g)
                total += size
            pos += 1
            if total >= target and abs(total - target) <= max(1, int(group_sizes.max())):
                break
        return selected, pos

    best: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    all_groups = np.arange(len(group_indices))
    for _ in range(trials):
        order = rng.permutation(all_groups)
        val_groups, pos = consume(order, 0, val_size)
        used = set(val_groups)
        remaining_order = np.asarray([g for g in order[pos:].tolist() + order[:pos].tolist() if int(g) not in used])
        hold_groups, _ = consume(remaining_order, 0, holdout_size)
        hold_set = set(hold_groups)
        train_groups = [int(g) for g in all_groups if int(g) not in used and int(g) not in hold_set]

        val = np.concatenate([group_indices[g] for g in val_groups]) if val_groups else np.empty(0, dtype=int)
        hold = np.concatenate([group_indices[g] for g in hold_groups]) if hold_groups else np.empty(0, dtype=int)
        train = np.concatenate([group_indices[g] for g in train_groups]) if train_groups else np.empty(0, dtype=int)

        size_penalty = 10.0 * ((len(val) - val_size) / max(val_size, 1)) ** 2
        size_penalty += 10.0 * ((len(hold) - holdout_size) / max(holdout_size, 1)) ** 2
        score = split_score(val, obj, presence, frac_val) + split_score(hold, obj, presence, frac_hold) + size_penalty
        if best is None or score < best[3]:
            best = (train.copy(), val.copy(), hold.copy(), float(score))
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("prepared/hsi16_clip320"))
    parser.add_argument("--groups", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--holdout-size", type=int, default=300)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.out}")

    ids, obj, presence, records = collect(args.source)
    group_data = json.loads(args.groups.read_text())
    mapping: dict[str, str] = group_data["image_to_group"]
    missing = [image_id for image_id in ids if image_id not in mapping]
    if missing:
        raise RuntimeError(f"Group mapping missing {len(missing)} ids, e.g. {missing[:5]}")

    grouped: dict[str, list[int]] = {}
    for idx, image_id in enumerate(ids):
        grouped.setdefault(mapping[image_id], []).append(idx)
    group_names = sorted(grouped)
    group_indices = [np.asarray(grouped[name], dtype=int) for name in group_names]

    train, val, hold, score = choose_group_split(
        obj, presence, group_indices, args.seed, args.val_size, args.holdout_size, args.trials
    )
    audit_report: dict[str, list[dict]] = {}
    for split, indices in (("train", train), ("val", val), ("holdout", hold)):
        for idx in indices:
            image_id = ids[int(idx)]
            image_src, label_src = records[image_id]
            safe_link(image_src, args.out / "images" / split / image_src.name)
            sanitize_label_file(label_src, args.out / "labels" / split / label_src.name, image_id, audit_report)
    for image_path in (args.source / "images" / "test").glob("*.tiff"):
        safe_link(image_path, args.out / "images" / "test" / image_path.name)

    split_sets = {
        "train": {mapping[ids[int(i)]] for i in train},
        "val": {mapping[ids[int(i)]] for i in val},
        "holdout": {mapping[ids[int(i)]] for i in hold},
    }
    overlaps = {
        "train_val": sorted(split_sets["train"] & split_sets["val"]),
        "train_holdout": sorted(split_sets["train"] & split_sets["holdout"]),
        "val_holdout": sorted(split_sets["val"] & split_sets["holdout"]),
    }

    def stats(indices: np.ndarray) -> dict:
        return {
            "images": int(len(indices)),
            "groups": len({mapping[ids[int(i)]] for i in indices}),
            "objects": obj[indices].sum(axis=0).tolist(),
            "presence": presence[indices].sum(axis=0).tolist(),
        }

    report = {
        "seed": args.seed,
        "score": score,
        "groups_file": str(args.groups),
        "total_groups": len(group_indices),
        "train": stats(train),
        "val": stats(val),
        "holdout": stats(hold),
        "group_overlap": overlaps,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_report.json").write_text(json.dumps(report, indent=2))
    (args.out / "annotation_audit.json").write_text(json.dumps(audit_report, indent=2))
    split_ids = {
        "train": [ids[int(i)] for i in train],
        "val": [ids[int(i)] for i in val],
        "holdout": [ids[int(i)] for i in hold],
    }
    (args.out / "split.json").write_text(json.dumps(split_ids, indent=2))
    yaml_lines = [
        f"path: {args.out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "channels: 16",
        "names:",
    ]
    yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
