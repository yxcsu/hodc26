from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

import prepare_yolo as base


_ZIP: zipfile.ZipFile | None = None
_GROUPS: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
    tuple(range(0, 5)),
    tuple(range(5, 11)),
    tuple(range(11, 16)),
)


def _init_worker(zip_path: str, groups: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> None:
    global _ZIP, _GROUPS
    _ZIP = zipfile.ZipFile(zip_path)
    _GROUPS = groups


def grouped_rgb(img: np.ndarray, groups: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]) -> np.ndarray:
    cube = base.x2cube(img).astype(np.float32)
    rgb = np.stack([cube[:, :, list(g)].mean(axis=2) for g in groups], axis=2)
    # Robust per-channel scaling avoids a few saturated pixels dominating the
    # dynamic range while retaining the broad spectral shape.
    for c in range(3):
        ch = rgb[:, :, c]
        lo, hi = np.percentile(ch, (1.0, 99.0))
        if hi > lo:
            rgb[:, :, c] = (ch - lo) * (255.0 / (hi - lo))
        else:
            rgb[:, :, c] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _convert_one(task: tuple[str, str]) -> str:
    image_name, out_name = task
    assert _ZIP is not None
    raw = np.array(Image.open(io.BytesIO(_ZIP.read(image_name))))
    rgb = grouped_rgb(raw, _GROUPS)
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path, compress_level=2)
    return str(out_path)


def parse_groups(text: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    groups = []
    for part in text.split(","):
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            groups.append(tuple(range(a, b + 1)))
        else:
            groups.append((int(part),))
    if len(groups) != 3 or any(not g for g in groups):
        raise ValueError("--groups must define exactly three non-empty groups, e.g. 0-4,5-10,11-15")
    flat = [b for g in groups for b in g]
    if any(b < 0 or b > 15 for b in flat):
        raise ValueError("band indices must be between 0 and 15")
    return tuple(groups)  # type: ignore[return-value]


def convert_tasks(zip_path: Path, tasks: list[tuple[str, str]], groups, workers: int, label: str) -> None:
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(zip_path), groups)) as ex:
        for i, _ in enumerate(ex.map(_convert_one, tasks, chunksize=8), start=1):
            if i % 100 == 0 or i == len(tasks):
                print(f"{label}: {i}/{len(tasks)} converted", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, default=Path("data/hyperspectral-object-detection-challenge-2026.zip"))
    parser.add_argument("--out", type=Path, default=Path("prepared/grouped_0_4_5_10_11_15"))
    parser.add_argument("--groups", default="0-4,5-10,11-15")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    groups = parse_groups(args.groups)

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path) as archive:
        classes = base.load_classes(archive)
        class_to_id = {name: i for i, name in enumerate(classes)}
        records = base.load_records(archive, class_to_id)
        train, val = base.split_records(records, args.val_fraction, args.seed)
        test_names = sorted(
            name for name in archive.namelist()
            if name.startswith("data_test/data_test/VIS/") and name.endswith(".png")
        )

    split_json = {"train": [r.image_id for r in train], "val": [r.image_id for r in val]}
    (args.out / "split.json").write_text(json.dumps(split_json, indent=2))
    base.write_labels(train, args.out / "labels" / "train")
    base.write_labels(val, args.out / "labels" / "val")

    for split, recs in (("train", train), ("val", val)):
        image_dir = args.out / "images" / split
        tasks = []
        for r in recs:
            target = image_dir / f"{r.image_id}.png"
            if not target.exists():
                tasks.append((r.image_name, str(target)))
        convert_tasks(args.zip_path, tasks, groups, args.workers, split)

    test_dir = args.out / "images" / "test"
    test_tasks = []
    for name in test_names:
        target = test_dir / Path(name).name
        if not target.exists():
            test_tasks.append((name, str(target)))
    convert_tasks(args.zip_path, test_tasks, groups, args.workers, "test")

    train_counts = Counter(cls for r in train for cls, *_ in r.objects)
    val_counts = Counter(cls for r in val for cls, *_ in r.objects)
    stats = {
        "groups": [list(g) for g in groups],
        "seed": args.seed,
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test_names),
        "train_objects": sum(train_counts.values()),
        "val_objects": sum(val_counts.values()),
        "train_class_counts": {classes[k]: train_counts[k] for k in range(len(classes))},
        "val_class_counts": {classes[k]: val_counts[k] for k in range(len(classes))},
    }
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2))

    abs_out = args.out.resolve()
    yaml_lines = [
        f"path: {abs_out}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(classes))
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
