from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import io
import json
from pathlib import Path
import random
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Record:
    image_id: str
    image_name: str
    xml_name: str
    width: int
    height: int
    objects: tuple[tuple[int, float, float, float, float], ...]


_ZIP: zipfile.ZipFile | None = None
_BANDS: tuple[int, int, int] = (5, 8, 13)


def _init_worker(zip_path: str, bands: tuple[int, int, int]) -> None:
    global _ZIP, _BANDS
    _ZIP = zipfile.ZipFile(zip_path)
    _BANDS = bands


def x2cube(img: np.ndarray) -> np.ndarray:
    h = (img.shape[0] // 4) * 4
    w = (img.shape[1] // 4) * 4
    img = img[:h, :w]
    return img.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(h // 4, w // 4, 16)


def pseudo_rgb(img: np.ndarray, bands: tuple[int, int, int]) -> np.ndarray:
    cube = x2cube(img)
    rgb = cube[:, :, bands].astype(np.float32)
    for c in range(3):
        ch = rgb[:, :, c]
        lo = float(ch.min())
        hi = float(ch.max())
        if hi > lo:
            rgb[:, :, c] = (ch - lo) * (255.0 / (hi - lo))
        else:
            rgb[:, :, c] = 0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _convert_one(task: tuple[str, str]) -> str:
    image_name, out_name = task
    assert _ZIP is not None
    raw = np.array(Image.open(io.BytesIO(_ZIP.read(image_name))))
    rgb = pseudo_rgb(raw, _BANDS)
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(out_path, compress_level=2)
    return str(out_path)


def load_classes(archive: zipfile.ZipFile) -> list[str]:
    return [x.strip() for x in archive.read("class.txt").decode().splitlines() if x.strip()]


def load_records(archive: zipfile.ZipFile, class_to_id: dict[str, int]) -> list[Record]:
    xml_names = sorted(name for name in archive.namelist() if name.endswith(".xml"))
    records: list[Record] = []
    for xml_name in xml_names:
        root = ET.fromstring(archive.read(xml_name))
        image_file = root.findtext("filename")
        image_id = Path(image_file).stem
        image_name = f"data_train/data_train/VIS/{image_file}"
        size = root.find("size")
        width = int(size.findtext("width"))
        height = int(size.findtext("height"))
        objects = []
        for obj in root.findall("object"):
            cls = class_to_id[obj.findtext("name")]
            box = obj.find("bndbox")
            x1 = float(box.findtext("xmin"))
            y1 = float(box.findtext("ymin"))
            x2 = float(box.findtext("xmax"))
            y2 = float(box.findtext("ymax"))
            objects.append((cls, x1, y1, x2, y2))
        records.append(Record(image_id, image_name, xml_name, width, height, tuple(objects)))
    return records


def split_records(records: list[Record], val_fraction: float, seed: int) -> tuple[list[Record], list[Record]]:
    # Rarity-aware deterministic split: prioritize images containing rare classes,
    # then fill remaining validation capacity from a seeded shuffle.
    rng = random.Random(seed)
    target_n = round(len(records) * val_fraction)
    total = Counter(cls for r in records for cls, *_ in r.objects)
    target = {cls: count * val_fraction for cls, count in total.items()}
    current: Counter[int] = Counter()
    pool = list(records)
    rng.shuffle(pool)

    def score(r: Record) -> float:
        per_image = Counter(cls for cls, *_ in r.objects)
        return sum(
            max(target[cls] - current[cls], 0.0) / max(target[cls], 1.0) * n
            for cls, n in per_image.items()
        )

    val: list[Record] = []
    while pool and len(val) < target_n:
        # Evaluating the whole pool is still cheap for 3k images and gives much
        # better rare-class coverage than a plain random split.
        best_i = max(range(len(pool)), key=lambda i: score(pool[i]))
        chosen = pool.pop(best_i)
        val.append(chosen)
        current.update(cls for cls, *_ in chosen.objects)
    train = pool
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def write_labels(records: list[Record], label_dir: Path) -> None:
    label_dir.mkdir(parents=True, exist_ok=True)
    for r in records:
        lines = []
        for cls, x1, y1, x2, y2 in r.objects:
            xc = ((x1 + x2) / 2.0) / r.width
            yc = ((y1 + y2) / 2.0) / r.height
            bw = (x2 - x1) / r.width
            bh = (y2 - y1) / r.height
            lines.append(f"{cls} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}")
        (label_dir / f"{r.image_id}.txt").write_text("\n".join(lines) + "\n")


def convert_split(zip_path: Path, records: list[Record], out_dir: Path, split: str, bands: tuple[int, int, int], workers: int) -> None:
    image_dir = out_dir / "images" / split
    tasks = []
    for r in records:
        target = image_dir / f"{r.image_id}.png"
        if not target.exists():
            tasks.append((r.image_name, str(target)))
    if not tasks:
        return
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(zip_path), bands)) as ex:
        for i, _ in enumerate(ex.map(_convert_one, tasks, chunksize=8), start=1):
            if i % 100 == 0 or i == len(tasks):
                print(f"{split}: {i}/{len(tasks)} converted", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, default=Path("data/hyperspectral-object-detection-challenge-2026.zip"))
    parser.add_argument("--out", type=Path, default=Path("prepared/pseudo_5_8_13"))
    parser.add_argument("--bands", type=int, nargs=3, default=(5, 8, 13))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    bands = tuple(args.bands)

    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip_path) as archive:
        classes = load_classes(archive)
        class_to_id = {name: i for i, name in enumerate(classes)}
        records = load_records(archive, class_to_id)
        train, val = split_records(records, args.val_fraction, args.seed)
        test_names = sorted(name for name in archive.namelist() if name.startswith("data_test/data_test/VIS/") and name.endswith(".png"))

    split_json = {"train": [r.image_id for r in train], "val": [r.image_id for r in val]}
    (args.out / "split.json").write_text(json.dumps(split_json, indent=2))
    write_labels(train, args.out / "labels" / "train")
    write_labels(val, args.out / "labels" / "val")

    convert_split(args.zip_path, train, args.out, "train", bands, args.workers)
    convert_split(args.zip_path, val, args.out, "val", bands, args.workers)

    test_tasks = []
    test_dir = args.out / "images" / "test"
    for name in test_names:
        target = test_dir / Path(name).name
        if not target.exists():
            test_tasks.append((name, str(target)))
    if test_tasks:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker, initargs=(str(args.zip_path), bands)) as ex:
            for i, _ in enumerate(ex.map(_convert_one, test_tasks, chunksize=8), start=1):
                if i % 100 == 0 or i == len(test_tasks):
                    print(f"test: {i}/{len(test_tasks)} converted", flush=True)

    train_counts = Counter(cls for r in train for cls, *_ in r.objects)
    val_counts = Counter(cls for r in val for cls, *_ in r.objects)
    stats = {
        "bands": bands,
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
