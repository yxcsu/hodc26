from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
from PIL import Image


_ZIP: zipfile.ZipFile | None = None
_BANDS = (5, 8, 13)


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
            rgb[:, :, c] = (ch - lo) / (hi - lo) * 255.0
        else:
            rgb[:, :, c] = 0.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _convert_one(task: tuple[str, str]) -> str:
    source_name, target_name = task
    assert _ZIP is not None
    raw = np.array(Image.open(io.BytesIO(_ZIP.read(source_name))))
    rgb = pseudo_rgb(raw, _BANDS)
    target = Path(target_name)
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(target, compress_level=2)
    return str(target)


def convert(tasks: list[tuple[str, str]], zip_path: Path, bands: tuple[int, int, int], workers: int, label: str) -> None:
    pending = [task for task in tasks if not Path(task[1]).exists()]
    if not pending:
        print(f"{label}: already complete", flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(zip_path), bands),
    ) as ex:
        for i, _ in enumerate(ex.map(_convert_one, pending, chunksize=8), start=1):
            if i % 100 == 0 or i == len(pending):
                print(f"{label}: {i}/{len(pending)} converted", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", dest="zip_path", type=Path, default=Path("data/hyperspectral-object-detection-challenge-2026.zip"))
    parser.add_argument("--split-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bands", type=int, nargs=3, default=(5, 8, 13))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    bands = tuple(args.bands)

    split = json.loads((args.split_source / "split.json").read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps(split, indent=2))

    for split_name in ("train", "val", "holdout"):
        if split_name not in split:
            continue
        src_labels = args.split_source / "labels" / split_name
        dst_labels = args.out / "labels" / split_name
        if dst_labels.exists():
            shutil.rmtree(dst_labels)
        shutil.copytree(src_labels, dst_labels)
        ids = [str(x) for x in split[split_name]]
        tasks = [
            (
                f"data_train/data_train/VIS/{image_id}.png",
                str(args.out / "images" / split_name / f"{image_id}.png"),
            )
            for image_id in ids
        ]
        convert(tasks, args.zip_path, bands, args.workers, split_name)

    with zipfile.ZipFile(args.zip_path) as archive:
        classes = [line.strip() for line in archive.read("class.txt").decode().splitlines() if line.strip()]
        test_names = sorted(
            name for name in archive.namelist()
            if name.startswith("data_test/data_test/VIS/") and name.endswith(".png")
        )
    test_tasks = [
        (name, str(args.out / "images" / "test" / Path(name).name))
        for name in test_names
    ]
    convert(test_tasks, args.zip_path, bands, args.workers, "test")

    lines = [
        f"path: {args.out.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {i}: {name}" for i, name in enumerate(classes))
    (args.out / "data.yaml").write_text("\n".join(lines) + "\n")
    meta = {
        "bands": bands,
        "split_source": str(args.split_source),
        "train_images": len(split.get("train", [])),
        "val_images": len(split.get("val", [])),
        "holdout_images": len(split.get("holdout", [])),
        "test_images": len(test_names),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
