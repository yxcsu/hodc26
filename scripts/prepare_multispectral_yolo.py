from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import io
import json
from pathlib import Path
import shutil
import zipfile

import cv2
import numpy as np
from PIL import Image


_ZIP: zipfile.ZipFile | None = None
_CLIP_MAX = 320.0
_PSEUDO_BANDS: tuple[int, int, int] | None = None


def _init_worker(
    zip_path: str,
    clip_max: float,
    pseudo_bands: tuple[int, int, int] | None,
) -> None:
    global _ZIP, _CLIP_MAX, _PSEUDO_BANDS
    _ZIP = zipfile.ZipFile(zip_path)
    _CLIP_MAX = clip_max
    _PSEUDO_BANDS = pseudo_bands


def x2cube(img: np.ndarray) -> np.ndarray:
    h = (img.shape[0] // 4) * 4
    w = (img.shape[1] // 4) * 4
    img = img[:h, :w]
    return img.reshape(h // 4, 4, w // 4, 4).transpose(0, 2, 1, 3).reshape(h // 4, w // 4, 16)


def scale_cube(cube: np.ndarray, clip_max: float) -> np.ndarray:
    # Use one fixed global scale for all bands so cross-band intensity ratios
    # remain meaningful. The sensor values are uint16 but observed data are
    # concentrated in roughly 0..320.
    out = cube.astype(np.float32) * (255.0 / clip_max)
    return np.clip(out, 0, 255).astype(np.uint8)


def pseudo_rgb(cube: np.ndarray, bands: tuple[int, int, int]) -> np.ndarray:
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
    raw_cube = x2cube(raw)
    cube = scale_cube(raw_cube, _CLIP_MAX)
    if _PSEUDO_BANDS is not None:
        cube = np.concatenate([cube, pseudo_rgb(raw_cube, _PSEUDO_BANDS)], axis=2)
    out_path = Path(out_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwritemulti(str(out_path), [cube[..., i] for i in range(cube.shape[2])])
    if not ok:
        raise RuntimeError(f"Failed to write {out_path}")
    return str(out_path)


def convert_tasks(
    zip_path: Path,
    tasks: list[tuple[str, str]],
    clip_max: float,
    pseudo_bands: tuple[int, int, int] | None,
    workers: int,
    label: str,
) -> None:
    pending = [(src, dst) for src, dst in tasks if not Path(dst).exists()]
    if not pending:
        print(f"{label}: already complete", flush=True)
        return
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(zip_path), clip_max, pseudo_bands),
    ) as ex:
        for i, _ in enumerate(ex.map(_convert_one, pending, chunksize=8), start=1):
            if i % 100 == 0 or i == len(pending):
                print(f"{label}: {i}/{len(pending)} converted", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("data/hyperspectral-object-detection-challenge-2026.zip"),
    )
    parser.add_argument("--split-source", type=Path, default=Path("prepared/pseudo_5_8_13"))
    parser.add_argument("--out", type=Path, default=Path("prepared/hsi16_clip320"))
    parser.add_argument("--clip-max", type=float, default=320.0)
    parser.add_argument("--pseudo-bands", type=int, nargs=3, default=None)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    pseudo_bands = tuple(args.pseudo_bands) if args.pseudo_bands is not None else None
    channels = 16 + (3 if pseudo_bands is not None else 0)

    split = json.loads((args.split_source / "split.json").read_text())
    train_ids = [str(x) for x in split["train"]]
    val_ids = [str(x) for x in split["val"]]

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split.json").write_text(json.dumps(split, indent=2))

    for split_name in ("train", "val"):
        src = args.split_source / "labels" / split_name
        dst = args.out / "labels" / split_name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)

    with zipfile.ZipFile(args.zip_path) as archive:
        classes = [x.strip() for x in archive.read("class.txt").decode().splitlines() if x.strip()]
        test_names = sorted(
            n
            for n in archive.namelist()
            if n.startswith("data_test/data_test/VIS/") and n.endswith(".png")
        )

    train_tasks = [
        (f"data_train/data_train/VIS/{image_id}.png", str(args.out / "images/train" / f"{image_id}.tiff"))
        for image_id in train_ids
    ]
    val_tasks = [
        (f"data_train/data_train/VIS/{image_id}.png", str(args.out / "images/val" / f"{image_id}.tiff"))
        for image_id in val_ids
    ]
    test_tasks = [
        (name, str(args.out / "images/test" / f"{Path(name).stem}.tiff"))
        for name in test_names
    ]

    convert_tasks(args.zip_path, train_tasks, args.clip_max, pseudo_bands, args.workers, "train")
    convert_tasks(args.zip_path, val_tasks, args.clip_max, pseudo_bands, args.workers, "val")
    convert_tasks(args.zip_path, test_tasks, args.clip_max, pseudo_bands, args.workers, "test")

    abs_out = args.out.resolve()
    yaml_lines = [
        f"path: {abs_out}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"channels: {channels}",
        "names:",
    ]
    yaml_lines.extend(f"  {i}: {name}" for i, name in enumerate(classes))
    (args.out / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    meta = {
        "clip_max": args.clip_max,
        "channels": channels,
        "pseudo_bands": pseudo_bands,
        "train_images": len(train_tasks),
        "val_images": len(val_tasks),
        "test_images": len(test_tasks),
        "source_split": str(args.split_source),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
