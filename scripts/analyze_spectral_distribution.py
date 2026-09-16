from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_multispectral_yolo import x2cube


def sample_cube(cube: np.ndarray, rng: np.random.Generator, pixels: int) -> np.ndarray:
    flat = cube.reshape(-1, cube.shape[-1])
    if len(flat) <= pixels:
        return flat.astype(np.float32)
    idx = rng.choice(len(flat), size=pixels, replace=False)
    return flat[idx].astype(np.float32)


def summarize(values: np.ndarray, clip_max: float) -> dict:
    clipped = np.clip(values / clip_max, 0.0, 1.0)
    quantiles = np.quantile(values, [0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995], axis=0)
    return {
        "samples": int(values.shape[0]),
        "raw_mean": values.mean(axis=0).tolist(),
        "raw_std": values.std(axis=0).tolist(),
        "raw_quantiles": {
            "q005": quantiles[0].tolist(),
            "q01": quantiles[1].tolist(),
            "q05": quantiles[2].tolist(),
            "q50": quantiles[3].tolist(),
            "q95": quantiles[4].tolist(),
            "q99": quantiles[5].tolist(),
            "q995": quantiles[6].tolist(),
        },
        "raw_max": values.max(axis=0).tolist(),
        "fraction_ge_clip_max": (values >= clip_max).mean(axis=0).tolist(),
        "clipped_mean": clipped.mean(axis=0).tolist(),
        "clipped_std": clipped.std(axis=0).tolist(),
    }


def ids_from_dir(path: Path) -> list[str]:
    ids = {p.stem for p in path.glob("*.tiff")}
    ids |= {p.stem for p in path.glob("*.tif")}
    ids |= {p.stem for p in path.glob("*.png")}
    return sorted(ids, key=int)


def load_raw(archive: zipfile.ZipFile, image_id: str, is_test: bool) -> np.ndarray:
    prefix = "data_test/data_test/VIS" if is_test else "data_train/data_train/VIS"
    name = f"{prefix}/{image_id}.png"
    raw = np.array(Image.open(io.BytesIO(archive.read(name))))
    return x2cube(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("data/hyperspectral-object-detection-challenge-2026.zip"),
    )
    parser.add_argument("--split-root", type=Path, default=Path("prepared/hsi16_strat_seed42_audited"))
    parser.add_argument("--clip-max", type=float, default=320.0)
    parser.add_argument("--pixels-per-image", type=int, default=512)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--out", type=Path, default=Path("results/spectral_distribution.json"))
    args = parser.parse_args()

    split_ids = {
        "train": ids_from_dir(args.split_root / "images" / "train"),
        "val": ids_from_dir(args.split_root / "images" / "val"),
        "holdout": ids_from_dir(args.split_root / "images" / "holdout"),
        "test": ids_from_dir(args.split_root / "images" / "test"),
    }
    rng = np.random.default_rng(args.seed)
    results: dict[str, dict] = {}
    with zipfile.ZipFile(args.zip_path) as archive:
        for split, ids in split_ids.items():
            chunks: list[np.ndarray] = []
            for i, image_id in enumerate(ids, start=1):
                cube = load_raw(archive, image_id, is_test=(split == "test"))
                chunks.append(sample_cube(cube, rng, args.pixels_per_image))
                if i % 250 == 0 or i == len(ids):
                    print(f"{split}: {i}/{len(ids)}", flush=True)
            values = np.concatenate(chunks, axis=0)
            results[split] = summarize(values, args.clip_max)

    # Simple train-relative shift diagnostics. These are descriptive only; they
    # do not establish sensor drift or causality.
    train_mean = np.asarray(results["train"]["raw_mean"], dtype=np.float64)
    train_std = np.asarray(results["train"]["raw_std"], dtype=np.float64)
    for split in ("val", "holdout", "test"):
        mean = np.asarray(results[split]["raw_mean"], dtype=np.float64)
        standardized = (mean - train_mean) / np.maximum(train_std, 1e-6)
        results[split]["mean_shift_in_train_std"] = standardized.tolist()

    report = {
        "clip_max": args.clip_max,
        "pixels_per_image": args.pixels_per_image,
        "seed": args.seed,
        "splits": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"report={args.out}")


if __name__ == "__main__":
    main()
