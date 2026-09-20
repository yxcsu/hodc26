from __future__ import annotations

import argparse
import os
from pathlib import Path


def replace_symlink(path: Path, target: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        raise RuntimeError(f"Refusing to replace non-symlink path: {path}")
    path.symlink_to(target.resolve(), target_is_directory=target.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the COCO2017 directory layout expected by official DINO."
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Prepared COCO dataset with train/, valid/, and optional test/ directories.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.resolve()
    out = args.out.resolve()
    train = source / "train"
    valid = source / "valid"
    if not train.is_dir() or not valid.is_dir():
        raise RuntimeError(f"Expected train/ and valid/ under {source}")

    replace_symlink(out / "train2017", train)
    replace_symlink(out / "val2017", valid)

    annotations = out / "annotations"
    annotations.mkdir(parents=True, exist_ok=True)
    replace_symlink(
        annotations / "instances_train2017.json",
        train / "_annotations.coco.json",
    )
    replace_symlink(
        annotations / "instances_val2017.json",
        valid / "_annotations.coco.json",
    )

    test = source / "test"
    if test.is_dir():
        replace_symlink(out / "test2017", test)

    print(f"DINO COCO layout ready: {out}")
    for path in (
        out / "train2017",
        out / "val2017",
        annotations / "instances_train2017.json",
        annotations / "instances_val2017.json",
    ):
        print(f"{path} -> {os.readlink(path)}")


if __name__ == "__main__":
    main()
