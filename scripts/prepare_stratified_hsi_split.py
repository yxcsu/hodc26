from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def safe_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        dst.symlink_to(src.resolve())


def collect(source: Path) -> tuple[list[str], np.ndarray, np.ndarray, dict[str, tuple[Path, Path]]]:
    records: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val"):
        for image_path in (source / "images" / split).glob("*.tiff"):
            label_path = source / "labels" / split / f"{image_path.stem}.txt"
            records[image_path.stem] = (image_path, label_path)

    ids = sorted(records, key=int)
    obj = np.zeros((len(ids), 18), dtype=np.int32)
    presence = np.zeros_like(obj)
    for i, image_id in enumerate(ids):
        label_path = records[image_id][1]
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:])
            x1, y1 = xc - bw / 2.0, yc - bh / 2.0
            x2, y2 = xc + bw / 2.0, yc + bh / 2.0
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(1.0, x2), min(1.0, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            obj[i, cls] += 1
        presence[i] = (obj[i] > 0).astype(np.int32)
    return ids, obj, presence, records


def sanitize_label_file(src: Path, dst: Path, image_id: str, report: dict[str, list[dict]]) -> None:
    """Drop degenerate YOLO boxes and clip boxes to [0, 1] reproducibly.

    The original challenge XMLs contain a small number of zero-area and
    out-of-bounds boxes. Those issues survive conversion into YOLO labels, so
    normalize them here before materializing the audited split.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        dst.write_text("")
        return

    clean_lines: list[str] = []
    for line_no, line in enumerate(src.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            report.setdefault("malformed", []).append(
                {"image_id": image_id, "line": line_no, "value": line}
            )
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = map(float, parts[1:])
        x1, y1 = xc - bw / 2.0, yc - bh / 2.0
        x2, y2 = xc + bw / 2.0, yc + bh / 2.0
        cx1, cy1 = max(0.0, x1), max(0.0, y1)
        cx2, cy2 = min(1.0, x2), min(1.0, y2)
        if cx2 <= cx1 or cy2 <= cy1:
            report.setdefault("dropped_degenerate", []).append(
                {"image_id": image_id, "line": line_no, "value": line}
            )
            continue
        if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
            report.setdefault("clipped", []).append(
                {"image_id": image_id, "line": line_no, "before": [x1, y1, x2, y2], "after": [cx1, cy1, cx2, cy2]}
            )
        nxc = (cx1 + cx2) / 2.0
        nyc = (cy1 + cy2) / 2.0
        nbw = cx2 - cx1
        nbh = cy2 - cy1
        clean_lines.append(f"{cls} {nxc:.8f} {nyc:.8f} {nbw:.8f} {nbh:.8f}")
    dst.write_text("\n".join(clean_lines) + ("\n" if clean_lines else ""))


def split_score(indices: np.ndarray, obj: np.ndarray, presence: np.ndarray, frac: float) -> float:
    target_obj = obj.sum(axis=0) * frac
    target_pre = presence.sum(axis=0) * frac
    got_obj = obj[indices].sum(axis=0)
    got_pre = presence[indices].sum(axis=0)
    # Relative errors with a small denominator floor keep rare classes important
    # without allowing a single tiny class to dominate the whole objective.
    obj_err = ((got_obj - target_obj) / np.maximum(target_obj, 3.0)) ** 2
    pre_err = ((got_pre - target_pre) / np.maximum(target_pre, 3.0)) ** 2
    return float(obj_err.mean() + pre_err.mean())


def choose_split(
    obj: np.ndarray,
    presence: np.ndarray,
    seed: int,
    val_size: int,
    holdout_size: int,
    trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    n = len(obj)
    frac_val = val_size / n
    frac_hold = holdout_size / n
    best: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None
    for _ in range(trials):
        perm = rng.permutation(n)
        val = perm[:val_size]
        hold = perm[val_size : val_size + holdout_size]
        train = perm[val_size + holdout_size :]
        score = split_score(val, obj, presence, frac_val) + split_score(hold, obj, presence, frac_hold)
        if best is None or score < best[3]:
            best = (train.copy(), val.copy(), hold.copy(), score)
    assert best is not None
    return best


def materialize(
    out: Path,
    split: str,
    indices: np.ndarray,
    ids: list[str],
    records: dict[str, tuple[Path, Path]],
    audit_report: dict[str, list[dict]],
) -> None:
    for idx in indices:
        image_id = ids[int(idx)]
        image_src, label_src = records[image_id]
        safe_link(image_src, out / "images" / split / image_src.name)
        sanitize_label_file(label_src, out / "labels" / split / label_src.name, image_id, audit_report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("prepared/hsi16_clip320"))
    parser.add_argument("--out", type=Path, default=Path("prepared/hsi16_strat_seed42"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=int, default=300)
    parser.add_argument("--holdout-size", type=int, default=300)
    parser.add_argument("--trials", type=int, default=5000)
    args = parser.parse_args()

    # This script writes corrected labels rather than linking them, so require
    # a fresh output directory to avoid accidentally overwriting an existing
    # split through symlinks/hardlinks.
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError(f"Output directory is not empty: {args.out}")

    ids, obj, presence, records = collect(args.source)
    train, val, hold, score = choose_split(
        obj, presence, args.seed, args.val_size, args.holdout_size, args.trials
    )

    audit_report: dict[str, list[dict]] = {}
    materialize(args.out, "train", train, ids, records, audit_report)
    materialize(args.out, "val", val, ids, records, audit_report)
    materialize(args.out, "holdout", hold, ids, records, audit_report)
    # Reuse the challenge test set unchanged.
    for image_path in (args.source / "images" / "test").glob("*.tiff"):
        safe_link(image_path, args.out / "images" / "test" / image_path.name)

    def stats(indices: np.ndarray) -> dict:
        return {
            "images": int(len(indices)),
            "objects": obj[indices].sum(axis=0).tolist(),
            "presence": presence[indices].sum(axis=0).tolist(),
        }

    report = {
        "seed": args.seed,
        "score": score,
        "train": stats(train),
        "val": stats(val),
        "holdout": stats(hold),
        "total_objects": obj.sum(axis=0).tolist(),
        "total_presence": presence.sum(axis=0).tolist(),
        "annotation_audit": {
            "dropped_degenerate": len(audit_report.get("dropped_degenerate", [])),
            "clipped": len(audit_report.get("clipped", [])),
            "malformed": len(audit_report.get("malformed", [])),
            "affected_image_ids": sorted(
                {
                    entry["image_id"]
                    for entries in audit_report.values()
                    for entry in entries
                },
                key=int,
            ),
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "split_report.json").write_text(json.dumps(report, indent=2))
    (args.out / "annotation_audit.json").write_text(json.dumps(audit_report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
