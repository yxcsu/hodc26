from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class UnionFind:
    parent: list[int]
    rank: list[int]

    @classmethod
    def create(cls, n: int) -> "UnionFind":
        return cls(list(range(n)), [0] * n)

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def decode_composite(path: Path) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.uint8)
    ok, frames = cv2.imdecodemulti(raw, cv2.IMREAD_UNCHANGED)
    if not ok or not frames:
        raise RuntimeError(f"Could not decode {path}")
    stack = np.stack(frames, axis=0).astype(np.float32)
    if stack.ndim != 3:
        raise RuntimeError(f"Expected CxHxW stack, got {stack.shape} for {path}")
    if stack.shape[0] >= 13:
        # Same representative bands used by the pseudo-RGB route, averaged to a
        # luminance-like image. Grouping is only for near-duplicate discovery;
        # it does not alter model inputs.
        image = stack[[4, 7, 12]].mean(axis=0)
    else:
        image = stack.mean(axis=0)
    lo, hi = np.percentile(image, [2.0, 98.0])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0)


def phash64(image: np.ndarray) -> int:
    resized = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    low = dct[:8, :8].copy()
    values = low.reshape(-1)
    median = np.median(values[1:])
    bits = values > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def scene_feature(image: np.ndarray, size: int = 24) -> np.ndarray:
    # Heavy downsampling suppresses object-level details and emphasizes scene
    # layout. Standardization makes the feature less sensitive to global gain.
    small = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= float(small.mean())
    std = float(small.std())
    if std > 1e-6:
        small /= std
    vec = small.reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-6:
        vec /= norm
    return vec


def collect_images(root: Path) -> list[Path]:
    paths: dict[str, Path] = {}
    for split in ("train", "val"):
        for ext in ("*.tiff", "*.tif", "*.png"):
            for path in (root / "images" / split).glob(ext):
                paths[path.stem] = path
    return [paths[key] for key in sorted(paths, key=int)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("prepared/hsi16_clip320"))
    parser.add_argument("--out", type=Path, default=Path("results/scene_groups.json"))
    parser.add_argument("--pairs-out", type=Path, default=Path("results/scene_group_candidates.csv"))
    parser.add_argument("--cos-threshold", type=float, default=0.985)
    parser.add_argument("--strict-cos", type=float, default=0.997)
    parser.add_argument("--hamming", type=int, default=12)
    parser.add_argument("--candidate-cos", type=float, default=0.97)
    parser.add_argument("--max-candidate-pairs", type=int, default=10000)
    parser.add_argument("--chunk", type=int, default=128)
    args = parser.parse_args()

    image_paths = collect_images(args.source)
    if not image_paths:
        raise RuntimeError(f"No images found under {args.source}")

    ids: list[str] = []
    hashes: list[int] = []
    features: list[np.ndarray] = []
    for i, path in enumerate(image_paths, start=1):
        image = decode_composite(path)
        ids.append(path.stem)
        hashes.append(phash64(image))
        features.append(scene_feature(image))
        if i % 250 == 0 or i == len(image_paths):
            print(f"features={i}/{len(image_paths)}", flush=True)

    feats = np.stack(features, axis=0).astype(np.float32)
    uf = UnionFind.create(len(ids))
    candidate_pairs: list[tuple[float, int, str, str]] = []
    linked = 0
    for start in range(0, len(ids), args.chunk):
        stop = min(len(ids), start + args.chunk)
        sim = feats[start:stop] @ feats.T
        for local_i, row in enumerate(sim):
            i = start + local_i
            js = np.flatnonzero(row >= args.candidate_cos)
            for j in js:
                j = int(j)
                if j <= i:
                    continue
                cosine = float(row[j])
                hamming = int((hashes[i] ^ hashes[j]).bit_count())
                candidate_pairs.append((cosine, hamming, ids[i], ids[j]))
                if cosine >= args.strict_cos or (cosine >= args.cos_threshold and hamming <= args.hamming):
                    uf.union(i, j)
                    linked += 1

    roots: dict[int, list[str]] = {}
    for i, image_id in enumerate(ids):
        roots.setdefault(uf.find(i), []).append(image_id)

    groups = sorted(
        (sorted(members, key=int) for members in roots.values()),
        key=lambda members: (int(members[0]), len(members)),
    )
    image_to_group: dict[str, str] = {}
    group_rows: list[dict[str, object]] = []
    for idx, members in enumerate(groups):
        group_id = f"g{idx:04d}"
        for image_id in members:
            image_to_group[image_id] = group_id
        group_rows.append({"group_id": group_id, "size": len(members), "image_ids": members})

    non_singletons = [g for g in group_rows if int(g["size"]) > 1]
    report = {
        "source": str(args.source),
        "image_count": len(ids),
        "group_count": len(groups),
        "non_singleton_group_count": len(non_singletons),
        "images_in_non_singletons": sum(int(g["size"]) for g in non_singletons),
        "largest_group": max((int(g["size"]) for g in group_rows), default=0),
        "link_count": linked,
        "thresholds": {
            "cos_threshold": args.cos_threshold,
            "strict_cos": args.strict_cos,
            "hamming": args.hamming,
            "candidate_cos": args.candidate_cos,
        },
        "image_to_group": image_to_group,
        "groups": group_rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    candidate_pairs.sort(key=lambda x: (-x[0], x[1], int(x[2]), int(x[3])))
    args.pairs_out.parent.mkdir(parents=True, exist_ok=True)
    with args.pairs_out.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cosine", "phash_hamming", "image_id_a", "image_id_b", "linked"])
        for cosine, hamming, a, b in candidate_pairs[: args.max_candidate_pairs]:
            is_linked = args.strict_cos <= cosine or (cosine >= args.cos_threshold and hamming <= args.hamming)
            writer.writerow([f"{cosine:.8f}", hamming, a, b, int(is_linked)])

    print(json.dumps({k: report[k] for k in report if k not in {"image_to_group", "groups"}}, indent=2))


if __name__ == "__main__":
    main()
