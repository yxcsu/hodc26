from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zip",
        dest="zip_path",
        type=Path,
        default=Path("data/hyperspectral-object-detection-challenge-2026.zip"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("prepared/annotation_audit.json"),
    )
    args = parser.parse_args()

    issue_images: dict[str, set[str]] = defaultdict(set)
    issue_details: dict[str, list[dict]] = defaultdict(list)
    class_counts: Counter[str] = Counter()
    object_counts: Counter[int] = Counter()

    with zipfile.ZipFile(args.zip_path) as archive:
        classes = {
            x.strip()
            for x in archive.read("class.txt").decode("utf-8").splitlines()
            if x.strip()
        }
        names = set(archive.namelist())
        xml_names = sorted(n for n in names if n.lower().endswith(".xml"))

        for xml_name in xml_names:
            root = ET.fromstring(archive.read(xml_name))
            image_file = (root.findtext("filename") or "").strip()
            image_id = Path(image_file).stem or Path(xml_name).stem
            xml_id = Path(xml_name).stem
            expected_image = f"data_train/data_train/VIS/{image_file}"

            def flag(kind: str, **detail: object) -> None:
                issue_images[kind].add(image_id)
                issue_details[kind].append({"image_id": image_id, "xml": xml_name, **detail})

            if not image_file:
                flag("missing_filename")
            elif expected_image not in names:
                flag("missing_image", filename=image_file)
            if image_file and xml_id != image_id:
                flag("xml_filename_mismatch", xml_id=xml_id, filename=image_file)

            size = root.find("size")
            if size is None:
                flag("missing_size")
                continue
            try:
                width = float(size.findtext("width") or "nan")
                height = float(size.findtext("height") or "nan")
            except ValueError:
                flag("invalid_size_text")
                continue
            if not (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0):
                flag("invalid_size", width=width, height=height)
                continue

            seen_boxes: Counter[tuple] = Counter()
            objects = root.findall("object")
            object_counts[len(objects)] += 1
            for obj_index, obj in enumerate(objects):
                cls = (obj.findtext("name") or "").strip()
                class_counts[cls] += 1
                if cls not in classes:
                    flag("unknown_class", object_index=obj_index, class_name=cls)

                box = obj.find("bndbox")
                if box is None:
                    flag("missing_box", object_index=obj_index, class_name=cls)
                    continue
                try:
                    x1 = float(box.findtext("xmin") or "nan")
                    y1 = float(box.findtext("ymin") or "nan")
                    x2 = float(box.findtext("xmax") or "nan")
                    y2 = float(box.findtext("ymax") or "nan")
                except ValueError:
                    flag("invalid_box_text", object_index=obj_index, class_name=cls)
                    continue

                coords = (x1, y1, x2, y2)
                if not all(math.isfinite(v) for v in coords):
                    flag("nonfinite_box", object_index=obj_index, class_name=cls, box=coords)
                    continue
                if x2 <= x1 or y2 <= y1:
                    flag("degenerate_box", object_index=obj_index, class_name=cls, box=coords)
                if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
                    clipped = (
                        min(max(x1, 0.0), width),
                        min(max(y1, 0.0), height),
                        min(max(x2, 0.0), width),
                        min(max(y2, 0.0), height),
                    )
                    flag(
                        "out_of_bounds_box",
                        object_index=obj_index,
                        class_name=cls,
                        box=coords,
                        size=(width, height),
                        clipped=clipped,
                    )
                if (x2 - x1) < 1 or (y2 - y1) < 1:
                    flag("subpixel_box", object_index=obj_index, class_name=cls, box=coords)

                key = (cls, round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))
                seen_boxes[key] += 1

            dupes = [box for box, count in seen_boxes.items() if count > 1]
            if dupes:
                flag("duplicate_box", duplicates=dupes)

    all_issue_ids = sorted(set().union(*issue_images.values()), key=lambda x: int(x) if x.isdigit() else x)
    report = {
        "xml_images": len(xml_names),
        "objects": sum(n * count for n, count in object_counts.items()),
        "objects_per_image": dict(sorted(object_counts.items())),
        "class_counts": dict(class_counts),
        "issue_image_count": len(all_issue_ids),
        "issue_image_ids": all_issue_ids,
        "issues": {
            kind: {
                "image_count": len(ids),
                "image_ids": sorted(ids, key=lambda x: int(x) if x.isdigit() else x),
                "details": issue_details[kind],
            }
            for kind, ids in sorted(issue_images.items())
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({
        "xml_images": report["xml_images"],
        "objects": report["objects"],
        "issue_image_count": report["issue_image_count"],
        "issue_image_ids": report["issue_image_ids"],
        "issue_counts": {k: v["image_count"] for k, v in report["issues"].items()},
    }, indent=2))


if __name__ == "__main__":
    main()
