from collections import Counter
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile


ZIP_PATH = Path("data/hyperspectral-object-detection-challenge-2026.zip")


def main() -> None:
    class_counts: Counter[str] = Counter()
    objects_per_image: Counter[int] = Counter()
    sizes: Counter[tuple[int, int]] = Counter()

    with zipfile.ZipFile(ZIP_PATH) as archive:
        xml_names = [name for name in archive.namelist() if name.endswith(".xml")]
        for name in xml_names:
            root = ET.fromstring(archive.read(name))
            objects = root.findall("object")
            objects_per_image[len(objects)] += 1
            class_counts.update(obj.findtext("name") for obj in objects)
            size = root.find("size")
            sizes[(int(size.findtext("width")), int(size.findtext("height")))] += 1

    print(f"images={sum(objects_per_image.values())}")
    print(f"objects={sum(k * v for k, v in objects_per_image.items())}")
    print(f"objects_per_image={dict(sorted(objects_per_image.items()))}")
    print(f"class_counts={dict(class_counts.most_common())}")
    print(f"unique_sizes={len(sizes)}")
    print(f"top_sizes={sizes.most_common(20)}")


if __name__ == "__main__":
    main()
