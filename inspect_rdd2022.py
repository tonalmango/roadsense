"""Read-only structural and label audit for an extracted RDD2022 dataset."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_SOURCE = Path(r"D:\21431547\RDD2022_released_through_CRDDC2022\RDD2022")
RDD_CLASSES = ("D00", "D10", "D20", "D40")


def inspect_dataset(source):
    """Return an annotation-based dataset report without changing source files."""
    source = Path(source)
    if not source.is_dir():
        raise FileNotFoundError(f"RDD2022 directory not found: {source}")
    report = {
        "source": str(source), "annotation_format": "Pascal VOC XML",
        "known_classes": list(RDD_CLASSES), "countries": {},
        "class_instances": Counter(), "class_images": Counter(),
        "missing_images": [], "malformed_annotations": [], "invalid_boxes": 0,
        "publisher_train_images": 0, "publisher_train_annotations": 0,
        "publisher_test_images": 0,
    }
    for country in sorted(path for path in source.iterdir() if path.is_dir()):
        # RDD2022 country folders contain another country-named release folder.
        candidates = [country, country / country.name]
        country_root = next((path for path in candidates if (path / "train").is_dir()), None)
        if country_root is None:
            continue
        image_dir = country_root / "train" / "images"
        xml_dir = country_root / "train" / "annotations" / "xmls"
        test_dir = country_root / "test" / "images"
        xml_paths = sorted(xml_dir.glob("*.xml")) if xml_dir.is_dir() else []
        train_images = sorted(path for path in image_dir.glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) if image_dir.is_dir() else []
        test_images = sorted(path for path in test_dir.glob("*.*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}) if test_dir.is_dir() else []
        country_counts = Counter()
        country_image_classes = Counter()
        report["publisher_train_images"] += len(train_images)
        report["publisher_train_annotations"] += len(xml_paths)
        report["publisher_test_images"] += len(test_images)
        for xml_path in xml_paths:
            image_path = image_dir / f"{xml_path.stem}.jpg"
            if not image_path.is_file():
                report["missing_images"].append(str(image_path))
                continue
            try:
                root = ET.parse(xml_path).getroot()
                size = root.find("size")
                width = float(size.findtext("width", "0")) if size is not None else 0
                height = float(size.findtext("height", "0")) if size is not None else 0
                if width <= 0 or height <= 0:
                    raise ValueError("missing or invalid image dimensions")
            except (ET.ParseError, ValueError, TypeError) as error:
                report["malformed_annotations"].append(f"{xml_path}: {error}")
                continue
            classes_in_image = set()
            for object_node in root.findall("object"):
                name = object_node.findtext("name", "")
                if name not in RDD_CLASSES:
                    continue
                box = object_node.find("bndbox")
                try:
                    xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
                    xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
                except (AttributeError, TypeError, ValueError):
                    report["invalid_boxes"] += 1
                    continue
                if xmin < 0 or ymin < 0 or xmax > width or ymax > height or xmax <= xmin or ymax <= ymin:
                    report["invalid_boxes"] += 1
                    continue
                country_counts[name] += 1
                classes_in_image.add(name)
            for name in classes_in_image:
                country_image_classes[name] += 1
        report["class_instances"].update(country_counts)
        report["class_images"].update(country_image_classes)
        report["countries"][country.name] = {
            "root": str(country_root), "train_images": len(train_images),
            "train_annotations": len(xml_paths), "test_images": len(test_images),
            "class_instances": dict(country_counts), "class_images": dict(country_image_classes),
        }
    report["class_instances"] = dict(report["class_instances"])
    report["class_images"] = dict(report["class_images"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Inspect RDD2022 without changing it.")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--json", type=Path, help="Optional report output path.")
    parser.add_argument("--quiet", action="store_true", help="Write only the optional JSON report.")
    args = parser.parse_args()
    report = inspect_dataset(args.source)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not args.quiet:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
