"""Create a validated four-class RDD2022 YOLO dataset without copying raw data.

The source RDD2022 files remain untouched.  On the same NTFS drive, images
are hard-linked into a derived dataset and only compact YOLO label files are
newly written.  The default balanced selection retains every D40/Pothole
image, caps dominant crack-image groups, and includes true no-defect images.
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import cv2


SOURCE_ROOT = Path(r"D:\21431547\RDD2022_released_through_CRDDC2022\RDD2022")
OUTPUT_ROOT = Path(r"D:\RoadSense_RDD2022_4Class")
CLASS_MAP = {"D40": (0, "Pothole"), "D00": (1, "Longitudinal Crack"), "D10": (2, "Transverse Crack"), "D20": (3, "Alligator Crack")}
CLASS_ID_NAMES = {class_id: name for class_id, name in CLASS_MAP.values()}
SEED = 2026


def _country_roots(source):
    for country in sorted(path for path in source.iterdir() if path.is_dir()):
        root = country / country.name
        if (root / "train" / "images").is_dir() and (root / "train" / "annotations" / "xmls").is_dir():
            yield country.name, root


def _read_record(country, xml_path, image_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width", "0")) if size is not None else 0
    height = float(size.findtext("height", "0")) if size is not None else 0
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid dimensions in {xml_path}")
    boxes, corrected_boxes, invalid_boxes = [], 0, 0
    for node in root.findall("object"):
        source_name = node.findtext("name", "")
        if source_name not in CLASS_MAP:
            continue
        box = node.find("bndbox")
        try:
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        except (AttributeError, TypeError, ValueError):
            invalid_boxes += 1
            continue
        original = (xmin, ymin, xmax, ymax)
        xmin, xmax = max(0.0, xmin), min(width, xmax)
        ymin, ymax = max(0.0, ymin), min(height, ymax)
        if (xmin, ymin, xmax, ymax) != original:
            corrected_boxes += 1
        if xmax <= xmin or ymax <= ymin:
            invalid_boxes += 1
            continue
        class_id, _ = CLASS_MAP[source_name]
        boxes.append((class_id, xmin, ymin, xmax, ymax))
    return {
        "country": country, "xml": xml_path, "image": image_path,
        "width": width, "height": height, "boxes": boxes,
        "classes": {item[0] for item in boxes},
        "corrected_boxes": corrected_boxes, "invalid_boxes": invalid_boxes,
    }


def discover_records(source):
    records, missing_images, malformed = [], [], []
    for country, root in _country_roots(source):
        image_dir, xml_dir = root / "train" / "images", root / "train" / "annotations" / "xmls"
        for xml_path in sorted(xml_dir.glob("*.xml")):
            image_path = image_dir / f"{xml_path.stem}.jpg"
            if not image_path.is_file():
                missing_images.append(str(image_path))
                continue
            try:
                records.append(_read_record(country, xml_path, image_path))
            except (ET.ParseError, ValueError, TypeError) as error:
                malformed.append(f"{xml_path}: {error}")
    if not records:
        raise ValueError("No readable labeled RDD2022 image/XML pairs found.")
    return records, missing_images, malformed


def select_records(records, crack_cap_per_class, background_images, seed):
    """Keep all pothole images and a deterministic balanced crack/background set."""
    rng = random.Random(seed)
    selected = {index for index, record in enumerate(records) if 0 in record["classes"]}
    for class_id in (1, 2, 3):
        candidates = [index for index, record in enumerate(records) if class_id in record["classes"]]
        rng.shuffle(candidates)
        selected.update(candidates[:min(len(candidates), crack_cap_per_class)])
    backgrounds = [index for index, record in enumerate(records) if not record["classes"]]
    rng.shuffle(backgrounds)
    selected.update(backgrounds[:min(len(backgrounds), background_images)])
    return [record for index, record in enumerate(records) if index in selected]


def split_records(records, val_fraction, seed):
    """Split per country and class-presence combination to avoid distribution collapse."""
    groups = defaultdict(list)
    for record in records:
        groups[(record["country"], tuple(sorted(record["classes"])))].append(record)
    rng = random.Random(seed)
    train, val = [], []
    for items in groups.values():
        rng.shuffle(items)
        if len(items) == 1:
            train.extend(items)
            continue
        split = max(1, min(len(items) - 1, round(len(items) * (1 - val_fraction))))
        train.extend(items[:split])
        val.extend(items[split:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def _link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError as error:
        raise OSError(
            f"Could not hard-link {source}. Keep the output on the same NTFS drive or explicitly implement copying."
        ) from error


def _label_lines(record):
    lines = []
    for class_id, xmin, ymin, xmax, ymax in record["boxes"]:
        width, height = record["width"], record["height"]
        xc, yc = (xmin + xmax) / (2 * width), (ymin + ymax) / (2 * height)
        bw, bh = (xmax - xmin) / width, (ymax - ymin) / height
        if not all(0 < value <= 1 for value in (xc, yc, bw, bh)):
            raise ValueError(f"Invalid normalized label for {record['xml']}")
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _write_split(records, split, output):
    for record in records:
        filename = f"{record['country']}_{record['image'].name}"
        _link(record["image"], output / "images" / split / filename)
        label_path = output / "labels" / split / f"{Path(filename).stem}.txt"
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(_label_lines(record)), encoding="utf-8")


def validate_dataset(output):
    report = {}
    train_names, val_names = set(), set()
    for split, destination_names in (("train", train_names), ("val", val_names)):
        images = sorted((output / "images" / split).glob("*.*"))
        labels = sorted((output / "labels" / split).glob("*.txt"))
        image_stems, label_stems = {path.stem for path in images}, {path.stem for path in labels}
        if not images or image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in {split}.")
        destination_names.update(image_stems)
        class_counts, empty_labels = Counter(), 0
        for label in labels:
            contents = label.read_text(encoding="utf-8").splitlines()
            if not contents:
                empty_labels += 1
            for line in contents:
                values = line.split()
                if len(values) != 5:
                    raise ValueError(f"Malformed YOLO label: {label}")
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
                if class_id not in range(4) or not all(0 < value <= 1 for value in coordinates):
                    raise ValueError(f"Invalid YOLO label: {label}")
                class_counts[class_id] += 1
        report[split] = {"images": len(images), "labels": len(labels), "empty_background_labels": empty_labels, "class_instances": dict(class_counts)}
    if train_names & val_names:
        raise ValueError("Train/validation image leakage detected.")
    return report


def write_sanity_check(output):
    previews = []
    for image_path in sorted((output / "images" / "val").glob("*.*"))[:16]:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in (output / "labels" / "val" / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
            class_id, xc, yc, bw, bh = map(float, line.split())
            x1, y1 = int((xc - bw / 2) * width), int((yc - bh / 2) * height)
            x2, y2 = int((xc + bw / 2) * width), int((yc + bh / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, CLASS_ID_NAMES[int(class_id)], (x1, max(16, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, .42, (0, 255, 0), 1)
        previews.append(cv2.resize(image, (320, 180)))
    if previews:
        while len(previews) % 4:
            previews.append(previews[-1] * 0)
        cv2.imwrite(str(output / "sanity_check.jpg"), cv2.vconcat([cv2.hconcat(previews[index:index + 4]) for index in range(0, len(previews), 4)]))


def main():
    parser = argparse.ArgumentParser(description="Prepare balanced four-class RDD2022 YOLO labels.")
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--crack-cap-per-class", type=int, default=3000)
    parser.add_argument("--background-images", type=int, default=1000)
    parser.add_argument("--val-fraction", type=float, default=.10)
    parser.add_argument("--finalize-existing", action="store_true", help="Validate and finish a complete set created before optional reporting failed.")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(f"RDD2022 source not found: {args.source}")
    existing_output = args.output.exists() and any(args.output.iterdir())
    if existing_output and not args.finalize_existing:
        raise FileExistsError(f"Refusing to mix data into existing output: {args.output}")
    if not 0 < args.val_fraction < 1 or args.crack_cap_per_class < 1 or args.background_images < 0:
        raise ValueError("Invalid split or selection values.")
    records, missing_images, malformed = discover_records(args.source)
    selected = select_records(records, args.crack_cap_per_class, args.background_images, SEED)
    train, val = split_records(selected, args.val_fraction, SEED)
    if not existing_output:
        _write_split(train, "train", args.output)
        _write_split(val, "val", args.output)
        (args.output / "data.yaml").write_text(
            f"path: {str(args.output).replace('\\', '/')}\ntrain: images/train\nval: images/val\nnc: 4\nnames:\n"
            + "".join(f"  {class_id}: {name}\n" for class_id, name in ((0, "Pothole"), (1, "Longitudinal Crack"), (2, "Transverse Crack"), (3, "Alligator Crack"))),
            encoding="utf-8",
        )
    validation = validate_dataset(args.output)
    write_sanity_check(args.output)
    report = {
        "source": str(args.source), "output": str(args.output), "source_annotation_format": "Pascal VOC XML",
        "class_mapping": {str(key): value for key, value in ((0, "Pothole"), (1, "Longitudinal Crack"), (2, "Transverse Crack"), (3, "Alligator Crack"))},
        "source_records": len(records), "selected_records": len(selected), "missing_images": missing_images,
        "malformed_annotations": malformed, "corrected_boxes": sum(item["corrected_boxes"] for item in selected),
        "discarded_invalid_boxes": sum(item["invalid_boxes"] for item in selected), "validation": validation,
        "selection": {"all_pothole_images_retained": True, "crack_cap_per_class": args.crack_cap_per_class, "background_images": args.background_images, "seed": SEED},
    }
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
