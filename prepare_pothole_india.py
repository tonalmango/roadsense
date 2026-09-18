"""Create a validated one-class pothole dataset from RDD2022 India.

This preparation utility never writes to RDD2022. It creates hard links to
the original images where NTFS supports them, and writes new YOLO labels only
under D:\\RoadSense_Pothole_India. It does not start model training.
"""

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2


INDIA_ROOT = Path(r"D:\21431547\RDD2022_released_through_CRDDC2022\RDD2022\India\India")
OUTPUT_ROOT = Path(r"D:\RoadSense_Pothole_India")
POTHOLE_CLASS = "D40"
CRACK_CLASSES = {"D00", "D10", "D20"}
RANDOM_SEED = 2026


def parse_xml(xml_path):
    """Read only genuine D40 boxes and report other source classes."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"Missing image size: {xml_path}")
    width, height = int(size.findtext("width", "0")), int(size.findtext("height", "0"))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {xml_path}")
    potholes, classes = [], []
    for item in root.findall("object"):
        name = item.findtext("name", "")
        classes.append(name)
        if name != POTHOLE_CLASS:
            continue
        box = item.find("bndbox")
        if box is None:
            continue
        try:
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid D40 box in {xml_path}") from error
        xmin, xmax = max(0.0, xmin), min(float(width), xmax)
        ymin, ymax = max(0.0, ymin), min(float(height), ymax)
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid/empty D40 box in {xml_path}")
        potholes.append((xmin, ymin, xmax, ymax))
    return width, height, potholes, set(classes)


def yolo_lines(boxes, width, height):
    lines = []
    for xmin, ymin, xmax, ymax in boxes:
        xc, yc = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height
        bw, bh = (xmax - xmin) / width, (ymax - ymin) / height
        if not all(0 < value <= 1 for value in (xc, yc, bw, bh)):
            raise ValueError("Generated normalized bounding box is outside (0, 1].")
        lines.append(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
    return lines


def link_image(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError as error:
        raise OSError(
            "Could not create an image hard link. Output must be on the same NTFS "
            "volume as RDD2022; do not use --copy unless you explicitly accept duplication."
        ) from error


def write_item(item, split, output_root):
    image_path, width, height, boxes = item
    image_destination = output_root / "images" / split / image_path.name
    label_destination = output_root / "labels" / split / f"{image_path.stem}.txt"
    link_image(image_path, image_destination)
    label_destination.parent.mkdir(parents=True, exist_ok=True)
    label_destination.write_text("\n".join(yolo_lines(boxes, width, height)), encoding="utf-8")


def validate_yolo_dataset(output_root):
    report = {}
    for split in ("train", "val"):
        image_paths = sorted((output_root / "images" / split).glob("*.jpg"))
        label_paths = sorted((output_root / "labels" / split).glob("*.txt"))
        image_stems = {path.stem for path in image_paths}
        label_stems = {path.stem for path in label_paths}
        if image_stems != label_stems:
            raise ValueError(f"Image/label mismatch in {split}.")
        potholes = 0
        for label_path in label_paths:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line:
                    continue
                values = line.split()
                if len(values) != 5 or values[0] != "0":
                    raise ValueError(f"Invalid YOLO row: {label_path}")
                coords = [float(value) for value in values[1:]]
                if not all(0 < value <= 1 for value in coords):
                    raise ValueError(f"Out-of-range YOLO row: {label_path}")
                potholes += 1
        report[split] = {"images": len(image_paths), "labels": len(label_paths), "pothole_instances": potholes}
    if set((output_root / "images" / "train").glob("*.jpg")) & set((output_root / "images" / "val").glob("*.jpg")):
        raise ValueError("Train/validation image leakage detected.")
    return report


def write_sanity_check(output_root):
    samples = sorted((output_root / "images" / "val").glob("*.jpg"))[:12]
    previews = []
    for image_path in samples:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        for line in (output_root / "labels" / "val" / f"{image_path.stem}.txt").read_text().splitlines():
            values = line.split()
            if len(values) != 5:
                continue
            _, xc, yc, bw, bh = map(float, values)
            x1, y1 = int((xc - bw / 2) * width), int((yc - bh / 2) * height)
            x2, y2 = int((xc + bw / 2) * width), int((yc + bh / 2) * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        previews.append(cv2.resize(image, (320, 180)))
    if previews:
        rows = [previews[index:index + 4] for index in range(0, len(previews), 4)]
        while len(rows[-1]) < 4:
            rows[-1].append(rows[-1][-1] * 0)
        cv2.imwrite(str(output_root / "sanity_check.jpg"), cv2.vconcat([cv2.hconcat(row) for row in rows]))


def main():
    parser = argparse.ArgumentParser(description="Prepare RDD2022 India D40 potholes as one YOLO class.")
    parser.add_argument("--source", type=Path, default=INDIA_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--negative-ratio", type=float, default=1.0,
                        help="Crack-only negatives per positive pothole image; 0 disables negatives.")
    args = parser.parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between 0 and 1.")
    xml_directory = args.source / "train" / "annotations" / "xmls"
    image_directory = args.source / "train" / "images"
    if not xml_directory.is_dir() or not image_directory.is_dir():
        raise FileNotFoundError("Expected India train/images and train/annotations/xmls were not found.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to mix outputs in existing directory: {args.output}")

    positives, negatives = [], []
    stats = Counter()
    for xml_path in sorted(xml_directory.glob("*.xml")):
        image_path = image_directory / f"{xml_path.stem}.jpg"
        if not image_path.is_file():
            raise FileNotFoundError(f"Image missing for annotation: {xml_path.name}")
        width, height, boxes, classes = parse_xml(xml_path)
        stats["labeled_images"] += 1
        if boxes:
            positives.append((image_path, width, height, boxes))
            stats["pothole_images"] += 1
            stats["pothole_instances"] += len(boxes)
            if classes & CRACK_CLASSES:
                stats["pothole_and_crack_images"] += 1
        elif classes & CRACK_CLASSES:
            negatives.append((image_path, width, height, []))
            stats["crack_only_images"] += 1
        else:
            stats["empty_images"] += 1

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    retained_negatives = negatives[:min(len(negatives), round(len(positives) * max(0, args.negative_ratio)))]
    split_positive = max(1, round(len(positives) * (1 - args.val_fraction)))
    split_negative = round(len(retained_negatives) * (1 - args.val_fraction))
    split_items = {
        "train": positives[:split_positive] + retained_negatives[:split_negative],
        "val": positives[split_positive:] + retained_negatives[split_negative:],
    }
    for split, items in split_items.items():
        rng.shuffle(items)
        for item in items:
            write_item(item, split, args.output)

    (args.output / "data.yaml").write_text(
        f"path: {str(args.output).replace('\\\\', '/')}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: pothole\n",
        encoding="utf-8",
    )
    validation = validate_yolo_dataset(args.output)
    write_sanity_check(args.output)
    report = {
        "source": str(args.source), "output": str(args.output), "class": "pothole", "source_class": "D40",
        "source_statistics": dict(stats), "retained_crack_only_negatives": len(retained_negatives),
        "validation": validation, "random_seed": RANDOM_SEED, "val_fraction": args.val_fraction,
    }
    (args.output / "preparation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
