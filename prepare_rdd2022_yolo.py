"""Prepare extracted RDD2022 Pascal VOC labels for local YOLO11 training.

This script never edits ``D:\\21431547``. It requires the nested archives to
be extracted first, then creates a separate ``D:\\21431547_yolo`` directory.
Images are hard-linked by default (no duplicate image bytes on the same NTFS
volume); use --copy-images only if hard links are not available.
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


SOURCE_ROOT = Path(r"D:\21431547\RDD2022_released_through_CRDDC2022\RDD2022")
OUTPUT_ROOT = Path(r"D:\21431547_yolo")
CLASS_NAMES = ["D00", "D10", "D20", "D40"]


def voc_to_yolo(xml_path, label_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = float(size.findtext("width", "0")) if size is not None else 0
    height = float(size.findtext("height", "0")) if size is not None else 0
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image dimensions in {xml_path}")
    lines = []
    for item in root.findall("object"):
        name = item.findtext("name")
        if name not in CLASS_NAMES:
            continue
        box = item.find("bndbox")
        if box is None:
            continue
        try:
            xmin, ymin = float(box.findtext("xmin")), float(box.findtext("ymin"))
            xmax, ymax = float(box.findtext("xmax")), float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue
        xmin, xmax = max(0, xmin), min(width, xmax)
        ymin, ymax = max(0, ymin), min(height, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        x_center, y_center = ((xmin + xmax) / 2 / width), ((ymin + ymax) / 2 / height)
        box_width, box_height = ((xmax - xmin) / width), ((ymax - ymin) / height)
        lines.append(f"{CLASS_NAMES.index(name)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")


def link_or_copy(source, destination, copy_images):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if copy_images:
        shutil.copy2(source, destination)
    else:
        try:
            destination.hardlink_to(source)
        except OSError as error:
            raise OSError(
                f"Could not create hard link for {source}. Re-run with --copy-images "
                "only if you accept duplicated image files."
            ) from error


def main():
    parser = argparse.ArgumentParser(description="Convert extracted RDD2022 VOC annotations to YOLO labels.")
    parser.add_argument("--source", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise FileNotFoundError(
            f"Extracted RDD2022 folder not found: {args.source}. Extract the outer archive and country ZIPs first."
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Output already exists and is non-empty: {args.output}")
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between zero and one.")

    pairs = []
    for xml_path in sorted(args.source.glob("*/train/annotations/xmls/*.xml")):
        image_path = xml_path.parents[2] / "images" / f"{xml_path.stem}.jpg"
        if image_path.is_file():
            pairs.append((xml_path, image_path))
    if not pairs:
        raise ValueError("No matching RDD2022 train XML/JPG pairs found.")
    random.Random(2026).shuffle(pairs)
    split = max(1, round(len(pairs) * (1 - args.val_fraction)))
    for split_name, subset in (("train", pairs[:split]), ("val", pairs[split:])):
        for xml_path, image_path in subset:
            # Country prefix makes filenames unique across the publisher splits.
            filename = f"{xml_path.parents[3].name}_{image_path.name}"
            link_or_copy(image_path, args.output / "images" / split_name / filename, args.copy_images)
            voc_to_yolo(xml_path, args.output / "labels" / split_name / f"{Path(filename).stem}.txt")
    yaml_path = args.output / "data.yaml"
    yaml_path.write_text(
        "path: " + str(args.output).replace("\\", "/") + "\n"
        "train: images/train\nval: images/val\n"
        "names:\n" + "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES)),
        encoding="utf-8",
    )
    print(f"Prepared {len(pairs)} labeled images: {split} train, {len(pairs) - split} val")
    print(f"YOLO configuration: {yaml_path}")


if __name__ == "__main__":
    main()
