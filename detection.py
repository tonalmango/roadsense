"""RoadSense YOLOv8 detection for individual images and extracted frames."""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from priority import calculate_priority, prioritize_repair
from gps_mapping import enrich_detections_with_gps, load_gps_records
from severity import calculate_severity, calculate_severity_score


MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
CONFIDENCE_THRESHOLD = 0.15

# Keep using the existing trained RoadSense model.
model = YOLO(str(MODEL_PATH))


def _damage_type_for_class(class_id):
    """Return a safe class label if the model has no name for an ID."""
    if isinstance(model.names, dict):
        return str(model.names.get(class_id, "unknown"))
    if 0 <= class_id < len(model.names):
        return str(model.names[class_id])
    return "unknown"


def _read_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
    return image


def _run_yolo(image_path):
    """Run the existing YOLO model with the project's current threshold."""
    return model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False,
    )[0]


def _box_coordinates(box, image_width, image_height):
    """Convert a YOLO box to integer image coordinates and derived dimensions."""
    raw_x1, raw_y1, raw_x2, raw_y2 = box.xyxy[0].tolist()
    x1 = max(0, min(image_width, int(raw_x1)))
    y1 = max(0, min(image_height, int(raw_y1)))
    x2 = max(0, min(image_width, int(raw_x2)))
    y2 = max(0, min(image_height, int(raw_y2)))

    bbox_width = max(0, x2 - x1)
    bbox_height = max(0, y2 - y1)
    bbox_area = bbox_width * bbox_height
    image_area = image_width * image_height

    return {
        "bbox": [x1, y1, x2, y2],
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_area,
        "image_width": image_width,
        "image_height": image_height,
        "bbox_area_ratio": round(bbox_area / image_area, 6) if image_area else 0.0,
    }


def _annotate_image(image, boxes):
    """Draw simple detection labels on a copy of an image for demonstrations."""
    annotated = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        label = f"{box['damage_type']} {box['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
    return annotated


def _image_detections(image_path):
    """Run detection and return box data plus an annotated image, if readable."""
    image_path = Path(image_path)
    image = _read_image(image_path)
    if image is None:
        return [], None

    image_height, image_width = image.shape[:2]
    result = _run_yolo(image_path)
    boxes = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            boxes.append(
                {
                    "damage_type": _damage_type_for_class(class_id),
                    "confidence": float(box.conf[0]),
                    **_box_coordinates(box, image_width, image_height),
                }
            )
    return boxes, _annotate_image(image, boxes)


def detect_image(image_path, output_folder="results"):
    """Detect damage in one image and preserve the legacy return fields.

    This function remains usable for the original image-by-image workflow.
    The metadata batch path below deliberately does not add severity or GPS to
    ``detections.json``.
    """
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {image_path.name}")

    boxes, annotated = _image_detections(image_path)
    if annotated is None:
        return []

    output_path = output_folder / image_path.name
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Could not save annotated image: {output_path}")

    detections = []
    for box in boxes:
        severity_score = calculate_severity_score(
            box["damage_type"],
            box["bbox_area_ratio"],
            box["bbox_width"],
            box["bbox_height"],
            box["confidence"],
        )
        severity = calculate_severity(
            box["damage_type"],
            box["confidence"],
            box["bbox_area_ratio"],
            box["bbox_width"],
            box["bbox_height"],
        )
        priority_data = prioritize_repair(
            severity_score,
            box["damage_type"],
            bbox_area_ratio=box["bbox_area_ratio"],
        )
        detection = {
            "damage_type": box["damage_type"],
            "confidence": box["confidence"],
            "severity_score": severity_score,
            "severity": severity,
            "priority": calculate_priority(severity),
            **priority_data,
            "bbox": box["bbox"],
        }
        detections.append(detection)
        print(
            f"Damage: {detection['damage_type']} | "
            f"Confidence: {detection['confidence']:.2%} | "
            f"Severity score: {detection['severity_score']:.2f} | "
            f"Severity: {detection['severity']} | "
            f"Priority: {detection['priority_level']} "
            f"({detection['priority_score']:.2f})"
        )

    if not detections:
        print("No road damage detected.")
    print(f"Result saved to: {output_path}")
    return detections


def _load_frame_metadata(metadata_path):
    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Frame metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid frame metadata JSON: {metadata_path}") from error

    if not isinstance(metadata, dict) or not isinstance(metadata.get("frames"), list):
        raise ValueError("Frame metadata must contain a top-level 'frames' list.")
    return metadata


def process_frames_with_metadata(
    frames_folder="frames",
    metadata_path="frames/frame_metadata.json",
    output_folder="results",
    gps_data_path=None,
    road_context=None,
    traffic_factor=None,
):
    """Process usable extracted frames and save all detections to JSON.

    Each usable metadata entry is matched to its filename in ``frames_folder``.
    Unusable, missing, and unreadable frames are skipped and reported. An
    annotated image is saved for every readable usable frame, including frames
    with no detections. If no GPS path is supplied, every detection explicitly
    receives unavailable GPS fields rather than a guessed location.
    """
    frames_folder = Path(frames_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    metadata = _load_frame_metadata(metadata_path)

    all_detections = []
    usable_frame_count = 0
    skipped_frames = []

    for frame_metadata in metadata["frames"]:
        if not isinstance(frame_metadata, dict):
            skipped_frames.append("invalid metadata entry")
            continue
        if not frame_metadata.get("usable", False):
            continue

        frame_name = frame_metadata.get("filename")
        if not isinstance(frame_name, str) or not frame_name:
            skipped_frames.append("metadata entry without filename")
            continue

        frame_path = frames_folder / frame_name
        if not frame_path.is_file():
            skipped_frames.append(f"missing frame: {frame_name}")
            continue

        usable_frame_count += 1
        boxes, annotated = _image_detections(frame_path)
        if annotated is None:
            skipped_frames.append(f"unreadable frame: {frame_name}")
            continue

        annotated_path = output_folder / frame_name
        if not cv2.imwrite(str(annotated_path), annotated):
            raise OSError(f"Could not save annotated image: {annotated_path}")

        frame_number = frame_metadata.get("frame_number")
        timestamp_seconds = frame_metadata.get("timestamp_seconds")
        for box in boxes:
            severity_score = calculate_severity_score(
                box["damage_type"],
                box["bbox_area_ratio"],
                box["bbox_width"],
                box["bbox_height"],
                box["confidence"],
            )
            priority_data = prioritize_repair(
                severity_score,
                box["damage_type"],
                road_context=road_context,
                traffic_factor=traffic_factor,
                bbox_area_ratio=box["bbox_area_ratio"],
            )
            all_detections.append(
                {
                    "id": len(all_detections) + 1,
                    "frame": frame_name,
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "damage_type": box["damage_type"],
                    "confidence": box["confidence"],
                    "severity_score": severity_score,
                    "severity": calculate_severity(
                        box["damage_type"],
                        box["confidence"],
                        box["bbox_area_ratio"],
                        box["bbox_width"],
                        box["bbox_height"],
                    ),
                    **priority_data,
                    "bbox": box["bbox"],
                    "bbox_width": box["bbox_width"],
                    "bbox_height": box["bbox_height"],
                    "bbox_area": box["bbox_area"],
                    "image_width": box["image_width"],
                    "image_height": box["image_height"],
                    "bbox_area_ratio": box["bbox_area_ratio"],
                }
            )

    gps_records = load_gps_records(gps_data_path)
    all_detections = enrich_detections_with_gps(all_detections, gps_records)

    detections_path = output_folder / "detections.json"
    with open(detections_path, "w", encoding="utf-8") as file:
        json.dump(all_detections, file, indent=2)

    print(f"Usable frames processed: {usable_frame_count}")
    print(f"Detections saved: {len(all_detections)}")
    if gps_data_path is None:
        print("GPS: unavailable (no GPS data file supplied).")
    elif not gps_records:
        print(f"GPS: unavailable (no valid records in {gps_data_path}).")
    else:
        print(f"GPS records loaded: {len(gps_records)} from {gps_data_path}")
    print(f"Detection JSON saved to: {detections_path}")
    for message in skipped_frames:
        print(f"Skipped: {message}")
    return all_detections


def process_all_images(image_folder="frames", output_folder="results"):
    """Preserve the original batch image workflow without requiring metadata."""
    image_folder = Path(image_folder)
    image_files = []
    for extension in ("*.jpg", "*.jpeg", "*.png"):
        image_files.extend(image_folder.glob(extension))

    if not image_files:
        print(f"No images found in {image_folder}.")
        return []

    all_results = []
    for image_path in sorted(image_files):
        all_results.extend(detect_image(image_path, output_folder))
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RoadSense frame detection.")
    parser.add_argument(
        "--gps-data",
        type=Path,
        help="Timestamped CSV/JSON GPS data. DEMO files must be labeled as such.",
    )
    parser.add_argument(
        "--road-context",
        help="Optional contextual category (motorway, arterial, collector, local).",
    )
    parser.add_argument(
        "--traffic-factor",
        type=float,
        help="Optional contextual traffic score from 0 to 100; not measured by RoadSense.",
    )
    arguments = parser.parse_args()
    metadata_file = Path("frames/frame_metadata.json")
    if metadata_file.is_file():
        process_frames_with_metadata(
            metadata_path=metadata_file,
            gps_data_path=arguments.gps_data,
            road_context=arguments.road_context,
            traffic_factor=arguments.traffic_factor,
        )
    else:
        process_all_images()
