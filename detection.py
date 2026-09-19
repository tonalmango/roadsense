"""RoadSense RAD YOLO11n detection for individual images and extracted frames."""

import argparse
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from priority import calculate_priority, prioritize_repair
from gps_mapping import enrich_detections_with_gps, load_gps_records
from model_config import RDD2022_SUBTYPES, active_model_profile
from damage_model import refine_road_damage_boxes
from occlusion import add_occlusion_indicators
from frame_preprocessing import bbox_to_original, letterbox_frame
from severity import ROAD_DAMAGE_CLASS, calculate_severity, calculate_severity_score


ACTIVE_MODEL_PROFILE, MODEL_CONFIG = active_model_profile()
MODEL_PATH = MODEL_CONFIG["path"]
# These are inference acceptance thresholds, not accuracy values. They are
# deliberately above the old 0.15 global cutoff to reduce weak one-frame hits.
# A caller may raise the floor for a particular demo run without changing code.
CONFIDENCE_THRESHOLD = 0.25
CLASS_CONFIDENCE_THRESHOLDS = MODEL_CONFIG["confidence_thresholds"]
ROAD_CONDITION_CLASSES = MODEL_CONFIG["road_condition_classes"]
MODEL_CLASS_NAMES = MODEL_CONFIG["class_names"]
RAD_CLASS_NAMES = MODEL_CLASS_NAMES  # Backward-compatible exported name.

# The original model/best.pt is intentionally retained but is not used here.
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Active {ACTIVE_MODEL_PROFILE} model was not found: {MODEL_PATH}. "
        "RoadSense will not fall back to model/best.pt."
    )
model = YOLO(str(MODEL_PATH))


def _model_class_names():
    """Normalize Ultralytics class names for a one-time taxonomy check."""
    if isinstance(model.names, dict):
        return {int(class_id): str(name) for class_id, name in model.names.items()}
    return {class_id: str(name) for class_id, name in enumerate(model.names)}


if _model_class_names() != MODEL_CLASS_NAMES:
    raise RuntimeError(
        f"Active model class names do not match the expected {ACTIVE_MODEL_PROFILE} taxonomy. "
        f"Expected {MODEL_CLASS_NAMES}; received {_model_class_names()}."
    )


def _damage_type_for_class(class_id):
    """Return a repair-compatible category without losing model subtype data."""
    model_label = MODEL_CLASS_NAMES.get(class_id, "unknown")
    return ROAD_DAMAGE_CLASS if ACTIVE_MODEL_PROFILE == "rdd2022" and model_label != "unknown" else model_label


def _model_label_for_class(class_id):
    return MODEL_CLASS_NAMES.get(class_id, "unknown")


def _read_image(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
    return image


def _run_yolo(source, minimum_confidence=None):
    """Run YOLO once, then apply transparent class-specific filtering locally."""
    inference_floor = min(CLASS_CONFIDENCE_THRESHOLDS.values())
    if minimum_confidence is not None:
        inference_floor = max(inference_floor, float(minimum_confidence))
    return model.predict(
        source=source,
        conf=inference_floor,
        save=False,
        verbose=False,
    )[0]


def _box_coordinates(box, image_width, image_height, preprocessing=None):
    """Convert a YOLO box to integer image coordinates and derived dimensions."""
    raw_x1, raw_y1, raw_x2, raw_y2 = box.xyxy[0].tolist()
    raw_bbox = [raw_x1, raw_y1, raw_x2, raw_y2]
    if preprocessing:
        raw_bbox = bbox_to_original(raw_bbox, preprocessing)
    x1 = max(0, min(image_width, int(raw_bbox[0])))
    y1 = max(0, min(image_height, int(raw_bbox[1])))
    x2 = max(0, min(image_width, int(raw_bbox[2])))
    y2 = max(0, min(image_height, int(raw_bbox[3])))

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


def _threshold_for_class(class_name, minimum_confidence=None):
    """Return the recorded acceptance threshold for one RAD class."""
    threshold = CLASS_CONFIDENCE_THRESHOLDS.get(class_name, CONFIDENCE_THRESHOLD)
    if minimum_confidence is not None:
        try:
            threshold = max(threshold, float(minimum_confidence))
        except (TypeError, ValueError):
            pass
    return threshold


def _detection_group(damage_type):
    return "road_defect_condition" if damage_type in ROAD_CONDITION_CLASSES else "road_context"


def _image_detections(image_path, minimum_confidence=None, preprocess=False, preprocessing_size=512):
    """Run detection and return accepted boxes, annotation, and diagnostics."""
    image_path = Path(image_path)
    image = _read_image(image_path)
    if image is None:
        return [], None, {"rejected_low_confidence": 0}

    image_height, image_width = image.shape[:2]
    preprocessing = None
    inference_source = str(image_path)
    if preprocess:
        inference_source, preprocessing = letterbox_frame(image, target_size=preprocessing_size)
    result = _run_yolo(inference_source, minimum_confidence)
    boxes = []
    rejected_low_confidence = 0
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            model_class = _model_label_for_class(class_id)
            damage_type = _damage_type_for_class(class_id)
            confidence = float(box.conf[0])
            threshold_used = _threshold_for_class(model_class, minimum_confidence)
            if confidence < threshold_used:
                rejected_low_confidence += 1
                continue
            boxes.append(
                {
                    "damage_type": damage_type,
                    "model_class": model_class,
                    "class_id": class_id,
                    "confidence": confidence,
                    "threshold_used": threshold_used,
                    "detection_group": _detection_group(model_class),
                    "damage_category": damage_type,
                    "damage_subtype": (
                        RDD2022_SUBTYPES.get(model_class, "Unknown / Not classified")
                        if damage_type == ROAD_DAMAGE_CLASS else "N/A"
                    ),
                    **_box_coordinates(box, image_width, image_height, preprocessing),
                    "preprocessing": preprocessing or {"enabled": False},
                }
            )
    boxes, refinement = refine_road_damage_boxes(image, boxes, minimum_confidence)
    boxes = add_occlusion_indicators(boxes)
    return boxes, _annotate_image(image, boxes), {
        "rejected_low_confidence": rejected_low_confidence,
        "damage_refinement": refinement,
    }


def _analysis_fields(box, road_context=None, traffic_factor=None):
    """Apply existing prototype scoring to repair-relevant damage labels only."""
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
        road_context=road_context,
        traffic_factor=traffic_factor,
        bbox_area_ratio=box["bbox_area_ratio"],
    )
    return {
        "severity_score": severity_score,
        "severity": severity,
        "severity_level": severity,
        **priority_data,
    }


def detect_image(image_path, output_folder="results"):
    """Detect damage in one image and preserve the legacy return fields.

    This function remains usable for the original image-by-image workflow.
    The metadata batch path below adds timestamps, GPS, and structured output.
    """
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"\nProcessing: {image_path.name}")

    boxes, annotated, _ = _image_detections(image_path)
    if annotated is None:
        return []

    output_path = output_folder / image_path.name
    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(f"Could not save annotated image: {output_path}")

    detections = []
    for box in boxes:
        analysis = _analysis_fields(box)
        detection = {
            "class_id": box["class_id"],
            "damage_type": box["damage_type"],
            "model_class": box["model_class"],
            "confidence": box["confidence"],
            **analysis,
            "priority": calculate_priority(analysis["severity"]),
            "bbox": box["bbox"],
        }
        detections.append(detection)
        priority_display = (
            f"{detection['priority_level']} ({detection['priority_score']:.2f})"
            if detection["priority_score"] is not None
            else "N/A"
        )
        print(
            f"Damage: {detection['damage_type']} | "
            f"Confidence: {detection['confidence']:.2%} | "
            f"Severity score: {detection['severity_score']} | "
            f"Severity: {detection['severity']} | "
            f"Priority: {priority_display}"
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


def _iou(first_bbox, second_bbox):
    """Calculate bounding-box IoU for lightweight video confirmation."""
    ax1, ay1, ax2, ay2 = first_bbox
    bx1, by1, bx2, by2 = second_bbox
    overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
    overlap = overlap_width * overlap_height
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - overlap
    return overlap / union if union else 0.0


def _temporally_confirm(candidates, required_frames=1, iou_threshold=0.10, max_gap_seconds=2.5):
    """Keep persistent road-condition detections; context objects stay visible.

    Sampling intervals can be large, so the default is one (disabled). Setting
    two or three requires same-class road conditions to overlap in nearby
    sampled frames, reducing isolated predictions without claiming tracking.
    """
    if required_frames <= 1:
        return candidates
    confirmed = []
    for candidate in candidates:
        if candidate.get("detection_group") != "road_defect_condition":
            confirmed.append(candidate)
            continue
        nearby = 1
        for other in candidates:
            if (
                other is candidate
                or other.get("model_class", other["damage_type"])
                != candidate.get("model_class", candidate["damage_type"])
            ):
                continue
            if abs((other.get("frame_number") or 0) - (candidate.get("frame_number") or 0)) == 0:
                continue
            candidate_time = candidate.get("timestamp_seconds")
            other_time = other.get("timestamp_seconds")
            if (
                candidate_time is not None and other_time is not None
                and abs(float(other_time) - float(candidate_time)) > max_gap_seconds
            ):
                continue
            if _iou(candidate["bbox"], other["bbox"]) >= iou_threshold:
                nearby += 1
        if nearby >= required_frames:
            candidate["temporal_confirmed"] = True
            confirmed.append(candidate)
    return confirmed


def process_frames_with_metadata(
    frames_folder="frames",
    metadata_path="frames/frame_metadata.json",
    output_folder="results",
    gps_data_path=None,
    gps_records=None,
    gps_source=None,
    road_context=None,
    traffic_factor=None,
    minimum_confidence=None,
    temporal_confirmation_frames=1,
    temporal_iou_threshold=0.10,
    max_gps_time_difference_seconds=5.0,
    preprocess_frames=False,
    preprocessing_size=512,
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

    candidate_detections = []
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
        boxes, annotated, diagnostics = _image_detections(
            frame_path, minimum_confidence, preprocess=preprocess_frames,
            preprocessing_size=preprocessing_size,
        )
        if annotated is None:
            skipped_frames.append(f"unreadable frame: {frame_name}")
            continue

        annotated_path = output_folder / frame_name
        if not cv2.imwrite(str(annotated_path), annotated):
            raise OSError(f"Could not save annotated image: {annotated_path}")

        frame_number = frame_metadata.get("frame_number")
        timestamp_seconds = frame_metadata.get("timestamp_seconds")
        for box in boxes:
            analysis = _analysis_fields(box, road_context, traffic_factor)
            candidate_detections.append(
                {
                    "frame": frame_name,
                    "frame_number": frame_number,
                    "timestamp_seconds": timestamp_seconds,
                    "class_id": box["class_id"],
                    "damage_type": box["damage_type"],
                    "model_class": box["model_class"],
                    "confidence": box["confidence"],
                    "threshold_used": box["threshold_used"],
                    "accepted": True,
                    "detection_group": box["detection_group"],
                    "damage_category": box["damage_category"],
                    "damage_subtype": box["damage_subtype"],
                    "damage_model_used": box.get("damage_model_used", False),
                    "damage_refinement_status": box.get("damage_refinement_status"),
                    "rad_class_id": box.get("rad_class_id"),
                    "rad_confidence": box.get("rad_confidence"),
                    "damage_model_class_id": box.get("damage_model_class_id"),
                    "damage_model_confidence": box.get("damage_model_confidence"),
                    "damage_model_match_iou": box.get("damage_model_match_iou"),
                    "occlusion_flag": box["occlusion_flag"],
                    "occlusion_reason": box["occlusion_reason"],
                    "occlusion_indicator_quality": box["occlusion_indicator_quality"],
                    "preprocessing": box["preprocessing"],
                    **analysis,
                    "bbox": box["bbox"],
                    "bbox_width": box["bbox_width"],
                    "bbox_height": box["bbox_height"],
                    "bbox_area": box["bbox_area"],
                    "image_width": box["image_width"],
                    "image_height": box["image_height"],
                    "bbox_area_ratio": box["bbox_area_ratio"],
                }
            )

    all_detections = _temporally_confirm(
        candidate_detections,
        required_frames=max(1, int(temporal_confirmation_frames)),
        iou_threshold=float(temporal_iou_threshold),
    )
    for identifier, detection in enumerate(all_detections, start=1):
        detection["id"] = identifier
        detection.setdefault("temporal_confirmed", temporal_confirmation_frames <= 1)

    gps_records = gps_records if gps_records is not None else load_gps_records(gps_data_path)
    all_detections = enrich_detections_with_gps(
        all_detections,
        gps_records,
        max_time_difference_seconds=max_gps_time_difference_seconds,
        gps_source=gps_source or ("External GPS file" if gps_data_path else None),
    )

    detections_path = output_folder / "detections.json"
    with open(detections_path, "w", encoding="utf-8") as file:
        json.dump(all_detections, file, indent=2)

    print(f"Usable frames processed: {usable_frame_count}")
    print(f"Detections saved: {len(all_detections)}")
    if not gps_records:
        print("GPS: unavailable (no valid embedded or external records).")
    else:
        print(f"GPS records loaded: {len(gps_records)} ({gps_source or 'GPS source'})")
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


def process_single_image(
    image_path,
    output_folder="results",
    road_context=None,
    traffic_factor=None,
    minimum_confidence=None,
    gps_data=None,
    preprocess=False,
    preprocessing_size=512,
):
    """Analyze one still image into the same record schema used by video.

    ``gps_data`` is expected to come from embedded EXIF extraction. An external
    timestamped GPS log is intentionally not guessed onto a still image.
    """
    image_path = Path(image_path)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    boxes, annotated, diagnostics = _image_detections(
        image_path, minimum_confidence, preprocess=preprocess,
        preprocessing_size=preprocessing_size,
    )
    if annotated is None:
        raise ValueError(f"Could not read image: {image_path}")
    annotated_path = output_folder / image_path.name
    if not cv2.imwrite(str(annotated_path), annotated):
        raise OSError(f"Could not save annotated image: {annotated_path}")

    unavailable_gps = {
        "latitude": None,
        "longitude": None,
        "gps_timestamp": None,
        "gps_match_method": "unavailable",
        "gps_time_difference_seconds": None,
        "gps_source": "Unavailable",
    }
    gps_data = gps_data or unavailable_gps
    detections = []
    for identifier, box in enumerate(boxes, start=1):
        analysis = _analysis_fields(box, road_context, traffic_factor)
        detections.append(
            {
                "id": identifier,
                "frame": image_path.name,
                "frame_number": None,
                "timestamp_seconds": None,
                "class_id": box["class_id"],
                "damage_type": box["damage_type"],
                "model_class": box["model_class"],
                "confidence": box["confidence"],
                "threshold_used": box["threshold_used"],
                "accepted": True,
                "detection_group": box["detection_group"],
                "damage_category": box["damage_category"],
                "damage_subtype": box["damage_subtype"],
                **analysis,
                **box,
                **gps_data,
                "temporal_confirmed": "not_applicable_image",
            }
        )
    with open(output_folder / "detections.json", "w", encoding="utf-8") as file:
        json.dump(detections, file, indent=2)
    return detections, diagnostics


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
