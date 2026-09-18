"""Lazy, failure-safe RDD2022 subtype refinement for RAD RoadDamages boxes.

The RAD detector remains the primary and mandatory model.  This optional model
only replaces a RoadDamages fallback when a confident specific damage box has
sufficient spatial overlap with that RAD region.
"""

from model_config import DAMAGE_CLASS_NAMES, DAMAGE_CONFIDENCE_THRESHOLDS, DAMAGE_MATCH_IOU_THRESHOLD, DAMAGE_MODEL_PATH, damage_model_is_ready


_CACHED_MODEL = None


def bbox_iou(first, second):
    """Return conventional IoU for pixel-space boxes."""
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - overlap
    return overlap / union if union else 0.0


def _load_model():
    """Load one verified secondary model only when its checkpoint exists."""
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL, None
    if not damage_model_is_ready():
        return None, "Damage model is unavailable or final validation is still in progress. RAD RoadDamages fallback remains active."
    try:
        from ultralytics import YOLO
        model = YOLO(str(DAMAGE_MODEL_PATH))
        names = {int(key): str(value) for key, value in (model.names.items() if isinstance(model.names, dict) else enumerate(model.names))}
        if names != DAMAGE_CLASS_NAMES:
            return None, f"Damage model taxonomy mismatch: expected {DAMAGE_CLASS_NAMES}, received {names}"
        _CACHED_MODEL = model
        return _CACHED_MODEL, None
    except Exception as error:  # Secondary model must never take down RAD inference.
        return None, f"Damage model load failed: {error}"


def _metrics(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    width, height = max(0, x2 - x1), max(0, y2 - y1)
    area = width * height
    return {
        "bbox": [x1, y1, x2, y2], "bbox_width": width, "bbox_height": height,
        "bbox_area": area, "image_width": image_width, "image_height": image_height,
        "bbox_area_ratio": round(area / (image_width * image_height), 6) if image_width and image_height else 0.0,
    }


def _specific_candidates(model, image, minimum_confidence=None):
    floor = min(DAMAGE_CONFIDENCE_THRESHOLDS.values())
    if minimum_confidence is not None:
        floor = max(floor, float(minimum_confidence))
    result = model.predict(source=image, conf=floor, save=False, verbose=False)[0]
    height, width = image.shape[:2]
    candidates = []
    for box in ([] if result.boxes is None else result.boxes):
        class_id = int(box.cls[0])
        label = DAMAGE_CLASS_NAMES.get(class_id)
        if label is None:
            continue
        confidence = float(box.conf[0])
        threshold = max(DAMAGE_CONFIDENCE_THRESHOLDS[label], float(minimum_confidence or 0))
        if confidence < threshold:
            continue
        raw = box.xyxy[0].tolist()
        bbox = [max(0, min(width, int(raw[0]))), max(0, min(height, int(raw[1]))), max(0, min(width, int(raw[2]))), max(0, min(height, int(raw[3])))]
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        candidates.append({"class_id": class_id, "label": label, "confidence": confidence, "threshold": threshold, **_metrics(bbox, width, height)})
    return candidates


def merge_specific_candidates(rad_boxes, candidates):
    """Pure spatial merge used by inference and unit tests."""
    boxes = [dict(box) for box in rad_boxes]
    refined = 0
    for box in boxes:
        box.setdefault("damage_model_used", False)
        box.setdefault("damage_refinement_status", "not_applicable" if box.get("damage_type") != "RoadDamages" else "fallback")
    available_candidates = list(candidates)
    for rad_box in (box for box in boxes if box.get("damage_type") == "RoadDamages"):
        matches = [(bbox_iou(rad_box["bbox"], candidate["bbox"]), candidate) for candidate in available_candidates]
        matches = [(iou, candidate) for iou, candidate in matches if iou >= DAMAGE_MATCH_IOU_THRESHOLD]
        if not matches:
            rad_box["damage_refinement_status"] = "fallback_no_spatial_match"
            continue
        iou, candidate = max(matches, key=lambda item: (item[0] * item[1]["confidence"], item[1]["confidence"]))
        available_candidates.remove(candidate)
        rad_box.update({
            "rad_class_id": rad_box.get("class_id"), "rad_confidence": rad_box.get("confidence"),
            "damage_model_class_id": candidate["class_id"], "damage_model_confidence": candidate["confidence"],
            "damage_model_match_iou": round(iou, 4), "damage_model_used": True,
            "damage_refinement_status": "specific_match", "damage_type": candidate["label"],
            "class_id": candidate["class_id"], "model_class": candidate["label"], "confidence": candidate["confidence"],
            "threshold_used": candidate["threshold"], "damage_category": "RoadDamages",
            "damage_subtype": candidate["label"], **_metrics(candidate["bbox"], rad_box["image_width"], rad_box["image_height"]),
        })
        refined += 1
    return boxes, refined


def refine_road_damage_boxes(image, rad_boxes, minimum_confidence=None):
    """Replace only spatially corresponding RAD RoadDamages boxes when safe.

    Returns a copy of RAD boxes and diagnostics.  Missing weights, load errors,
    inference errors, empty output, and poor overlap all leave RAD detections
    unchanged as the documented fallback.
    """
    if not any(box.get("damage_type") == "RoadDamages" for box in rad_boxes):
        boxes, refined = merge_specific_candidates(rad_boxes, [])
        return boxes, {"damage_model": "not_needed", "refined": refined}
    model, load_error = _load_model()
    if model is None:
        boxes, refined = merge_specific_candidates(rad_boxes, [])
        return boxes, {"damage_model": "unavailable", "reason": load_error, "refined": refined}
    try:
        candidates = _specific_candidates(model, image, minimum_confidence)
    except Exception as error:
        boxes, refined = merge_specific_candidates(rad_boxes, [])
        return boxes, {"damage_model": "inference_failed", "reason": str(error), "refined": refined}
    boxes, refined = merge_specific_candidates(rad_boxes, candidates)
    return boxes, {"damage_model": "used", "candidates": len(candidates), "refined": refined}
