"""Prototype, explainable road-damage severity heuristic.

This module does not measure physical depth or structural condition. It ranks
detected image regions for a hackathon demonstration using bounding-box extent,
simple geometry, damage type, and a small confidence reliability component.
"""


SEVERITY_THRESHOLDS = {
    "Minor": 0,
    "Moderate": 40,
    "Severe": 70,
}

# These are prototype risk weights, not scientific calibration or model metrics.
DAMAGE_TYPE_RISK = {
    "pothole": 1.20,
    "manhole": 1.10,
    "crack": 0.90,
    "unknown": 1.00,
}


def _number_in_range(value, minimum, maximum, default=0.0):
    """Convert a value to a finite number constrained to a known range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:  # NaN does not equal itself.
        return default
    return max(minimum, min(maximum, number))


def calculate_severity_score(
    damage_type,
    bbox_area_ratio=None,
    bbox_width=None,
    bbox_height=None,
    confidence=None,
):
    """Calculate a 0--100 prototype severity score.

    Formula:
        score = min(100, (0.70 * extent + 0.20 * geometry
                          + 0.10 * reliability) * type_risk)

    * ``extent`` reaches 100 when the bounding box covers 10% of the image.
    * ``geometry`` is used for cracks only: a longer, thinner bounding box
      receives a higher score, reaching 100 at an aspect ratio of 8:1.
    * ``reliability`` is YOLO confidence times 100. It supplies at most ten
      points before the type-risk weight, so it cannot be the main driver.

    This is a transparent prioritisation heuristic, not a calibrated estimate
    of physical road damage. Missing or invalid measurements become zero.
    """
    damage_type = str(damage_type or "unknown").lower()
    risk_factor = DAMAGE_TYPE_RISK.get(damage_type, DAMAGE_TYPE_RISK["unknown"])

    area_ratio = _number_in_range(bbox_area_ratio, 0.0, 1.0)
    extent_score = min(100.0, (area_ratio / 0.10) * 100.0)

    width = _number_in_range(bbox_width, 0.0, float("inf"))
    height = _number_in_range(bbox_height, 0.0, float("inf"))
    geometry_score = 0.0
    if damage_type == "crack" and width > 0 and height > 0:
        aspect_ratio = max(width / height, height / width)
        geometry_score = min(100.0, max(0.0, (aspect_ratio - 1.0) / 7.0 * 100.0))

    reliability_score = _number_in_range(confidence, 0.0, 1.0) * 100.0
    raw_score = (
        0.70 * extent_score
        + 0.20 * geometry_score
        + 0.10 * reliability_score
    )
    return round(min(100.0, raw_score * risk_factor), 2)


def classify_severity(severity_score):
    """Map the documented prototype score bands to a severity label."""
    score = _number_in_range(severity_score, 0.0, 100.0)
    if score >= SEVERITY_THRESHOLDS["Severe"]:
        return "Severe"
    if score >= SEVERITY_THRESHOLDS["Moderate"]:
        return "Moderate"
    return "Minor"


def calculate_severity(
    damage_type,
    confidence=None,
    bbox_area_ratio=None,
    bbox_width=None,
    bbox_height=None,
):
    """Backward-compatible label helper for existing callers.

    Callers should pass bounding-box measurements whenever available. Confidence
    alone only affects the small reliability component and cannot imply severe
    physical damage.
    """
    score = calculate_severity_score(
        damage_type,
        bbox_area_ratio,
        bbox_width,
        bbox_height,
        confidence,
    )
    return classify_severity(score)
