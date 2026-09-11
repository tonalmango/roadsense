"""Prototype, explainable RoadDamages severity heuristic.

This module does not measure physical depth or structural condition. It ranks
the RAD RoadDamages class for a hackathon demonstration using bounding-box
extent and a small confidence reliability component.
"""


ROAD_DAMAGE_CLASS = "RoadDamages"

SEVERITY_THRESHOLDS = {
    "Minor": 0,
    "Moderate": 40,
    "Severe": 70,
}

# These are prototype risk weights, not scientific calibration or model metrics.
DAMAGE_TYPE_RISK = {"roaddamages": 1.00}


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
    * ``geometry`` is retained as zero for the broad RAD RoadDamages class.
      RAD does not provide a separate crack-geometry label.
    * ``reliability`` is YOLO confidence times 100. It supplies at most ten
      points before the type-risk weight, so it cannot be the main driver.

    This is a transparent prioritisation heuristic, not a calibrated estimate
    of physical road damage. It applies only to the RAD ``RoadDamages`` class;
    other RAD classes return ``None`` because they are not repair defects.
    Missing or invalid RoadDamages measurements become zero.
    """
    damage_type = str(damage_type or "unknown").lower()
    if damage_type != ROAD_DAMAGE_CLASS.lower():
        return None
    risk_factor = DAMAGE_TYPE_RISK[damage_type]

    area_ratio = _number_in_range(bbox_area_ratio, 0.0, 1.0)
    extent_score = min(100.0, (area_ratio / 0.10) * 100.0)

    # The RAD taxonomy has only the broad RoadDamages class, so no crack-
    # specific shape inference is made from this bounding box.
    geometry_score = 0.0

    reliability_score = _number_in_range(confidence, 0.0, 1.0) * 100.0
    raw_score = (
        0.70 * extent_score
        + 0.20 * geometry_score
        + 0.10 * reliability_score
    )
    return round(min(100.0, raw_score * risk_factor), 2)


def classify_severity(severity_score):
    """Map the documented prototype score bands to a severity label."""
    if severity_score is None:
        return "N/A"
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
    physical damage. Non-RoadDamages classes return ``N/A``.
    """
    score = calculate_severity_score(
        damage_type,
        bbox_area_ratio,
        bbox_width,
        bbox_height,
        confidence,
    )
    return classify_severity(score)
