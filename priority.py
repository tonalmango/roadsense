"""Prototype, explainable RoadDamages repair-prioritisation heuristic."""

from severity import ROAD_DAMAGE_CLASS


# Configurable prototype weights. They sum to 1.0 (100% of priority score).
SEVERITY_WEIGHT = 0.60
DAMAGE_RISK_WEIGHT = 0.20
ROAD_CONTEXT_WEIGHT = 0.12
TRAFFIC_WEIGHT = 0.08

# Prototype risk values: not measured traffic, crash, or repair data.
DAMAGE_TYPE_RISK = {"roaddamages": 100.0}

# Optional contextual inputs. Supplying a category is a user/context decision,
# not a claim that RoadSense measured the road category or traffic level.
ROAD_CONTEXT_SCORES = {
    "motorway": 100.0,
    "arterial": 80.0,
    "collector": 60.0,
    "local": 40.0,
    "unknown": 50.0,
}


def _score(value, default=50.0):
    """Safely constrain a supplied normalized score to the 0--100 range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return max(0.0, min(100.0, number))


def _road_context_score(road_context):
    if road_context is None:
        return 50.0, "not supplied (neutral prototype baseline)"
    if isinstance(road_context, str):
        category = road_context.lower()
        return ROAD_CONTEXT_SCORES.get(category, 50.0), category
    return _score(road_context), f"custom score {road_context}"


def _traffic_score(traffic_factor):
    if traffic_factor is None:
        return 50.0, "not supplied (neutral prototype baseline)"
    return _score(traffic_factor), f"provided score {traffic_factor}"


def classify_priority(priority_score):
    """Classify the documented 0--100 priority bands."""
    score = _score(priority_score, default=0.0)
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Low"


def prioritize_repair(
    severity_score,
    damage_type,
    road_context=None,
    traffic_factor=None,
    bbox_area_ratio=None,
):
    """Return explainable prototype repair priority for one detection.

    Formula for the RAD ``RoadDamages`` class:
        priority = (0.60 * severity + 0.20 * damage risk
                    + 0.12 * road context + 0.08 * traffic)

    All inputs are normalized to 0--100. ``road_context`` and
    ``traffic_factor`` are optional contextual inputs; omitted values use a
    neutral 50-point baseline and are clearly stated in the reason. They are
    never presented as data measured by this project. Other RAD classes are
    contextual objects/features, not repair defects, and return N/A fields.
    """
    if str(damage_type or "").lower() != ROAD_DAMAGE_CLASS.lower():
        return {
            "priority_score": None,
            "priority_level": "N/A",
            "priority_reason": "Not applicable: this RAD class is not RoadDamages.",
            "priority_reasoning": "Not applicable: this RAD class is not RoadDamages.",
        }

    severity_value = _score(severity_score, default=0.0)
    damage_type_label = str(damage_type or ROAD_DAMAGE_CLASS)
    normalized_type = damage_type_label.lower()
    damage_risk = DAMAGE_TYPE_RISK[normalized_type]
    road_score, road_description = _road_context_score(road_context)
    traffic_score, traffic_description = _traffic_score(traffic_factor)

    severity_component = severity_value * SEVERITY_WEIGHT
    damage_risk_component = damage_risk * DAMAGE_RISK_WEIGHT
    road_context_component = road_score * ROAD_CONTEXT_WEIGHT
    traffic_component = traffic_score * TRAFFIC_WEIGHT
    priority_score = round(
        severity_component
        + damage_risk_component
        + road_context_component
        + traffic_component,
        2,
    )

    extent_text = ""
    if bbox_area_ratio is not None:
        try:
            extent = _score(float(bbox_area_ratio) * 100.0, default=0.0)
        except (TypeError, ValueError):
            extent = 0.0
        extent_text = f"; bbox extent {extent:.1f}% of image"

    priority_level = classify_priority(priority_score)
    priority_reason = (
        f"{damage_type_label} with severity score {severity_value:.1f}/100 "
        f"and damage-risk score {damage_risk:.0f}/100"
        f"{extent_text}; road context {road_description} "
        f"({road_score:.0f}/100); traffic {traffic_description} "
        f"({traffic_score:.0f}/100)."
    )
    return {
        "priority_score": priority_score,
        "priority_level": priority_level,
        "priority_reason": priority_reason,
        "priority_reasoning": priority_reason,
    }


def calculate_priority(severity):
    """Preserve the legacy UI mapping for callers without structured inputs."""
    legacy_map = {
        "critical": "Immediate",
        "high": "High",
        "medium": "Moderate",
        "low": "Low",
        "severe": "High",
        "moderate": "Moderate",
        "minor": "Low",
    }
    return legacy_map.get(str(severity).lower(), "Unknown")
