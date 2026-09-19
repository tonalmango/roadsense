"""Explicit local model profiles for RoadSense.

The RAD YOLO11m profile is the default. The RDD2022 profile is deliberately
separate because it uses the publisher's D00/D10/D20/D40 taxonomy. Set
``ROADSENSE_MODEL_PROFILE=rdd2022`` only after its training run creates the
configured checkpoint.
"""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent

RAD_CLASS_NAMES = {
    0: "HMV", 1: "LMV", 2: "Pedestrian", 3: "RoadDamages",
    4: "SpeedBump", 5: "UnsurfacedRoad",
}
RDD2022_CLASS_NAMES = {
    0: "D00", 1: "D10", 2: "D20", 3: "D40",
}
POTHOLE_CLASS_NAMES = {
    0: "Pothole", 1: "Crack", 2: "Manhole",
}

# These descriptions follow the official RDD2022 labels. They are distinct
# from RAD and are retained as subtype metadata, not silently discarded.
RDD2022_SUBTYPES = {
    "D00": "Longitudinal crack",
    "D10": "Transverse crack",
    "D20": "Alligator crack",
    "D40": "Pothole",
}

# A second model is deliberately separate from the active RAD profile.  It is
# loaded only when a RAD RoadDamages region needs a more specific label.
DAMAGE_MODEL_PATH = PROJECT_ROOT / "training" / "rdd_yolo11n" / "weights" / "best.pt"
DAMAGE_CLASS_NAMES = {
    0: "Pothole",
    1: "Longitudinal Crack",
    2: "Transverse Crack",
    3: "Alligator Crack",
}
DAMAGE_CONFIDENCE_THRESHOLDS = {name: 0.35 for name in DAMAGE_CLASS_NAMES.values()}
DAMAGE_MATCH_IOU_THRESHOLD = 0.10


def damage_model_is_ready():
    """Return true only after the four-class run has finished final validation.

    Ultralytics writes ``results.png`` after final checkpoint validation.  An
    intermediate best.pt is intentionally not activated while training is live.
    """
    return DAMAGE_MODEL_PATH.is_file() and (DAMAGE_MODEL_PATH.parents[1] / "results.png").is_file()

MODEL_PROFILES = {
    "rad": {
        "path": PROJECT_ROOT / "training" / "rad_yolo11m" / "weights" / "best.pt",
        "class_names": RAD_CLASS_NAMES,
        "confidence_thresholds": {
            "HMV": 0.25, "LMV": 0.25, "Pedestrian": 0.30,
            "RoadDamages": 0.25, "SpeedBump": 0.35, "UnsurfacedRoad": 0.30,
        },
        "road_condition_classes": {"RoadDamages", "SpeedBump", "UnsurfacedRoad"},
    },
    "rdd2022": {
        "path": PROJECT_ROOT / "training" / "rdd2022_yolo11n" / "weights" / "best.pt",
        "class_names": RDD2022_CLASS_NAMES,
        "confidence_thresholds": {name: 0.25 for name in RDD2022_CLASS_NAMES.values()},
        "road_condition_classes": set(RDD2022_CLASS_NAMES.values()),
    },
    "pothole": {
        "path": PROJECT_ROOT / "model" / "best.pt",
        "class_names": POTHOLE_CLASS_NAMES,
        "confidence_thresholds": {name: 0.25 for name in POTHOLE_CLASS_NAMES.values()},
        "road_condition_classes": set(POTHOLE_CLASS_NAMES.values()),
    },
}


def active_model_profile():
    """Return the requested profile, using the local compatible model if needed."""
    requested = os.environ.get("ROADSENSE_MODEL_PROFILE", "").strip().lower()
    if requested:
        name = requested
    elif MODEL_PROFILES["rad"]["path"].is_file():
        name = "rad"
    elif MODEL_PROFILES["pothole"]["path"].is_file():
        name = "pothole"
    else:
        name = "rad"
    if name not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown ROADSENSE_MODEL_PROFILE={name!r}. "
            f"Choose one of: {', '.join(MODEL_PROFILES)}."
        )
    return name, MODEL_PROFILES[name]
