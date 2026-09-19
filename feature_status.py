"""Transparent RoadSense capability status for the demo UI."""

from model_config import DAMAGE_MODEL_PATH, damage_model_is_ready


def build_feature_status(media_type=None, has_gps=False, has_capture_date=False):
    """Return capability rows based on implemented local functionality."""
    damage_checkpoint = DAMAGE_MODEL_PATH
    return [
        {"Capability": "Temporal frame extraction", "Status": "Implemented", "Evidence": "frame_extraction.py"},
        {"Capability": "Configurable sampling", "Status": "Implemented", "Evidence": "frame_extraction.py / pipeline.py"},
        {"Capability": "Blur and lighting quality checks", "Status": "Implemented", "Evidence": "frame_extraction.py"},
        {"Capability": "Occlusion indicator", "Status": "Implemented", "Evidence": "occlusion.py (conservative visual flag)"},
        {"Capability": "Explicit frame preprocessing", "Status": "Implemented", "Evidence": "frame_preprocessing.py (optional letterbox)"},
        {"Capability": "Bounding boxes and confidence", "Status": "Implemented", "Evidence": "detection.py"},
        {"Capability": "RAD road-condition detection", "Status": "Implemented", "Evidence": "active RAD model profile"},
        {"Capability": "Specific pothole/crack refinement", "Status": "Implemented" if damage_model_is_ready() else ("Training in progress" if damage_checkpoint.is_file() else "Not available for this input"), "Evidence": str(damage_checkpoint) if damage_checkpoint.is_file() else "four-class model checkpoint not yet created"},
        {"Capability": "Rule-based RoadDamages severity", "Status": "Implemented", "Evidence": "severity.py"},
        {"Capability": "Timestamp GPS synchronization", "Status": "Implemented" if has_gps else "Not available for this input", "Evidence": "gps_mapping.py"},
        {"Capability": "EXIF GPS", "Status": "Available when metadata exists" if media_type == "image" else "Not available for this input", "Evidence": "gps_mapping.py"},
        {"Capability": "Embedded MP4 GPS", "Status": "Available when metadata exists" if media_type == "video" else "Not applicable", "Evidence": "FFprobe/QuickTime metadata and visible GPS overlay fallback"},
        {"Capability": "Reliable capture-date filter", "Status": "Available when metadata exists" if has_capture_date else "Not available for this input", "Evidence": "capture_metadata.py"},
        {"Capability": "Interactive map and filters", "Status": "Implemented", "Evidence": "app.py / PyDeck"},
        {"Capability": "Prototype repair prioritization", "Status": "Implemented", "Evidence": "priority.py"},
        {"Capability": "Evidence and JSON outputs", "Status": "Implemented", "Evidence": "annotated images + JSON artifacts"},
    ]
