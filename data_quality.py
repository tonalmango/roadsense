"""GPS and inspection-data quality diagnostics."""

from collections import Counter


def assess_data_quality(detections, gps_records=None):
    total = len(detections)
    located = sum(item.get("latitude") is not None and item.get("longitude") is not None for item in detections)
    timestamps = [float(item["timestamp_seconds"]) for item in detections if item.get("timestamp_seconds") is not None]
    gps = gps_records or []
    duplicate_timestamps = len(timestamps) - len(set(timestamps))
    gps_methods = Counter(item.get("gps_match_method", "unavailable") for item in detections)
    ordered_gps_times = [float(item.get("timestamp_seconds")) for item in gps if item.get("timestamp_seconds") is not None]
    gps_gaps = [right - left for left, right in zip(ordered_gps_times, ordered_gps_times[1:])]
    return {"detection_count": total, "gps_coverage": round(located / max(total, 1), 3), "timestamp_coverage": round(len(timestamps) / max(total, 1), 3), "duplicate_detection_timestamps": duplicate_timestamps, "gps_record_count": len(gps), "gps_match_methods": dict(gps_methods), "gps_max_gap_seconds": max(gps_gaps) if gps_gaps else None, "status": "Good" if total and located / total >= 0.8 and duplicate_timestamps == 0 else "Review"}


def quality_flags(quality):
    """Translate quality measurements into actionable review flags."""
    flags = []
    if quality.get("gps_coverage", 0) < 0.8:
        flags.append("low GPS coverage")
    if quality.get("timestamp_coverage", 0) < 0.8:
        flags.append("missing timestamps")
    if quality.get("duplicate_detection_timestamps", 0):
        flags.append("duplicate timestamps")
    if quality.get("gps_max_gap_seconds") is not None and quality["gps_max_gap_seconds"] > 5:
        flags.append("large GPS time gap")
    return flags
