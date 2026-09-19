"""Aggregate detections into geographic road segments."""

from collections import Counter


def analyze_segments(detections, precision=3):
    """Aggregate detections by rounded coordinates and calculate segment risk."""
    segments = {}
    for item in detections:
        lat, lon = item.get("latitude"), item.get("longitude")
        if lat is None or lon is None:
            continue
        key = (round(float(lat), precision), round(float(lon), precision))
        bucket = segments.setdefault(key, {"segment_id": f"segment-{key[0]}-{key[1]}", "latitude": key[0], "longitude": key[1], "detections": 0, "damage_count": 0, "priority_total": 0.0, "severity_total": 0.0, "confidence_total": 0.0, "types": Counter()})
        bucket["detections"] += 1
        bucket["confidence_total"] += float(item.get("confidence") or 0)
        if item.get("priority_score") is not None:
            bucket["damage_count"] += 1
            bucket["priority_total"] += float(item["priority_score"])
            bucket["severity_total"] += float(item.get("severity_score") or 0)
        label = item.get("damage_type", "Unknown")
        bucket["types"][label] += 1
    for bucket in segments.values():
        bucket["average_priority"] = round(bucket["priority_total"] / max(bucket["damage_count"], 1), 1)
        bucket["average_severity"] = round(bucket["severity_total"] / max(bucket["damage_count"], 1), 1)
        bucket["average_confidence"] = round(bucket["confidence_total"] / max(bucket["detections"], 1), 3)
        bucket["risk_band"] = "High" if bucket["average_priority"] >= 60 else "Medium" if bucket["average_priority"] >= 40 else "Low"
        bucket["types"] = dict(bucket["types"])
    return list(segments.values())
