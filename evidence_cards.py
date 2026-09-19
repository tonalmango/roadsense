"""Create reviewable, explainable evidence cards from detections."""

from collections import Counter


def _location(detection):
    return {
        "latitude": detection.get("latitude"),
        "longitude": detection.get("longitude"),
        "source": detection.get("gps_source", "Unavailable"),
        "match_method": detection.get("gps_match_method", "unavailable"),
    }


def build_evidence_cards(detections, include_context=False):
    """Build cards suitable for UI review, export, and human labeling."""
    cards = []
    for index, detection in enumerate(detections, 1):
        if not include_context and detection.get("detection_group") == "road_context":
            continue
        confidence = float(detection.get("confidence") or 0)
        cards.append({
            "id": detection.get("id", f"evidence-{index}"),
            "track_id": detection.get("track_id"),
            "frame": detection.get("frame"),
            "timestamp_seconds": detection.get("timestamp_seconds"),
            "damage_type": detection.get("damage_type", "Unknown"),
            "severity": detection.get("severity", "N/A"),
            "priority_level": detection.get("priority_level", "N/A"),
            "confidence": detection.get("confidence"),
            "location": _location(detection),
            "image_path": detection.get("annotated_image") or detection.get("image_path"),
            "bbox": detection.get("bbox"),
            "confidence_band": "High" if confidence >= 0.75 else "Review" if confidence < 0.5 else "Moderate",
            "risk_score": detection.get("risk_score", detection.get("priority_score")),
            "explanation": detection.get("explanation") or detection.get("priority_reason") or "No explanation recorded",
            "review_status": detection.get("review_status", "Unreviewed"),
        })
    return cards


def summarize_evidence(cards):
    """Return counts used by the review dashboard."""
    return {
        "total": len(cards),
        "by_damage_type": dict(Counter(card.get("damage_type", "Unknown") for card in cards)),
        "unreviewed": sum(card.get("review_status") == "Unreviewed" for card in cards),
        "mapped": sum(card.get("location", {}).get("latitude") is not None for card in cards),
    }
