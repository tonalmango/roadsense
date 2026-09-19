"""Reliability calibration helpers for labeled validation results."""

import math


def calibration_bins(records, bins=10):
    """Return reliability-curve bins with count, confidence, and accuracy."""
    if bins < 1:
        raise ValueError("bins must be at least 1")
    result = [{"lower": index / bins, "upper": (index + 1) / bins, "count": 0, "accuracy": None, "confidence": None} for index in range(bins)]
    for item in records:
        confidence = max(0.0, min(0.999999, float(item.get("confidence", 0))))
        bucket = result[min(bins - 1, int(confidence * bins))]
        bucket["count"] += 1
        bucket.setdefault("_correct", 0)
        bucket["_confidence_total"] = bucket.get("_confidence_total", 0) + confidence
        bucket["_correct"] += int(bool(item.get("correct", False)))
    for bucket in result:
        if bucket["count"]:
            bucket["accuracy"] = round(bucket["_correct"] / bucket["count"], 3)
            bucket["confidence"] = round(bucket["_confidence_total"] / bucket["count"], 3)
        bucket.pop("_correct", None)
        bucket.pop("_confidence_total", None)
    return result


def expected_calibration_error(records, bins=10):
    calibrated = calibration_bins(records, bins)
    total = len(records)
    return round(sum((bucket["count"] / total) * abs(bucket["accuracy"] - bucket["confidence"]) for bucket in calibrated if bucket["count"]), 4) if total else None


def maximum_calibration_error(records, bins=10):
    """Return the largest occupied-bin confidence/accuracy gap."""
    calibrated = calibration_bins(records, bins)
    gaps = [abs(item["accuracy"] - item["confidence"]) for item in calibrated if item["count"]]
    return round(max(gaps), 4) if gaps else None


def calibration_summary(records, bins=10):
    """Return calibration metrics and a human-readable quality band."""
    ece = expected_calibration_error(records, bins)
    return {"sample_count": len(records), "expected_calibration_error": ece, "maximum_calibration_error": maximum_calibration_error(records, bins), "quality": "Good" if ece is not None and ece <= 0.05 else "Review" if ece is not None and ece <= 0.1 else "Insufficient data" if ece is None else "Poor"}
