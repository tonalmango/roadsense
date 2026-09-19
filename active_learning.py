"""Select uncertain or high-impact samples for human labeling."""


def select_feedback_samples(detections, limit=25, uncertainty_band=(0.35, 0.7)):
    """Rank samples by uncertainty, impact, and missing review labels."""
    def score(item):
        confidence = float(item.get("confidence") or 0)
        uncertainty = 1.0 - abs(confidence - 0.5) * 2
        impact = max(0.0, min(1.0, float(item.get("priority_score") or 0) / 100))
        novelty = 1.0 if item.get("review_status") in {None, "Unreviewed"} else 0.0
        return uncertainty * 0.55 + impact * 0.3 + novelty * 0.15
    selected = sorted(detections, key=score, reverse=True)
    low, high = uncertainty_band
    return [{**item, "feedback_status": item.get("feedback_status", "Needs review"), "active_learning_score": round(score(item), 3), "uncertainty": round(1.0 - abs(float(item.get("confidence") or 0) - 0.5) * 2, 3), "selection_reason": "uncertain confidence" if low <= float(item.get("confidence") or 0) <= high else "high maintenance impact"} for item in selected[:limit]]


def record_feedback(sample, label, reviewer=None, notes=""):
    """Return a labeled feedback record without mutating the source sample."""
    if not label:
        raise ValueError("label is required")
    return {**sample, "feedback_status": "Reviewed", "review_label": label, "reviewer": reviewer, "review_notes": notes}
