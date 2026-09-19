"""Road-condition health scoring on a transparent 0-100 scale."""


def _band(score):
    if score >= 80:
        return "Good"
    if score >= 60:
        return "Watch"
    if score >= 35:
        return "Poor"
    return "Critical"


def calculate_health_score(detections, usable_frames=0, total_frames=0):
    """Calculate a bounded score with explicit penalty components."""
    damage = [item for item in detections if item.get("priority_score") is not None]
    severity_values = [max(0.0, min(100.0, float(item.get("severity_score") or 0))) for item in damage]
    average_severity = sum(severity_values) / len(severity_values) if severity_values else 0.0
    severity_penalty = min(45.0, average_severity * 0.45)
    density_penalty = min(40.0, len(damage) * 4.0)
    coverage = max(0.0, min(1.0, usable_frames / total_frames)) if total_frames else 1.0
    quality_penalty = max(0.0, 1.0 - coverage) * 15.0
    score = round(max(0.0, min(100.0, 100.0 - severity_penalty - density_penalty - quality_penalty)), 1)
    return {"score": score, "band": _band(score), "damage_count": len(damage), "average_severity": round(average_severity, 1), "coverage": round(coverage, 3), "penalties": {"severity": round(severity_penalty, 1), "density": round(density_penalty, 1), "quality": round(quality_penalty, 1)}}
