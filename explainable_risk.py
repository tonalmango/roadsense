"""Explain severity and risk outputs without hiding heuristic inputs."""


def explain_detection(detection):
    factors = []
    confidence = float(detection.get("confidence") or 0)
    area = float(detection.get("bbox_area_ratio") or 0)
    severity = float(detection.get("severity_score") or 0)
    priority = float(detection.get("priority_score") or 0)
    if confidence >= 0.75:
        factors.append("high model confidence")
    elif confidence < 0.4:
        factors.append("limited model confidence")
    if area >= 0.08:
        factors.append("large image footprint")
    if detection.get("priority_level") in {"High", "Critical"}:
        factors.append(f"{detection['priority_level'].lower()} maintenance priority")
    if detection.get("gps_match_method") == "unavailable":
        factors.append("location unavailable")
    if detection.get("occlusion_possible"):
        factors.append("possible occlusion")
    return {"risk_score": round(max(priority, severity), 1), "severity_score": round(severity, 1), "priority_score": round(priority, 1), "factors": factors or ["baseline rule-based assessment"], "explanation": "; ".join(factors) if factors else "baseline rule-based assessment", "disclaimer": "This is decision support from model confidence and geometric/context heuristics; it is not a structural engineering assessment."}
