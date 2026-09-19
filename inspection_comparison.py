"""Compare two inspection runs using stable summary metrics."""

from collections import Counter


def compare_inspections(previous, current):
    def summary(records):
        damage = [item for item in records if item.get("priority_score") is not None]
        return {"detections": len(records), "damage": len(damage), "average_severity": round(sum(float(item.get("severity_score") or 0) for item in damage) / max(len(damage), 1), 1), "critical": sum(item.get("priority_level") == "Critical" for item in damage), "types": dict(Counter(item.get("damage_type", "Unknown") for item in damage))}
    before, after = summary(previous), summary(current)
    delta = {key: after[key] - before[key] for key in before if isinstance(after[key], (int, float))}
    return {"before": before, "after": after, "delta": delta, "status": "Improved" if (after["damage"], after["critical"], after["average_severity"]) < (before["damage"], before["critical"], before["average_severity"]) else "Worsened" if (after["damage"], after["critical"], after["average_severity"]) > (before["damage"], before["critical"], before["average_severity"]) else "Unchanged"}
