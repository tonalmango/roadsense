"""Prioritized and deduplicated maintenance work queue."""


VALID_STATUSES = {"Open", "Assigned", "In progress", "Completed", "Deferred"}


def build_repair_queue(detections, limit=None):
    """Create a stable queue with urgency, location, and work status."""
    candidates = [item for item in detections if item.get("priority_score") is not None]
    queue = sorted(candidates, key=lambda item: (float(item.get("priority_score", 0)), float(item.get("confidence", 0))), reverse=True)
    result = []
    seen = set()
    for rank, item in enumerate(queue, 1):
        key = item.get("track_id") or (round(float(item.get("latitude", 0) or 0), 4), round(float(item.get("longitude", 0) or 0), 4), item.get("damage_type"))
        if key in seen:
            continue
        seen.add(key)
        status = item.get("work_status", "Open")
        if status not in VALID_STATUSES:
            status = "Open"
        urgency = "Immediate" if float(item.get("priority_score", 0)) >= 80 else "Soon" if float(item.get("priority_score", 0)) >= 60 else "Routine"
        result.append({**item, "queue_rank": len(result) + 1, "urgency": urgency, "work_status": status, "location_key": key})
        if limit and len(result) >= limit:
            break
    return result


def update_work_status(queue, item_key, status):
    """Return a copied queue with one work item status updated."""
    if status not in VALID_STATUSES:
        raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
    updated = []
    for item in queue:
        key = item.get("track_id") or item.get("location_key")
        updated.append({**item, "work_status": status if key == item_key else item.get("work_status", "Open")})
    return updated
