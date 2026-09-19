"""Replay a mission in capture order for auditing and review."""


def replay_mission(detections, start_seconds=0, end_seconds=None):
    selected = [item for item in detections if item.get("timestamp_seconds") is not None and float(item["timestamp_seconds"]) >= start_seconds and (end_seconds is None or float(item["timestamp_seconds"]) <= end_seconds)]
    replay = []
    for index, item in enumerate(sorted(selected, key=lambda item: float(item["timestamp_seconds"])), 1):
        previous = replay[-1]["timestamp_seconds"] if replay else None
        replay.append({**item, "replay_index": index, "elapsed_seconds": round(float(item["timestamp_seconds"]) - float(selected[0]["timestamp_seconds"]), 3), "gap_seconds": round(float(item["timestamp_seconds"]) - float(previous), 3) if previous is not None else 0.0})
    return replay


def mission_summary(replay):
    """Summarize a replay window for a timeline header."""
    timestamps = [float(item["timestamp_seconds"]) for item in replay]
    return {"events": len(replay), "start_seconds": min(timestamps) if timestamps else None, "end_seconds": max(timestamps) if timestamps else None, "duration_seconds": round(max(timestamps) - min(timestamps), 3) if timestamps else 0.0, "tracks": len({item.get("track_id") for item in replay if item.get("track_id") is not None})}
