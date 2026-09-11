"""GPS timestamp matching for RoadSense detections.

The legacy image-name lookup below is DEMO-only and remains for the current
Streamlit page. Timestamp matching is the path used for video detections.
"""

import csv
import json
from bisect import bisect_left
from pathlib import Path


# DEMO ONLY: these coordinates are not captured from a RoadSense dashcam.
DEMO_IMAGE_GPS_DATA = {
    "road1.jpeg.jpeg": {"latitude": 20.2961, "longitude": 85.8245},
    "road2.jpeg.jpeg": {"latitude": 20.2965, "longitude": 85.8250},
    "road3.jpeg.jpeg": {"latitude": 20.2970, "longitude": 85.8255},
    "road4.jpeg.jpeg": {"latitude": 20.2975, "longitude": 85.8260},
}

# gps/demo_gps_timestamps.csv is also DEMO-only sample data. It is provided to
# exercise synchronization and must not be represented as dashcam GPS capture.
DEMO_TIMESTAMP_GPS_CSV = Path("gps/demo_gps_timestamps.csv")

# Backward-compatible name used by the existing Streamlit UI.
GPS_DATA = DEMO_IMAGE_GPS_DATA


def get_gps(image_name):
    """Return DEMO image-name GPS data for the existing image dashboard only."""
    return GPS_DATA.get(image_name, {"latitude": None, "longitude": None})


def save_gps_data(output_file="gps/gps_data.json"):
    """Save the legacy DEMO image-name GPS mapping; it is not video GPS data."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(GPS_DATA, file, indent=2)
    print(f"DEMO image GPS data saved to: {output_path}")


def _as_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number else None  # Reject NaN.


def _normalise_records(records):
    """Validate, sort, and deterministically merge duplicate timestamps.

    Duplicate timestamp coordinates are averaged. This avoids choosing an
    arbitrary duplicate record while preserving all records' contribution.
    """
    grouped = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        timestamp = _as_float(record.get("timestamp_seconds"))
        latitude = _as_float(record.get("latitude"))
        longitude = _as_float(record.get("longitude"))
        if timestamp is None or latitude is None or longitude is None:
            continue
        grouped.setdefault(timestamp, []).append((latitude, longitude))

    normalised = []
    for timestamp in sorted(grouped):
        locations = grouped[timestamp]
        normalised.append(
            {
                "timestamp_seconds": timestamp,
                "latitude": sum(location[0] for location in locations) / len(locations),
                "longitude": sum(location[1] for location in locations) / len(locations),
            }
        )
    return normalised


def load_gps_records(gps_path):
    """Load timestamped GPS records from CSV or JSON.

    CSV columns must be ``timestamp_seconds,latitude,longitude``. JSON may be
    a list of records or an object containing ``records`` or ``gps_records``.
    A missing path returns an empty list so callers can mark GPS unavailable.
    """
    if gps_path is None:
        return []
    gps_path = Path(gps_path)
    if not gps_path.is_file():
        return []

    if gps_path.suffix.lower() == ".csv":
        with open(gps_path, "r", encoding="utf-8", newline="") as file:
            records = list(csv.DictReader(file))
    elif gps_path.suffix.lower() == ".json":
        with open(gps_path, "r", encoding="utf-8") as file:
            content = json.load(file)
        if isinstance(content, list):
            records = content
        elif isinstance(content, dict):
            records = content.get("records", content.get("gps_records", []))
        else:
            raise ValueError("GPS JSON must be a list or contain records/gps_records.")
    else:
        raise ValueError("GPS data must be a CSV or JSON file.")

    return _normalise_records(records)


def match_gps_timestamp(timestamp_seconds, gps_records):
    """Match one detection timestamp to GPS using interpolation where possible."""
    detection_time = _as_float(timestamp_seconds)
    records = _normalise_records(gps_records or [])
    unavailable = {
        "latitude": None,
        "longitude": None,
        "gps_timestamp": None,
        "gps_match_method": "unavailable",
        "gps_time_difference_seconds": None,
    }
    if detection_time is None or not records:
        return unavailable

    timestamps = [record["timestamp_seconds"] for record in records]
    right_index = bisect_left(timestamps, detection_time)

    # Exact matches, and detections outside the recorded range, use nearest.
    if right_index == 0:
        nearest = records[0]
    elif right_index == len(records):
        nearest = records[-1]
    elif timestamps[right_index] == detection_time:
        nearest = records[right_index]
    else:
        before = records[right_index - 1]
        after = records[right_index]
        fraction = (
            (detection_time - before["timestamp_seconds"])
            / (after["timestamp_seconds"] - before["timestamp_seconds"])
        )
        return {
            "latitude": before["latitude"] + fraction * (after["latitude"] - before["latitude"]),
            "longitude": before["longitude"] + fraction * (after["longitude"] - before["longitude"]),
            "gps_timestamp": detection_time,
            "gps_match_method": "interpolated",
            "gps_time_difference_seconds": 0.0,
        }

    return {
        "latitude": nearest["latitude"],
        "longitude": nearest["longitude"],
        "gps_timestamp": nearest["timestamp_seconds"],
        "gps_match_method": "nearest",
        "gps_time_difference_seconds": abs(detection_time - nearest["timestamp_seconds"]),
    }


def enrich_detections_with_gps(detections, gps_records):
    """Return copies of detection dictionaries enriched from timestamped GPS."""
    return [
        {
            **detection,
            **match_gps_timestamp(detection.get("timestamp_seconds"), gps_records),
        }
        for detection in detections
        if isinstance(detection, dict)
    ]
