"""GPS timestamp matching for RoadSense detections.

The legacy image-name lookup below is DEMO-only and remains for the current
Streamlit page. Timestamp matching is the path used for video detections.
"""

import csv
import json
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from bisect import bisect_left
from pathlib import Path

from PIL import ExifTags, Image


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
        timestamp = _as_float(record.get("timestamp_seconds", record.get("timestamp")))
        latitude = _as_float(record.get("latitude"))
        longitude = _as_float(record.get("longitude"))
        if timestamp is None or latitude is None or longitude is None:
            continue
        grouped.setdefault(timestamp, []).append(
            {
                "latitude": latitude,
                "longitude": longitude,
                "altitude": record.get("altitude"),
                "speed": record.get("speed"),
                "heading": record.get("heading"),
            }
        )

    normalised = []
    for timestamp in sorted(grouped):
        locations = grouped[timestamp]
        normalised.append(
            {
                "timestamp_seconds": timestamp,
                "latitude": sum(location["latitude"] for location in locations) / len(locations),
                "longitude": sum(location["longitude"] for location in locations) / len(locations),
                "altitude": _average_field(locations, "altitude"),
                "speed": _average_field(locations, "speed"),
                "heading": _average_field(locations, "heading"),
            }
        )
    return normalised


def _average_field(records, field_name):
    values = [_as_float(record.get(field_name)) for record in records]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def load_gps_records(gps_path):
    """Load timestamped GPS records from embedded records, CSV, JSON, or GPX.

    CSV columns must include ``timestamp_seconds,latitude,longitude``. JSON may
    be a list of records or an object containing ``records`` or ``gps_records``.
    GPX track points use elapsed seconds from the first point as their timeline.
    A missing path returns an empty list so callers can mark GPS unavailable.
    """
    if gps_path is None:
        return []
    if isinstance(gps_path, (list, tuple)):
        return _normalise_records(gps_path)
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
    elif gps_path.suffix.lower() == ".gpx":
        root = ET.parse(gps_path).getroot()
        records = []
        first_time = None
        for point in root.iter():
            if point.tag.rsplit("}", 1)[-1] != "trkpt":
                continue
            latitude = _as_float(point.attrib.get("lat"))
            longitude = _as_float(point.attrib.get("lon"))
            time_node = next((child for child in point if child.tag.rsplit("}", 1)[-1] == "time"), None)
            if latitude is None or longitude is None or time_node is None or not time_node.text:
                continue
            from datetime import datetime
            timestamp = datetime.fromisoformat(time_node.text.strip().replace("Z", "+00:00"))
            if first_time is None:
                first_time = timestamp
            elevation_node = next((child for child in point if child.tag.rsplit("}", 1)[-1] == "ele"), None)
            records.append(
                {
                    "timestamp_seconds": (timestamp - first_time).total_seconds(),
                    "latitude": latitude,
                    "longitude": longitude,
                    "altitude": elevation_node.text if elevation_node is not None else None,
                }
            )
    else:
        raise ValueError("GPS data must be a CSV, JSON, or GPX file.")

    return _normalise_records(records)


def match_gps_timestamp(timestamp_seconds, gps_records, max_time_difference_seconds=None):
    """Match one detection timestamp to GPS using interpolation where possible."""
    detection_time = _as_float(timestamp_seconds)
    records = _normalise_records(gps_records or [])
    unavailable = {
        "latitude": None,
        "longitude": None,
        "altitude": None,
        "speed": None,
        "heading": None,
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
            "altitude": _interpolate_optional(before, after, "altitude", fraction),
            "speed": _interpolate_optional(before, after, "speed", fraction),
            "heading": _interpolate_optional(before, after, "heading", fraction),
            "gps_timestamp": detection_time,
            "gps_match_method": "interpolated",
            "gps_time_difference_seconds": 0.0,
        }

    difference = abs(detection_time - nearest["timestamp_seconds"])
    if (
        max_time_difference_seconds is not None
        and difference > float(max_time_difference_seconds)
    ):
        return unavailable
    return {
        "latitude": nearest["latitude"],
        "longitude": nearest["longitude"],
        "altitude": nearest.get("altitude"),
        "speed": nearest.get("speed"),
        "heading": nearest.get("heading"),
        "gps_timestamp": nearest["timestamp_seconds"],
        "gps_match_method": "nearest",
        "gps_time_difference_seconds": difference,
    }


def _interpolate_optional(before, after, field_name, fraction):
    before_value = before.get(field_name)
    after_value = after.get(field_name)
    if before_value is None and after_value is None:
        return None
    if before_value is None:
        return after_value
    if after_value is None:
        return before_value
    return before_value + fraction * (after_value - before_value)


def enrich_detections_with_gps(
    detections, gps_records, max_time_difference_seconds=None, gps_source=None
):
    """Return copies of detection dictionaries enriched from timestamped GPS."""
    return [
        {
            **detection,
            **match_gps_timestamp(
                detection.get("timestamp_seconds"),
                gps_records,
                max_time_difference_seconds=max_time_difference_seconds,
            ),
            "gps_source": (
                gps_source
                if gps_source and gps_records
                else "Unavailable"
            ),
        }
        for detection in detections
        if isinstance(detection, dict)
    ]


def _decimal_degrees(value, reference):
    """Convert EXIF DMS coordinates to decimal degrees without guessing."""
    if not value or not reference:
        return None
    try:
        degrees, minutes, seconds = value[:3]
        decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    reference = str(reference).upper()
    if reference in {"S", "W"}:
        decimal *= -1
    elif reference not in {"N", "E"}:
        return None
    return decimal


def extract_image_exif_gps(image_path):
    """Return embedded JPEG EXIF GPS, or explicit unavailable fields.

    This reads only coordinates actually embedded in the image. PNG/WebP and
    images without valid EXIF continue through vision analysis without a map.
    """
    unavailable = {
        "latitude": None,
        "longitude": None,
        "gps_timestamp": None,
        "gps_match_method": "unavailable",
        "gps_time_difference_seconds": None,
        "gps_source": "Unavailable",
    }
    try:
        with Image.open(image_path) as image:
            exif = image.getexif()
            gps_ifd = exif.get_ifd(34853) if exif else {}
    except (OSError, ValueError, AttributeError):
        return unavailable
    if not gps_ifd:
        return unavailable

    tags = {ExifTags.GPSTAGS.get(key, key): value for key, value in gps_ifd.items()}
    latitude = _decimal_degrees(tags.get("GPSLatitude"), tags.get("GPSLatitudeRef"))
    longitude = _decimal_degrees(tags.get("GPSLongitude"), tags.get("GPSLongitudeRef"))
    if latitude is None or longitude is None:
        return unavailable
    # EXIF date/time is retained as source metadata when present. It is not
    # converted to a video-relative timestamp because no reliable offset exists.
    date = tags.get("GPSDateStamp")
    time_value = tags.get("GPSTimeStamp")
    timestamp = None
    if date and time_value:
        try:
            timestamp = f"{date} {int(float(time_value[0])):02d}:{int(float(time_value[1])):02d}:{float(time_value[2]):06.3f} UTC"
        except (TypeError, ValueError, IndexError):
            timestamp = str(date)
    return {
        "latitude": latitude,
        "longitude": longitude,
        "gps_timestamp": timestamp,
        "gps_match_method": "image_exif",
        "gps_time_difference_seconds": 0.0,
        "gps_source": "Embedded image EXIF",
    }


def inspect_video_embedded_gps(video_path):
    """Inspect FFprobe metadata/data packets and extract real video GPS.

    FFprobe is intentionally invoked as an external tool because OpenCV does
    not expose MP4 data streams or timed telemetry. Packet timestamps are
    treated as video-relative seconds, which aligns them with frame metadata.
    """
    ffprobe = shutil.which("ffprobe")
    result = {
        "available": False,
        "source": "Unavailable",
        "format": None,
        "records": [],
        "message": "",
        "video": str(video_path),
    }
    if ffprobe is None:
        result["message"] = "FFprobe is not installed or is not available on PATH."
        return result
    try:
        completed = subprocess.run(
            [
                ffprobe, "-v", "error", "-show_format", "-show_streams",
                "-show_chapters", "-show_programs", "-show_packets", "-show_data",
                "-of", "json", str(video_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        probe = json.loads(completed.stdout or "{}")
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        result["message"] = f"FFprobe inspection failed: {error}"
        return result

    result["format"] = probe.get("format", {})
    records = _extract_ffprobe_records(probe)
    result["records"] = _normalise_records(records)
    if result["records"]:
        result["available"] = True
        result["source"] = "Embedded video telemetry"
        result["message"] = "GPS coordinates extracted from FFprobe metadata or timed data."
    else:
        result["message"] = "No extractable GPS coordinates were found in video metadata or data streams."
    return result


def _extract_ffprobe_records(probe):
    records = []
    format_tags = (probe.get("format") or {}).get("tags") or {}
    location = format_tags.get("location") or format_tags.get("com.apple.quicktime.location.ISO6709")
    coordinate_pair = _parse_iso6709(location)
    if coordinate_pair:
        records.append({"timestamp_seconds": 0.0, **coordinate_pair})

    for stream in probe.get("streams", []):
        tags = stream.get("tags") or {}
        pair = _coordinates_from_mapping(tags)
        if pair:
            records.append({"timestamp_seconds": 0.0, **pair})
    for packet in probe.get("packets", []):
        timestamp = _as_float(packet.get("pts_time", packet.get("dts_time")))
        if timestamp is None:
            continue
        data = _decode_ffprobe_data(packet.get("data", ""))
        records.extend(_parse_telemetry_text(data, timestamp))
        records.extend(_parse_telemetry_text(str(packet.get("tags") or ""), timestamp))
    return records


def _coordinates_from_mapping(mapping):
    latitude = _as_float(mapping.get("latitude", mapping.get("lat")))
    longitude = _as_float(mapping.get("longitude", mapping.get("lon", mapping.get("lng"))))
    if latitude is None or longitude is None:
        return None
    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": mapping.get("altitude", mapping.get("elevation")),
        "speed": mapping.get("speed"),
        "heading": mapping.get("heading", mapping.get("course")),
    }


def _parse_iso6709(value):
    if not value:
        return None
    match = re.search(r"([+-]\d{2,3}(?:\.\d+)?)([+-]\d{2,3}(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?", str(value))
    if not match:
        return None
    return {"latitude": float(match.group(1)), "longitude": float(match.group(2))}


def _decode_ffprobe_data(value):
    if not value:
        return ""
    lines = []
    for line in str(value).splitlines():
        payload = line.split(":", 1)[-1].split("|", 1)[0]
        hex_bytes = re.findall(r"\b[0-9a-fA-F]{2}\b", payload)
        if hex_bytes:
            lines.append(bytes.fromhex("".join(hex_bytes)).decode("utf-8", errors="ignore"))
    return "\n".join(lines) or str(value)


def _parse_telemetry_text(text, timestamp):
    records = []
    for sentence in str(text).splitlines():
        sentence = sentence.strip().strip("\x00")
        fields = sentence.split(",")
        if len(fields) >= 10 and fields[0].lstrip("$").endswith(("GGA", "GNS")):
            latitude = _nmea_coordinate(fields[2], fields[3])
            longitude = _nmea_coordinate(fields[4], fields[5])
            if latitude is not None and longitude is not None:
                records.append({"timestamp_seconds": timestamp, "latitude": latitude, "longitude": longitude, "altitude": fields[9]})
        elif len(fields) >= 9 and fields[0].lstrip("$").endswith("RMC"):
            latitude = _nmea_coordinate(fields[3], fields[4])
            longitude = _nmea_coordinate(fields[5], fields[6])
            if latitude is not None and longitude is not None:
                speed = _as_float(fields[7])
                records.append({"timestamp_seconds": timestamp, "latitude": latitude, "longitude": longitude, "speed": speed * 0.514444 if speed is not None else None, "heading": fields[8]})
    for match in re.finditer(r"(?:latitude|lat)\s*[:=]\s*(-?\d+(?:\.\d+)?).*?(?:longitude|lon|lng)\s*[:=]\s*(-?\d+(?:\.\d+)?)", str(text), re.I | re.S):
        records.append({"timestamp_seconds": timestamp, "latitude": float(match.group(1)), "longitude": float(match.group(2))})
    return records


def _nmea_coordinate(value, reference):
    numeric = _as_float(value)
    if numeric is None or not reference:
        return None
    degrees = int(numeric / 100)
    decimal = degrees + (numeric - degrees * 100) / 60
    return -decimal if reference.upper() in {"S", "W"} else decimal
