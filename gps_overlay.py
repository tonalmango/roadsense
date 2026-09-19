"""Extract GPS coordinates from RoadSense photos and videos.

Attempts, in order: image EXIF, video container metadata, and the GPS Map
Camera coordinate overlay rendered into the video frame.
"""
import json
import os
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image
from PIL.ExifTags import GPSTAGS


LAT_LON_RE = re.compile(
    r"Lat\s*[:.]?\s*(-?\d{1,2}\.\d{3,})\D{0,6}?\s*Long\s*[:.]?\s*(-?\d{1,3}\.\d{3,})",
    re.I,
)
STAMP_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2}:\d{2})\s*([AP]M)?", re.I)


def gps_from_exif(image_path):
    """Return decimal EXIF GPS coordinates for a JPEG, when present."""
    exif = Image.open(image_path)._getexif() or {}
    gps_raw = exif.get(34853)
    if not gps_raw:
        return None
    tags = {GPSTAGS.get(key, key): value for key, value in gps_raw.items()}

    def dms(value, reference):
        degrees, minutes, seconds = (float(part) for part in value)
        decimal = degrees + minutes / 60 + seconds / 3600
        return -decimal if reference in ("S", "W") else decimal

    return (
        dms(tags["GPSLatitude"], tags["GPSLatitudeRef"]),
        dms(tags["GPSLongitude"], tags["GPSLongitudeRef"]),
    )


def gps_from_video_metadata(video_path):
    """Read ISO-6709 coordinates from supported MP4/MOV metadata tags."""
    try:
        output = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        info = json.loads(output or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        return None

    tag_sets = [info.get("format", {}).get("tags", {})] + [
        stream.get("tags", {}) for stream in info.get("streams", [])
    ]
    for tags in tag_sets:
        for key, value in tags.items():
            if key.lower() in ("location", "com.apple.quicktime.location.iso6709", "location-eng"):
                match = re.match(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)", value)
                if match:
                    return float(match.group(1)), float(match.group(2))
    return None


def _ocr_frame(frame, bottom_fraction=0.45):
    """OCR the lower GPS-overlay portion of one video frame."""
    try:
        import pytesseract
    except ImportError:
        return ""

    # An existing Streamlit process does not inherit PATH changes made by the
    # Windows Tesseract installer, so use its standard location if necessary.
    if not shutil.which("tesseract"):
        installed = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.isfile(installed):
            pytesseract.pytesseract.tesseract_cmd = installed

    height, _ = frame.shape[:2]
    roi = frame[int(height * (1 - bottom_fraction)):, :]
    roi = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    try:
        return pytesseract.image_to_string(255 - grayscale, config="--psm 6").replace("\n", " ")
    except pytesseract.TesseractNotFoundError:
        return ""


def gps_from_video_overlay(video_path, samples=8, bbox=None):
    """OCR sampled GPS Map Camera frames and majority-vote valid coordinates."""
    lat_min, lat_max, lon_min, lon_max = bbox or (-90, 90, -180, 180)
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    votes, stamps = Counter(), []
    try:
        for index in [int(i * (frame_count - 1) / max(samples - 1, 1)) for i in range(samples)]:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            readable, frame = capture.read()
            if not readable:
                continue
            match = LAT_LON_RE.search(_ocr_frame(frame))
            if not match:
                continue
            latitude, longitude = float(match.group(1)), float(match.group(2))
            if lat_min <= latitude <= lat_max and lon_min <= longitude <= lon_max:
                votes[(round(latitude, 5), round(longitude, 5))] += 1
    finally:
        capture.release()

    if not votes:
        return None
    (latitude, longitude), count = votes.most_common(1)[0]
    return {"lat": latitude, "lon": longitude, "votes": count, "frames": sum(votes.values())}


def get_gps(path, bbox=None):
    """Return embedded GPS data as a source-labelled dictionary, or ``None``."""
    path = Path(path)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        coordinates = gps_from_exif(path)
        return {"lat": coordinates[0], "lon": coordinates[1], "source": "exif"} if coordinates else None
    coordinates = gps_from_video_metadata(path)
    if coordinates:
        return {"lat": coordinates[0], "lon": coordinates[1], "source": "container-metadata"}
    coordinates = gps_from_video_overlay(path, bbox=bbox)
    return {**coordinates, "source": "ocr-overlay"} if coordinates else None
