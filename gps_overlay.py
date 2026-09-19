"""
gps_overlay.py - get GPS coordinates out of RoadSense inputs.

Order of attempts
  1. Photos : EXIF GPS tags                     (GPS Map Camera .jpg has these)
  2. Videos : container metadata (©xyz/location) (many phones write this - yours don't)
  3. Videos : OCR of the burned-in "Lat ... Long ..." overlay (what GPS Map Camera videos need)

Requires: opencv-python, pytesseract (+ tesseract binary), Pillow
"""
import re
import json
import os
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image
from PIL.ExifTags import GPSTAGS

LAT_LON_RE = re.compile(
    r"(?:lat(?:itude)?)[\s:.=]*(-?\d{1,2}(?:\.\d{3,})?)\D{0,16}?"
    r"(?:long(?:itude)?|lon|lng)[\s:.=]*(-?\d{1,3}(?:\.\d{3,})?)",
    re.I,
)
STAMP_RE = re.compile(r"(\d{2}/\d{2}/\d{4})\s+(\d{1,2}:\d{2}:\d{2})\s*([AP]M)?", re.I)


# ---------------------------------------------------------------- photos
def gps_from_exif(image_path):
    exif = Image.open(image_path)._getexif() or {}
    gps_raw = exif.get(34853)  # GPSInfo
    if not gps_raw:
        return None
    g = {GPSTAGS.get(k, k): v for k, v in gps_raw.items()}

    def dms(v, ref):
        d, m, s = (float(x) for x in v)
        val = d + m / 60 + s / 3600
        return -val if ref in ("S", "W") else val

    return dms(g["GPSLatitude"], g["GPSLatitudeRef"]), dms(g["GPSLongitude"], g["GPSLongitudeRef"])


# ---------------------------------------------------- video: metadata
def gps_from_video_metadata(video_path):
    """Looks for an ISO-6709 string such as +20.4593+085.9016/ in the mp4 tags."""
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "-show_streams", str(video_path)],
            capture_output=True, text=True, check=False,
        ).stdout
        info = json.loads(out or "{}")
    except (FileNotFoundError, json.JSONDecodeError):
        info = {}
    tag_sets = [info.get("format", {}).get("tags", {})] + [s.get("tags", {}) for s in info.get("streams", [])]
    for tags in tag_sets:
        for key, val in tags.items():
            if key.lower() in ("location", "com.apple.quicktime.location.iso6709", "location-eng"):
                pair = _iso6709_pair(val)
                if pair:
                    return pair
    # WhatsApp/phone MP4 files may retain the QuickTime location atom even when
    # ffprobe is not installed. Scan it directly as a dependency-free fallback.
    try:
        with open(video_path, "rb") as file:
            raw = file.read().decode("latin-1", errors="ignore")
    except OSError:
        return None
    for match in re.finditer(r"(?:location|ISO6709|©xyz).{0,96}?([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)", raw, re.I | re.S):
        pair = _validated_pair(match.group(1), match.group(2))
        if pair:
            return pair
    return None


def _iso6709_pair(value):
    match = re.search(r"([+-]\d{1,2}(?:\.\d+)?)([+-]\d{1,3}(?:\.\d+)?)", str(value))
    return _validated_pair(*match.groups()) if match else None


def _validated_pair(latitude, longitude):
    latitude, longitude = float(latitude), float(longitude)
    if -90 <= latitude <= 90 and -180 <= longitude <= 180:
        return latitude, longitude
    return None


# ------------------------------------------------- video: OCR overlay
def _ocr_frame(frame, bottom_fraction=0.45):
    try:
        import pytesseract
    except ImportError:
        return ""
    # Windows installers do not update the PATH of an already-running
    # Streamlit process.  Use the standard installer location when needed.
    executable = shutil.which("tesseract")
    if not executable:
        for installed in (
            r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
            r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe",
        ):
            if os.path.isfile(installed):
                executable = installed
                break
    if not executable:
        return ""
    pytesseract.pytesseract.tesseract_cmd = executable
    h, w = frame.shape[:2]
    roi = frame[int(h * (1 - bottom_fraction)):, :]          # overlay lives in the lower part
    roi = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    try:
        # GPS Map Camera uses light text on a dark translucent panel. Otsu
        # thresholding isolates it much more consistently than a raw frame.
        _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.:+-° "
        return pytesseract.image_to_string(thresholded, config=config).replace("\n", " ")
    except pytesseract.TesseractNotFoundError:
        return ""


def gps_from_video_overlay(video_path, samples=8, bbox=None):
    """
    Sample `samples` frames evenly, OCR each, and majority-vote the result.
    bbox=(lat_min, lat_max, lon_min, lon_max) rejects OCR garbage (default: whole world).
    Returns dict(lat, lon, stamp, votes, frames) or None.
    """
    lat_min, lat_max, lon_min, lon_max = bbox or (-90, 90, -180, 180)
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)     # honour the phone's 90-degree rotation flag
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    votes, stamps = Counter(), []
    for idx in [int(i * (n - 1) / max(samples - 1, 1)) for i in range(samples)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            continue
        text = _ocr_frame(frame)
        pair = parse_overlay_coordinates(text)
        if not pair:
            continue
        lat, lon = pair
        if lat_min <= lat <= lat_max and lon_min <= lon <= lon_max:
            votes[(round(lat, 5), round(lon, 5))] += 1
            s = STAMP_RE.search(text)
            if s:
                stamps.append(" ".join(g for g in s.groups() if g))
    cap.release()
    if not votes:
        return None
    (lat, lon), count = votes.most_common(1)[0]
    return {"lat": lat, "lon": lon, "stamp": stamps[0] if stamps else None,
            "votes": count, "frames": sum(votes.values())}


def parse_overlay_coordinates(text):
    """Parse GPS Map Camera's ``Lat …° Long …°`` text safely."""
    match = LAT_LON_RE.search(text or "")
    if not match:
        return None
    return _validated_pair(match.group(1), match.group(2))


def ocr_engine_available():
    """Whether the optional local Tesseract executable can be used."""
    return bool(
        shutil.which("tesseract")
        or os.path.isfile(r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe")
        or os.path.isfile(r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe")
    )


# -------------------------------------------------------------- entry
def get_gps(path, bbox=None):
    p = Path(path)
    if p.suffix.lower() in (".jpg", ".jpeg"):
        r = gps_from_exif(p)
        return {"lat": r[0], "lon": r[1], "source": "exif"} if r else None
    r = gps_from_video_metadata(p)
    if r:
        return {"lat": r[0], "lon": r[1], "source": "container-metadata"}
    r = gps_from_video_overlay(p, bbox=bbox)
    return {**r, "source": "ocr-overlay"} if r else None


if __name__ == "__main__":
    import sys
    INDIA = (6.0, 38.0, 68.0, 98.0)
    for f in sys.argv[1:]:
        print(f, "->", get_gps(f, bbox=INDIA))
