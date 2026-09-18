"""Read reliable capture dates without inventing a date from file timestamps."""

from datetime import datetime
from pathlib import Path

from PIL import Image


EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


def parse_capture_date(value):
    """Return ISO ``YYYY-MM-DD`` only for an explicitly parseable EXIF date."""
    if not isinstance(value, str):
        return None
    for date_format in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value.strip(), date_format).date().isoformat()
        except ValueError:
            continue
    return None


def extract_image_capture_date(image_path):
    """Read a JPEG EXIF capture date, returning ``None`` when absent/unusable."""
    try:
        with Image.open(Path(image_path)) as image:
            exif = image.getexif()
            for tag in EXIF_DATE_TAGS:
                capture_date = parse_capture_date(exif.get(tag))
                if capture_date:
                    return capture_date
    except (OSError, ValueError):
        return None
    return None
