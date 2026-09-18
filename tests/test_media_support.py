"""Focused tests for the local image/GPS and temporal-filter helpers."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gps_mapping import extract_image_exif_gps, inspect_video_embedded_gps


class MediaSupportTests(unittest.TestCase):
    def test_image_without_exif_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plain.jpg"
            Image.new("RGB", (10, 10), "white").save(path)
            result = extract_image_exif_gps(path)
        self.assertEqual(result["gps_match_method"], "unavailable")
        self.assertEqual(result["gps_source"], "Unavailable")
        self.assertIsNone(result["latitude"])

    def test_video_telemetry_capability_does_not_invent_coordinates(self):
        result = inspect_video_embedded_gps("video.mp4")
        self.assertFalse(result["available"])
        self.assertEqual(result["source"], "Unavailable")


if __name__ == "__main__":
    unittest.main()
