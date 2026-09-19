"""Focused tests for the local image/GPS and temporal-filter helpers."""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gps_mapping import extract_image_exif_gps, inspect_video_embedded_gps
from gps_overlay import parse_overlay_coordinates


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

    def test_parses_gps_map_camera_coordinate_stamp(self):
        coordinates = parse_overlay_coordinates("Lat 20.456071° Long 85.901673°")
        self.assertEqual(coordinates, (20.456071, 85.901673))


if __name__ == "__main__":
    unittest.main()
