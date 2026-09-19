"""Unit tests for RoadSense timestamp-based GPS matching."""

import tempfile
import unittest
from pathlib import Path

from gps_mapping import enrich_detections_with_gps, load_gps_records, match_gps_timestamp


GPS_RECORDS = [
    {"timestamp_seconds": 0.0, "latitude": 20.0, "longitude": 85.0},
    {"timestamp_seconds": 2.0, "latitude": 22.0, "longitude": 87.0},
]


class GpsSynchronizationTests(unittest.TestCase):
    def test_exact_timestamp_match(self):
        result = match_gps_timestamp(2.0, GPS_RECORDS)
        self.assertEqual(result["gps_match_method"], "nearest")
        self.assertEqual(result["gps_time_difference_seconds"], 0.0)
        self.assertEqual(result["latitude"], 22.0)

    def test_interpolation_between_records(self):
        result = match_gps_timestamp(1.0, GPS_RECORDS)
        self.assertEqual(result["gps_match_method"], "interpolated")
        self.assertEqual(result["latitude"], 21.0)
        self.assertEqual(result["longitude"], 86.0)
        self.assertIsNone(result["altitude"])
        self.assertEqual(result["gps_time_difference_seconds"], 0.0)

    def test_nearest_timestamp_after_last_record(self):
        result = match_gps_timestamp(3.5, GPS_RECORDS)
        self.assertEqual(result["gps_match_method"], "nearest")
        self.assertEqual(result["gps_timestamp"], 2.0)
        self.assertEqual(result["gps_time_difference_seconds"], 1.5)

    def test_timestamp_before_first_record(self):
        result = match_gps_timestamp(-1.0, GPS_RECORDS)
        self.assertEqual(result["gps_match_method"], "nearest")
        self.assertEqual(result["gps_timestamp"], 0.0)
        self.assertEqual(result["gps_time_difference_seconds"], 1.0)

    def test_missing_gps_is_explicitly_unavailable(self):
        result = match_gps_timestamp(1.0, [])
        self.assertEqual(result["gps_match_method"], "unavailable")
        self.assertIsNone(result["latitude"])
        self.assertIsNone(result["gps_time_difference_seconds"])

    def test_duplicate_timestamps_are_averaged(self):
        duplicate_records = [
            {"timestamp_seconds": 1.0, "latitude": 20.0, "longitude": 85.0},
            {"timestamp_seconds": 1.0, "latitude": 22.0, "longitude": 87.0},
        ]
        result = match_gps_timestamp(1.0, duplicate_records)
        self.assertEqual(result["latitude"], 21.0)
        self.assertEqual(result["longitude"], 86.0)

    def test_enrichment_preserves_detection_fields(self):
        detection = {"id": 1, "timestamp_seconds": 1.0, "damage_type": "pothole"}
        enriched = enrich_detections_with_gps([detection], GPS_RECORDS)[0]
        self.assertEqual(enriched["id"], 1)
        self.assertEqual(enriched["gps_match_method"], "interpolated")
        self.assertEqual(enriched["latitude"], 21.0)

    def test_optional_telemetry_fields_are_interpolated(self):
        records = [
            {"timestamp_seconds": 0, "latitude": 20, "longitude": 85, "altitude": 10, "speed": 2, "heading": 90},
            {"timestamp_seconds": 2, "latitude": 22, "longitude": 87, "altitude": 14, "speed": 6, "heading": 100},
        ]
        result = match_gps_timestamp(1, records)
        self.assertEqual(result["altitude"], 12)
        self.assertEqual(result["speed"], 4)
        self.assertEqual(result["heading"], 95)

    def test_gpx_track_points_use_elapsed_time(self):
        gpx = """<gpx xmlns=\"http://www.topografix.com/GPX/1/1\"><trk><trkseg>
        <trkpt lat=\"20.0\" lon=\"85.0\"><ele>10</ele><time>2026-09-19T00:00:00Z</time></trkpt>
        <trkpt lat=\"20.2\" lon=\"85.2\"><ele>14</ele><time>2026-09-19T00:00:02Z</time></trkpt>
        </trkseg></trk></gpx>"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.gpx"
            path.write_text(gpx, encoding="utf-8")
            records = load_gps_records(path)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[1]["timestamp_seconds"], 2.0)
        self.assertEqual(records[1]["altitude"], 14.0)


if __name__ == "__main__":
    unittest.main()
