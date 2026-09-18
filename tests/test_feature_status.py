import unittest

from feature_status import build_feature_status


class FeatureStatusTests(unittest.TestCase):
    def test_status_panel_never_claims_embedded_mp4_gps(self):
        rows = build_feature_status(media_type="video", has_gps=False, has_capture_date=False)
        status = {row["Capability"]: row["Status"] for row in rows}
        self.assertEqual(status["Embedded MP4 GPS"], "Not implemented")
        self.assertEqual(status["Timestamp GPS synchronization"], "Not available for this input")


    def test_image_exif_and_capture_date_are_conditionally_available(self):
        rows = build_feature_status(media_type="image", has_gps=True, has_capture_date=True)
        status = {row["Capability"]: row["Status"] for row in rows}
        self.assertEqual(status["EXIF GPS"], "Available when metadata exists")
        self.assertEqual(status["Reliable capture-date filter"], "Available when metadata exists")
