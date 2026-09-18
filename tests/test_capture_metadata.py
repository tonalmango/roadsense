import unittest

from capture_metadata import extract_image_capture_date, parse_capture_date


class CaptureMetadataTests(unittest.TestCase):
    def test_parses_exif_capture_date(self):
        self.assertEqual(parse_capture_date("2026:09:18 08:15:00"), "2026-09-18")


    def test_invalid_capture_date_returns_none(self):
        self.assertIsNone(parse_capture_date("not a date"))


    def test_image_without_exif_returns_none(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "no_exif.jpg"
            image_path.write_bytes(b"not a readable image")
            self.assertIsNone(extract_image_capture_date(image_path))

    def test_reads_valid_embedded_exif_capture_date(self):
        import tempfile
        from pathlib import Path
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "dated.jpg"
            exif = Image.Exif()
            exif[36867] = "2026:09:18 08:15:00"
            Image.new("RGB", (10, 10), "white").save(image_path, exif=exif)
            self.assertEqual(extract_image_capture_date(image_path), "2026-09-18")
