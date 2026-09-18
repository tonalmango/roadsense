import unittest

import numpy as np

from frame_preprocessing import bbox_to_original, bbox_to_preprocessed, letterbox_frame


class FramePreprocessingTests(unittest.TestCase):
    def test_landscape_letterbox_preserves_aspect_ratio(self):
        image, transform = letterbox_frame(np.zeros((720, 1280, 3), dtype=np.uint8), 512)
        self.assertEqual(image.shape, (512, 512, 3))
        self.assertEqual(transform["scale"], 0.4)
        self.assertEqual(transform["pad_top"], 112)


    def test_portrait_letterbox_preserves_aspect_ratio(self):
        image, transform = letterbox_frame(np.zeros((1280, 720, 3), dtype=np.uint8), 512)
        self.assertEqual(image.shape, (512, 512, 3))
        self.assertEqual(transform["pad_left"], 112)


    def test_square_letterbox_has_no_padding(self):
        _, transform = letterbox_frame(np.zeros((640, 640, 3), dtype=np.uint8), 512)
        self.assertEqual(transform["pad_left"], 0)
        self.assertEqual(transform["pad_top"], 0)


    def test_coordinate_round_trip_returns_original_box(self):
        _, transform = letterbox_frame(np.zeros((720, 1280, 3), dtype=np.uint8), 512)
        original = [100, 50, 300, 250]
        self.assertEqual(bbox_to_original(bbox_to_preprocessed(original, transform), transform), original)
