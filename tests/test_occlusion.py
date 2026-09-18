import unittest

from occlusion import add_occlusion_indicators


def _box(bbox):
    return {"bbox": bbox, "image_width": 100, "image_height": 100}


class OcclusionIndicatorTests(unittest.TestCase):
    def test_boundary_box_gets_possible_occlusion_indicator(self):
        result = add_occlusion_indicators([_box([0, 20, 40, 60])])[0]
        self.assertTrue(result["occlusion_flag"])
        self.assertEqual(result["occlusion_indicator_quality"], "low")
        self.assertIn("left frame edge", result["occlusion_reason"])


    def test_overlapping_boxes_get_possible_occlusion_indicator(self):
        results = add_occlusion_indicators([_box([20, 20, 80, 80]), _box([25, 25, 85, 85])])
        self.assertTrue(all(item["occlusion_flag"] for item in results))
        self.assertIn("overlaps another detection", results[0]["occlusion_reason"])


    def test_clear_box_has_no_occlusion_indicator(self):
        result = add_occlusion_indicators([_box([20, 20, 40, 40])])[0]
        self.assertFalse(result["occlusion_flag"])
        self.assertEqual(result["occlusion_indicator_quality"], "none")
