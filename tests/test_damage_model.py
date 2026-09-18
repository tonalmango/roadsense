import unittest

from damage_model import merge_specific_candidates


def _rad_box():
    return {
        "class_id": 3, "damage_type": "RoadDamages", "model_class": "RoadDamages",
        "confidence": .81, "threshold_used": .25, "damage_category": "RoadDamages",
        "damage_subtype": "Unknown / Not classified", "bbox": [10, 10, 90, 90],
        "bbox_width": 80, "bbox_height": 80, "bbox_area": 6400,
        "bbox_area_ratio": .64, "image_width": 100, "image_height": 100,
    }


class DamageModelMergeTests(unittest.TestCase):
    def test_spatial_secondary_match_replaces_generic_final_class(self):
        candidate = {"class_id": 0, "label": "Pothole", "confidence": .91, "threshold": .35, "bbox": [20, 20, 70, 70]}
        merged, refined = merge_specific_candidates([_rad_box()], [candidate])
        self.assertEqual(refined, 1)
        self.assertEqual(merged[0]["damage_type"], "Pothole")
        self.assertTrue(merged[0]["damage_model_used"])
        self.assertEqual(merged[0]["class_id"], 0)
        self.assertEqual(merged[0]["rad_class_id"], 3)
        self.assertAlmostEqual(merged[0]["confidence"], .91)

    def test_unmatched_secondary_box_keeps_roaddamages_fallback(self):
        candidate = {"class_id": 0, "label": "Pothole", "confidence": .91, "threshold": .35, "bbox": [91, 91, 99, 99]}
        merged, refined = merge_specific_candidates([_rad_box()], [candidate])
        self.assertEqual(refined, 0)
        self.assertEqual(merged[0]["damage_type"], "RoadDamages")
        self.assertFalse(merged[0]["damage_model_used"])

    def test_context_detection_is_never_converted_to_road_damage(self):
        context = {**_rad_box(), "class_id": 1, "damage_type": "LMV", "model_class": "LMV"}
        candidate = {"class_id": 0, "label": "Pothole", "confidence": .91, "threshold": .35, "bbox": [20, 20, 70, 70]}
        merged, refined = merge_specific_candidates([context], [candidate])
        self.assertEqual(refined, 0)
        self.assertEqual(merged[0]["damage_type"], "LMV")

    def test_one_specific_box_is_not_reused_for_two_generic_boxes(self):
        second = {**_rad_box(), "bbox": [15, 15, 85, 85]}
        candidate = {"class_id": 0, "label": "Pothole", "confidence": .91, "threshold": .35, "bbox": [20, 20, 70, 70]}
        merged, refined = merge_specific_candidates([_rad_box(), second], [candidate])
        self.assertEqual(refined, 1)
        self.assertEqual(sum(item["damage_type"] == "Pothole" for item in merged), 1)
