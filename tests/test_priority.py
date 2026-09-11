"""Unit tests for RAD RoadDamages repair prioritisation."""

import unittest

from priority import classify_priority, prioritize_repair


class RepairPriorityTests(unittest.TestCase):
    def test_high_severity_road_damage_on_arterial_is_critical(self):
        result = prioritize_repair(
            85, "RoadDamages", road_context="arterial", traffic_factor=90, bbox_area_ratio=0.09
        )
        self.assertEqual(result["priority_score"], 87.8)
        self.assertEqual(result["priority_level"], "Critical")
        self.assertIn("road context arterial", result["priority_reason"])

    def test_low_severity_road_damage_with_no_context_is_medium(self):
        result = prioritize_repair(20, "RoadDamages")
        self.assertEqual(result["priority_score"], 42.0)
        self.assertEqual(result["priority_level"], "Medium")
        self.assertIn("neutral prototype baseline", result["priority_reason"])

    def test_non_damage_rad_classes_are_not_prioritised_for_repair(self):
        result = prioritize_repair(90, "Pedestrian")
        self.assertIsNone(result["priority_score"])
        self.assertEqual(result["priority_level"], "N/A")
        self.assertIn("not RoadDamages", result["priority_reason"])

    def test_invalid_optional_context_is_bounded(self):
        result = prioritize_repair(100, "RoadDamages", road_context=200, traffic_factor=-10)
        self.assertEqual(result["priority_score"], 92.0)
        self.assertEqual(result["priority_level"], "Critical")

    def test_priority_thresholds(self):
        self.assertEqual(classify_priority(39.99), "Low")
        self.assertEqual(classify_priority(40), "Medium")
        self.assertEqual(classify_priority(60), "High")
        self.assertEqual(classify_priority(80), "Critical")


if __name__ == "__main__":
    unittest.main()
