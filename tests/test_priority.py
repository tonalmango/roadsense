"""Unit tests for the prototype RoadSense repair-priority heuristic."""

import unittest

from priority import classify_priority, prioritize_repair


class RepairPriorityTests(unittest.TestCase):
    def test_high_severity_pothole_on_arterial_is_critical(self):
        result = prioritize_repair(
            85, "pothole", road_context="arterial", traffic_factor=90, bbox_area_ratio=0.09
        )
        self.assertEqual(result["priority_score"], 87.8)
        self.assertEqual(result["priority_level"], "Critical")
        self.assertIn("road context arterial", result["priority_reason"])
        self.assertIn("traffic provided score 90", result["priority_reason"])

    def test_low_severity_crack_with_no_context_is_low(self):
        result = prioritize_repair(20, "crack")
        self.assertEqual(result["priority_score"], 34.0)
        self.assertEqual(result["priority_level"], "Low")
        self.assertIn("neutral prototype baseline", result["priority_reason"])

    def test_unknown_damage_type_uses_neutral_damage_risk(self):
        result = prioritize_repair(45, "unlisted", road_context="local", traffic_factor=30)
        self.assertEqual(result["priority_score"], 44.2)
        self.assertEqual(result["priority_level"], "Medium")

    def test_invalid_optional_context_is_bounded(self):
        result = prioritize_repair(100, "pothole", road_context=200, traffic_factor=-10)
        self.assertEqual(result["priority_score"], 92.0)
        self.assertEqual(result["priority_level"], "Critical")

    def test_priority_thresholds(self):
        self.assertEqual(classify_priority(39.99), "Low")
        self.assertEqual(classify_priority(40), "Medium")
        self.assertEqual(classify_priority(60), "High")
        self.assertEqual(classify_priority(80), "Critical")


if __name__ == "__main__":
    unittest.main()
