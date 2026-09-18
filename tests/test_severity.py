"""Unit tests for the RAD RoadDamages severity heuristic."""

import unittest

from severity import calculate_severity, calculate_severity_score, classify_severity


class SeverityHeuristicTests(unittest.TestCase):
    def test_small_road_damage_is_minor(self):
        score = calculate_severity_score("RoadDamages", 0.002, 100, 20, 0.90)
        self.assertEqual(score, 10.4)
        self.assertEqual(classify_severity(score), "Minor")

    def test_large_road_damage_is_severe(self):
        score = calculate_severity_score("RoadDamages", 0.12, 1000, 50, 0.90)
        self.assertEqual(score, 79.0)
        self.assertEqual(classify_severity(score), "Severe")

    def test_invalid_road_damage_bbox_does_not_crash(self):
        score = calculate_severity_score("RoadDamages", None, -10, 0, None)
        self.assertEqual(score, 0.0)
        self.assertEqual(classify_severity(score), "Minor")

    def test_contextual_rad_classes_have_no_severity(self):
        for class_name in ("HMV", "LMV", "Pedestrian", "SpeedBump", "UnsurfacedRoad"):
            self.assertIsNone(calculate_severity_score(class_name, 0.1, 100, 100, 0.9))
            self.assertEqual(calculate_severity(class_name, 0.9, 0.1, 100, 100), "N/A")

    def test_unknown_class_has_no_severity(self):
        self.assertIsNone(calculate_severity_score("unknown", 0.05, 100, 100, 0.5))
        self.assertEqual(classify_severity(None), "N/A")

    def test_specific_pothole_uses_same_reproducible_damage_heuristic(self):
        score = calculate_severity_score("Pothole", 0.002, 100, 20, 0.90)
        self.assertEqual(score, 10.4)
        self.assertEqual(classify_severity(score), "Minor")


if __name__ == "__main__":
    unittest.main()
