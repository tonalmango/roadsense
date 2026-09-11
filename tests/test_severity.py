"""Unit tests for the prototype RoadSense severity heuristic."""

import unittest

from severity import calculate_severity_score, classify_severity


class SeverityHeuristicTests(unittest.TestCase):
    def test_small_crack_is_minor(self):
        score = calculate_severity_score("crack", 0.002, 100, 20, 0.90)
        self.assertLess(score, 40)
        self.assertEqual(classify_severity(score), "Minor")

    def test_large_crack_is_severe(self):
        score = calculate_severity_score("crack", 0.12, 1000, 50, 0.90)
        self.assertGreaterEqual(score, 70)
        self.assertEqual(classify_severity(score), "Severe")

    def test_small_pothole_is_minor(self):
        score = calculate_severity_score("pothole", 0.004, 50, 50, 0.90)
        self.assertLess(score, 40)
        self.assertEqual(classify_severity(score), "Minor")

    def test_large_pothole_is_severe(self):
        score = calculate_severity_score("pothole", 0.09, 400, 400, 0.80)
        self.assertGreaterEqual(score, 70)
        self.assertEqual(classify_severity(score), "Severe")

    def test_missing_or_invalid_bbox_values_do_not_crash(self):
        score = calculate_severity_score("pothole", None, -10, 0, None)
        self.assertEqual(score, 0.0)
        self.assertEqual(classify_severity(score), "Minor")

    def test_unknown_damage_type_uses_neutral_risk(self):
        score = calculate_severity_score("debris", 0.05, 100, 100, 0.50)
        self.assertEqual(score, 40.0)
        self.assertEqual(classify_severity(score), "Moderate")


if __name__ == "__main__":
    unittest.main()
