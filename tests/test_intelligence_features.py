import unittest

from active_learning import select_feedback_samples
from active_learning import record_feedback
from confidence_calibration import calibration_summary, expected_calibration_error
from data_quality import assess_data_quality, quality_flags
from evidence_cards import build_evidence_cards, summarize_evidence
from health_score import calculate_health_score
from inspection_comparison import compare_inspections
from inspection_report import report_payload
from mission_replay import mission_summary, replay_mission
from repair_queue import build_repair_queue
from segment_analytics import analyze_segments
from temporal_tracking import build_tracks, suppress_duplicates


class IntelligenceFeatureTests(unittest.TestCase):
    def setUp(self):
        self.records = [
            {"damage_type": "RoadDamages", "confidence": 0.8, "severity_score": 70, "priority_score": 80, "priority_level": "Critical", "bbox": [0, 0, 10, 10], "frame": 1, "latitude": 20.0, "longitude": 85.0, "timestamp_seconds": 1},
            {"damage_type": "RoadDamages", "confidence": 0.4, "severity_score": 30, "priority_score": 45, "priority_level": "Medium", "bbox": [20, 20, 30, 30], "frame": 2, "latitude": 20.1, "longitude": 85.1, "timestamp_seconds": 2},
        ]

    def test_core_feature_modules(self):
        self.assertEqual(len(suppress_duplicates(self.records + [self.records[0]])), 2)
        self.assertEqual({item["track_id"] for item in build_tracks(self.records)}, {1, 2})
        self.assertEqual(len(build_repair_queue(self.records)), 2)
        self.assertEqual(len(analyze_segments(self.records)), 2)
        self.assertEqual(calculate_health_score(self.records, 9, 10)["band"], "Watch")
        self.assertEqual(compare_inspections(self.records, self.records[:1])["status"], "Improved")
        self.assertEqual(assess_data_quality(self.records)["gps_coverage"], 1.0)
        self.assertEqual(expected_calibration_error([{"confidence": 1.0, "correct": True}]), 0.0)
        self.assertEqual(len(select_feedback_samples(self.records, 1)), 1)

    def test_expanded_feature_outputs(self):
        cards = build_evidence_cards(self.records)
        self.assertEqual(summarize_evidence(cards)["total"], 2)
        self.assertIn("urgency", build_repair_queue(self.records)[0])
        self.assertIn("risk_band", analyze_segments(self.records)[0])
        self.assertIn("average_severity", calculate_health_score(self.records))
        self.assertIn("types", compare_inspections([], self.records)["after"])
        self.assertEqual(mission_summary(replay_mission(self.records))["events"], 2)
        self.assertEqual(report_payload("Test", self.records)["detection_count"], 2)
        quality = assess_data_quality(self.records)
        self.assertEqual(quality_flags(quality), [])
        self.assertIn("quality", calibration_summary([{"confidence": 1.0, "correct": True}]))
        self.assertEqual(record_feedback(self.records[0], "RoadDamages")["feedback_status"], "Reviewed")


if __name__ == "__main__":
    unittest.main()