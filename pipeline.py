"""End-to-end RoadSense orchestration for one MP4 dashcam video.

This module coordinates the existing extraction and metadata-aware detection
functions. It does not load a second YOLO model or duplicate scoring logic.
"""

import argparse
import json
import time
from pathlib import Path

def _write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(content, file, indent=2)


def _build_intelligence_artifacts(detections, results_folder, usable_inputs, total_inputs, gps_records=None):
    """Run the model-agnostic intelligence layer over pipeline detections."""
    from active_learning import select_feedback_samples
    from confidence_calibration import calibration_summary, expected_calibration_error
    from data_quality import assess_data_quality, quality_flags
    from evidence_cards import build_evidence_cards, summarize_evidence
    from explainable_risk import explain_detection
    from health_score import calculate_health_score
    from inspection_comparison import compare_inspections
    from inspection_report import create_pdf_report
    from mission_replay import mission_summary, replay_mission
    from repair_queue import build_repair_queue
    from segment_analytics import analyze_segments
    from temporal_tracking import build_tracks, suppress_duplicates

    tracked = [{**item, **explain_detection(item)} for item in build_tracks(detections)]
    unique = suppress_duplicates(tracked)
    health = calculate_health_score(unique, usable_inputs, total_inputs)
    quality = assess_data_quality(unique, gps_records)
    evidence_cards = build_evidence_cards(unique)
    replay = replay_mission(unique)
    quality = assess_data_quality(unique, gps_records)
    labeled_records = [item for item in unique if "correct" in item]
    summary = {
        "health_score": health,
        "data_quality": quality,
        "data_quality_flags": quality_flags(quality),
        "evidence_cards": evidence_cards,
        "evidence_summary": summarize_evidence(evidence_cards),
        "repair_queue": build_repair_queue(unique),
        "segments": analyze_segments(unique),
        "mission_replay": replay,
        "mission_summary": mission_summary(replay),
        "active_learning": select_feedback_samples(unique),
        "confidence_calibration": calibration_summary(labeled_records),
        "confidence_calibration_error": expected_calibration_error(labeled_records),
        "comparison_baseline": compare_inspections([], unique),
        "unique_detection_count": len(unique),
        "suppressed_detection_count": len(detections) - len(unique),
    }
    results_folder = Path(results_folder)
    _write_json(results_folder / "intelligence_summary.json", summary)
    _write_json(results_folder / "evidence_cards.json", summary["evidence_cards"])
    _write_json(results_folder / "repair_queue.json", summary["repair_queue"])
    _write_json(results_folder / "segment_analytics.json", summary["segments"])
    _write_json(results_folder / "mission_replay.json", summary["mission_replay"])
    try:
        create_pdf_report(
            results_folder / "inspection_report.pdf",
            "RoadSense Inspection Report",
            unique,
            summary=health,
        )
        summary["pdf_report"] = "inspection_report.pdf"
    except RuntimeError as error:
        summary["pdf_report"] = None
        summary["pdf_report_error"] = str(error)
    _write_json(results_folder / "intelligence_summary.json", summary)
    return tracked, summary


def _statistics(video_path, frame_metadata, detections, elapsed_seconds):
    sampled_frames = len(frame_metadata.get("frames", []))
    usable_frames = sum(
        1 for frame in frame_metadata.get("frames", []) if frame.get("usable", False)
    )
    return {
        "video": str(video_path),
        "total_video_frames": frame_metadata.get("video", {}).get("total_frames", 0),
        "sampled_frames": sampled_frames,
        "usable_frames": usable_frames,
        "skipped_frames": sampled_frames - usable_frames,
        "detections": len(detections),
        "processing_time_seconds": round(elapsed_seconds, 3),
    }


def run_pipeline(
    video_path,
    frames_folder="frames",
    results_folder="results",
    gps_data_path=None,
    every_seconds=1.0,
    every_n_frames=None,
    road_context=None,
    traffic_factor=None,
    minimum_confidence=None,
    temporal_confirmation_frames=1,
    preprocess_frames=False,
    preprocessing_size=512,
):
    """Run extraction, quality filtering, detection, GPS, and prioritisation.

    The existing detector owns YOLO inference plus severity, GPS, and priority
    enrichment. This function only coordinates those modules and saves final
    hackathon-friendly JSON artifacts.

    Returns a dictionary with ``success``, output paths, and processing stats.
    Errors are caught and returned so command-line demonstrations fail clearly.
    """
    video_path = Path(video_path)
    frames_folder = Path(frames_folder)
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    started_at = time.perf_counter()
    metadata_path = frames_folder / "frame_metadata.json"
    final_path = results_folder / "final_detections.json"
    statistics_path = results_folder / "processing_statistics.json"

    try:
        # Imports are delayed so a missing runtime dependency is reported as a
        # pipeline failure instead of preventing even the command help screen.
        from detection import CLASS_CONFIDENCE_THRESHOLDS, process_frames_with_metadata
        from frame_extraction import extract_frames
        from gps_mapping import inspect_video_embedded_gps, load_gps_records

        print("[1/6] Extracting frames...")
        frame_metadata = extract_frames(
            video_path=video_path,
            output_folder=frames_folder,
            every_n_frames=every_n_frames,
            every_seconds=None if every_n_frames is not None else every_seconds,
            metadata_path=metadata_path,
        )

        print("[2/6] Checking frame quality...")
        sampled_frames = len(frame_metadata["frames"])
        usable_frames = sum(frame["usable"] for frame in frame_metadata["frames"])
        print(
            f"Quality filter: {usable_frames}/{sampled_frames} sampled frames are usable; "
            f"{sampled_frames - usable_frames} are retained but skipped for detection."
        )

        # The metadata-aware detector reuses the existing YOLO, severity, GPS,
        # and priority implementations instead of reimplementing them here.
        print("[3/6] Running YOLO detection...")
        print("[4/6] Calculating severity...")
        print("[5/6] Synchronizing GPS...")
        print("[6/6] Calculating repair priority...")
        detections = process_frames_with_metadata(
            frames_folder=frames_folder,
            metadata_path=metadata_path,
            output_folder=results_folder,
            gps_data_path=gps_data_path,
            road_context=road_context,
            traffic_factor=traffic_factor,
            minimum_confidence=minimum_confidence,
            temporal_confirmation_frames=temporal_confirmation_frames,
            preprocess_frames=preprocess_frames,
            preprocessing_size=preprocessing_size,
        )

        detections, intelligence = _build_intelligence_artifacts(
            detections,
            results_folder,
            usable_frames,
            sampled_frames,
            load_gps_records(gps_data_path),
        )
        _write_json(final_path, detections)
        elapsed_seconds = time.perf_counter() - started_at
        statistics = _statistics(video_path, frame_metadata, detections, elapsed_seconds)
        telemetry = inspect_video_embedded_gps(video_path)
        statistics.update(
            {
                "media_type": "video",
                "confidence_floor": minimum_confidence,
                "class_confidence_thresholds": CLASS_CONFIDENCE_THRESHOLDS,
                "temporal_confirmation_frames": temporal_confirmation_frames,
                "explicit_preprocessing": bool(preprocess_frames),
                "preprocessing_size": preprocessing_size if preprocess_frames else None,
                "gps_source": "External GPS file" if gps_data_path else "Unavailable",
                "embedded_video_gps": telemetry,
                "intelligence_summary": str(results_folder / "intelligence_summary.json"),
            }
        )
        _write_json(statistics_path, statistics)
        print(f"Final detections saved to: {final_path}")
        print(f"Processing statistics saved to: {statistics_path}")
        return {
            "success": True,
            "final_detections_path": str(final_path),
            "statistics_path": str(statistics_path),
            "statistics": statistics,
        }
    except Exception as error:
        elapsed_seconds = time.perf_counter() - started_at
        failure = {
            "success": False,
            "video": str(video_path),
            "error": str(error),
            "processing_time_seconds": round(elapsed_seconds, 3),
        }
        _write_json(statistics_path, failure)
        print(f"Pipeline failed: {error}")
        print(f"Failure details saved to: {statistics_path}")
        return failure


def run_image_pipeline(
    image_path,
    results_folder="results",
    gps_data_path=None,
    road_context=None,
    traffic_factor=None,
    minimum_confidence=None,
    preprocess=False,
    preprocessing_size=512,
):
    """Run the compatible still-image path without treating an image as video."""
    image_path = Path(image_path)
    results_folder = Path(results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    statistics_path = results_folder / "processing_statistics.json"
    final_path = results_folder / "final_detections.json"
    metadata_path = results_folder / "image_metadata.json"
    started_at = time.perf_counter()
    try:
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            raise ValueError("Unsupported image format. Use JPG, JPEG, PNG, or WebP.")
        from detection import CLASS_CONFIDENCE_THRESHOLDS, process_single_image
        from frame_extraction import assess_frame_quality
        from gps_mapping import extract_image_exif_gps
        from capture_metadata import extract_image_capture_date
        import cv2

        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]
        quality = assess_frame_quality(image)
        exif_gps = extract_image_exif_gps(image_path)
        capture_date = extract_image_capture_date(image_path)
        metadata = {
            "source_image": str(image_path),
            "media_type": "image",
            "width": width,
            "height": height,
            **quality,
            "gps": exif_gps,
            "capture_date": capture_date,
        }
        _write_json(metadata_path, metadata)
        # A supplied timestamped GPS file has no trustworthy offset for a still
        # image, so EXIF is preferred and external GPS is not guessed.
        detections, diagnostics = process_single_image(
            image_path=image_path,
            output_folder=results_folder,
            road_context=road_context,
            traffic_factor=traffic_factor,
            minimum_confidence=minimum_confidence,
            gps_data=exif_gps,
            preprocess=preprocess,
            preprocessing_size=preprocessing_size,
        )
        if capture_date:
            for detection in detections:
                detection["capture_date"] = capture_date
            _write_json(results_folder / "detections.json", detections)
        detections, intelligence = _build_intelligence_artifacts(
            detections,
            results_folder,
            int(quality["usable"]),
            1,
            [],
        )
        _write_json(final_path, detections)
        elapsed = time.perf_counter() - started_at
        statistics = {
            "media_type": "image",
            "image": str(image_path),
            "images_processed": 1,
            "usable_inputs": int(quality["usable"]),
            "detections": len(detections),
            "rejected_low_confidence": diagnostics["rejected_low_confidence"],
            "confidence_floor": minimum_confidence,
            "class_confidence_thresholds": CLASS_CONFIDENCE_THRESHOLDS,
            "gps_source": exif_gps["gps_source"],
            "capture_date": capture_date,
            "explicit_preprocessing": bool(preprocess),
            "preprocessing_size": preprocessing_size if preprocess else None,
            "external_gps_ignored": bool(gps_data_path),
            "intelligence_summary": str(results_folder / "intelligence_summary.json"),
            "processing_time_seconds": round(elapsed, 3),
        }
        _write_json(statistics_path, statistics)
        return {
            "success": True,
            "final_detections_path": str(final_path),
            "statistics_path": str(statistics_path),
            "statistics": statistics,
        }
    except Exception as error:
        failure = {
            "success": False,
            "media_type": "image",
            "image": str(image_path),
            "error": str(error),
            "processing_time_seconds": round(time.perf_counter() - started_at, 3),
        }
        _write_json(statistics_path, failure)
        print(f"Image pipeline failed: {error}")
        return failure


def _parse_arguments():
    parser = argparse.ArgumentParser(description="Run the RoadSense MP4 pipeline.")
    parser.add_argument("video", type=Path, help="Path to an MP4 dashcam video.")
    parser.add_argument("--frames-folder", type=Path, default=Path("frames"))
    parser.add_argument("--results-folder", type=Path, default=Path("results"))
    parser.add_argument("--gps-data", type=Path, help="Timestamped CSV/JSON GPS data.")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--every-frames", type=int, help="Save every Nth frame.")
    sampling.add_argument(
        "--every-seconds", type=float, default=1.0, help="Save every X seconds (default: 1)."
    )
    parser.add_argument(
        "--road-context",
        help="Optional prototype context: motorway, arterial, collector, or local.",
    )
    parser.add_argument(
        "--min-confidence", type=float,
        help="Optional confidence floor; class-specific thresholds still apply.",
    )
    parser.add_argument(
        "--temporal-confirmation-frames", type=int, default=1,
        help="Road-condition confirmations required across sampled frames (default: 1).",
    )
    parser.add_argument(
        "--traffic-factor",
        type=float,
        help="Optional prototype traffic score from 0 to 100; not measured by RoadSense.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    result = run_pipeline(
        video_path=arguments.video,
        frames_folder=arguments.frames_folder,
        results_folder=arguments.results_folder,
        gps_data_path=arguments.gps_data,
        every_n_frames=arguments.every_frames,
        every_seconds=arguments.every_seconds,
        road_context=arguments.road_context,
        traffic_factor=arguments.traffic_factor,
        minimum_confidence=arguments.min_confidence,
        temporal_confirmation_frames=arguments.temporal_confirmation_frames,
    )
    raise SystemExit(0 if result["success"] else 1)
