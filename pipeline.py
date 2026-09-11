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
        from detection import process_frames_with_metadata
        from frame_extraction import extract_frames

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
        )

        _write_json(final_path, detections)
        elapsed_seconds = time.perf_counter() - started_at
        statistics = _statistics(video_path, frame_metadata, detections, elapsed_seconds)
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
    )
    raise SystemExit(0 if result["success"] else 1)
