"""Extract sampled dashcam frames and record explainable quality metadata.

Example:
    python frame_extraction.py videos/dashcam.mp4 --every-seconds 1
"""

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_BLUR_THRESHOLD = 40.0
DEFAULT_MIN_BRIGHTNESS = 40.0
DEFAULT_MAX_BRIGHTNESS = 215.0


def assess_frame_quality(
    frame,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    min_brightness=DEFAULT_MIN_BRIGHTNESS,
    max_brightness=DEFAULT_MAX_BRIGHTNESS,
):
    """Return simple, non-ML frame-quality measurements.

    ``blur_score`` is the variance of the grayscale Laplacian: lower values
    generally indicate fewer sharp edges and therefore a blurrier frame.
    ``brightness_score`` is the average grayscale pixel value (0--255).
    The thresholds are flags only; callers still save borderline frames.
    """
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    brightness_score = float(grayscale.mean())

    usable = (
        blur_score >= blur_threshold
        and min_brightness <= brightness_score <= max_brightness
    )

    return {
        "blur_score": round(blur_score, 2),
        "brightness_score": round(brightness_score, 2),
        "usable": usable,
    }


def _get_video_properties(capture):
    """Read video properties and validate the FPS needed for timestamps."""
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or fps != fps:  # NaN does not equal itself.
        raise ValueError("Video has an invalid or unavailable FPS value.")
    if total_frames <= 0:
        raise ValueError("Video contains no readable frames.")
    if width <= 0 or height <= 0:
        raise ValueError("Video has invalid frame dimensions.")

    return {
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
        "width": width,
        "height": height,
    }


def extract_frames(
    video_path,
    output_folder="frames",
    every_n_frames=None,
    every_seconds=1.0,
    metadata_path=None,
    blur_threshold=DEFAULT_BLUR_THRESHOLD,
    min_brightness=DEFAULT_MIN_BRIGHTNESS,
    max_brightness=DEFAULT_MAX_BRIGHTNESS,
):
    """Extract sampled MP4 frames and write their metadata as JSON.

    Choose one sampling method: ``every_n_frames`` or ``every_seconds``.
    The default saves one frame per second. Every sampled frame is written even
    when its quality is flagged as unusable.

    Returns:
        The metadata dictionary that is also written to ``metadata_path``.
    """
    video_path = Path(video_path)
    output_folder = Path(output_folder)

    if video_path.suffix.lower() != ".mp4":
        raise ValueError("RoadSense frame extraction accepts MP4 video files.")
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if every_n_frames is not None and every_seconds is not None:
        raise ValueError("Choose either every_n_frames or every_seconds, not both.")
    if every_n_frames is None and every_seconds is None:
        raise ValueError("Set every_n_frames or every_seconds.")
    if every_n_frames is not None and every_n_frames <= 0:
        raise ValueError("every_n_frames must be greater than zero.")
    if every_seconds is not None and every_seconds <= 0:
        raise ValueError("every_seconds must be greater than zero.")
    if min_brightness > max_brightness:
        raise ValueError("min_brightness cannot be greater than max_brightness.")

    output_folder.mkdir(parents=True, exist_ok=True)
    metadata_path = Path(metadata_path) if metadata_path else output_folder / "frame_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"Could not open video: {video_path}")

    try:
        video = _get_video_properties(capture)
        # A time interval becomes a stable number of frames for this video.
        sample_interval = (
            int(every_n_frames)
            if every_n_frames is not None
            else max(1, round(every_seconds * video["fps"]))
        )

        video_summary = {
            key: round(value, 3) if isinstance(value, float) else value
            for key, value in video.items()
        }
        video_summary["sampling"] = {
            "mode": "frames" if every_n_frames is not None else "seconds",
            "interval": every_n_frames if every_n_frames is not None else every_seconds,
            "effective_frame_interval": sample_interval,
        }
        metadata = {
            "source_video": str(video_path),
            "video": video_summary,
            "quality_thresholds": {
                "blur_laplacian_variance_minimum": blur_threshold,
                "brightness_minimum": min_brightness,
                "brightness_maximum": max_brightness,
            },
            "frames": [],
        }

        frame_number = 0
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_number % sample_interval == 0:
                frame_name = f"frame_{frame_number:06d}.jpg"
                frame_path = output_folder / frame_name
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Could not save frame: {frame_path}")

                frame_height, frame_width = frame.shape[:2]
                quality = assess_frame_quality(
                    frame, blur_threshold, min_brightness, max_brightness
                )
                metadata["frames"].append(
                    {
                        "frame_number": frame_number,
                        "timestamp_seconds": round(frame_number / video["fps"], 3),
                        "filename": frame_name,
                        "width": frame_width,
                        "height": frame_height,
                        **quality,
                    }
                )
            frame_number += 1

        if frame_number == 0:
            raise ValueError("Video contained no readable frames.")

        metadata["frames_read"] = frame_number
        metadata["frames_saved"] = len(metadata["frames"])
        if frame_number < video["total_frames"]:
            metadata["read_warning"] = (
                "Reading ended before the video's reported frame count. "
                "The remaining frames may be unreadable."
            )
        with open(metadata_path, "w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)

        print(f"Video: {video_path}")
        print(f"FPS: {video['fps']:.3f} | Duration: {video['duration_seconds']:.3f} seconds")
        print(f"Frames read: {frame_number} | Frames saved: {len(metadata['frames'])}")
        if "read_warning" in metadata:
            print(f"Warning: {metadata['read_warning']}")
        print(f"Metadata saved to: {metadata_path}")
        return metadata
    finally:
        capture.release()


def _parse_arguments():
    parser = argparse.ArgumentParser(
        description="Extract sampled frames from an MP4 dashcam video."
    )
    parser.add_argument("video", type=Path, help="Path to an MP4 dashcam video.")
    parser.add_argument("--output-folder", type=Path, default=Path("frames"))
    sampling_group = parser.add_mutually_exclusive_group()
    sampling_group.add_argument("--every-frames", type=int, help="Save one frame every N frames.")
    sampling_group.add_argument(
        "--every-seconds",
        type=float,
        default=1.0,
        help="Save one frame every X seconds (default: 1).",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        help="JSON output path (default: OUTPUT_FOLDER/frame_metadata.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = _parse_arguments()
    extract_frames(
        video_path=arguments.video,
        output_folder=arguments.output_folder,
        every_n_frames=arguments.every_frames,
        every_seconds=None if arguments.every_frames is not None else arguments.every_seconds,
        metadata_path=arguments.metadata_path,
    )
