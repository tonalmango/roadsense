"""Inspect a video for embedded GPS telemetry and report extracted samples."""

import argparse
import json
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gps_mapping import inspect_video_embedded_gps


def main():
    parser = argparse.ArgumentParser(description="Inspect embedded video GPS telemetry.")
    parser.add_argument("video", type=Path, help="Path to a video file.")
    arguments = parser.parse_args()
    if not arguments.video.is_file():
        raise SystemExit(f"Video not found: {arguments.video}")

    capture = cv2.VideoCapture(str(arguments.video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    capture.release()
    duration = frame_count / fps if fps > 0 else None
    telemetry = inspect_video_embedded_gps(arguments.video)
    records = telemetry.get("records", [])

    print(f"Video: {arguments.video}")
    print(f"Duration: {duration:.3f} seconds" if duration is not None else "Duration: unavailable")
    print(f"FPS: {fps:.3f}" if fps > 0 else "FPS: unavailable")
    print(f"GPS detected: {'YES' if records else 'NO'}")
    print(f"GPS format: {telemetry.get('source', 'Unavailable')}")
    print(f"GPS source: {telemetry.get('source', 'Unavailable')}")
    print(f"Number of GPS samples: {len(records)}")
    if records:
        first = records[0]
        last = records[-1]
        print("First GPS sample:")
        print(f"timestamp: {first['timestamp_seconds']}")
        print(f"latitude: {first['latitude']}")
        print(f"longitude: {first['longitude']}")
        print("Last GPS sample:")
        print(f"timestamp: {last['timestamp_seconds']}")
        print(f"latitude: {last['latitude']}")
        print(f"longitude: {last['longitude']}")
    else:
        print(f"Reason: {telemetry.get('message', 'No extractable GPS coordinates found.')}")
    print("FFprobe format details:")
    print(json.dumps(telemetry.get("format") or {}, indent=2))


if __name__ == "__main__":
    main()
