"""Road Sense - Video Frame Extraction Module"""

import cv2
from pathlib import Path


def extract_frames(video_path, output_folder, frame_interval=10):
    """
    Extract frames from a video.

    Parameters:
        video_path: Path to input video.
        output_folder: Folder where frames will be saved.
        frame_interval: Save one frame after every N frames.

    Returns:
        List of saved frame paths.
    """

    video_path = Path(video_path)
    output_folder = Path(output_folder)

    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    saved_frames = []
    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        # Save every Nth frame
        if frame_number % frame_interval == 0:

            frame_name = f"frame_{frame_number:06d}.jpg"
            frame_path = output_folder / frame_name

            cv2.imwrite(str(frame_path), frame)

            saved_frames.append(str(frame_path))

        frame_number += 1

    cap.release()

    print(f"Total frames processed: {frame_number}")
    print(f"Frames saved: {len(saved_frames)}")

    return saved_frames


# --------------------------------------------------
# TEST FRAME EXTRACTION
# --------------------------------------------------

if __name__ == "__main__":

    print("Road Sense Frame Extraction Module")
    print("----------------------------------")

    # Video folder
    video_folder = Path("videos")

    # Frame output folder
    frame_folder = Path("frames")

    # Find video files
    video_files = list(video_folder.glob("*.mp4"))

    if not video_files:
        print("No .mp4 video found in the videos folder.")
        print("Put a road video inside:")
        print("Road_Sense/videos/")

    else:

        video_path = video_files[0]

        print(f"Using video: {video_path}")

        extract_frames(
            video_path=video_path,
            output_folder=frame_folder,
            frame_interval=10
        )
