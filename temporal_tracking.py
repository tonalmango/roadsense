"""Temporal tracking for detections and optional video-clip classification.

The detection pipeline uses :func:`build_tracks` and
:func:`suppress_duplicates` for JSON detection records. The optional clip
classifier uses :func:`predict_video` when a compatible ``video_best.pt`` and
its model architecture are available.
"""

import importlib
from math import hypot
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "video_best.pt"
CLASS_NAMES = ["HMV", "LMV", "Pedestrian", "RoadDamages", "SpeedBump", "UnsurfacedRoad"]
ROAD_DAMAGE_CLASS = "RoadDamages"
DEFAULT_CONFIDENCE = 0.50
_video_model = None


def _torch_available():
    """Check optional video-classification support without importing it eagerly."""
    return importlib.util.find_spec("torch") is not None


def _device_name():
    if not _torch_available():
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


DEVICE_NAME = _device_name()


def _video_dependencies():
    try:
        import cv2
        import numpy as np
        import torch
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Video classification requires torch, OpenCV, NumPy, and Pillow."
        ) from error
    return cv2, np, torch, Image


def _iou(first, second):
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - overlap
    return overlap / union if union else 0.0


def suppress_duplicates(detections, iou_threshold=0.45, distance_threshold=0.0002):
    """Keep the strongest detection for the same frame/location/class."""
    kept = []
    for item in sorted(detections, key=lambda value: float(value.get("confidence", 0)), reverse=True):
        duplicate = False
        for existing in kept:
            same_class = item.get("damage_type") == existing.get("damage_type")
            same_frame = item.get("frame") == existing.get("frame")
            gps_close = all(item.get(key) is not None and existing.get(key) is not None for key in ("latitude", "longitude")) and hypot(float(item["latitude"]) - float(existing["latitude"]), float(item["longitude"]) - float(existing["longitude"])) <= distance_threshold
            boxes_overlap = item.get("bbox") and existing.get("bbox") and _iou(item["bbox"], existing["bbox"]) >= iou_threshold
            duplicate = same_class and (same_frame and boxes_overlap or gps_close)
            if duplicate:
                break
        if not duplicate:
            kept.append({**item, "duplicate_suppressed": False})
    return kept


def build_tracks(detections, iou_threshold=0.25):
    """Assign stable track IDs to detections ordered by timestamp/frame."""
    tracks = []
    next_id = 1
    result = []
    for item in sorted(detections, key=lambda value: (value.get("timestamp_seconds", 0), value.get("frame", 0))):
        candidates = [track for track in tracks if track["class"] == item.get("damage_type") and _iou(track["bbox"], item.get("bbox", [])) >= iou_threshold]
        track = max(candidates, key=lambda value: value["score"], default=None)
        if track is None:
            track = {"id": next_id, "class": item.get("damage_type"), "bbox": item.get("bbox", []), "score": 0}
            tracks.append(track)
            next_id += 1
        track["bbox"] = item.get("bbox", track["bbox"])
        track["score"] = float(item.get("confidence", 0))
        result.append({**item, "track_id": track["id"]})
    return result


def _frame_tensor(frame, image_size=160):
    """Convert an OpenCV BGR frame to a normalized RGB tensor."""
    cv2, np, torch, _ = _video_dependencies()
    resized = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    array = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_video_model(model_factory=None, model_path=MODEL_PATH):
    """Load a compatible video model once.

    ``model_factory`` must return the architecture used during training. It is
    required for a state-dict checkpoint; TorchScript and serialized module
    checkpoints can be loaded without it.
    """
    global _video_model
    if _video_model is not None:
        return _video_model
    _, _, torch, _ = _video_dependencies()
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Video model not found: {model_path}. Place video_best.pt in model/."
        )
    if model_factory is not None:
        _video_model = model_factory()
        checkpoint = torch.load(model_path, map_location=DEVICE_NAME)
        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        _video_model.load_state_dict(state_dict)
    else:
        try:
            _video_model = torch.jit.load(str(model_path), map_location=DEVICE_NAME)
        except RuntimeError as error:
            checkpoint = torch.load(model_path, map_location=DEVICE_NAME, weights_only=False)
            if not hasattr(checkpoint, "eval"):
                raise RuntimeError(
                    "The video checkpoint contains weights only. Pass the trained model architecture as model_factory."
                ) from error
            _video_model = checkpoint
    _video_model = _video_model.to(DEVICE_NAME)
    _video_model.eval()
    return _video_model


def extract_video_frames(video_path, num_frames=8, image_size=160):
    """Extract evenly spaced frames as ``[1, T, C, H, W]`` tensor."""
    cv2, np, _, _ = _video_dependencies()
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if num_frames < 1:
        raise ValueError("num_frames must be at least 1")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        capture.release()
        raise RuntimeError("Video contains no readable frames.")
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    frames = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = capture.read()
        if success:
            frames.append(_frame_tensor(frame, image_size))
    capture.release()
    if not frames:
        raise RuntimeError("Could not extract frames from video.")
    while len(frames) < num_frames:
        frames.append(frames[-1].clone())
    _, _, torch, _ = _video_dependencies()
    return torch.stack(frames[:num_frames]).unsqueeze(0)


def predict_clip(video_path, confidence_threshold=DEFAULT_CONFIDENCE, model_factory=None, num_frames=8):
    """Classify one clip and return all classes above the confidence threshold."""
    _, _, torch, _ = _video_dependencies()
    model = load_video_model(model_factory=model_factory)
    video_tensor = extract_video_frames(video_path, num_frames=num_frames).to(DEVICE_NAME)
    with torch.no_grad():
        output = model(video_tensor)
    probabilities = torch.sigmoid(output).squeeze(0).detach().cpu().numpy()
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(f"Video model returned {len(probabilities)} outputs; expected {len(CLASS_NAMES)}.")
    predictions = [
        {"class_name": CLASS_NAMES[index], "confidence": float(probability)}
        for index, probability in enumerate(probabilities)
        if probability >= confidence_threshold
    ]
    return {"video": str(video_path), "predictions": predictions}


def predict_video(video_path, confidence_threshold=DEFAULT_CONFIDENCE, model_factory=None):
    """Return whether the optional video classifier detects RoadDamages."""
    result = predict_clip(video_path, confidence_threshold, model_factory=model_factory)
    road_damage = next(
        (item for item in result["predictions"] if item["class_name"] == ROAD_DAMAGE_CLASS),
        None,
    )
    return {
        "video": str(video_path),
        "detected": road_damage is not None,
        "class_name": road_damage["class_name"] if road_damage else None,
        "confidence": road_damage["confidence"] if road_damage else 0.0,
        "predictions": result["predictions"],
    }
