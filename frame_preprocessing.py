"""Optional, aspect-ratio-preserving frame normalization for inference."""

import cv2
import numpy as np


def letterbox_frame(frame, target_size=512, pad_color=(114, 114, 114)):
    """Letterbox a BGR frame and return its reversible transform metadata.

    This runs in memory: original extracted frames are never overwritten or
    duplicated.  The returned metadata may be persisted with detections so
    their boxes remain auditable in original-frame coordinates.
    """
    if frame is None or getattr(frame, "ndim", 0) < 2:
        raise ValueError("A readable image frame is required for preprocessing.")
    if isinstance(target_size, int):
        target_width = target_height = target_size
    else:
        target_width, target_height = target_size
    if target_width <= 0 or target_height <= 0:
        raise ValueError("Target preprocessing dimensions must be positive.")

    original_height, original_width = frame.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, round(original_width * scale))
    resized_height = max(1, round(original_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2
    pad_right = target_width - resized_width - pad_left
    pad_bottom = target_height - resized_height - pad_top
    normalized = cv2.copyMakeBorder(resized, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=pad_color)
    return normalized, {
        "enabled": True,
        "method": "aspect_ratio_letterbox",
        "original_width": original_width,
        "original_height": original_height,
        "target_width": target_width,
        "target_height": target_height,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
    }


def bbox_to_preprocessed(bbox, transform):
    """Map an original-frame box to letterboxed coordinates."""
    scale = transform["scale"]
    return [
        bbox[0] * scale + transform["pad_left"], bbox[1] * scale + transform["pad_top"],
        bbox[2] * scale + transform["pad_left"], bbox[3] * scale + transform["pad_top"],
    ]


def bbox_to_original(bbox, transform):
    """Map letterboxed coordinates back to clipped original-frame pixels."""
    scale = transform["scale"]
    if scale <= 0:
        raise ValueError("Preprocessing scale must be positive.")
    width, height = transform["original_width"], transform["original_height"]
    x1 = round((bbox[0] - transform["pad_left"]) / scale)
    y1 = round((bbox[1] - transform["pad_top"]) / scale)
    x2 = round((bbox[2] - transform["pad_left"]) / scale)
    y2 = round((bbox[3] - transform["pad_top"]) / scale)
    return [max(0, min(width, x1)), max(0, min(height, y1)), max(0, min(width, x2)), max(0, min(height, y2))]
