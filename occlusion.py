"""Explainable visual occlusion indicators for RoadSense detections.

This module does not assert that a physical road feature is occluded.  It only
flags observable image evidence that can make a detection incomplete or less
easy to inspect: a box cut by a frame edge or a strongly overlapping box.
"""


def bbox_iou(first_bbox, second_bbox):
    """Return intersection-over-union for two ``[x1, y1, x2, y2]`` boxes."""
    ax1, ay1, ax2, ay2 = first_bbox
    bx1, by1, bx2, by2 = second_bbox
    overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
    overlap = overlap_width * overlap_height
    union = max(0, ax2 - ax1) * max(0, ay2 - ay1) + max(0, bx2 - bx1) * max(0, by2 - by1) - overlap
    return overlap / union if union else 0.0


def add_occlusion_indicators(detections, edge_margin_pixels=4, overlap_iou_threshold=0.50):
    """Return detection copies enriched with conservative occlusion indicators.

    The input is a same-frame collection.  A flag is raised only for an image
    boundary truncation or high bounding-box overlap; neither is presented as
    proof of real-world occlusion.
    """
    enriched = []
    for detection in detections:
        item = dict(detection)
        bbox = item.get("bbox")
        width, height = item.get("image_width"), item.get("image_height")
        reasons = []
        if (
            isinstance(bbox, (list, tuple)) and len(bbox) == 4
            and isinstance(width, (int, float)) and isinstance(height, (int, float))
        ):
            x1, y1, x2, y2 = bbox
            edges = []
            if x1 <= edge_margin_pixels:
                edges.append("left")
            if y1 <= edge_margin_pixels:
                edges.append("top")
            if x2 >= width - edge_margin_pixels:
                edges.append("right")
            if y2 >= height - edge_margin_pixels:
                edges.append("bottom")
            if edges:
                reasons.append(f"bounding box reaches the {', '.join(edges)} frame edge")

            for other in detections:
                if other is detection or other.get("bbox") == bbox:
                    continue
                if bbox_iou(bbox, other.get("bbox", [0, 0, 0, 0])) >= overlap_iou_threshold:
                    reasons.append("bounding box substantially overlaps another detection")
                    break

        item["occlusion_flag"] = bool(reasons)
        item["occlusion_indicator_quality"] = "medium" if len(reasons) > 1 else ("low" if reasons else "none")
        item["occlusion_reason"] = (
            "Possible occlusion indicator: " + "; ".join(reasons) + "."
            if reasons else "No boundary-truncation or high-overlap indicator observed."
        )
        enriched.append(item)
    return enriched
