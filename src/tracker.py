"""
Kitchen-Cam: Tracker Wrapper
Extracts tracking data (track IDs, bounding boxes, classes, confidences)
from ultralytics YOLO results with BoT-SORT.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TrackedObject:
    """A single tracked detection in a frame."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2) in pixel coordinates
    center: tuple  # (cx, cy) center point


@dataclass
class TrackingResult:
    """All tracked objects from a single frame."""

    persons: List[TrackedObject] = field(default_factory=list)
    gloves: List[TrackedObject] = field(default_factory=list)
    no_gloves: List[TrackedObject] = field(default_factory=list)
    hairnets: List[TrackedObject] = field(default_factory=list)
    no_hairnets: List[TrackedObject] = field(default_factory=list)
    chef_hats: List[TrackedObject] = field(default_factory=list)
    all_objects: List[TrackedObject] = field(default_factory=list)


def parse_tracking_results(
    results: object,
    class_map: Dict[int, str],
) -> TrackingResult:
    """Parse ultralytics Results (with tracking) into structured TrackedObjects.

    Args:
        results: A single ultralytics Results object from model.predict/track.
        class_map: Mapping of class_id → class_name from config.

    Returns:
        TrackingResult with objects categorized by type.
    """
    tracking = TrackingResult()

    if results is None:
        return tracking

    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return tracking

    # Extract arrays
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
    cls_ids = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes.cls, "cpu") else np.array(boxes.cls, dtype=int)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)

    # Track IDs (may be None if tracking not enabled)
    track_ids: Optional[np.ndarray] = None
    if boxes.id is not None:
        track_ids = boxes.id.cpu().numpy().astype(int) if hasattr(boxes.id, "cpu") else np.array(boxes.id, dtype=int)

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        cls_id = int(cls_ids[i])
        conf = float(confs[i])
        tid = int(track_ids[i]) if track_ids is not None else -1
        cls_name = class_map.get(cls_id, f"class_{cls_id}")

        obj = TrackedObject(
            track_id=tid,
            class_id=cls_id,
            class_name=cls_name,
            confidence=conf,
            bbox=(float(x1), float(y1), float(x2), float(y2)),
            center=(float((x1 + x2) / 2), float((y1 + y2) / 2)),
        )

        tracking.all_objects.append(obj)

        # Categorize by class name
        name_lower = cls_name.lower()
        if name_lower == "person":
            tracking.persons.append(obj)
        elif name_lower == "glove":
            tracking.gloves.append(obj)
        elif name_lower == "no_glove":
            tracking.no_gloves.append(obj)
        elif name_lower == "hairnet":
            tracking.hairnets.append(obj)
        elif name_lower == "no_hairnet":
            tracking.no_hairnets.append(obj)
        elif name_lower == "chef_hat":
            tracking.chef_hats.append(obj)

    return tracking


def offset_tracking_results(
    tracking: TrackingResult,
    offset_x: float,
    offset_y: float,
) -> TrackingResult:
    """Offset all bounding boxes in a tracking result by a given amount.

    This is used when detections are made on a cropped image and need
    to be mapped back to the original frame's coordinate space.

    Args:
        tracking: The tracking result to offset (modified in-place).
        offset_x: X coordinate offset.
        offset_y: Y coordinate offset.

    Returns:
        The modified tracking result.
    """
    for obj in tracking.all_objects:
        x1, y1, x2, y2 = obj.bbox
        obj.bbox = (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y)
        cx, cy = obj.center
        obj.center = (cx + offset_x, cy + offset_y)
    
    return tracking
