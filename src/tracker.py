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


def associate_gear_to_person(
    tracking: TrackingResult,
    iou_threshold: float = 0.05,
) -> Dict[int, Dict[str, bool]]:
    """Associate detected gear (gloves, hairnets) to the nearest person.

    Uses spatial overlap (IoU or containment) to link gear detections
    to person bounding boxes.

    Args:
        tracking: Parsed tracking result for one frame.
        iou_threshold: Minimum IoU to consider a gear detection linked to a person.

    Returns:
        Dict mapping person track_id → {"glove": bool, "hairnet": bool}
    """
    person_status: Dict[int, Dict[str, bool]] = {}

    for person in tracking.persons:
        pid = person.track_id
        person_status[pid] = {
            "glove": False,
            "hairnet": False,
        }

        # Check gloves — if a "glove" detection overlaps with this person
        for glove in tracking.gloves:
            if _bbox_overlap(person.bbox, glove.bbox) > iou_threshold:
                person_status[pid]["glove"] = True
                break

        # Check no_glove — explicit negative detection
        for no_glove in tracking.no_gloves:
            if _bbox_overlap(person.bbox, no_glove.bbox) > iou_threshold:
                person_status[pid]["glove"] = False
                break  # Explicit negative overrides

        # Check hairnet / chef_hat
        for hairnet in tracking.hairnets + tracking.chef_hats:
            if _bbox_overlap(person.bbox, hairnet.bbox) > iou_threshold:
                person_status[pid]["hairnet"] = True
                break

        # Check no_hairnet — explicit negative
        for no_hairnet in tracking.no_hairnets:
            if _bbox_overlap(person.bbox, no_hairnet.bbox) > iou_threshold:
                person_status[pid]["hairnet"] = False
                break

    return person_status


def _bbox_overlap(box_a: tuple, box_b: tuple) -> float:
    """Compute the ratio of box_b's area that overlaps with box_a.

    This is NOT strict IoU — it measures how much of the smaller box (gear)
    is contained within the larger box (person). This works better for
    associating small gear detections with large person boxes.

    Args:
        box_a: (x1, y1, x2, y2) — typically the person box.
        box_b: (x1, y1, x2, y2) — typically the gear box.

    Returns:
        Overlap ratio [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    # Intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_b = max((bx2 - bx1) * (by2 - by1), 1e-6)

    return intersection / area_b
