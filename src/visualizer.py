"""
Kitchen-Cam: Frame Visualizer
Renders annotated frames with bounding boxes, track IDs,
compliance status, and FPS overlay.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.config import DisplayConfig
from src.state_machine import StateMachine
from src.tracker import TrackedObject, TrackingResult


class Visualizer:
    """Draws annotated overlays on video frames for monitoring display."""

    def __init__(self, config: DisplayConfig, state_machine: StateMachine) -> None:
        self._config = config
        self._state_machine = state_machine

        # FPS calculation
        self._frame_times: List[float] = []
        self._fps: float = 0.0

    # ── Public API ──

    def annotate_frame(
        self,
        frame: np.ndarray,
        tracking: TrackingResult,
        person_statuses: Dict[int, Dict[str, bool]],
        pest_detections: Optional[List[TrackedObject]] = None,
    ) -> np.ndarray:
        """Draw all annotations on a frame.

        Args:
            frame: BGR image to annotate (will be modified in-place).
            tracking: Parsed tracking results for this frame.
            person_statuses: Per-person compliance dict from tracker.
            pest_detections: Optional list of pest TrackedObjects.

        Returns:
            Annotated frame.
        """
        # Update FPS
        self._update_fps()

        # Draw person boxes with compliance status
        for person in tracking.persons:
            self._draw_person(frame, person, person_statuses)

        # Draw gear detections (gloves, hairnets, etc.)
        for obj in tracking.all_objects:
            if obj.class_name.lower() != "person":
                self._draw_gear(frame, obj)

        # Draw pest detections
        if pest_detections:
            for pest in pest_detections:
                self._draw_pest(frame, pest)

        # Draw FPS overlay
        if self._config.show_fps:
            self._draw_fps(frame)

        return frame

    def show(self, frame: np.ndarray) -> bool:
        """Display the frame in a window.

        Args:
            frame: Annotated frame to display.

        Returns:
            False if the user pressed 'q' to quit, True otherwise.
        """
        scale = 1.0
        if hasattr(self._config, "visualization_scale"):
            # Access from parent config if available
            pass

        cv2.imshow(self._config.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q")

    def cleanup(self) -> None:
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()

    @property
    def fps(self) -> float:
        """Current processing FPS."""
        return self._fps

    # ── Private Drawing Methods ──

    def _draw_person(
        self,
        frame: np.ndarray,
        person: TrackedObject,
        person_statuses: Dict[int, Dict[str, bool]],
    ) -> None:
        """Draw a person bounding box with compliance status."""
        x1, y1, x2, y2 = [int(v) for v in person.bbox]
        tid = person.track_id

        # Determine compliance color
        status = person_statuses.get(tid, {})
        summary = self._state_machine.get_violation_summary(tid)

        has_violation = any(v == "VIOLATION" for v in summary.values())
        has_warning = any(v == "warning" for v in summary.values())

        if has_violation:
            color = self._config.colors.violation
        elif has_warning:
            # Yellow for warnings (in-progress violations)
            color = (0, 200, 255)
        else:
            color = self._config.colors.compliant

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._config.bbox_thickness)

        # Build label
        label_parts = []
        if self._config.show_track_ids and tid >= 0:
            label_parts.append(f"Chef #{tid}")

        for attr, state in summary.items():
            icon = "✓" if state == "compliant" else ("⚠" if state == "warning" else "✗")
            label_parts.append(f"{attr}:{icon}")

        label = " | ".join(label_parts)

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._config.font_scale
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            font, scale, self._config.colors.text, 1, cv2.LINE_AA,
        )

    def _draw_gear(self, frame: np.ndarray, obj: TrackedObject) -> None:
        """Draw a small gear detection box (glove, hairnet, etc.)."""
        x1, y1, x2, y2 = [int(v) for v in obj.bbox]
        name = obj.class_name.lower()

        # Green for positive detections, red for negative
        if name.startswith("no_"):
            color = self._config.colors.violation
        else:
            color = self._config.colors.compliant

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

        label = f"{obj.class_name} {obj.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._config.font_scale * 0.8
        cv2.putText(
            frame, label, (x1, y1 - 3),
            font, scale, color, 1, cv2.LINE_AA,
        )

    def _draw_pest(self, frame: np.ndarray, pest: TrackedObject) -> None:
        """Draw a pest detection with distinct orange styling."""
        x1, y1, x2, y2 = [int(v) for v in pest.bbox]
        color = self._config.colors.pest

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, self._config.bbox_thickness)

        label = f"PEST: {pest.class_name} {pest.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._config.font_scale
        (tw, th), _ = cv2.getTextSize(label, font, scale, 1)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            frame, label, (x1 + 2, y1 - 4),
            font, scale, (0, 0, 0), 1, cv2.LINE_AA,
        )

    def _draw_fps(self, frame: np.ndarray) -> None:
        """Draw FPS counter in the top-left corner."""
        label = f"FPS: {self._fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = self._config.font_scale * 1.2
        color = self._config.colors.text

        # Black background for readability
        (tw, th), _ = cv2.getTextSize(label, font, scale, 2)
        cv2.rectangle(frame, (8, 8), (16 + tw, 16 + th), (0, 0, 0), -1)
        cv2.putText(frame, label, (12, 12 + th), font, scale, color, 2, cv2.LINE_AA)

    def _update_fps(self) -> None:
        """Update rolling FPS calculation."""
        now = time.time()
        self._frame_times.append(now)

        # Keep only last 30 frame timestamps
        if len(self._frame_times) > 30:
            self._frame_times = self._frame_times[-30:]

        if len(self._frame_times) >= 2:
            elapsed = self._frame_times[-1] - self._frame_times[0]
            if elapsed > 0:
                self._fps = (len(self._frame_times) - 1) / elapsed
