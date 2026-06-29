"""
Kitchen-Cam: JSON Telemetry Logger
Writes violation events and pest detections to daily-rotated JSON log files.
Each log file is a JSON array of event objects for easy client-side parsing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AppConfig, LoggingConfig, PROJECT_ROOT
from src.state_machine import ViolationEvent


class TelemetryLogger:
    """Writes structured JSON logs for hygiene violations and pest detections.

    Log files are stored as:
        logs/<source_name>_<YYYY-MM-DD>.json

    Each file contains a JSON array of event dicts. New events are appended
    by reading, extending, and rewriting the array. This approach is chosen
    over JSONL for easier client-side consumption with simple JSON parsers.
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config.logging
        self._source_name = self._derive_source_name(config.camera.source)
        self._log_dir = PROJECT_ROOT / self._config.output_dir

        # Ensure log directory exists
        self._log_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──

    def log_violation(
        self,
        event: ViolationEvent,
        bbox: Optional[tuple] = None,
        confidence: Optional[float] = None,
    ) -> None:
        """Log a confirmed hygiene violation event.

        Args:
            event: ViolationEvent from the state machine.
            bbox: Optional bounding box (x1, y1, x2, y2) of the person.
            confidence: Optional detection confidence score.
        """
        payload: Dict[str, Any] = {
            "event_type": "hygiene_violation",
            "timestamp": datetime.fromtimestamp(event.timestamp).isoformat(),
            "unix_timestamp": event.timestamp,
            "source": self._source_name,
            "track_id": event.track_id,
            "violation_type": event.violation_type,
            "duration_seconds": event.duration_seconds,
        }

        if self._config.include_bbox and bbox is not None:
            payload["bbox"] = {
                "x1": round(bbox[0], 1),
                "y1": round(bbox[1], 1),
                "x2": round(bbox[2], 1),
                "y2": round(bbox[3], 1),
            }

        if self._config.include_confidence and confidence is not None:
            payload["confidence"] = round(confidence, 4)

        self._append_event(payload)
        print(
            f"[Logger] Violation logged: Track #{event.track_id} — "
            f"{event.violation_type} ({event.duration_seconds}s)"
        )

    def log_pest_detection(
        self,
        class_name: str,
        confidence: float,
        bbox: tuple,
        timestamp: Optional[float] = None,
    ) -> None:
        """Log a pest detection event.

        Args:
            class_name: Type of pest detected (e.g., "cockroach", "fly").
            confidence: Detection confidence score.
            bbox: Bounding box (x1, y1, x2, y2).
            timestamp: Optional Unix timestamp. Defaults to now.
        """
        if not self._config.log_pest_detections:
            return

        import time as _time

        ts = timestamp if timestamp is not None else _time.time()

        payload: Dict[str, Any] = {
            "event_type": "pest_detection",
            "timestamp": datetime.fromtimestamp(ts).isoformat(),
            "unix_timestamp": ts,
            "source": self._source_name,
            "pest_type": class_name,
            "confidence": round(confidence, 4),
        }

        if self._config.include_bbox:
            payload["bbox"] = {
                "x1": round(bbox[0], 1),
                "y1": round(bbox[1], 1),
                "x2": round(bbox[2], 1),
                "y2": round(bbox[3], 1),
            }

        self._append_event(payload)
        print(f"[Logger] Pest detected: {class_name} (conf={confidence:.3f})")

    # ── Private Helpers ──

    def _get_log_path(self) -> Path:
        """Get the log file path for today (daily rotation)."""
        if self._config.rotate_daily:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"{self._source_name}_{date_str}.json"
        else:
            filename = f"{self._source_name}.json"
        return self._log_dir / filename

    def _append_event(self, payload: Dict[str, Any]) -> None:
        """Append an event to the daily log file.

        Reads existing events, appends the new one, and rewrites the file.
        This ensures the file is always a valid JSON array.
        """
        log_path = self._get_log_path()

        # Read existing events
        events: List[Dict[str, Any]] = []
        if log_path.exists():
            try:
                with open(log_path, "r", encoding="utf-8") as f:
                    events = json.load(f)
            except (json.JSONDecodeError, ValueError):
                # Corrupted file — start fresh
                events = []

        # Append new event
        events.append(payload)

        # Write back
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)

    @staticmethod
    def _derive_source_name(source: str) -> str:
        """Derive a filesystem-safe source name from the camera source path/URL.

        Examples:
            "input/kitchen_front.mp4"  →  "kitchen_front"
            "rtsp://192.168.1.10/cam1" →  "rtsp_192_168_1_10_cam1"
        """
        if source.startswith("rtsp://"):
            # Sanitize RTSP URL
            name = source.replace("rtsp://", "rtsp_")
            name = name.replace("/", "_").replace(":", "_").replace(".", "_")
            return name

        # File path — use the stem
        return Path(source).stem
