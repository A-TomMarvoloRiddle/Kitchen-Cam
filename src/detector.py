"""
Kitchen-Cam: Detection Pipeline Orchestrator
Wires together the hygiene model, pest model (via SAHI), tracker,
state machine, and logger into a single per-frame processing pipeline.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import AppConfig
from src.logger import TelemetryLogger
from src.models import ModelManager
from src.state_machine import StateMachine, ViolationEvent
from src.tracker import (
    TrackedObject,
    TrackingResult,
    associate_gear_to_person,
    parse_tracking_results,
)


class Detector:
    """Orchestrates the full detection pipeline for each video frame.

    Pipeline per frame:
        1. Run hygiene model (YOLOv8s + BoT-SORT tracking)
        2. Parse detections → associate gear to persons
        3. Update state machine with compliance status
        4. Log any confirmed violations
        5. Periodically run pest detection (SAHI sliced inference)
        6. Log any pest sightings
    """

    def __init__(
        self,
        config: AppConfig,
        model_manager: ModelManager,
        state_machine: StateMachine,
        logger: TelemetryLogger,
    ) -> None:
        self._config = config
        self._models = model_manager
        self._state_machine = state_machine
        self._logger = logger

        self._frame_index: int = 0
        self._sahi_model: Any = None  # Lazy-initialized SAHI wrapper

    # ── Public API ──

    def process_frame(self, frame: np.ndarray) -> DetectionOutput:
        """Run the full detection pipeline on a single frame.

        Args:
            frame: BGR image as numpy array.

        Returns:
            DetectionOutput containing all results for visualization.
        """
        self._frame_index += 1
        output = DetectionOutput()

        # ── Step 1: Hygiene Detection + Tracking ──
        perf = self._config.performance
        if self._frame_index % perf.process_every_n_frames == 0:
            hygiene_results = self._models.predict_hygiene(
                frame,
                tracker_config=self._config.tracker.config,
                persist=self._config.tracker.persist,
            )

            if hygiene_results is not None:
                # ── Step 2: Parse & Associate ──
                class_map = self._config.hygiene_model.classes
                tracking = parse_tracking_results(hygiene_results, class_map)
                person_statuses = associate_gear_to_person(tracking)

                # ── Step 3: Update State Machine ──
                now = time.time()
                violations = self._state_machine.update(person_statuses, now)

                # ── Step 4: Log Violations ──
                for v in violations:
                    # Find the person's bbox and confidence for logging
                    person_bbox = None
                    person_conf = None
                    for p in tracking.persons:
                        if p.track_id == v.track_id:
                            person_bbox = p.bbox
                            person_conf = p.confidence
                            break

                    self._logger.log_violation(
                        event=v,
                        bbox=person_bbox,
                        confidence=person_conf,
                    )

                output.tracking = tracking
                output.person_statuses = person_statuses
                output.violations = violations

        # ── Step 5: Pest Detection (periodic, via SAHI) ──
        pest_interval = self._config.performance.pest_check_every_n_frames
        if self._config.performance.enable_pest_detection and self._frame_index % pest_interval == 0:
            pest_detections = self._run_pest_detection(frame)
            output.pest_detections = pest_detections

            # ── Step 6: Log Pest Sightings ──
            for pest in pest_detections:
                self._logger.log_pest_detection(
                    class_name=pest.class_name,
                    confidence=pest.confidence,
                    bbox=pest.bbox,
                )

        return output

    @property
    def frame_index(self) -> int:
        """Current frame index."""
        return self._frame_index

    # ── Private Helpers ──

    def _run_pest_detection(self, frame: np.ndarray) -> List[TrackedObject]:
        """Run pest detection using SAHI sliced inference.

        Falls back to direct inference if SAHI is not available.

        Args:
            frame: BGR image.

        Returns:
            List of pest TrackedObjects.
        """
        pest_objects: List[TrackedObject] = []
        pest_class_map = self._config.pest_model.classes

        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction

            # Lazy-initialize SAHI model wrapper
            if self._sahi_model is None:
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model=self._models.pest_model,
                    confidence_threshold=self._config.pest_model.conf_threshold,
                    device=self._config.pest_model.device,
                )

            sahi_cfg = self._config.sahi
            result = get_sliced_prediction(
                image=frame,
                detection_model=self._sahi_model,
                slice_height=sahi_cfg.slice_height,
                slice_width=sahi_cfg.slice_width,
                overlap_height_ratio=sahi_cfg.overlap_ratio,
                overlap_width_ratio=sahi_cfg.overlap_ratio,
                verbose=0,
            )

            for pred in result.object_prediction_list:
                bbox_obj = pred.bbox
                x1, y1, x2, y2 = bbox_obj.minx, bbox_obj.miny, bbox_obj.maxx, bbox_obj.maxy
                cls_id = pred.category.id
                cls_name = pest_class_map.get(cls_id, pred.category.name)

                pest_objects.append(
                    TrackedObject(
                        track_id=-1,
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=pred.score.value,
                        bbox=(float(x1), float(y1), float(x2), float(y2)),
                        center=(float((x1 + x2) / 2), float((y1 + y2) / 2)),
                    )
                )

        except ImportError:
            # SAHI not installed — fall back to direct inference
            pest_results = self._models.predict_pest(frame)
            if pest_results is not None:
                tracking = parse_tracking_results(pest_results, pest_class_map)
                pest_objects = tracking.all_objects

        return pest_objects


class DetectionOutput:
    """Container for all detection results from a single frame."""

    def __init__(self) -> None:
        self.tracking: Optional[TrackingResult] = None
        self.person_statuses: Dict[int, Dict[str, bool]] = {}
        self.violations: List[ViolationEvent] = []
        self.pest_detections: List[TrackedObject] = []
