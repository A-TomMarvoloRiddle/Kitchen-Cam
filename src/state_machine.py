"""
Kitchen-Cam: State Machine
Tracks per-chef compliance state over time with temporal thresholding.
A violation is only confirmed after it persists for N consecutive seconds,
preventing false alarms from momentary detection flickers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.config import StateMachineConfig


@dataclass
class ChefState:
    """Tracks the compliance state of a single chef (by track ID)."""

    track_id: int
    last_seen: float = 0.0  # Unix timestamp of last detection

    # Current frame-level status
    glove: bool = True
    hairnet: bool = True

    # Temporal violation tracking — records when a violation *started*
    # If the chef becomes compliant again, the timer resets.
    violation_start: Dict[str, Optional[float]] = field(default_factory=dict)

    # Confirmed (logged) violations — to avoid duplicate logging
    confirmed_violations: Dict[str, bool] = field(default_factory=dict)


@dataclass
class ViolationEvent:
    """A confirmed hygiene violation ready for logging."""

    track_id: int
    violation_type: str          # e.g., "missing_glove", "missing_hairnet"
    duration_seconds: float      # How long the violation has persisted
    timestamp: float             # Unix timestamp when violation was confirmed


class StateMachine:
    """Manages per-chef compliance states with temporal thresholding.

    A violation is only flagged after the chef is seen non-compliant
    for `violation_threshold_seconds` continuously. This prevents
    single-frame false positives from triggering alerts.
    """

    def __init__(self, config: StateMachineConfig) -> None:
        self._config = config
        self._chefs: Dict[int, ChefState] = {}

    # ── Public API ──

    def update(
        self,
        person_statuses: Dict[int, Dict[str, bool]],
        current_time: Optional[float] = None,
    ) -> List[ViolationEvent]:
        """Update chef states with new frame detections and check for violations.

        Args:
            person_statuses: Dict mapping track_id → {"glove": bool, "hairnet": bool}
                             as produced by tracker.associate_gear_to_person().
            current_time: Override timestamp (for testing). Defaults to time.time().

        Returns:
            List of newly confirmed ViolationEvents (empty if no new violations).
        """
        now = current_time if current_time is not None else time.time()
        new_violations: List[ViolationEvent] = []

        for track_id, status in person_statuses.items():
            chef = self._get_or_create(track_id)
            chef.last_seen = now

            # Check each compliance attribute
            for attr in self._config.compliance_attributes:
                is_compliant = status.get(attr, True)
                violation_key = f"missing_{attr}"

                if is_compliant:
                    # Reset violation timer — chef is now compliant
                    chef.violation_start[violation_key] = None
                    chef.confirmed_violations[violation_key] = False
                    setattr(chef, attr, True)
                else:
                    setattr(chef, attr, False)

                    # Start violation timer if not already running
                    if chef.violation_start.get(violation_key) is None:
                        chef.violation_start[violation_key] = now

                    # Check if violation has exceeded the threshold
                    start = chef.violation_start[violation_key]
                    if start is not None:
                        duration = now - start

                        if (
                            duration >= self._config.violation_threshold_seconds
                            and not chef.confirmed_violations.get(violation_key, False)
                        ):
                            # Confirm the violation
                            chef.confirmed_violations[violation_key] = True
                            new_violations.append(
                                ViolationEvent(
                                    track_id=track_id,
                                    violation_type=violation_key,
                                    duration_seconds=round(duration, 2),
                                    timestamp=now,
                                )
                            )

        # Prune stale tracks
        self._prune_stale(now)

        return new_violations

    def get_chef_state(self, track_id: int) -> Optional[ChefState]:
        """Get the current state of a specific chef."""
        return self._chefs.get(track_id)

    def get_all_states(self) -> Dict[int, ChefState]:
        """Get all active chef states."""
        return dict(self._chefs)

    def get_violation_summary(self, track_id: int) -> Dict[str, str]:
        """Get a human-readable violation summary for a chef.

        Returns:
            Dict mapping attribute → "compliant" | "warning" | "VIOLATION"
        """
        chef = self._chefs.get(track_id)
        if chef is None:
            return {}

        summary: Dict[str, str] = {}
        now = time.time()

        for attr in self._config.compliance_attributes:
            violation_key = f"missing_{attr}"
            is_compliant = getattr(chef, attr, True)

            if is_compliant:
                summary[attr] = "compliant"
            elif chef.confirmed_violations.get(violation_key, False):
                summary[attr] = "VIOLATION"
            else:
                # Violation in progress but not yet confirmed
                summary[attr] = "warning"

        return summary

    def reset(self) -> None:
        """Clear all tracked chef states."""
        self._chefs.clear()

    # ── Private Helpers ──

    def _get_or_create(self, track_id: int) -> ChefState:
        """Get existing chef state or create a new one."""
        if track_id not in self._chefs:
            self._chefs[track_id] = ChefState(
                track_id=track_id,
                last_seen=time.time(),
                violation_start={},
                confirmed_violations={},
            )
        return self._chefs[track_id]

    def _prune_stale(self, now: float) -> None:
        """Remove chef states that haven't been seen recently."""
        stale_ids = [
            tid
            for tid, chef in self._chefs.items()
            if (now - chef.last_seen) > self._config.stale_track_timeout_seconds
        ]
        for tid in stale_ids:
            del self._chefs[tid]
