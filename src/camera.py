"""
Kitchen-Cam: Camera Stream Module
Flexible video ingestion — supports local files and live RTSP streams.
RTSP mode uses a daemon thread + single-slot queue to always fetch the latest frame.
"""

from __future__ import annotations

import threading
import time
from queue import Queue
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import CameraConfig


class CameraStream:
    """Unified video source for file playback and live RTSP streams.

    File mode:  Sequential frame reads with optional resize.
    RTSP mode:  Background thread continuously grabs frames into a
                maxsize=1 queue, ensuring the consumer always gets the
                freshest frame without latency buildup.
    """

    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_queue: Optional[Queue] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()

        self._fps: float = 30.0
        self._total_frames: int = 0
        self._frame_count: int = 0

    # ── Public API ──

    def open(self) -> None:
        """Open the video source."""
        source = self._config.source

        if self._config.mode == "rtsp":
            # RTSP: use FFMPEG backend for better stream handling
            self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        else:
            # File mode: default backend
            self._cap = cv2.VideoCapture(source)

        if not self._cap.isOpened():
            raise IOError(f"[CameraStream] Cannot open video source: {source}")

        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._frame_count = 0

        print(f"[CameraStream] Opened: {source}")
        print(f"  → FPS: {self._fps:.1f}  |  Total frames: {self._total_frames}")

        # Start background grabber thread for RTSP
        if self._config.mode == "rtsp":
            self._frame_queue = Queue(maxsize=self._config.rtsp_queue_size)
            self._stopped.clear()
            self._thread = threading.Thread(
                target=self._rtsp_grab_loop, daemon=True
            )
            self._thread.start()
            print("  → RTSP grab thread started.")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame (or latest frame for RTSP).

        Returns:
            (success, frame) tuple. Frame is resized to configured dimensions.
        """
        if self._config.mode == "rtsp" and self._frame_queue is not None:
            return self._read_rtsp()
        return self._read_file()

    def release(self) -> None:
        """Release the video source and stop any background threads."""
        self._stopped.set()

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        if self._cap is not None:
            self._cap.release()
            self._cap = None

        print("[CameraStream] Released.")

    @property
    def fps(self) -> float:
        """Source FPS (original video / stream FPS)."""
        return self._fps

    @property
    def total_frames(self) -> int:
        """Total frame count (0 for live streams)."""
        return self._total_frames

    @property
    def frame_count(self) -> int:
        """Number of frames read so far."""
        return self._frame_count

    @property
    def is_opened(self) -> bool:
        """Whether the video source is currently open."""
        return self._cap is not None and self._cap.isOpened()

    # ── Context Manager ──

    def __enter__(self) -> "CameraStream":
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()

    # ── Private Helpers ──

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize frame to configured processing dimensions."""
        target_w = self._config.frame_width
        target_h = self._config.frame_height
        h, w = frame.shape[:2]

        if w != target_w or h != target_h:
            frame = cv2.resize(
                frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
        return frame

    def _read_file(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Sequential read for file-based playback."""
        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            return False, None

        self._frame_count += 1
        return True, self._resize(frame)

    def _read_rtsp(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the latest frame from the RTSP queue."""
        if self._frame_queue is None:
            return False, None

        try:
            frame = self._frame_queue.get(timeout=5.0)
            self._frame_count += 1
            return True, self._resize(frame)
        except Exception:
            return False, None

    def _rtsp_grab_loop(self) -> None:
        """Background thread: continuously grabs frames from RTSP stream.

        Uses a maxsize=1 queue so the consumer always gets the latest frame.
        Old frames are discarded when the queue is full.
        """
        while not self._stopped.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Replace stale frame — always keep only the latest
            if self._frame_queue is not None:
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except Exception:
                        pass
                self._frame_queue.put(frame)
