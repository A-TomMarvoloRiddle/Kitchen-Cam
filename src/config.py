"""
Kitchen-Cam: Configuration Loader
Loads and validates settings from config/settings.yaml using Pydantic models.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, field_validator


# ── Project root (two levels up from src/config.py) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"


# ── Pydantic Sub-Models ──

class CameraConfig(BaseModel):
    source: str = "input/sample.mp4"
    mode: str = "file"  # "file" | "rtsp"
    frame_width: int = 640
    frame_height: int = 480
    rtsp_queue_size: int = 1

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        if v not in ("file", "rtsp"):
            raise ValueError(f"camera.mode must be 'file' or 'rtsp', got '{v}'")
        return v


class ModelConfig(BaseModel):
    weights: str
    openvino_dir: str
    imgsz: int = 640
    conf_threshold: float = 0.35
    iou_threshold: float = 0.45
    device: str = "cpu"
    half: bool = False
    classes: Dict[int, str] = Field(default_factory=dict)


class SAHIConfig(BaseModel):
    enabled: bool = True
    slice_width: int = 320
    slice_height: int = 320
    overlap_ratio: float = 0.2


class TrackerConfig(BaseModel):
    config: str = "botsort.yaml"
    persist: bool = True


class StateMachineConfig(BaseModel):
    violation_threshold_seconds: float = 5.0
    stale_track_timeout_seconds: float = 30.0
    compliance_attributes: List[str] = Field(
        default_factory=lambda: ["glove", "hairnet"]
    )


class LoggingConfig(BaseModel):
    output_dir: str = "logs"
    rotate_daily: bool = True
    log_pest_detections: bool = True
    include_bbox: bool = True
    include_confidence: bool = True


class PerformanceConfig(BaseModel):
    process_every_n_frames: int = 2
    pest_check_every_n_frames: int = 10
    max_batch_size: int = 1
    num_threads: int = 4
    enable_visualization: bool = True
    visualization_scale: float = 1.0


class DisplayColors(BaseModel):
    compliant: Tuple[int, int, int] = (0, 200, 0)
    violation: Tuple[int, int, int] = (0, 0, 255)
    pest: Tuple[int, int, int] = (0, 165, 255)
    text: Tuple[int, int, int] = (255, 255, 255)

    @field_validator("compliant", "violation", "pest", "text", mode="before")
    @classmethod
    def coerce_list_to_tuple(cls, v: Any) -> Tuple[int, int, int]:
        if isinstance(v, list):
            return tuple(v)  # type: ignore[return-value]
        return v


class DisplayConfig(BaseModel):
    window_name: str = "Kitchen-Cam Monitor"
    show_fps: bool = True
    show_track_ids: bool = True
    bbox_thickness: int = 2
    font_scale: float = 0.5
    colors: DisplayColors = Field(default_factory=DisplayColors)


class OutputConfig(BaseModel):
    save_video: bool = False
    output_dir: str = "output"
    filename: str = "output.mp4"

# ── Root Config ──

class AppConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    hygiene_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            weights="models/hygiene_yolov8s.pt",
            openvino_dir="models/hygiene_yolov8s_openvino_model",
        )
    )
    pest_model: ModelConfig = Field(
        default_factory=lambda: ModelConfig(
            weights="models/pest_yolov8n.pt",
            openvino_dir="models/pest_yolov8n_openvino_model",
        )
    )
    sahi: SAHIConfig = Field(default_factory=SAHIConfig)
    tracker: TrackerConfig = Field(default_factory=TrackerConfig)
    state_machine: StateMachineConfig = Field(default_factory=StateMachineConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """Load configuration from a YAML file and validate it.

    Args:
        config_path: Path to the YAML config file.
                     Defaults to config/settings.yaml relative to project root.

    Returns:
        Validated AppConfig instance.
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if not path.exists():
        print(f"[Config] Warning: {path} not found — using defaults.")
        return AppConfig()

    with open(path, "r", encoding="utf-8") as f:
        raw: Dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(**raw)
