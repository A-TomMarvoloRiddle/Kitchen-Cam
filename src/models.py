"""
Kitchen-Cam: Model Manager
Handles loading YOLOv8 models, exporting to OpenVINO format,
and running inference optimized for Intel Iris Xe iGPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.config import AppConfig, ModelConfig, PROJECT_ROOT


class ModelManager:
    """Loads, exports, and manages YOLOv8 models with OpenVINO optimization."""

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._hygiene_model: Any = None
        self._pest_model: Any = None

        # Set OpenVINO thread count for Intel i7-1165G7 (4 physical cores)
        os.environ["OMP_NUM_THREADS"] = str(config.performance.num_threads)
        os.environ["OPENVINO_NUM_THREADS"] = str(config.performance.num_threads)

    # ── Public API ──

    def load_all(self) -> None:
        """Load both hygiene and pest models (OpenVINO preferred, PyTorch fallback)."""
        print("[ModelManager] Loading hygiene model...")
        self._hygiene_model = self._load_model(self._config.hygiene_model)

        print("[ModelManager] Loading pest model...")
        self._pest_model = self._load_model(self._config.pest_model)

        print("[ModelManager] All models loaded successfully.")

    def export_all_to_openvino(self) -> None:
        """Export both PyTorch models to OpenVINO IR format for Intel acceleration."""
        self._export_to_openvino(self._config.hygiene_model, tag="hygiene")
        self._export_to_openvino(self._config.pest_model, tag="pest")

    @property
    def hygiene_model(self) -> Any:
        """Access the loaded hygiene detection model."""
        if self._hygiene_model is None:
            raise RuntimeError("Hygiene model not loaded. Call load_all() first.")
        return self._hygiene_model

    @property
    def pest_model(self) -> Any:
        """Access the loaded pest detection model."""
        if self._pest_model is None:
            raise RuntimeError("Pest model not loaded. Call load_all() first.")
        return self._pest_model

    def predict_hygiene(
        self,
        frame: np.ndarray,
        tracker_config: Optional[str] = None,
        persist: bool = True,
    ) -> Any:
        """Run hygiene model inference with optional tracking.

        Args:
            frame: BGR image as numpy array.
            tracker_config: Path to tracker YAML config (e.g., 'botsort.yaml').
            persist: Whether to persist tracker state across frames.

        Returns:
            ultralytics Results object.
        """
        cfg = self._config.hygiene_model
        kwargs: Dict[str, Any] = {
            "imgsz": cfg.imgsz,
            "conf": cfg.conf_threshold,
            "iou": cfg.iou_threshold,
            "verbose": False,
        }

        if tracker_config:
            kwargs["tracker"] = tracker_config
            kwargs["persist"] = persist

        results = self._hygiene_model.predict(frame, **kwargs)
        return results[0] if results else None

    def predict_pest(self, frame: np.ndarray) -> Any:
        """Run pest model inference (no tracking needed).

        Args:
            frame: BGR image as numpy array.

        Returns:
            ultralytics Results object.
        """
        cfg = self._config.pest_model
        results = self._pest_model.predict(
            frame,
            imgsz=cfg.imgsz,
            conf=cfg.conf_threshold,
            iou=cfg.iou_threshold,
            verbose=False,
        )
        return results[0] if results else None

    # ── Private Helpers ──

    def _load_model(self, model_cfg: ModelConfig) -> Any:
        """Load a model — prefer OpenVINO export, fall back to PyTorch weights.

        Args:
            model_cfg: Model configuration section.

        Returns:
            Loaded YOLO model instance.
        """
        from ultralytics import YOLO

        openvino_path = PROJECT_ROOT / model_cfg.openvino_dir
        weights_path = PROJECT_ROOT / model_cfg.weights

        # Prefer OpenVINO export if it exists
        if openvino_path.exists():
            # Look for the .xml file inside the OpenVINO directory
            xml_files = list(openvino_path.glob("*.xml"))
            if xml_files:
                model_file = str(xml_files[0])
                print(f"  → Loading OpenVINO model: {model_file}")
                model = YOLO(model_file, task="detect")
                return model

        # Fall back to PyTorch weights
        if weights_path.exists():
            print(f"  → Loading PyTorch weights: {weights_path}")
            model = YOLO(str(weights_path), task="detect")
            return model

        # Neither exists — load a pretrained base model (for initial setup/testing)
        base_name = Path(model_cfg.weights).stem  # e.g., "hygiene_yolov8s"
        # Extract the YOLO variant (e.g., "yolov8s" from "hygiene_yolov8s")
        variant = base_name.split("_")[-1] if "_" in base_name else "yolov8n"
        print(f"  → Warning: No custom weights found. Loading pretrained {variant}.")
        model = YOLO(f"{variant}.pt")
        return model

    def _export_to_openvino(self, model_cfg: ModelConfig, tag: str) -> None:
        """Export a PyTorch YOLO model to OpenVINO IR format.

        Args:
            model_cfg: Model configuration section.
            tag: Human-readable tag for logging.
        """
        from ultralytics import YOLO

        weights_path = PROJECT_ROOT / model_cfg.weights

        if not weights_path.exists():
            print(f"[Export] Skipping {tag}: weights not found at {weights_path}")
            return

        print(f"[Export] Exporting {tag} model to OpenVINO format...")
        model = YOLO(str(weights_path))

        # Export with Intel-optimized settings
        export_path = model.export(
            format="openvino",
            imgsz=model_cfg.imgsz,
            half=model_cfg.half,  # FP16 disabled for CPU/iGPU
        )

        print(f"[Export] {tag} model exported to: {export_path}")
