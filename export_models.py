"""
Kitchen-Cam: Model Export Script
Downloads pretrained YOLOv8 models and exports them to OpenVINO IR format
for Intel Iris Xe iGPU acceleration.

Usage:
    python export_models.py                          # Export all models
    python export_models.py --hygiene-only           # Export hygiene model only
    python export_models.py --pest-only              # Export pest model only
    python export_models.py --config config/custom.yaml  # Custom config
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import AppConfig, ModelConfig, PROJECT_ROOT, load_config


def export_model(model_cfg: ModelConfig, tag: str) -> None:
    """Export a single YOLOv8 model to OpenVINO format.

    Args:
        model_cfg: Model configuration section.
        tag: Human-readable label for logging.
    """
    from ultralytics import YOLO

    weights_path = PROJECT_ROOT / model_cfg.weights

    if not weights_path.exists():
        # Download the pretrained model variant
        variant = Path(model_cfg.weights).stem.split("_")[-1]
        print(f"[Export] Downloading pretrained {variant}...")

        # Ensure the models directory exists
        weights_path.parent.mkdir(parents=True, exist_ok=True)

        model = YOLO(f"{variant}.pt")
        # Save to the expected location
        import shutil
        src = Path(f"{variant}.pt")
        if src.exists():
            shutil.move(str(src), str(weights_path))
            model = YOLO(str(weights_path))
        print(f"[Export] Downloaded to: {weights_path}")
    else:
        print(f"[Export] Loading existing weights: {weights_path}")
        model = YOLO(str(weights_path))

    # Export to OpenVINO IR format
    print(f"[Export] Exporting {tag} model to OpenVINO format...")
    print(f"  → Image size: {model_cfg.imgsz}")
    print(f"  → FP16 (half): {model_cfg.half}")

    export_path = model.export(
        format="openvino",
        imgsz=model_cfg.imgsz,
        half=model_cfg.half,
    )

    print(f"[Export] ✓ {tag} model exported to: {export_path}")
    return export_path


def main() -> None:
    """Parse arguments and run model export."""
    parser = argparse.ArgumentParser(
        description="Export YOLOv8 models to OpenVINO format for Intel acceleration",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--hygiene-only",
        action="store_true",
        help="Export only the hygiene model",
    )
    parser.add_argument(
        "--pest-only",
        action="store_true",
        help="Export only the pest model",
    )
    args = parser.parse_args()

    config_path = Path(args.config) if args.config else None
    config: AppConfig = load_config(config_path)

    print("=" * 60)
    print("  Kitchen-Cam: Model Export (OpenVINO)")
    print("  Optimized for: Intel i7-1165G7 / Iris Xe iGPU")
    print("=" * 60)

    if not args.pest_only:
        print("\n── Hygiene Model ──")
        export_model(config.hygiene_model, "Hygiene")

    if not args.hygiene_only:
        print("\n── Pest Model ──")
        export_model(config.pest_model, "Pest")

    print("\n" + "=" * 60)
    print("  ✓ Export complete! Models ready for OpenVINO inference.")
    print("=" * 60)


if __name__ == "__main__":
    main()
