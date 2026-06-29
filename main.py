"""
Kitchen-Cam: Main Entry Point
Wires all modules together and runs the monitoring pipeline.

Usage:
    python main.py                              # Use default config
    python main.py --config config/custom.yaml  # Use custom config
    python main.py --source input/video.mp4     # Override video source
    python main.py --no-display                 # Run headless (no GUI)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src.camera import CameraStream
from src.config import AppConfig, load_config
from src.detector import Detector
from src.logger import TelemetryLogger
from src.models import ModelManager
from src.state_machine import StateMachine
from src.tracker import TrackingResult
from src.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Kitchen-Cam: AI-Powered Kitchen Hygiene Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (default: config/settings.yaml)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override video source (file path or RTSP URL)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["file", "rtsp"],
        default=None,
        help="Override camera mode",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without GUI visualization",
    )
    return parser.parse_args()


def main() -> None:
    """Main application loop."""
    args = parse_args()

    # ── Load Configuration ──
    config_path = Path(args.config) if args.config else None
    config: AppConfig = load_config(config_path)

    # Apply CLI overrides
    if args.source:
        config.camera.source = args.source
    if args.mode:
        config.camera.mode = args.mode
    if args.no_display:
        config.performance.enable_visualization = False

    print("=" * 60)
    print("  Kitchen-Cam: AI-Powered Kitchen Hygiene Monitor")
    print("=" * 60)
    print(f"  Source:      {config.camera.source}")
    print(f"  Mode:        {config.camera.mode}")
    print(f"  Resolution:  {config.camera.frame_width}x{config.camera.frame_height}")
    print(f"  Skip frames: process every {config.performance.process_every_n_frames}")
    print(f"  Pest check:  every {config.performance.pest_check_every_n_frames} frames")
    print(f"  Visualization: {'ON' if config.performance.enable_visualization else 'OFF'}")
    print("=" * 60)

    # ── Initialize Modules ──
    print("\n[Init] Loading models...")
    model_manager = ModelManager(config)
    model_manager.load_all()

    state_machine = StateMachine(config.state_machine)
    logger = TelemetryLogger(config)
    detector = Detector(config, model_manager, state_machine, logger)

    visualizer: Visualizer | None = None
    if config.performance.enable_visualization or config.output.save_video:
        visualizer = Visualizer(config.display, state_machine)

    # ── Main Processing Loop ──
    print("\n[Run] Starting processing loop... (press 'q' to quit)\n")

    camera = CameraStream(config.camera)
    camera.open()

    video_writer = None
    if config.output.save_video:
        out_dir = Path(config.output.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Derive output filename from input source
        source_name = TelemetryLogger._derive_source_name(config.camera.source)
        out_filename = f"{source_name}-output.mp4"
        out_path = out_dir / out_filename
        
        fps = camera.fps if camera.fps > 0 else 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        video_writer = cv2.VideoWriter(
            str(out_path), 
            fourcc, 
            fps, 
            (config.camera.frame_width, config.camera.frame_height)
        )
        print(f"  → Saving video output to {out_path} at {fps} FPS")

    try:
        while True:
            ret, frame = camera.read()
            if not ret:
                if config.camera.mode == "file":
                    print("\n[Run] End of video file reached.")
                    break
                else:
                    # RTSP: brief retry
                    time.sleep(0.1)
                    continue

            # ── Run Detection Pipeline ──
            output = detector.process_frame(frame)

            # ── Visualize & Save ──
            frame_to_write = frame
            
            if visualizer is not None and output.tracking is not None:
                frame_to_write = visualizer.annotate_frame(
                    frame,
                    output.tracking,
                    output.person_statuses,
                    output.pest_detections or None,
                )
            
            if config.performance.enable_visualization and visualizer is not None:
                if not visualizer.show(frame_to_write):
                    print("\n[Run] User quit (pressed 'q').")
                    break

            if video_writer is not None:
                video_writer.write(frame_to_write)

            # Progress indicator for file mode
            if config.camera.mode == "file" and camera.total_frames > 0:
                progress = (camera.frame_count / camera.total_frames) * 100
                if camera.frame_count % 100 == 0:
                    print(
                        f"  Progress: {progress:.1f}% "
                        f"({camera.frame_count}/{camera.total_frames} frames)"
                    )

    except KeyboardInterrupt:
        print("\n[Run] Interrupted by user.")

    finally:
        camera.release()
        if video_writer is not None:
            video_writer.release()
        if visualizer:
            visualizer.cleanup()
        print("\n[Done] Kitchen-Cam stopped. Check logs/ for event records.")


if __name__ == "__main__":
    main()
