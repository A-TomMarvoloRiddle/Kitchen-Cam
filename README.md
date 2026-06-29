# Kitchen-Cam 🍳🔍

**AI-Powered Kitchen Hygiene & Pest Monitoring System**

Real-time computer vision application that monitors kitchen staff compliance (gloves, hairnets) and detects pests, optimized for Intel hardware using OpenVINO acceleration.

---

## Architecture

```
Kitchen-Cam/
├── config/
│   └── settings.yaml          # All tunable parameters
├── src/
│   ├── __init__.py
│   ├── config.py              # Pydantic config loader
│   ├── models.py              # YOLOv8 model manager (OpenVINO)
│   ├── camera.py              # Video ingestion (file + RTSP)
│   ├── tracker.py             # BoT-SORT tracking & gear association
│   ├── state_machine.py       # Temporal violation logic
│   ├── detector.py            # Pipeline orchestrator
│   ├── logger.py              # JSON telemetry writer
│   └── visualizer.py          # Annotated frame rendering
├── models/                    # Model weights & OpenVINO exports
├── input/                     # Input video files
├── logs/                      # Daily JSON event logs
├── main.py                    # CLI entry point
├── export_models.py           # Model → OpenVINO export script
└── requirements.txt           # Python dependencies
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment (already done)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Export Models to OpenVINO (one-time)

```bash
python export_models.py
```

This downloads pretrained YOLOv8 models and exports them to OpenVINO IR format for Intel Iris Xe acceleration.

### 3. Run Monitoring

```bash
# Process a video file
python main.py --source input/kitchen_video.mp4

# Use RTSP camera stream
python main.py --source "rtsp://192.168.1.10/stream" --mode rtsp

# Run headless (no display window)
python main.py --source input/video.mp4 --no-display

# Use custom config
python main.py --config config/custom_settings.yaml
```

## Configuration

All parameters are in `config/settings.yaml`:

| Section | Key Settings |
|---------|-------------|
| `camera` | Source path/URL, resolution, mode (file/rtsp) |
| `hygiene_model` | Weights path, confidence threshold, class mapping |
| `pest_model` | Weights path, SAHI slice dimensions |
| `state_machine` | Violation duration threshold (seconds) |
| `performance` | Frame skip, thread count, pest check interval |
| `display` | Window name, colors, font scale |

## How It Works

1. **Video Ingestion** — Reads frames from file or RTSP stream (threaded for live feeds)
2. **Hygiene Detection** — YOLOv8s detects persons, gloves, hairnets via OpenVINO
3. **Tracking** — BoT-SORT assigns persistent IDs to each chef across frames
4. **Gear Association** — Spatial overlap links gear detections to person bounding boxes
5. **Temporal Filtering** — Violations only confirmed after 5s continuous non-compliance
6. **Pest Detection** — SAHI-sliced YOLOv8n runs periodically for small pest detection
7. **JSON Logging** — Events written to daily-rotated JSON files in `logs/`

## Hardware Optimization

Optimized for **Intel i7-1165G7 / Iris Xe iGPU / 16GB DDR4**:

- **OpenVINO** instead of TensorRT (Intel-native acceleration)
- **4 inference threads** matching physical core count
- **Frame skipping** (process every 2nd frame) for real-time throughput
- **640×480 processing** resolution to reduce memory pressure
- **Periodic pest checks** (every 10th frame) to conserve resources

## Log Format

Logs are stored as JSON arrays in `logs/<source>_<date>.json`:

```json
[
  {
    "event_type": "hygiene_violation",
    "timestamp": "2026-06-29T10:30:15",
    "source": "kitchen_front",
    "track_id": 3,
    "violation_type": "missing_glove",
    "duration_seconds": 5.2,
    "bbox": {"x1": 120.5, "y1": 80.3, "x2": 340.1, "y2": 420.7},
    "confidence": 0.8734
  }
]
```
