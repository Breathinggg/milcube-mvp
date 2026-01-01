# MilCube MVP

Real-time multi-camera crowd monitoring system with edge-device simulation.

> Simulates Jetson Nano inference constraints on PC for algorithm validation before deployment.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![ONNX Runtime](https://img.shields.io/badge/ONNX-Runtime-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## Features

- **Multi-camera support** - Independent processes per camera with unified dashboard
- **Edge simulation** - Token Bucket scheduler enforces Jetson-like inference budget (8 Hz)
- **Dual FOV** - Wide (overview) + Narrow (ROI zoom) software-defined field of view
- **Real-time tracking** - IOU-based multi-object tracking with trajectory visualization
- **Event detection** - Loitering, fall detection, posture classification (stand/sit/fall)
- **Cross-camera Re-ID** - OSNet feature matching for person re-identification
- **Web dashboard** - MJPEG streaming + live metrics + heatmap overlay

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MilCube MVP                              │
└─────────────────────────────────────────────────────────────────┘

  run_all.bat (Launcher)
       │
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
   [Camera 1]     [Camera 2]     [Camera N]    [Dashboard]
   Port 5000      Port 5001      Port 500X     Port 9000
       │              │              │              │
       └──────────────┴──────────────┴──────────────┘
                           │
                    REST API (JSON)
                           │
                           ▼
               ┌─────────────────────┐
               │   Browser Client    │
               │  - Live video feed  │
               │  - KPI cards        │
               │  - Heatmap          │
               │  - Event logs       │
               └─────────────────────┘

Per-Camera Pipeline:
┌────────┐   ┌───────────┐   ┌─────────┐   ┌────────────┐
│ Video  │──▶│ YOLOv8n   │──▶│ IOU     │──▶│ Event      │
│ Source │   │ Detector  │   │ Tracker │   │ Detector   │
└────────┘   └───────────┘   └─────────┘   └────────────┘
                  │                              │
                  ▼                              ▼
           ┌───────────┐                  ┌────────────┐
           │ Scheduler │                  │ Posture    │
           │ (Token    │                  │ Classifier │
           │  Bucket)  │                  └────────────┘
           └───────────┘                        │
                  │                              │
                  └──────────────┬───────────────┘
                                 ▼
                          ┌────────────┐
                          │ Web View   │
                          │ (MJPEG)    │
                          └────────────┘
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download models

```bash
python download_model.py
```

This downloads:
- `yolov8n.onnx` (~6MB) - Person detection
- `osnet_x0_25.onnx` (~20MB) - Re-ID features

### 3. Run single camera

```bash
python main.py --src your_video.mp4 --port 5000 --name cam1 --gpu
```

### 4. Run multi-camera with dashboard

```bash
run_all.bat
```

Then open: http://127.0.0.1:9000

---

## Command Line Options

```
python main.py [OPTIONS]

Options:
  --src PATH          Video source (file path, camera index, or RTSP URL)
  --model PATH        YOLOv8 ONNX model path (default: models/yolov8n.onnx)
  --port INT          Web server port (default: 5000)
  --name STR          Camera identifier (default: cam0)
  --gpu               Enable GPU inference (CUDA)
  --headless          Run without GUI window
  --reid_model PATH   OSNet ONNX model for Re-ID
  --no_reid           Disable Re-ID, use color histogram instead
```

---

## Project Structure

```
milcube_mvp/
├── main.py                 # Main processing loop
├── config.py               # All hyperparameters
├── dashboard_server.py     # Multi-camera web dashboard
├── run_all.bat             # Multi-process launcher
│
├── modules/
│   ├── yolo_onnx.py        # YOLOv8 ONNX inference wrapper
│   ├── tracker_iou.py      # IOU-based multi-object tracker
│   ├── metrics.py          # Event detection (loiter, fall)
│   ├── posture.py          # Posture classification
│   ├── scheduler.py        # Token Bucket inference scheduler
│   ├── video_source.py     # Multi-source video input
│   ├── logger.py           # CSV + JSONL logging
│   └── viz.py              # Visualization utilities
│
├── tool/
│   ├── web_view.py         # Flask MJPEG server
│   └── reid_extractor.py   # OSNet Re-ID feature extraction
│
├── models/                 # ONNX models (auto-downloaded)
└── runs/                   # Logs output directory
```

---

## Key Design Decisions

### Token Bucket Scheduler

Simulates edge device constraints by limiting total inference frequency:

```python
# Total budget: 8 Hz shared across all streams
scheduler = TokenBucketScheduler(
    total_hz=8.0,
    stream_weights={"wide": 0.65, "narrow": 0.35}
)

# Before each inference
if scheduler.can_run("wide", now_ms):
    result = detector.infer(frame)
    scheduler.consume("wide", now_ms)
```

### Dual FOV Strategy

- **Wide view** (12 fps render, ~5 Hz infer): Full frame overview
- **Narrow view** (2 fps): Center-cropped ROI for detail confirmation

Narrow inference triggers on:
- Event candidates (loitering, fall suspect)
- Person count changes

### Confirmation Policy

Handles uncertainty from low inference rate:

| Policy | Behavior |
|--------|----------|
| 0 (none) | All detections counted equally |
| 1 (soft) | Mark unconfirmed, don't reduce count |
| 2 (weighted) | Unconfirmed = 0.5 weight in effective_count |
| 3 (strict) | Only confirmed detections in confirmed_count |

---

## Performance

Tested on: Intel i7-12700H + RTX 3060 Laptop

| Mode | Inference Rate | Render FPS | Latency | Notes |
|------|---------------|------------|---------|-------|
| GPU (uncapped) | 45 Hz | 45 fps | ~22ms | Full speed |
| Jetson simulation | 8 Hz | 12 fps | ~125ms | Token bucket limited |
| CPU only | 6 Hz | 10 fps | ~160ms | No CUDA |

---

## Output

### Web Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /video` | MJPEG live stream |
| `GET /metrics` | JSON metrics (count, fps, events) |
| `POST /control/pause` | Pause/resume processing |
| `GET /crop/<tid>` | Cropped image of track ID |

### Log Files

```
runs/
└── cam1/
    └── 20241229_201800/
        ├── metrics.csv      # Per-second performance metrics
        └── events.jsonl     # Detected events (loiter, fall, etc.)
```

---

## Extending

### Add new event type

1. Edit `modules/metrics.py`
2. Add detection logic in `EventEngine.tick()`
3. Events auto-logged to `events.jsonl`

### Custom visualization

Edit `modules/viz.py`:
- `draw_tracks()` - Bounding boxes and trajectories
- `draw_grid_overlay()` - Heatmap
- `draw_kpi_cards_on_panel()` - Dashboard cards

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [OSNet](https://github.com/KaiyangZhou/deep-person-reid) - Person Re-ID
- [ONNX Runtime](https://onnxruntime.ai/) - Cross-platform inference
