# Architecture Design

## Overview

MilCube MVP is designed to validate crowd monitoring algorithms under edge-device constraints before actual Jetson Nano deployment.

## Design Philosophy

### 1. Constraint-First Development

Instead of developing on powerful hardware and hoping it works on edge devices, we simulate constraints upfront:

```
PC Development          vs         Edge Reality
─────────────────                 ─────────────
RTX 3060: 45 fps                  Jetson Nano: 8 fps
Unlimited memory                  4GB shared RAM
Low latency                       Network + inference delay
```

By enforcing these constraints in software, we catch performance issues early.

### 2. Token Bucket Scheduler

The core insight: edge devices can't run inference on every frame. We need a **budget**.

```
Total Budget: 8 Hz (inferences per second)
├── Wide stream:   65% = 5.2 Hz (overview)
└── Narrow stream: 35% = 2.8 Hz (detail confirmation)
```

Token Bucket algorithm:
- Tokens accumulate at configured rate (Hz)
- Each inference consumes 1 token
- Burst cap prevents token hoarding (max 2.0)
- Global lock ensures serial inference (no parallel)

```python
class TokenBucketScheduler:
    def can_run(self, stream_id, now_ms):
        if now_ms < self.busy_until_ms:  # Still processing
            return False
        return self.tokens[stream_id] >= 1.0
```

### 3. Dual FOV Strategy

Problem: Low inference rate means we might miss events.

Solution: Two virtual "cameras" from one stream:

| View | Coverage | Inference | Purpose |
|------|----------|-----------|---------|
| Wide | Full frame | 5 Hz | Count, track, detect candidates |
| Narrow | Center 45% crop | 2 Hz | Confirm events, Re-ID |

Narrow triggers on:
- Event candidate (loitering > 2s, fall suspect)
- Person count change (someone entered/left)

### 4. Confirmation Policy

With 8 Hz inference on 25 fps video, ~68% of frames have no fresh detection. How do we count?

```
Frame 0:  Detection ✓   count=5
Frame 1:  No inference  count=? (stale)
Frame 2:  No inference  count=?
Frame 3:  Detection ✓   count=6
```

Policies:
- **Policy 0**: Trust last detection (may lag)
- **Policy 1**: Mark staleness, keep count
- **Policy 2**: Weight stale detections lower (`effective_count`)
- **Policy 3**: Only count confirmed detections

We use Policy 2 by default:
```python
effective_count = confirmed * 1.0 + unconfirmed * 0.5
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                      MAIN LOOP                               │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. VIDEO SOURCE                                             │
│    - Read frame from file/camera/RTSP                       │
│    - Handle disconnection with exponential backoff          │
│    - Maintain original frame for cropping                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SCHEDULER DECISION                                       │
│    if scheduler.can_run("wide"):                            │
│        run_wide_inference = True                            │
│    if narrow_triggered and scheduler.can_run("narrow"):     │
│        run_narrow_inference = True                          │
└─────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ 3A. WIDE INFERENCE      │     │ 3B. NARROW INFERENCE    │
│     - Resize to 320x320 │     │     - Crop center 45%   │
│     - YOLOv8n detect    │     │     - Resize to 320x320 │
│     - NMS filtering     │     │     - YOLOv8n detect    │
│     - Output: [xyxy]    │     │     - Map coords back   │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. TRACKING (IOU Tracker)                                   │
│    - Match detections to existing tracks by IOU             │
│    - Update matched tracks (position, velocity, aspect)     │
│    - Create new tracks for unmatched detections             │
│    - Age out tracks with no matches                         │
│    - Output: [track_id, xyxy, hits, age, trajectory]        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. EVENT DETECTION                                          │
│    For each track:                                          │
│    - Loiter: position stable > 2s in ROI                    │
│    - Fall: aspect ratio drop + velocity spike               │
│    - Output: [(track_id, event_type, confidence)]           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. POSTURE CLASSIFICATION                                   │
│    Based on bbox shape + velocity (no pose model):          │
│    - STAND: tall aspect ratio (h/w > 1.1)                   │
│    - SIT: medium ratio, low velocity                        │
│    - FALL?: low ratio (h/w < 0.75), high velocity           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 7. RE-ID (Optional)                                         │
│    - Extract OSNet features for each person crop            │
│    - Match against feature cache from other cameras         │
│    - Update cross-camera identity mapping                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 8. VISUALIZATION & OUTPUT                                   │
│    - Draw bounding boxes + trajectories                     │
│    - Overlay heatmap (density grid)                         │
│    - Compose KPI cards panel                                │
│    - Encode MJPEG frame                                     │
│    - Update web endpoints (/video, /metrics)                │
│    - Log to CSV + JSONL                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Responsibilities

### `main.py` (Orchestrator)
- Parse arguments
- Initialize all modules
- Run main loop with FPS limiting
- Coordinate scheduler decisions

### `modules/scheduler.py` (Resource Manager)
- Enforce inference budget
- Allocate Hz across streams
- Prevent inference overlap

### `modules/yolo_onnx.py` (Detector)
- Load ONNX model
- Preprocess (letterbox, normalize)
- Run inference (CPU or CUDA)
- Postprocess (NMS, filter by class=person)

### `modules/tracker_iou.py` (Tracker)
- Maintain track state machine
- Compute velocities and trajectories
- Handle track lifecycle (create/update/delete)

### `modules/metrics.py` (Event Engine)
- Stateful event detection per track
- Emit events with timestamps
- Debounce to prevent spam

### `modules/posture.py` (Posture Classifier)
- Heuristic classification from bbox
- State machine with debounce
- No deep learning (fast)

### `tool/web_view.py` (Web Server)
- Flask-based MJPEG streaming
- REST API for metrics
- Control endpoints (pause/resume)

### `tool/reid_extractor.py` (Re-ID)
- OSNet ONNX inference
- Feature vector extraction
- Cosine similarity matching

---

## Why These Choices?

### Why IOU Tracker instead of DeepSORT?

- **Simplicity**: 133 lines vs thousands
- **Speed**: No Re-ID per frame (we do Re-ID separately on demand)
- **Edge-friendly**: Minimal compute overhead

Trade-off: Less robust to occlusion. Acceptable for controlled environments.

### Why ONNX instead of PyTorch?

- **Deployment parity**: Same runtime on Jetson
- **No Python dependency hell**: Just onnxruntime
- **Faster cold start**: No JIT compilation

### Why MJPEG instead of WebRTC?

- **Simplicity**: No STUN/TURN servers
- **Compatibility**: Works everywhere
- **Latency acceptable**: ~100-200ms is fine for monitoring

Trade-off: Higher bandwidth. Acceptable for LAN deployment.

### Why Multi-Process instead of Multi-Thread?

- **GIL avoidance**: True parallelism
- **Isolation**: One camera crash doesn't kill others
- **Simple scaling**: Add cameras = add processes

Trade-off: Higher memory. Acceptable with 16GB+ RAM.

---

## Future Improvements

1. **WebRTC streaming** for lower latency
2. **TensorRT** for faster Jetson inference
3. **Pose estimation** for better fall detection
4. **Zone analytics** with polygon ROI definition
5. **Alert system** with webhook notifications
