# Boundary Monitor v3

**IG Defence / Constems-AI Grade — Real-time perimeter surveillance with Groq LLM tactical analysis.**

A computer vision pipeline that detects, tracks, and threat-scores targets crossing defined boundary zones. Integrates YOLOv8 object detection, a 6-DOF Kinematic Kalman filter, MOG2 background subtraction, and live tactical assessment via the Groq LLM API.

---

## Features

- **YOLOv8s ONNX detection** — 640px inference, auto-downloaded on first run
- **Multi-object Kalman tracking** — 6-state kinematics (position, velocity, acceleration), FPS-adaptive `dt`
- **Dual threat zones** — Outer (warning) and inner (critical) polygons, redefinable at runtime
- **Groq LLM analyst** — Non-blocking background thread; sends scene snapshots every N seconds or on breach
- **Night / low-light mode** — Gamma correction + aggressive CLAHE
- **Optional denoising & ECC stabilisation** — For noisy or shaky real footage
- **Tkinter launcher** — Dark-themed GUI for source selection before pipeline starts
- **Interactive zone editor** — Click to redefine zones without restarting
- **MP4 recording** — Toggle with `[S]` key

---

## Project Structure

```
boundary_monitor/
├── main.py                  # Entry point — loads .env, opens launcher
├── pipeline.py              # Main video processing loop
├── requirements.txt
├── .env.example             # Copy to .env and add your API key
├── .gitignore
│
├── core/
│   ├── tracking.py          # KinematicKalman filter + Track dataclass
│   ├── tracker.py           # Multi-object TrackManager (IoU matching)
│   ├── analyst.py           # GroqAnalyst — non-blocking LLM integration
│   ├── zones.py             # Zone geometry, drawing, ZoneEditor
│   └── demo_scene.py        # Synthetic airfield scene for testing
│
├── detection/
│   ├── vision.py            # VisibilityEnhancer + MotionTrigger (MOG2)
│   └── detector.py          # YOLODetector (ONNX via OpenCV DNN)
│
├── ui/
│   ├── renderer.py          # TacticalRenderer — all HUD drawing
│   └── launcher.py          # Tkinter launcher GUI
│
└── utils/
    ├── config.py            # CFG dict, COCO labels, constants
    └── geometry.py          # IoU, point-in-polygon, gamma LUT
```

---

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/boundary-monitor.git
cd boundary-monitor
pip install -r requirements.txt
```

### 2. Set up your API key

```bash
cp .env.example .env
# Open .env and replace "your_groq_api_key_here" with your real key
```

Get a free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Run

```bash
python main.py
```

The launcher window will open. Select a source (demo / video file / webcam) and press **LAUNCH MONITOR**.

---

## Key Bindings (in the monitor window)

| Key | Action |
|-----|--------|
| `P` | Pause / resume |
| `R` | Reset all tracks |
| `O` | Toggle forced occlusion (Kalman-only mode) |
| `S` | Toggle MP4 recording |
| `G` | Manually trigger Groq analysis |
| `Z` | Open zone editor (click 4 outer + 4 inner corners) |
| `N` | Toggle night / low-light mode |
| `Q` / `ESC` | Quit |

---

## Configuration

All tunable parameters are in `utils/config.py` under the `CFG` dict. Key values:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `yolo_conf` | `0.30` | YOLO detection confidence threshold |
| `yolo_input_sz` | `640` | YOLO input resolution |
| `groq_interval_s` | `8.0` | Seconds between automatic Groq analyses |
| `groq_breach_cooldown` | `15.0` | Min seconds between breach-triggered analyses |
| `track_coast_frames` | `75` | Frames to coast a track before pruning |
| `night_mode` | `False` | Enable gamma lift for dark scenes |
| `denoise` | `False` | Enable slow denoising pre-pass |
| `stabilise` | `False` | Enable ECC frame stabilisation |

---

## Requirements

- Python 3.9+
- OpenCV 4.8+ (with DNN module)
- A Groq API key (free tier available)
- The YOLO `.onnx` model is downloaded automatically on first run (~6 MB for nano, ~22 MB for small)

---

## License

MIT — see `LICENSE` for details.
