"""
config.py — Central configuration and constants for Boundary Monitor v3.
"""

import os
import sys
from pathlib import Path

# ── Locate project root (two levels up from utils/config.py) ─────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"

# ── Load .env ─────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    loaded = load_dotenv(dotenv_path=ENV_PATH, override=True)
    if loaded:
        print(f"[CONFIG] .env loaded from: {ENV_PATH}")
    else:
        print(f"[CONFIG] WARNING: .env not found at: {ENV_PATH}")
        print(f"[CONFIG] Create it with:  GROQ_API_KEY=your_key_here")
except ImportError:
    print("[CONFIG] WARNING: python-dotenv not installed.")
    print("[CONFIG] Run: pip install python-dotenv")

# ── Groq API Key ──────────────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "").strip()

# Startup diagnostic — prints once so user knows the status immediately
_placeholder_values = {"", "your_groq_api_key_here", "YOUR_GROQ_API_KEY_HERE"}
if GROQ_API_KEY in _placeholder_values:
    print("=" * 60)
    print("[GROQ] API KEY NOT SET")
    print(f"[GROQ] Expected .env file at: {ENV_PATH}")
    print("[GROQ] File must contain:  GROQ_API_KEY=gsk_xxxxxxxxxxxx")
    print("[GROQ] Get a free key at:  https://console.groq.com")
    print("[GROQ] Groq analysis will be DISABLED until key is set.")
    print("=" * 60)
else:
    masked = GROQ_API_KEY[:8] + "..." + GROQ_API_KEY[-4:]
    print(f"[CONFIG] GROQ_API_KEY loaded OK: {masked}")

# ── Main configuration dict ───────────────────────────────────────────────────
CFG = {
    "dt":                  1 / 30,
    "process_noise_pos":   5e-3,
    "process_noise_vel":   1e-1,
    "process_noise_acc":   2e-1,
    "measure_noise":       8.0,
    "occlusion_conf_thresh": 0.30,
    "occlusion_max_frames":  90,
    "min_motion_area":       800,
    "clahe_clip":  3.5,
    "clahe_tile":  (8, 8),
    "sigma_s":     30,
    "sigma_r":     0.25,
    "denoise":     False,
    "night_mode":  False,
    "stabilise":   False,
    "yolo_conf":      0.30,
    "yolo_nms":       0.45,
    "yolo_input_sz":  640,
    "yolo_warmup":    90,
    "trail_len":          120,
    "pred_horizon":       18,
    "max_tracks":         12,
    "iou_match_thresh":   0.20,
    "track_init_frames":  3,
    "track_coast_frames": 75,
    # ByteTrack-specific
    "high_det_thresh":    0.50,   # conf ≥ this → HIGH pool (Round-1 matching)
    "low_det_thresh":     0.10,   # conf ≥ this → LOW pool  (Round-2 rescue)
    "byte_iou_low":       0.50,   # IoU gate for Round-2 (LOW det ↔ lost track)
    "max_display_width":  1280,
    "groq_interval_s":      8.0,
    "groq_model":           "llama-3.3-70b-versatile",
    "groq_max_tokens":      300,
    "groq_breach_cooldown": 15.0,
}

# ── COCO class labels ─────────────────────────────────────────────────────────
COCO_LABELS = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra",
    "giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass","cup",
    "fork","knife","spoon","bowl","banana","apple","sandwich","orange",
    "broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier",
    "toothbrush"
]

TRACK_COLORS = [
    (0, 255, 120), (0, 220, 255), (255, 160, 0), (180, 0, 255),
    (0, 140, 255), (255, 80, 160),(100, 255, 255),(60, 255, 60),
    (0, 200, 200), (200, 200, 0), (255, 100, 100),(150, 255, 150),
]


def tactical_label(coco_cls: str) -> str:
    cls = coco_cls.lower()
    if cls in {"airplane", "bird", "kite", "frisbee"}: return "UAV"
    if cls == "person":                                 return "Person"
    if cls in {"car","motorcycle","truck","bus","bicycle"}: return "Vehicle"
    if cls in {"dog","cat","horse","cow","sheep"}:      return "Animal"
    return "Object"
