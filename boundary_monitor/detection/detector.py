"""
detector.py — YOLOv8 ONNX object detector with automatic model download.
"""

import os
import urllib.request

import cv2
import numpy as np

from utils.config import CFG, COCO_LABELS, tactical_label


class YOLODetector:
    """
    Loads a YOLOv8 ONNX model (nano or small) from disk, or downloads it.
    Falls back gracefully if no model is available.
    """

    # Candidates tried in order: nano first (faster), then small (more accurate)
    MODEL_CANDIDATES = [
        (
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.onnx",
            "yolov8n.onnx",
        ),
        (
            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.onnx",
            "yolov8s.onnx",
        ),
    ]

    def __init__(self):
        self.net = None
        self._load_model()

    def _load_model(self):
        # First pass: load any already-downloaded model
        for _, path in self.MODEL_CANDIDATES:
            if os.path.exists(path):
                try:
                    self.net = cv2.dnn.readNetFromONNX(path)
                    self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                    print(f"[YOLO] Loaded existing model: {path}")
                    return
                except Exception as e:
                    print(f"[YOLO] Existing model load error ({path}): {e}")
                    try:
                        os.remove(path)
                    except OSError:
                        pass

        # Second pass: download each candidate
        print("[YOLO] No local model found — attempting download …")
        for url, path in self.MODEL_CANDIDATES:
            print(f"[YOLO] Trying: {url}")
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as r, open(path, "wb") as f:
                    f.write(r.read())
                self.net = cv2.dnn.readNetFromONNX(path)
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
                print(f"[YOLO] Downloaded and loaded: {path}")
                return
            except Exception as e:
                print(f"[YOLO] Failed ({url}): {e}")
                try:
                    os.remove(path)
                except OSError:
                    pass

        print("[YOLO] *** All download attempts failed. Detection disabled. ***")
        print("[YOLO] Manual fix: place yolov8n.onnx in the project root.")

    def detect(self, frame: np.ndarray) -> list:
        """Returns list of (x, y, w, h, conf, tactical_label) tuples."""
        if self.net is None:
            return []

        fh, fw = frame.shape[:2]
        sz = CFG["yolo_input_sz"]
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (sz, sz), swapRB=True, crop=False)
        self.net.setInput(blob)
        raw = self.net.forward()[0].T

        xs, ys = fw / sz, fh / sz
        boxes, scores, class_ids = [], [], []

        for row in raw:
            cs = row[4:]
            cid = int(np.argmax(cs))
            conf = float(cs[cid])
            if conf < CFG["yolo_conf"]:
                continue
            cx, cy, bw, bh = row[:4]
            cx *= xs; cy *= ys; bw *= xs; bh *= ys
            boxes.append([int(cx - bw / 2), int(cy - bh / 2), int(bw), int(bh)])
            scores.append(conf)
            class_ids.append(cid)

        if not boxes:
            return []

        idx = cv2.dnn.NMSBoxes(boxes, scores, CFG["yolo_conf"], CFG["yolo_nms"])
        out = []
        for i in (idx.flatten() if len(idx) else []):
            x, y, w, h = boxes[i]
            lbl = tactical_label(
                COCO_LABELS[class_ids[i]] if class_ids[i] < len(COCO_LABELS) else "object"
            )
            out.append((x, y, w, h, scores[i], lbl))
        return out
