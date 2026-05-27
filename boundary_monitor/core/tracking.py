"""
tracking.py — Kinematic Kalman filter and Track data object.
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple

import cv2
import numpy as np

from utils.config import CFG, TRACK_COLORS
from utils.geometry import point_in_polygon


class KinematicKalman:
    """
    6-DOF Kalman filter: state = [px, py, vx, vy, ax, ay], measurement = [px, py].
    Transition matrix is updated per-frame using the actual measured dt.
    """

    def __init__(self, cx: float, cy: float):
        dt = CFG["dt"]
        self.kf = cv2.KalmanFilter(6, 2)
        self._update_transition(dt)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32
        )
        qp = CFG["process_noise_pos"]
        qv = CFG["process_noise_vel"]
        qa = CFG["process_noise_acc"]
        self.kf.processNoiseCov = np.diag([qp, qp, qv, qv, qa, qa]).astype(np.float32)
        r = CFG["measure_noise"]
        self.kf.measurementNoiseCov = np.array([[r, 0], [0, r]], np.float32)
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 500
        self.kf.statePost = np.array([[cx], [cy], [0], [0], [0], [0]], np.float32)

    def _update_transition(self, dt: float):
        dt2 = 0.5 * dt * dt
        self.kf.transitionMatrix = np.array([
            [1, 0, dt, 0, dt2, 0],
            [0, 1, 0, dt, 0, dt2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], np.float32)

    def predict(self, dt: float = None):
        if dt:
            self._update_transition(dt)
        p = self.kf.predict()
        return float(p[0][0]), float(p[1][0])

    def correct(self, cx: float, cy: float):
        e = self.kf.correct(np.array([[cx], [cy]], np.float32))
        return float(e[0][0]), float(e[1][0])

    def state(self) -> np.ndarray:
        return self.kf.statePost.flatten()

    def cov_ellipse(self):
        P = self.kf.errorCovPost[:2, :2]
        evals, evecs = np.linalg.eigh(P)
        evals = np.maximum(evals, 0)
        angle = math.degrees(math.atan2(evecs[1, 0], evecs[0, 0]))
        axes = (int(2 * math.sqrt(evals[0]) + 1), int(2 * math.sqrt(evals[1]) + 1))
        return axes, angle

    def forecast(self, horizon: int) -> List[Tuple[float, float]]:
        state = self.kf.statePost.copy()
        F = self.kf.transitionMatrix
        pts = []
        for _ in range(horizon):
            state = F @ state
            pts.append((float(state[0][0]), float(state[1][0])))
        return pts


_track_counter = 0


@dataclass
class Track:
    tid:        int
    label:      str
    kf:         KinematicKalman
    box:        Tuple[int, int, int, int]
    trail:      deque = field(default_factory=lambda: deque(maxlen=CFG["trail_len"]))
    conf:       float = 0.0
    occ_frames: int   = 0
    age:        int   = 0
    hits:       int   = 0
    threat:     float = 0.0
    first_seen: float = field(default_factory=time.time)
    last_zone:  str   = "outside"

    @property
    def color(self):
        return TRACK_COLORS[self.tid % len(TRACK_COLORS)]

    @property
    def confirmed(self) -> bool:
        return self.hits >= CFG["track_init_frames"]

    @property
    def occluded(self) -> bool:
        return self.occ_frames > 0

    def threat_level(self, zone_inner, zone_outer, W: int, H: int) -> float:
        sv = self.kf.state()
        fps = 1.0 / max(CFG["dt"], 1e-6)
        speed = math.hypot(sv[2], sv[3]) * fps
        speed_score = min(speed / 300.0, 1.0)
        cx, cy = sv[0], sv[1]
        in_inner = point_in_polygon((cx, cy), zone_inner)
        in_outer = point_in_polygon((cx, cy), zone_outer)
        zone_score = 1.0 if in_inner else (0.5 if in_outer else 0.0)
        ax, ay = sv[4], sv[5]
        accel_score = min(math.hypot(ax, ay) * 10, 0.2)
        self.threat = min(0.45 * speed_score + 0.45 * zone_score + accel_score, 1.0)
        self.last_zone = "inner" if in_inner else ("outer" if in_outer else "outside")
        return self.threat

    def to_dict(self) -> dict:
        """Serialise track state for Groq prompt."""
        sv = self.kf.state()
        fps = 1.0 / max(CFG["dt"], 1e-6)
        return {
            "id":             str(f"T{self.tid:02d}"),
            "label":          str(self.label),
            "position_px":    [float(round(float(sv[0]), 1)), float(round(float(sv[1]), 1))],
            "velocity_px_s":  [float(round(float(sv[2] * fps), 1)), float(round(float(sv[3] * fps), 1))],
            "speed_px_s":     float(round(float(math.hypot(float(sv[2]), float(sv[3])) * fps), 1)),
            "zone":           str(self.last_zone),
            "threat_score":   float(round(float(self.threat), 3)),
            "age_frames":     int(self.age),
            "occluded":       bool(self.occluded),
        }
