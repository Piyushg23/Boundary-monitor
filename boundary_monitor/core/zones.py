"""
zones.py — Zone geometry, drawing, and interactive zone editor.
"""

from typing import List, Tuple

import cv2
import numpy as np


def build_zones(W: int, H: int):
    """Return default outer and inner zone polygons for a frame of size W×H."""
    m_o = 0.08
    outer = [
        [int(W * m_o),       int(H * m_o)],
        [int(W * (1 - m_o)), int(H * m_o)],
        [int(W * (1 - m_o)), int(H * (1 - m_o))],
        [int(W * m_o),       int(H * (1 - m_o))],
    ]
    m_i = 0.22
    inner = [
        [int(W * m_i),       int(H * m_i)],
        [int(W * (1 - m_i)), int(H * m_i)],
        [int(W * (1 - m_i)), int(H * (1 - m_i))],
        [int(W * m_i),       int(H * (1 - m_i))],
    ]
    return outer, inner


def draw_zones(frame: np.ndarray, outer: list, inner: list):
    """Overlay translucent zone polygons onto frame (in-place)."""
    ov = frame.copy()
    cv2.polylines(ov, [np.array(outer, np.int32)], True, (180, 100, 0), 2, cv2.LINE_AA)
    cv2.polylines(ov, [np.array(inner, np.int32)], True, (0, 60, 200),  2, cv2.LINE_AA)
    for pt in outer:
        cv2.drawMarker(ov, tuple(pt), (140, 80, 0), cv2.MARKER_TILTED_CROSS, 12, 1)
    for pt in inner:
        cv2.drawMarker(ov, tuple(pt), (0, 50, 180), cv2.MARKER_TILTED_CROSS, 12, 1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)


class ZoneEditor:
    """
    Interactive zone editor.
    Click 4 outer corners, then 4 inner corners to redefine zones.
    Bind to a named OpenCV window via bind_window() after the window is created.
    """

    def __init__(self, W: int, H: int, outer: list, inner: list):
        self.W, self.H = W, H
        self.outer = [list(p) for p in outer]
        self.inner = [list(p) for p in inner]
        self.active = False
        self._clicks: List[Tuple[int, int]] = []
        self._stage = 0  # 0 = outer, 1 = inner
        self._window_name = None

    def bind_window(self, window_name: str):
        self._window_name = window_name
        cv2.setMouseCallback(window_name, self._on_mouse)

    def _on_mouse(self, event, x, y, flags, param):
        if not self.active:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self._clicks.append((x, y))
            if self._stage == 0 and len(self._clicks) == 4:
                self.outer = [list(p) for p in self._clicks]
                self._clicks = []
                self._stage = 1
                print("[ZONE] Outer zone set. Click 4 inner corners.")
            elif self._stage == 1 and len(self._clicks) == 4:
                self.inner = [list(p) for p in self._clicks]
                self._clicks = []
                self._stage = 0
                self.active = False
                print("[ZONE] Inner zone set. Zone editor closed.")

    def start(self):
        self.active = True
        self._stage = 0
        self._clicks = []
        print("[ZONE] Zone editor ON — click 4 outer corners then 4 inner corners.")

    def draw_guide(self, frame: np.ndarray):
        if not self.active:
            return
        label = "OUTER ZONE" if self._stage == 0 else "INNER ZONE"
        need = 4 - len(self._clicks)
        cv2.putText(
            frame,
            f"ZONE EDITOR: Click {need} more {label} corners",
            (frame.shape[1] // 2 - 200, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )
        for pt in self._clicks:
            cv2.circle(frame, pt, 6, (0, 255, 0), -1)
            cv2.drawMarker(frame, pt, (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
