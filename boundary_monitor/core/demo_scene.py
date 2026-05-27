"""
demo_scene.py — Synthetic airfield scene for testing without a real video source.
"""

import math
from typing import List, Tuple

import cv2
import numpy as np


class SyntheticScene:
    """
    Generates a synthetic airfield background with two moving targets:
      - A UAV following a sinusoidal arc (briefly occluded mid-sequence)
      - A Person doing a slow perimeter walk
    """

    OCC_START = 180
    OCC_END   = 255

    def __init__(self, W: int, H: int):
        self.W, self.H = W, H
        self.base = self._build_base()

    def _build_base(self) -> np.ndarray:
        W, H = self.W, self.H
        img = np.zeros((H, W, 3), np.uint8)
        img[:] = (28, 30, 28)

        # Runway
        cv2.rectangle(img, (int(W * .10), int(H * .42)), (int(W * .90), int(H * .58)), (40, 42, 40), -1)
        for x in range(int(W * .10), int(W * .90), 60):
            cv2.rectangle(img, (x, int(H * .495)), (x + 30, int(H * .505)), (55, 58, 55), -1)

        # Grid
        for gx in range(0, W, 60): cv2.line(img, (gx, 0), (gx, H), (35, 37, 35), 1)
        for gy in range(0, H, 60): cv2.line(img, (0, gy), (W, gy), (35, 37, 35), 1)

        # Hangar A
        cv2.rectangle(img, (int(W * .65), int(H * .12)), (int(W * .88), int(H * .38)), (55, 52, 48), -1)
        cv2.rectangle(img, (int(W * .65), int(H * .12)), (int(W * .88), int(H * .38)), (70, 67, 62),  1)
        cv2.putText(img, "HANGAR A", (int(W * .66), int(H * .28)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (80, 80, 70), 1, cv2.LINE_AA)

        # Control Tower
        cv2.rectangle(img, (int(W * .06), int(H * .08)), (int(W * .22), int(H * .36)), (52, 50, 55), -1)
        cv2.rectangle(img, (int(W * .06), int(H * .08)), (int(W * .22), int(H * .36)), (70, 68, 72),  1)
        cv2.putText(img, "TWR", (int(W * .09), int(H * .24)),
                    cv2.FONT_HERSHEY_SIMPLEX, .42, (80, 78, 85), 1, cv2.LINE_AA)

        return img

    def get_frame(self, frame_idx: int) -> np.ndarray:
        return self.base.copy()

    def targets(self, frame_idx: int) -> List[Tuple]:
        """Return list of (cx, cy, w, h, label, is_occluded) for each target."""
        W, H = self.W, self.H
        t = frame_idx / 30.0
        results = []

        # UAV — sinusoidal arc
        uav_cx = int(W * .10 + (W * .80) * ((t / 20.0) % 1.0))
        uav_cy = int(H * .35 + H * .18 * math.sin(t * 1.0))
        uav_cx = min(max(uav_cx, 40), W - 40)
        uav_cy = min(max(uav_cy, 40), H - 40)
        uav_occ = self.OCC_START <= frame_idx <= self.OCC_END
        results.append((uav_cx, uav_cy, 62, 28, "UAV", uav_occ))

        # Person — slow perimeter walk
        per_cx = int(W * .15 + W * .55 * (math.sin(t * .3) + 1) / 2)
        per_cy = int(H * .70 + H * .05 * math.cos(t * .7))
        per_cx = min(max(per_cx, 40), W - 40)
        per_cy = min(max(per_cy, 40), H - 40)
        results.append((per_cx, per_cy, 22, 50, "Person", False))

        return results

    def draw_targets(self, frame: np.ndarray, targets: List[Tuple]):
        """Render ground-truth target shapes (dark outlines, no annotation)."""
        for cx, cy, w, h, lbl, occ in targets:
            if occ:
                continue
            color = (60, 60, 60)
            if lbl == "UAV":
                pts = np.array([[cx - w // 2, cy], [cx, cy - h // 2],
                                 [cx + w // 2, cy], [cx, cy + h // 2]])
                cv2.polylines(frame, [pts], True, color, 1)
                cv2.circle(frame, (cx, cy), 5, color, -1)
            else:
                cv2.rectangle(frame, (cx - w // 2, cy - h // 2),
                              (cx + w // 2, cy + h // 2), color, 1)
