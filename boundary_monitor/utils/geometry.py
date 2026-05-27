"""
geometry.py — Lightweight geometry helpers used across the pipeline.
"""

import numpy as np
import cv2


def iou(bA, bB) -> float:
    """Intersection-over-Union for two (x, y, w, h) boxes."""
    ax1, ay1 = bA[0], bA[1]
    ax2, ay2 = bA[0] + bA[2], bA[1] + bA[3]
    bx1, by1 = bB[0], bB[1]
    bx2, by2 = bB[0] + bB[2], bB[1] + bB[3]
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    union = bA[2] * bA[3] + bB[2] * bB[3] - inter
    return inter / union if union > 0 else 0.0


def point_in_polygon(pt, poly) -> bool:
    """Ray-casting point-in-polygon test."""
    x, y = pt
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
            inside = not inside
        j = i
    return inside


def apply_gamma(frame: np.ndarray, gamma: float = 1.8) -> np.ndarray:
    """Apply gamma correction via a LUT."""
    inv = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)], np.uint8)
    return cv2.LUT(frame, lut)
