"""
vision.py — Frame pre-processing: visibility enhancement and motion detection.
"""

import cv2
import numpy as np

from utils.config import CFG
from utils.geometry import apply_gamma


class VisibilityEnhancer:
    """
    Applies CLAHE, optional gamma lift (night mode), denoising,
    and optional ECC frame stabilisation.
    """

    def __init__(self):
        self.clahe = cv2.createCLAHE(
            clipLimit=CFG["clahe_clip"], tileGridSize=CFG["clahe_tile"]
        )
        self._prev_gray = None
        self._warp_mode = cv2.MOTION_EUCLIDEAN
        self._criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
        self._warp_mat = np.eye(2, 3, dtype=np.float32)

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        if CFG["denoise"]:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 6, 6, 7, 21)

        if CFG["night_mode"]:
            frame = apply_gamma(frame, gamma=2.2)

        # CLAHE on L channel (LAB colour space)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self.clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2BGR)

        if CFG["stabilise"]:
            enhanced = self._stabilise(enhanced)

        return enhanced

    def _stabilise(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return frame
        try:
            _, self._warp_mat = cv2.findTransformECC(
                self._prev_gray, gray, self._warp_mat,
                self._warp_mode, self._criteria
            )
            h, w = frame.shape[:2]
            stabilised = cv2.warpAffine(
                frame, self._warp_mat, (w, h),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP
            )
            self._prev_gray = gray
            return stabilised
        except cv2.error:
            self._prev_gray = gray
            self._warp_mat = np.eye(2, 3, dtype=np.float32)
            return frame


class MotionTrigger:
    """
    MOG2 background subtractor with shadow removal.
    Returns (has_motion: bool, foreground_mask: ndarray).
    """

    def __init__(self):
        self.mog2 = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=40, detectShadows=True
        )
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def process(self, frame: np.ndarray):
        raw = self.mog2.apply(frame)
        # Discard shadow pixels (127); keep full foreground (255)
        _, fgmask = cv2.threshold(raw, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        has_motion = any(cv2.contourArea(c) > CFG["min_motion_area"] for c in contours)
        return has_motion, mask
