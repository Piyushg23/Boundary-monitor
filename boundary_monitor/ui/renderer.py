"""
renderer.py — Tactical HUD renderer: tracks, detection boxes, Groq panel, status.
"""

import math
import time
from typing import List

import cv2
import numpy as np

from utils.config import CFG
from core.tracking import Track
from core.analyst import GroqAnalyst


class TacticalRenderer:
    # Colour palette (BGR)
    BRIGHT   = (0, 255, 100)
    DIM      = (0, 160,  60)
    DIMMER   = (0,  90,  35)
    WHITE    = (220, 230, 220)
    RED      = (0,   0, 255)
    ORANGE   = (0, 140, 255)
    YELLOW   = (0, 220, 240)
    CYAN     = (220, 210,  0)
    GROQ_COL = (255, 220, 100)

    def threat_color(self, threat: float):
        if threat > 0.7: return self.RED
        if threat > 0.4: return self.ORANGE
        return self.BRIGHT

    # ── Detection box ─────────────────────────────────────────────────────────
    def draw_detection_box(self, frame, x, y, w, h, label, conf, color, threat):
        tc = self.threat_color(threat)
        sz = 14
        segs = [
            [(x, y + sz), (x, y), (x + sz, y)],
            [(x + w - sz, y), (x + w, y), (x + w, y + sz)],
            [(x + w, y + h - sz), (x + w, y + h), (x + w - sz, y + h)],
            [(x + sz, y + h), (x, y + h), (x, y + h - sz)],
        ]
        for seg in segs:
            for i in range(len(seg) - 1):
                cv2.line(frame, seg[i], seg[i + 1], tc, 2, cv2.LINE_AA)
        tag = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.50, 1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), tc, 1)
        cv2.putText(frame, tag, (x + 4, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.50, tc, 1, cv2.LINE_AA)
        bar_h = int(h * threat)
        if bar_h > 0:
            cv2.rectangle(frame, (x + w + 3, y + h - bar_h), (x + w + 7, y + h), tc, -1)

    # ── Track overlay ─────────────────────────────────────────────────────────
    def draw_track(self, frame, tr: Track):
        if not tr.confirmed:
            return
        sv = tr.kf.state()
        kx, ky = int(sv[0]), int(sv[1])
        color = tr.color
        tc = self.threat_color(tr.threat)

        # Fading trail
        trail = list(tr.trail)
        n = len(trail)
        for i in range(1, n):
            alpha = i / n
            c = tuple(int(ch * alpha) for ch in color)
            thick = 2 if tr.occluded and i > n - 8 else 1
            cv2.line(
                frame,
                (int(trail[i - 1][0]), int(trail[i - 1][1])),
                (int(trail[i][0]),     int(trail[i][1])),
                c, thick, cv2.LINE_AA,
            )

        # Covariance ellipse
        axes, angle = tr.kf.cov_ellipse()
        axes = (max(axes[0], 4), max(axes[1], 4))
        if axes[0] < 200 and axes[1] < 200:
            cv2.ellipse(frame, (kx, ky), axes, angle, 0, 360, self.DIM, 1, cv2.LINE_AA)

        # Velocity arrow
        vx_px, vy_px = sv[2] * 8, sv[3] * 8
        if abs(vx_px) + abs(vy_px) > 3:
            ex, ey = int(kx + vx_px), int(ky + vy_px)
            cv2.arrowedLine(frame, (kx, ky), (ex, ey), tc, 2, cv2.LINE_AA, tipLength=0.35)

        cv2.circle(frame, (kx, ky), 5, color, -1)
        cv2.circle(frame, (kx, ky), 8, color,  1)
        cv2.putText(frame, f"T{tr.tid:02d}", (kx + 12, ky - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)

        # Forecast trajectory
        pts = tr.kf.forecast(CFG["pred_horizon"])
        for i in range(1, len(pts)):
            alpha = 1 - i / len(pts)
            c = tuple(int(ch * alpha * 0.7) for ch in color)
            cv2.line(
                frame,
                (int(pts[i - 1][0]), int(pts[i - 1][1])),
                (int(pts[i][0]),     int(pts[i][1])),
                c, 1, cv2.LINE_AA,
            )
        if pts:
            marker_col = self.RED if tr.occluded else self.DIM
            cv2.drawMarker(frame, (int(pts[-1][0]), int(pts[-1][1])),
                           marker_col, cv2.MARKER_DIAMOND, 10, 1)

    # ── Groq panel ────────────────────────────────────────────────────────────
    def draw_groq_panel(self, frame, analyst: GroqAnalyst):
        H, W = frame.shape[:2]
        pw = 320
        px0 = W - pw - 4
        lines = analyst.lines
        ph = 28 + 18 * min(len(lines), 20)
        py0 = 26

        cv2.rectangle(frame, (px0 - 2, py0), (W - 2, py0 + ph), (0, 0, 0), -1)
        cv2.rectangle(frame, (px0 - 2, py0), (W - 2, py0 + ph), self.GROQ_COL, 1)

        status = "THINKING..." if analyst.busy else "GROQ AI"
        cv2.putText(frame, f"[*] {status}", (px0 + 6, py0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.GROQ_COL, 1, cv2.LINE_AA)
        cv2.line(frame, (px0 - 2, py0 + 18), (W - 2, py0 + 18), self.GROQ_COL, 1)

        row = py0 + 32
        for ln in lines[:20]:
            col = self.GROQ_COL
            ln_up = ln.upper()
            if "CRITICAL" in ln_up:  col = self.RED
            elif "HIGH"    in ln_up: col = (0, 60, 255)
            elif "MEDIUM"  in ln_up: col = self.ORANGE
            elif "LOW"     in ln_up: col = self.BRIGHT
            elif "ACTION"  in ln_up: col = self.YELLOW
            cv2.putText(frame, ln, (px0 + 4, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
            row += 17

    # ── Main HUD ──────────────────────────────────────────────────────────────
    def draw_hud(self, frame, tracks: List[Track], fps: float, frame_idx: int,
                 motion: bool, paused: bool, any_breach: bool,
                 night_mode: bool = False, recording: bool = False, warmup: bool = False):
        H, W = frame.shape[:2]

        # Top bar
        cv2.rectangle(frame, (0, 0), (W, 22), (0, 0, 0), -1)
        cv2.line(frame, (0, 22), (W, 22), self.DIM, 1)
        ts = time.strftime("%H:%M:%S UTC")
        extra = ""
        if night_mode: extra += " [NIGHT]"
        if warmup:     extra += " [WARMUP]"
        cv2.putText(
            frame,
            f"PERIMETER MONITOR v3{extra}  |  {ts}  |  FPS:{fps:.0f}  |  F:{frame_idx:05d}",
            (8, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.BRIGHT, 1, cv2.LINE_AA,
        )
        if paused:
            cv2.putText(frame, "[[ PAUSED ]]", (W // 2 - 55, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.YELLOW, 1, cv2.LINE_AA)
        if recording:
            cv2.circle(frame, (W - 20, 12), 6, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (W - 40, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        # Bottom track table
        conf_tracks = [t for t in tracks if t.confirmed]
        ph = 28 + 24 * min(len(conf_tracks), 4)
        py0 = H - ph - 2
        cv2.rectangle(frame, (0, py0), (W, H), (0, 0, 0), -1)
        cv2.line(frame, (0, py0), (W, py0), self.DIM, 1)
        row = py0 + 16
        for tr in conf_tracks[:4]:
            sv = tr.kf.state()
            fps_r = 1.0 / max(CFG["dt"], 1e-6)
            speed = math.hypot(sv[2], sv[3]) * fps_r
            tc = self.threat_color(tr.threat)
            line = (
                f"T{tr.tid:02d} {tr.label:<7}  "
                f"pos({sv[0]:5.0f},{sv[1]:5.0f})  "
                f"spd:{speed:5.1f}px/s  "
                f"zone:{tr.last_zone:<8}  thr:{tr.threat:.2f}"
            )
            cv2.putText(frame, line, (8, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc, 1, cv2.LINE_AA)
            row += 22
        if not conf_tracks:
            cv2.putText(frame, "-- NO CONFIRMED TRACKS --", (8, row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.DIMMER, 1, cv2.LINE_AA)

        # Motion / breach indicators
        indicators = [
            ("MOTION", motion,     self.BRIGHT, self.DIMMER),
            ("BREACH", any_breach, self.RED,    self.DIMMER),
        ]
        ix = W - 340
        for label_i, active, on_col, off_col in indicators:
            col = on_col if active else off_col
            cv2.rectangle(frame, (ix, 4), (ix + 95, 19), col, 1)
            cv2.putText(frame, label_i, (ix + 8, 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1, cv2.LINE_AA)
            ix -= 105

        # Occlusion banner
        occ_tracks = [t for t in tracks if t.confirmed and t.occluded]
        if occ_tracks:
            ids = " ".join(f"T{t.tid:02d}" for t in occ_tracks)
            banner = f"  SENSOR BLACKOUT — KALMAN EXTRAPOLATING: {ids}  "
            (bw, bh), _ = cv2.getTextSize(banner, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            bx = W // 2 - bw // 2
            cv2.rectangle(frame, (bx - 4, 26), (bx + bw + 4, 26 + bh + 8), (0, 30, 100), -1)
            cv2.rectangle(frame, (bx - 4, 26), (bx + bw + 4, 26 + bh + 8), (0, 80, 200), 1)
            cv2.putText(frame, banner, (bx, 26 + bh + 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.ORANGE, 1, cv2.LINE_AA)

        # Legend
        items = [
            (self.BRIGHT,   "Confirmed track"),
            (self.DIM,      "Covariance ellipse"),
            (self.ORANGE,   "Occluded / predict"),
            (self.RED,      "Threat HIGH"),
            ((0, 100, 200), "Inner zone (critical)"),
            ((140, 80, 0),  "Outer zone (warning)"),
            (self.GROQ_COL, "Groq AI analysis"),
        ]
        lx, ly = 8, 30
        for col, txt in items:
            cv2.circle(frame, (lx + 5, ly + 4), 4, col, -1)
            cv2.putText(frame, txt, (lx + 14, ly + 9),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)
            ly += 17

        # Scan-line CRT effect
        scan = np.zeros_like(frame)
        scan[::4, :] = (0, 15, 5)
        cv2.add(frame, scan, frame)
