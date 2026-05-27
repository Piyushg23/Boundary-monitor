"""
pipeline.py — Main video processing loop for Boundary Monitor v3.
"""

import sys
import time

import cv2

from utils.config import CFG, GROQ_API_KEY
from utils.geometry import iou
from detection.vision import VisibilityEnhancer, MotionTrigger
from detection.detector import YOLODetector
from core.tracker import TrackManager
from core.tracking import KinematicKalman
from core.analyst import GroqAnalyst
from core.zones import build_zones, draw_zones, ZoneEditor
from core.demo_scene import SyntheticScene
from ui.renderer import TacticalRenderer


def run(
    source=None,
    demo_mode: bool = True,
    save_output: bool = False,
    conf_override: float = None,
    nms_override: float = None,
):
    if conf_override:
        CFG["yolo_conf"] = conf_override
    if nms_override:
        CFG["yolo_nms"] = nms_override

    WINDOW = "Boundary Monitor v3 — Multi-Track / Kalman / Threat / Groq"

    # ── Source setup ──────────────────────────────────────────────────────────
    W, H = 1280, 720
    cap = None
    if not demo_mode:
        src = source if source is not None else 0
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"[ERR] Cannot open source: {src}")
            sys.exit(1)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        real_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        CFG["dt"] = 1.0 / real_fps
        print(f"[SRC] {W}x{H} @ {real_fps:.1f} FPS  source={src}")

    disp_scale = 1.0
    if W > CFG["max_display_width"]:
        disp_scale = CFG["max_display_width"] / W
        dW = int(W * disp_scale)
        dH = int(H * disp_scale)
        print(f"[SRC] Display scaled to {dW}x{dH}")
    else:
        dW, dH = W, H

    # ── Module instantiation ──────────────────────────────────────────────────
    enhancer = VisibilityEnhancer()
    motion_t = MotionTrigger()
    detector = YOLODetector()
    tracker  = TrackManager()
    renderer = TacticalRenderer()
    analyst  = GroqAnalyst(GROQ_API_KEY)

    outer_zone, inner_zone = build_zones(dW, dH)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, dW, dH)

    zone_ed = ZoneEditor(dW, dH, outer_zone, inner_zone)
    zone_ed.bind_window(WINDOW)

    # ── Optional video writer ─────────────────────────────────────────────────
    writer = None
    recording = save_output

    def init_writer():
        out_path = f"boundary_output_{int(time.time())}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps_w = 1.0 / CFG["dt"]
        w = cv2.VideoWriter(out_path, fourcc, fps_w, (dW, dH))
        print(f"[REC] Recording to {out_path}")
        return w

    if recording:
        writer = init_writer()

    scene = SyntheticScene(dW, dH) if demo_mode else None

    # ── State ─────────────────────────────────────────────────────────────────
    frame_idx      = 0
    fps_disp       = 30.0
    t_prev         = time.time()
    paused         = False
    force_occ      = False
    groq_last_auto = 0.0
    groq_breach_t  = 0.0

    print("\n[SYS] Boundary Monitor v3 running.")
    print("      Keys: [P] pause  [R] reset  [O] occlude  [S] rec  [G] Groq")
    print("            [Z] zones  [N] night  [Q/ESC] quit\n")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        if key == ord('p'):
            paused = not paused
        if key == ord('r'):
            tracker.reset()
            print("[SYS] Tracks reset.")
        if key == ord('o'):
            force_occ = not force_occ
            print(f"[SYS] Force-occlude: {force_occ}")
        if key == ord('n'):
            CFG["night_mode"] = not CFG["night_mode"]
            print(f"[SYS] Night mode: {CFG['night_mode']}")
        if key == ord('g'):
            analyst.trigger(list(tracker.tracks.values()), frame_idx, reason="MANUAL")
            print("[GROQ] Manual analysis triggered.")
        if key == ord('z'):
            zone_ed.start()
        if key == ord('s'):
            recording = not recording
            if recording and writer is None:
                writer = init_writer()
            elif not recording and writer:
                writer.release()
                writer = None
                print("[REC] Stopped.")

        outer_zone = zone_ed.outer
        inner_zone = zone_ed.inner

        if paused:
            time.sleep(0.03)
            continue

        # ── Acquire frame ──────────────────────────────────────────────────
        if demo_mode and scene:
            frame = scene.get_frame(frame_idx)
        else:
            ret, frame = cap.read()
            if not ret:
                print("[SRC] End of stream.")
                break

        if disp_scale < 1.0:
            frame = cv2.resize(frame, (dW, dH), interpolation=cv2.INTER_AREA)

        enhanced = enhancer.enhance(frame)
        has_motion, _ = motion_t.process(enhanced)

        # ── Detection ─────────────────────────────────────────────────────
        warmup = (frame_idx < CFG["yolo_warmup"]) and not demo_mode
        detections = []
        global_occ = force_occ

        if demo_mode and scene:
            targets = scene.targets(frame_idx)
            scene.draw_targets(enhanced, targets)
            for cx, cy, w, h, lbl, occ in targets:
                if occ:
                    global_occ = True
                else:
                    x, y = cx - w // 2, cy - h // 2
                    detections.append((x, y, w, h, 0.91, lbl))
        elif not warmup:
            detections = detector.detect(enhanced)

        # ── FPS-adaptive dt ───────────────────────────────────────────────
        t_now = time.time()
        real_dt = max(t_now - t_prev, 1e-4)
        CFG["dt"] = 0.8 * CFG["dt"] + 0.2 * real_dt
        fps_disp = 0.9 * fps_disp + 0.1 / real_dt
        t_prev = t_now

        # ── Tracking & threat scoring ─────────────────────────────────────
        tracks = tracker.update(detections, dt=CFG["dt"], occluded_force=global_occ)
        draw_zones(enhanced, outer_zone, inner_zone)

        any_breach = False
        for tr in tracks:
            if tr.confirmed:
                threat = tr.threat_level(inner_zone, outer_zone, dW, dH)
                if threat > 0.5:
                    any_breach = True

        # ── Groq auto-trigger ─────────────────────────────────────────────
        analyst.poll()
        now = time.time()
        if now - groq_last_auto > CFG["groq_interval_s"]:
            analyst.trigger(tracks, frame_idx, reason="PERIODIC")
            groq_last_auto = now
        if any_breach and (now - groq_breach_t > CFG["groq_breach_cooldown"]):
            analyst.trigger(tracks, frame_idx, reason="BREACH_ALERT")
            groq_breach_t = now

        # ── Render ────────────────────────────────────────────────────────
        for d in detections:
            x, y, w, h, conf, lbl = d
            # Match by IoU first; fall back to nearest confirmed track by centre distance
            best_tr = next(
                (tr for tr in tracks if tr.confirmed and iou(tr.box, (x, y, w, h)) > 0.05),
                None,
            )
            if best_tr is None:
                # Nearest confirmed track whose Kalman position is close to detection centre
                cx_d, cy_d = x + w / 2, y + h / 2
                best_tr = min(
                    (tr for tr in tracks if tr.confirmed),
                    key=lambda tr: (tr.kf.state()[0] - cx_d) ** 2 + (tr.kf.state()[1] - cy_d) ** 2,
                    default=None,
                )
            color  = best_tr.color  if best_tr else renderer.BRIGHT
            threat = best_tr.threat if best_tr else 0.0
            renderer.draw_detection_box(enhanced, x, y, w, h, lbl, conf, color, threat)

        for tr in tracks:
            renderer.draw_track(enhanced, tr)

        renderer.draw_groq_panel(enhanced, analyst)
        renderer.draw_hud(
            enhanced, tracks, fps_disp, frame_idx,
            has_motion, paused, any_breach,
            night_mode=CFG["night_mode"],
            recording=recording,
            warmup=warmup,
        )
        zone_ed.draw_guide(enhanced)

        cv2.imshow(WINDOW, enhanced)
        if recording and writer:
            writer.write(enhanced)

        frame_idx += 1

    # ── Cleanup ───────────────────────────────────────────────────────────────
    if cap:    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()
    print("[SYS] Shutdown.")
