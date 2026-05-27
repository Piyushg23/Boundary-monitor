"""
tracker.py — ByteTrack multi-object tracker.

ByteTrack reference:
  Zhang et al., "ByteTrack: Multi-Object Tracking by Associating Every
  Detection Box", ECCV 2022.  https://arxiv.org/abs/2110.06864

Algorithm per frame
───────────────────
1. Split detections into HIGH (conf ≥ high_thresh) and LOW pools.
2. Predict all active tracks forward with the Kalman filter.
3. Round-1  — match HIGH detections → all active tracks  (IoU ≥ iou_thresh).
4. Round-2  — match LOW  detections → unmatched active tracks (lower IoU bar).
5. Round-3  — match remaining HIGH detections → lost tracks (re-ID rescue).
6. Unmatched HIGH detections → new tentative tracks.
7. Tracks with no match: move to "lost" pool; remove after max_lost frames.
8. Tentative tracks confirmed after min_hits consecutive matches.

The public API (TrackManager.update) is identical to the old greedy tracker
so no other file needs to change.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from utils.config import CFG
from utils.geometry import iou
from core.tracking import Track, KinematicKalman

# ── Global ID counter (survives reset so IDs are always unique) ───────────────
_counter = 0


def _new_tid() -> int:
    global _counter
    _counter += 1
    return _counter


# ── Hungarian / linear-sum assignment (scipy if available, else greedy) ───────
try:
    from scipy.optimize import linear_sum_assignment as _lsa
    def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
        """Return (row, col) pairs that minimise total cost (maximise IoU)."""
        rows, cols = _lsa(cost)
        return list(zip(rows.tolist(), cols.tolist()))
except ImportError:
    def _assign(cost: np.ndarray) -> List[Tuple[int, int]]:
        """Greedy fallback: repeatedly pick the minimum-cost cell."""
        taken_r, taken_c, pairs = set(), set(), []
        flat = np.argsort(cost, axis=None)
        for idx in flat:
            r, c = divmod(int(idx), cost.shape[1])
            if r not in taken_r and c not in taken_c:
                pairs.append((r, c))
                taken_r.add(r)
                taken_c.add(c)
        return pairs


def _iou_matrix(tracks: List[Track], dets: list) -> np.ndarray:
    """Build an (n_tracks × n_dets) IoU matrix."""
    mat = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, tr in enumerate(tracks):
        for j, det in enumerate(dets):
            mat[i, j] = iou(tr.box, det[:4])
    return mat


def _match(
    tracks: List[Track],
    dets: list,
    iou_thresh: float,
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Match tracks to detections via linear assignment on IoU.

    Returns
    -------
    matches      : list of (track_idx, det_idx) pairs above iou_thresh
    unmatched_tr : track indices with no valid match
    unmatched_dt : detection indices with no valid match
    """
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))

    iou_mat = _iou_matrix(tracks, dets)
    cost_mat = 1.0 - iou_mat                      # minimise cost = maximise IoU

    pairs = _assign(cost_mat)

    matched_tr, matched_dt = set(), set()
    matches = []
    for r, c in pairs:
        if iou_mat[r, c] >= iou_thresh:
            matches.append((r, c))
            matched_tr.add(r)
            matched_dt.add(c)

    unmatched_tr = [i for i in range(len(tracks)) if i not in matched_tr]
    unmatched_dt = [j for j in range(len(dets))   if j not in matched_dt]
    return matches, unmatched_tr, unmatched_dt


# ── TrackState enum ───────────────────────────────────────────────────────────
class _State:
    TENTATIVE = "tentative"   # not yet confirmed
    CONFIRMED = "confirmed"   # enough consecutive hits
    LOST      = "lost"        # temporarily unmatched


class TrackManager:
    """
    ByteTrack-style multi-object tracker.

    Reads from CFG:
        high_det_thresh     — conf threshold separating HIGH / LOW pools
                              (falls back to yolo_conf if absent)
        low_det_thresh      — minimum conf to keep a detection at all
                              (falls back to 0.1 if absent)
        iou_match_thresh    — Round-1 & Round-3 IoU gate
        byte_iou_low        — Round-2 IoU gate for LOW detections
                              (falls back to 0.5 if absent)
        track_init_frames   — hits needed to confirm a tentative track
        track_coast_frames  — lost frames before a track is deleted
        max_tracks          — hard cap on simultaneous tracks
    """

    def __init__(self):
        self._active: Dict[int, Track] = {}   # confirmed + tentative tracks
        self._lost:   Dict[int, Track] = {}   # recently lost, eligible for re-ID
        self.tracks:  Dict[int, Track] = {}   # union exposed to the rest of the app

    def reset(self):
        self._active.clear()
        self._lost.clear()
        self.tracks.clear()

    # ── public update ─────────────────────────────────────────────────────────
    def update(
        self,
        detections: list,
        dt: float = None,
        occluded_force: bool = False,
    ) -> List[Track]:
        real_dt = dt or CFG["dt"]

        # Step 1 — predict all tracks forward
        for tr in list(self._active.values()) + list(self._lost.values()):
            tr.kf.predict(dt=real_dt)
            tr.age += 1
            tr.occ_frames += 1

        if occluded_force or not detections:
            self._prune()
            self._sync()
            return list(self.tracks.values())

        # Step 2 — split detections by confidence
        high_thresh = CFG.get("high_det_thresh", CFG.get("yolo_conf", 0.30))
        low_thresh  = CFG.get("low_det_thresh",  0.10)
        byte_iou_lo = CFG.get("byte_iou_low",    0.50)

        high_dets = [d for d in detections if d[4] >= high_thresh]
        low_dets  = [d for d in detections if low_thresh <= d[4] < high_thresh]

        active_list = list(self._active.values())

        # ── Round 1: HIGH dets  ←→  active tracks ────────────────────────────
        matches1, unmatched_tr1, unmatched_hi = _match(
            active_list, high_dets, CFG["iou_match_thresh"]
        )
        for ti, di in matches1:
            self._update_track(active_list[ti], high_dets[di], real_dt)

        # ── Round 2: LOW dets  ←→  unmatched active tracks ───────────────────
        remaining_active = [active_list[i] for i in unmatched_tr1]
        matches2, unmatched_tr2, _ = _match(
            remaining_active, low_dets, byte_iou_lo
        )
        for ti, di in matches2:
            self._update_track(remaining_active[ti], low_dets[di], real_dt)

        # Tracks still unmatched after both rounds → move to lost
        still_unmatched_active = [remaining_active[i] for i in unmatched_tr2]
        for tr in still_unmatched_active:
            self._lost[tr.tid] = self._active.pop(tr.tid)

        # ── Round 3: remaining HIGH dets  ←→  lost tracks (re-ID) ────────────
        lost_list = list(self._lost.values())
        unmatched_hi_dets = [high_dets[j] for j in unmatched_hi]
        matches3, _, still_unmatched_hi_idx = _match(
            lost_list, unmatched_hi_dets, CFG["iou_match_thresh"]
        )
        for ti, di in matches3:
            recovered = lost_list[ti]
            self._update_track(recovered, unmatched_hi_dets[di], real_dt)
            self._active[recovered.tid] = self._lost.pop(recovered.tid)

        # ── Step 6: spawn new tentative tracks for remaining HIGH dets ────────
        final_unmatched = [unmatched_hi_dets[j] for j in still_unmatched_hi_idx]
        for det in final_unmatched:
            if len(self._active) + len(self._lost) >= CFG["max_tracks"]:
                break
            x, y, w, h, conf, lbl = det
            cx, cy = x + w / 2, y + h / 2
            tid = _new_tid()
            kf  = KinematicKalman(cx, cy)
            tr  = Track(tid=tid, label=lbl, kf=kf, box=(x, y, w, h), conf=conf, hits=1)
            tr.trail.append((cx, cy))
            self._active[tid] = tr

        self._prune()
        self._sync()
        return list(self.tracks.values())

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _update_track(tr: Track, det: tuple, dt: float):
        x, y, w, h, conf, lbl = det
        cx, cy = x + w / 2, y + h / 2
        tr.kf.correct(cx, cy)
        tr.box        = (x, y, w, h)
        tr.conf       = conf
        tr.occ_frames = 0
        tr.hits      += 1
        sv = tr.kf.state()
        tr.trail.append((float(sv[0]), float(sv[1])))

    def _prune(self):
        max_lost = CFG["track_coast_frames"]

        # Remove dead lost tracks
        dead = [tid for tid, tr in self._lost.items()
                if tr.occ_frames > max_lost]
        for tid in dead:
            del self._lost[tid]

        # Remove active tracks that somehow coasted too long (safety net)
        dead = [tid for tid, tr in self._active.items()
                if tr.occ_frames > max_lost]
        for tid in dead:
            del self._active[tid]

    def _sync(self):
        """Expose confirmed + tentative active tracks via self.tracks."""
        self.tracks = {
            tid: tr for tid, tr in self._active.items()
        }
