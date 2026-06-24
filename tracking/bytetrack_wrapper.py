# -*- coding: utf-8 -*-
"""
tracking/bytetrack_wrapper.py
Lightweight ByteTrack-style multi-object tracker using pure NumPy/SciPy.
Provides persistent integer IDs across frames without requiring a full
ByteTrack install — drop-in ready for the MVP.

When you install the real ByteTrack later, swap out the _assign() method.
"""

import numpy as np
from typing import List, Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.yoloworld_engine import Detection

# Real ByteTrack optional imports
try:
    from yolox.tracker.byte_tracker import BYTETracker
    class RealByteTrackerArgs:
        def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
            self.track_thresh = track_thresh
            self.track_buffer = track_buffer
            self.match_thresh = match_thresh
            self.mot20 = False
    HAS_REAL_BYTETRACK = True
except ImportError:
    HAS_REAL_BYTETRACK = False


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes [x1,y1,x2,y2]."""
    n, m = len(boxes_a), len(boxes_b)
    if n == 0 or m == 0:
        return np.zeros((n, m))
    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]
    ix1 = np.maximum(ax1[:, None], bx1[None, :])
    iy1 = np.maximum(ay1[:, None], by1[None, :])
    ix2 = np.minimum(ax2[:, None], bx2[None, :])
    iy2 = np.minimum(ay2[:, None], by2[None, :])
    inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-6)


class Track:
    """Single tracked object state."""

    def __init__(self, det: Detection, track_id: int):
        self.track_id  = track_id
        self.class_name = det.class_name
        self.bbox       = np.array([det.x1, det.y1, det.x2, det.y2], dtype=float)
        self.confidence = det.confidence
        self.age        = 0          # frames since last match
        self.hits       = 1
        self.history: List[Tuple[int,int]] = [det.center]   # center points

    def update(self, det: Detection):
        self.bbox       = np.array([det.x1, det.y1, det.x2, det.y2], dtype=float)
        self.confidence = det.confidence
        self.class_name = det.class_name
        self.age        = 0
        self.hits      += 1
        self.history.append(det.center)
        if len(self.history) > 60:
            self.history.pop(0)

    def predict(self):
        """Simple constant-velocity prediction (advance age)."""
        self.age += 1

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= 2

    @property
    def center(self) -> Tuple[int, int]:
        return (int((self.bbox[0] + self.bbox[2]) / 2),
                int((self.bbox[1] + self.bbox[3]) / 2))


class ByteTracker:
    """
    IoU-based multi-object tracker.
    Matches detections to existing tracks via Hungarian assignment on IoU.
    Supports real ByteTrack if installed.
    """

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30,
                 class_consistent: bool = True, track_thresh: float = 0.5,
                 track_buffer: int = 30, match_thresh: float = 0.8):
        self.iou_threshold = iou_threshold
        self.max_age       = max_age
        self.class_consistent = class_consistent
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1   # per-instance ID counter (per camera)

        self.use_real_bytetrack = HAS_REAL_BYTETRACK
        if self.use_real_bytetrack:
            args = RealByteTrackerArgs(track_thresh, track_buffer, match_thresh)
            self._real_tracker = BYTETracker(args, frame_rate=25)

    def _new_track(self, det: Detection) -> Track:
        t = Track(det, self._next_id)
        self._next_id += 1
        self.tracks[t.track_id] = t
        return t

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Match detections to tracks.
        Returns the same detections list with track_id filled in.
        """
        if not self.use_real_bytetrack:
            # Predict (age up all tracks)
            for t in self.tracks.values():
                t.predict()

            if not detections:
                self._remove_stale()
                return detections

            det_boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
            active_tracks = list(self.tracks.values())

            if active_tracks:
                trk_boxes = np.array([t.bbox for t in active_tracks])
                iou = _iou_matrix(trk_boxes, det_boxes)   # (n_tracks, n_dets)

                # Zero out IoU between mismatched classes (A5 fix)
                if self.class_consistent:
                    for ti, trk in enumerate(active_tracks):
                        for di, det in enumerate(detections):
                            if trk.class_name != det.class_name:
                                iou[ti, di] = 0.0

                matched_trk, matched_det = self._assign(iou)

                # Update matched tracks
                matched_det_ids = set()
                for ti, di in zip(matched_trk, matched_det):
                    active_tracks[ti].update(detections[di])
                    detections[di].track_id = active_tracks[ti].track_id
                    matched_det_ids.add(di)

                # Create new tracks for unmatched detections
                for di, det in enumerate(detections):
                    if di not in matched_det_ids:
                        t = self._new_track(det)
                        det.track_id = t.track_id
            else:
                for det in detections:
                    t = self._new_track(det)
                    det.track_id = t.track_id

            self._remove_stale()
            return detections
        else:
            if not detections:
                self._real_tracker.update(np.empty((0, 5)), [720, 1280], [720, 1280])
                active_ids = {t.track_id for t in self._real_tracker.tracked_stracks}
                self.tracks = {tid: t for tid, t in self.tracks.items() if tid in active_ids}
                return detections

            dets = np.array([[d.x1, d.y1, d.x2, d.y2, d.confidence] for d in detections])
            h = max(int(max(d.y2 for d in detections)), 1)
            w = max(int(max(d.x2 for d in detections)), 1)

            online_targets = self._real_tracker.update(dets, [h, w], [h, w])

            active_ids = set()
            for t in online_targets:
                tlbr = t.tlbr
                track_id = t.track_id
                active_ids.add(track_id)

                idx = -1
                best_iou = -1.0
                boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in detections])
                ious = _iou_matrix(boxes, np.array([tlbr]))
                if len(ious) > 0:
                    idx = np.argmax(ious[:, 0])
                    best_iou = ious[idx, 0]

                if idx >= 0 and best_iou > 0.3:
                    det = detections[idx]
                    det.track_id = track_id

                    if track_id in self.tracks:
                        self.tracks[track_id].update(det)
                    else:
                        self.tracks[track_id] = Track(det, track_id)
                else:
                    mock_det = Detection(track_id, "vehicle", t.score, tlbr[0], tlbr[1], tlbr[2], tlbr[3])
                    if track_id in self.tracks:
                        self.tracks[track_id].update(mock_det)
                    else:
                        self.tracks[track_id] = Track(mock_det, track_id)

            self.tracks = {tid: t for tid, t in self.tracks.items() if tid in active_ids}
            return detections

    def _assign(self, iou_matrix: np.ndarray):
        """Greedy assignment: match highest-IoU pairs above threshold."""
        matched_trk, matched_det = [], []
        if iou_matrix.size == 0:
            return matched_trk, matched_det
        used_trk, used_det = set(), set()
        # Sort all pairs by IoU descending
        pairs = sorted(
            [(i, j) for i in range(iou_matrix.shape[0])
                    for j in range(iou_matrix.shape[1])],
            key=lambda p: iou_matrix[p[0], p[1]],
            reverse=True
        )
        for ti, di in pairs:
            if iou_matrix[ti, di] < self.iou_threshold:
                break
            if ti in used_trk or di in used_det:
                continue
            matched_trk.append(ti)
            matched_det.append(di)
            used_trk.add(ti)
            used_det.add(di)
        return matched_trk, matched_det

    def _remove_stale(self):
        stale = [tid for tid, t in self.tracks.items() if t.age > self.max_age]
        for tid in stale:
            del self.tracks[tid]

    def get_track(self, track_id: int) -> Track:
        return self.tracks.get(track_id)
