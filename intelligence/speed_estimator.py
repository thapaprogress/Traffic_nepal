# -*- coding: utf-8 -*-
"""
intelligence/speed_estimator.py
Camera-calibrated vehicle speed estimation.

Two virtual horizontal lines (Line A and Line B).
Speed = real_distance / (frames_between_crossings / source_fps).

Uses FRAME-BASED timing (not wall-clock) so it is accurate regardless of
inference latency or frame-skip — critical for correctness (fix A1).
Line crossing is detected by checking if a tracked center moved ACROSS the
line between two updates (fix A2), and stale entries are expired (fix A3).
"""

import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.yoloworld_engine import Detection
from config.settings import SPEED_REAL_DISTANCE_M, SPEED_LIMIT_KMH, VEHICLE_CLASSES


@dataclass
class SpeedRecord:
    track_id:    int
    class_name:  str
    speed_kmh:   float
    is_violation: bool
    timestamp:   float


class SpeedEstimator:
    """
    Estimates speed using FRAME-BASED timing between two virtual lines.

    line_y_a: y-pixel of Line A (vehicle enters)
    line_y_b: y-pixel of Line B (vehicle exits) — must be BELOW line_a (larger y)
    real_distance_m: real-world gap between Line A and Line B in metres
    source_fps: frames per second of the SOURCE video (set by pipeline)
    frames_processed_ratio: account for FRAME_SKIP (effective fps = source_fps / skip)
    """

    def __init__(
        self,
        line_y_a: int = 300,
        line_y_b: int = 450,
        real_distance_m: float = SPEED_REAL_DISTANCE_M,
        speed_limit_kmh: float = SPEED_LIMIT_KMH,
        source_fps: float = 25.0,
        stale_frames: int = 300,   # expire un-completed crossings after N frames
        perspective=None,          # optional PerspectiveMapper for accurate distance
    ):
        self.line_y_a        = line_y_a
        self.line_y_b        = line_y_b
        self.real_distance_m = real_distance_m
        self.speed_limit_kmh = speed_limit_kmh
        self.source_fps      = max(source_fps, 1.0)
        self.stale_frames    = stale_frames
        self.perspective     = perspective

        # track_id → (frame_index when crossing Line A)
        self._enter_frames: Dict[int, int] = {}
        # track_id → center point at Line A crossing (for perspective distance)
        self._enter_points: Dict[int, Tuple[int, int]] = {}
        # track_id → previous center-y (to detect crossing direction)
        self._prev_cy: Dict[int, int] = {}
        # completed speed records
        self.records: List[SpeedRecord] = []

    def set_fps(self, fps: float):
        self.source_fps = max(fps, 1.0)

    def update(self, detections: List[Detection], frame_index: int) -> List[SpeedRecord]:
        """
        Call every PROCESSED frame with the source frame index.
        Returns new speed records generated this frame.
        """
        new_records = []
        seen_ids = set()

        for d in detections:
            if d.class_name not in VEHICLE_CLASSES or d.track_id < 0:
                continue
            tid = d.track_id
            seen_ids.add(tid)
            cx, cy = d.center
            prev_cy = self._prev_cy.get(tid)
            self._prev_cy[tid] = cy

            if prev_cy is None:
                continue

            # Crossed Line A downward (entering speed zone)
            if prev_cy < self.line_y_a <= cy:
                self._enter_frames[tid] = frame_index
                self._enter_points[tid] = (cx, cy)

            # Crossed Line B downward (exiting speed zone)
            if prev_cy < self.line_y_b <= cy and tid in self._enter_frames:
                frames_elapsed = frame_index - self._enter_frames.pop(tid)
                enter_pt = self._enter_points.pop(tid, (cx, self.line_y_a))
                if frames_elapsed > 0:
                    seconds = frames_elapsed / self.source_fps
                    # Use perspective (homography) distance if calibrated,
                    # else the fixed line distance.
                    if self.perspective is not None:
                        distance_m = self.perspective.world_distance(enter_pt, (cx, cy))
                    else:
                        distance_m = self.real_distance_m
                    speed_ms  = distance_m / seconds
                    speed_kmh = speed_ms * 3.6
                    # Sanity clamp — reject absurd readings
                    if 1.0 <= speed_kmh <= 250.0:
                        rec = SpeedRecord(
                            track_id     = tid,
                            class_name   = d.class_name,
                            speed_kmh    = round(speed_kmh, 1),
                            is_violation = speed_kmh > self.speed_limit_kmh,
                            timestamp    = time.time(),
                        )
                        self.records.append(rec)
                        new_records.append(rec)

        # ── Cleanup stale entries (fix A3 — memory leak) ───────────────
        self._cleanup(frame_index, seen_ids)
        return new_records

    def _cleanup(self, frame_index: int, seen_ids: set):
        """Expire crossings that never completed and prune prev positions."""
        stale = [tid for tid, f in self._enter_frames.items()
                 if frame_index - f > self.stale_frames]
        for tid in stale:
            self._enter_frames.pop(tid, None)
            self._enter_points.pop(tid, None)
        # Drop prev-positions of vehicles not seen this frame and not pending
        gone = [tid for tid in self._prev_cy
                if tid not in seen_ids and tid not in self._enter_frames]
        for tid in gone:
            self._prev_cy.pop(tid, None)

    def draw_lines(self, frame, new_records: List[SpeedRecord]):
        """Draw calibration lines and speed labels on frame."""
        import cv2
        h, w = frame.shape[:2]
        cv2.line(frame, (0, self.line_y_a), (w, self.line_y_a), (255, 100, 0), 2)
        cv2.putText(frame, "SPEED LINE A", (10, self.line_y_a - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 1)
        cv2.line(frame, (0, self.line_y_b), (w, self.line_y_b), (0, 165, 255), 2)
        cv2.putText(frame, "SPEED LINE B", (10, self.line_y_b - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
        for rec in new_records:
            color = (0, 0, 255) if rec.is_violation else (0, 255, 0)
            label = f"#{rec.track_id} {rec.speed_kmh} km/h {'SPEED!' if rec.is_violation else ''}"
            cv2.putText(frame, label, (w // 2 - 100, self.line_y_b + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
        return frame
