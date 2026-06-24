# -*- coding: utf-8 -*-
"""
intelligence/congestion_monitor.py
Zone-based traffic congestion analysis.

Define a polygon ROI per camera; count vehicles inside it each frame.
Smooth counts over a rolling window to avoid flickering.
"""

import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import List, Tuple, Deque
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.yoloworld_engine import Detection
from config.settings import CONGESTION_LOW, CONGESTION_MEDIUM, VEHICLE_CLASSES


@dataclass
class CongestionState:
    camera_id:        str
    vehicle_count:    int
    congestion_level: str   # LOW / MEDIUM / HIGH
    smoothed_count:   float


class CongestionMonitor:
    """
    Monitors vehicle density inside a configurable ROI polygon.

    Usage:
        monitor = CongestionMonitor(camera_id="cam_01")
        monitor.set_roi([(100,400),(900,400),(900,700),(100,700)])
        state = monitor.update(detections)
    """

    def __init__(
        self,
        camera_id: str = "cam_01",
        window_size: int = 30,       # rolling average window (frames)
        roi_polygon: List[Tuple[int, int]] = None,
    ):
        self.camera_id   = camera_id
        self.roi_polygon = np.array(roi_polygon, dtype=np.int32) if roi_polygon else None
        self._counts: Deque[int] = deque(maxlen=window_size)

    def set_roi(self, polygon: List[Tuple[int, int]]):
        """Set the region-of-interest polygon [(x,y), ...]."""
        self.roi_polygon = np.array(polygon, dtype=np.int32)

    def _point_in_roi(self, cx: int, cy: int) -> bool:
        if self.roi_polygon is None:
            return True   # no ROI = whole frame
        return cv2.pointPolygonTest(self.roi_polygon, (float(cx), float(cy)), False) >= 0

    def update(self, detections: List[Detection]) -> CongestionState:
        """Count vehicles in ROI and return congestion state."""
        count = 0
        for d in detections:
            if d.class_name in VEHICLE_CLASSES:
                cx, cy = d.center
                if self._point_in_roi(cx, cy):
                    count += 1
        self._counts.append(count)
        smoothed = sum(self._counts) / len(self._counts)

        if smoothed <= CONGESTION_LOW:
            level = "LOW"
        elif smoothed <= CONGESTION_MEDIUM:
            level = "MEDIUM"
        else:
            level = "HIGH"

        return CongestionState(
            camera_id        = self.camera_id,
            vehicle_count    = count,
            congestion_level = level,
            smoothed_count   = round(smoothed, 1),
        )

    def draw_roi(self, frame: np.ndarray, state: CongestionState,
                 in_place: bool = False) -> np.ndarray:
        """Overlay ROI polygon and congestion level on frame."""
        out = frame if in_place else frame.copy()
        LEVEL_COLORS = {"LOW": (0, 220, 80), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 230)}
        color = LEVEL_COLORS.get(state.congestion_level, (200, 200, 200))

        if self.roi_polygon is not None:
            overlay = out.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], (*color, 40))
            cv2.addWeighted(overlay, 0.2, out, 0.8, 0, out)
            cv2.polylines(out, [self.roi_polygon], isClosed=True, color=color, thickness=2)

        # Status badge
        badge_text = f"  Vehicles: {state.vehicle_count}  Traffic: {state.congestion_level}  "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        bx1, by1 = 10, frame.shape[0] - th - 20
        cv2.rectangle(out, (bx1 - 4, by1 - 4), (bx1 + tw + 4, by1 + th + 8), (20, 20, 20), -1)
        cv2.putText(out, badge_text, (bx1, by1 + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        return out
