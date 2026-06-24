# -*- coding: utf-8 -*-
"""
intelligence/wrong_lane.py
Detects vehicles moving against the allowed traffic direction.
Uses track history (center points) to compute movement vector,
compares against the configured allowed direction per lane.
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tracking.bytetrack_wrapper import ByteTracker, Track
from config.settings import VEHICLE_CLASSES


@dataclass
class WrongLaneViolation:
    track_id:      int
    class_name:    str
    direction_deg: float   # actual movement direction
    allowed_deg:   float   # expected direction
    center:        Tuple[int, int]


class WrongLaneDetector:
    """
    Checks if tracked vehicles are moving in the wrong direction.

    allowed_direction_deg: compass angle of allowed traffic flow
        0 = up, 90 = right, 180 = down, 270 = left
    tolerance_deg: how far off the allowed direction before flagging (default 120)
    min_history: minimum track history length before checking
    """

    def __init__(
        self,
        allowed_direction_deg: float = 180.0,  # default: traffic goes down
        tolerance_deg: float = 120.0,
        min_history: int = 5,
    ):
        self.allowed_deg = allowed_direction_deg
        self.tolerance   = tolerance_deg
        self.min_history = min_history
        self._flagged: set = set()  # track IDs already flagged

    def _angle_between(self, p1: Tuple[int,int], p2: Tuple[int,int]) -> float:
        """Calculate movement direction in degrees (0=up, clockwise)."""
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        # atan2 gives angle from positive x-axis; convert to compass
        rad = math.atan2(dx, -dy)  # negate dy so up=0
        deg = math.degrees(rad) % 360
        return deg

    def _angular_diff(self, a: float, b: float) -> float:
        """Smallest angular difference between two angles."""
        diff = abs(a - b) % 360
        return min(diff, 360 - diff)

    def check(self, tracker: ByteTracker) -> List[WrongLaneViolation]:
        """
        Check all active tracks for wrong-direction movement.
        Returns new violations (each track flagged only once).
        """
        violations: List[WrongLaneViolation] = []

        for tid, track in tracker.tracks.items():
            if tid in self._flagged:
                continue
            if track.class_name not in VEHICLE_CLASSES:
                continue
            if len(track.history) < self.min_history:
                continue

            # Compute average movement direction from recent history
            start = track.history[-self.min_history]
            end   = track.history[-1]
            if start == end:
                continue

            direction = self._angle_between(start, end)
            diff = self._angular_diff(direction, self.allowed_deg)

            if diff > self.tolerance:
                self._flagged.add(tid)
                violations.append(WrongLaneViolation(
                    track_id       = tid,
                    class_name     = track.class_name,
                    direction_deg  = round(direction, 1),
                    allowed_deg    = self.allowed_deg,
                    center         = track.center,
                ))

        return violations
