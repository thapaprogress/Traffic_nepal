# -*- coding: utf-8 -*-
"""
intelligence/perspective.py
Perspective (homography) calibration for accurate speed across the frame.

A camera sees the road at an angle, so a fixed pixels-to-metres ratio is wrong:
objects far away move fewer pixels for the same real distance. By mapping 4
image points to a known real-world rectangle on the road plane, we can convert
any pixel point to real-world metres and measure true distance travelled.
"""

import numpy as np
from typing import List, Tuple, Optional


class PerspectiveMapper:
    """
    Maps image pixels → real-world ground-plane coordinates (metres) via a
    4-point homography.

    src_points: 4 image points (px) of a known road rectangle, order:
                top-left, top-right, bottom-right, bottom-left
    real_width_m:  real width  of that rectangle (metres)
    real_length_m: real length of that rectangle (metres)
    """

    def __init__(self,
                 src_points: List[Tuple[float, float]],
                 real_width_m: float,
                 real_length_m: float):
        import cv2
        if len(src_points) != 4:
            raise ValueError("src_points must have exactly 4 points")
        self.real_width_m = real_width_m
        self.real_length_m = real_length_m
        src = np.array(src_points, dtype=np.float32)
        # Destination = real-world rectangle (metres), same corner order
        dst = np.array([
            [0.0, 0.0],
            [real_width_m, 0.0],
            [real_width_m, real_length_m],
            [0.0, real_length_m],
        ], dtype=np.float32)
        self._H = cv2.getPerspectiveTransform(src, dst)

    def to_world(self, px: float, py: float) -> Tuple[float, float]:
        """Convert an image pixel (px, py) to world metres (x_m, y_m)."""
        pt = np.array([px, py, 1.0], dtype=np.float64)
        w = self._H @ pt
        if abs(w[2]) < 1e-9:
            return (0.0, 0.0)
        return (float(w[0] / w[2]), float(w[1] / w[2]))

    def world_distance(self, p1: Tuple[float, float],
                       p2: Tuple[float, float]) -> float:
        """Real-world distance (metres) between two image points."""
        x1, y1 = self.to_world(*p1)
        x2, y2 = self.to_world(*p2)
        return float(np.hypot(x2 - x1, y2 - y1))
