# -*- coding: utf-8 -*-
"""
tests/test_congestion.py
Tests for intelligence/congestion_monitor.py — CongestionMonitor.
"""
import numpy as np
from detection.yoloworld_engine import Detection
from intelligence.congestion_monitor import CongestionMonitor, CongestionState


def _det(track_id: int, cls: str, cx: int, cy: int) -> Detection:
    """Helper — create a Detection centered at (cx, cy)."""
    return Detection(track_id=track_id, class_name=cls,
                     confidence=0.9,
                     x1=cx - 20, y1=cy - 20, x2=cx + 20, y2=cy + 20)


def test_no_roi_counts_all_vehicles():
    """Without a ROI polygon, every vehicle in the frame is counted."""
    monitor = CongestionMonitor(camera_id="cam_test")
    dets = [
        _det(1, "car", 100, 100),
        _det(2, "motorcycle", 200, 200),
        _det(3, "person", 300, 300),   # person is not a vehicle
    ]
    state = monitor.update(dets)
    assert state.vehicle_count == 2
    assert state.camera_id == "cam_test"


def test_roi_polygon_filters_outside_vehicles():
    """Vehicles outside the ROI polygon should NOT be counted."""
    # ROI = small box (50-150, 50-150)
    roi = [(50, 50), (150, 50), (150, 150), (50, 150)]
    monitor = CongestionMonitor(camera_id="cam_roi", roi_polygon=roi)
    dets = [
        _det(1, "car", 100, 100),   # inside ROI
        _det(2, "car", 300, 300),   # outside ROI
        _det(3, "bus", 80, 80),     # inside ROI
    ]
    state = monitor.update(dets)
    assert state.vehicle_count == 2


def test_congestion_level_low_with_few_vehicles():
    """Small vehicle counts should produce LOW congestion."""
    monitor = CongestionMonitor()
    for _ in range(10):    # fill rolling window
        state = monitor.update([_det(1, "car", 100, 100)])
    assert state.congestion_level == "LOW"


def test_congestion_level_high_with_many_vehicles():
    """Counts above CONGESTION_MEDIUM threshold should produce HIGH."""
    from config.settings import CONGESTION_MEDIUM
    monitor = CongestionMonitor()
    # Generate enough detections to surpass HIGH threshold
    many_vehicles = [
        _det(i, "car", 100 + i * 5, 100)
        for i in range(CONGESTION_MEDIUM + 5)
    ]
    for _ in range(30):  # fill window with high counts
        state = monitor.update(many_vehicles)
    assert state.congestion_level == "HIGH"


def test_smoothed_count_averages_over_window():
    """Smoothed count should be a rolling average, not instant spike."""
    monitor = CongestionMonitor(window_size=5)
    # Push 3 frames with 0 vehicles, then 1 frame with 10
    for _ in range(3):
        monitor.update([])
    state = monitor.update([_det(i, "car", 50, 50) for i in range(10)])
    # With 3 zeros + 1 ten in window of 4, average = 10/4 = 2.5
    assert state.smoothed_count < 10.0
    assert state.smoothed_count > 0.0


def test_draw_roi_returns_frame_same_shape():
    """draw_roi should return an array of the same shape as the input."""
    monitor = CongestionMonitor(roi_polygon=[(50, 50), (200, 50), (200, 200), (50, 200)])
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    state = CongestionState(
        camera_id="cam",
        vehicle_count=3,
        congestion_level="MEDIUM",
        smoothed_count=3.0,
    )
    out = monitor.draw_roi(frame, state)
    assert out.shape == frame.shape
