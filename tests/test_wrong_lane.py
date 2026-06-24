# -*- coding: utf-8 -*-
"""Tests for wrong-lane detection."""
from detection.yoloworld_engine import Detection
from tracking.bytetrack_wrapper import ByteTracker
from intelligence.wrong_lane import WrongLaneDetector


def _car(cy):
    # 60px-tall box so consecutive frames overlap (tracker keeps the ID)
    return Detection(-1, "car", 0.9, 40, cy, 60, cy + 60)


def test_detects_wrong_direction():
    """Allowed = Down (180°); a car moving UP should be flagged."""
    tracker = ByteTracker()
    wl = WrongLaneDetector(allowed_direction_deg=180.0,
                           tolerance_deg=90.0, min_history=3)
    # Car moving upward (cy decreasing) in small steps so IoU tracking holds
    for cy in [400, 385, 370, 355, 340, 325, 310, 295]:
        tracker.update([_car(cy)])
        wl.check(tracker)
    assert len(wl._flagged) >= 1


def test_correct_direction_not_flagged():
    """Car moving DOWN (allowed) should not be flagged."""
    tracker = ByteTracker()
    wl = WrongLaneDetector(allowed_direction_deg=180.0,
                           tolerance_deg=90.0, min_history=3)
    for cy in [100, 115, 130, 145, 160, 175, 190, 205]:
        tracker.update([_car(cy)])
        wl.check(tracker)
    assert len(wl._flagged) == 0
