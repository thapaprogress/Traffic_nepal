# -*- coding: utf-8 -*-
"""Tests for the SpeedEstimator (A1 + A2 fixes)."""
from detection.yoloworld_engine import Detection
from intelligence.speed_estimator import SpeedEstimator


def _veh(tid, cy):
    d = Detection(-1, "car", 0.9, 40, cy - 5, 60, cy + 5)
    d.track_id = tid
    return d


def test_frame_based_speed():
    """20 m crossed in 10 frames @ 25 fps = 0.4 s = 180 km/h."""
    se = SpeedEstimator(line_y_a=100, line_y_b=200,
                        real_distance_m=20, speed_limit_kmh=60, source_fps=25)
    se.update([_veh(5, 90)], 1)    # prime prev position
    se.update([_veh(5, 105)], 2)   # cross line A
    recs = se.update([_veh(5, 205)], 12)  # cross line B at frame 12
    assert len(recs) == 1
    assert recs[0].speed_kmh == 180.0
    assert recs[0].is_violation is True


def test_no_speed_without_full_crossing():
    """A vehicle that only crosses line A produces no record."""
    se = SpeedEstimator(line_y_a=100, line_y_b=200, source_fps=25)
    se.update([_veh(7, 90)], 1)
    recs = se.update([_veh(7, 105)], 2)   # only crossed A
    assert recs == []


def test_absurd_speed_rejected():
    """Crossing both lines in 1 frame would be absurd → rejected by clamp."""
    se = SpeedEstimator(line_y_a=100, line_y_b=200,
                        real_distance_m=20, source_fps=25)
    se.update([_veh(9, 90)], 1)
    # jump across both lines in consecutive frame
    recs = se.update([_veh(9, 205)], 2)
    # Either no crossing registered or clamped out; must not record absurd value
    assert all(1.0 <= r.speed_kmh <= 250.0 for r in recs)
