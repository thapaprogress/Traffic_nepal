# -*- coding: utf-8 -*-
"""Tests for the ByteTracker (A4 + A5 fixes)."""
from detection.yoloworld_engine import Detection
from tracking.bytetrack_wrapper import ByteTracker


def _det(cls, x1, y1, x2, y2, conf=0.9):
    return Detection(-1, cls, conf, x1, y1, x2, y2)


def test_per_instance_ids():
    """Two trackers must not share ID counters (A4)."""
    t1, t2 = ByteTracker(), ByteTracker()
    t1.update([_det("car", 0, 0, 40, 40)])
    t2.update([_det("car", 0, 0, 40, 40)])
    assert list(t1.tracks.keys()) == [1]
    assert list(t2.tracks.keys()) == [1]


def test_persistent_id_across_frames():
    """Same object keeps its ID across frames."""
    t = ByteTracker()
    d1 = t.update([_det("car", 0, 0, 40, 40)])
    tid = d1[0].track_id
    d2 = t.update([_det("car", 5, 5, 45, 45)])  # moved slightly
    assert d2[0].track_id == tid


def test_class_consistency():
    """A car track should not be reassigned to a person (A5)."""
    t = ByteTracker()
    car = t.update([_det("car", 0, 0, 40, 40)])
    car_id = car[0].track_id
    # Person appears at same spot — must get a NEW id, not the car's
    out = t.update([_det("person", 0, 0, 40, 40)])
    assert out[0].track_id != car_id


def test_stale_track_removed():
    t = ByteTracker(max_age=2)
    t.update([_det("car", 0, 0, 40, 40)])
    for _ in range(5):
        t.update([])   # no detections → age up
    assert len(t.tracks) == 0
