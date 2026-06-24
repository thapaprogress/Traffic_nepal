# -*- coding: utf-8 -*-
"""Tests for the PerspectiveMapper (#2)."""
import pytest
from intelligence.perspective import PerspectiveMapper


def test_corner_maps_to_origin():
    """Top-left source corner maps to world (0,0)."""
    pm = PerspectiveMapper(
        src_points=[(100, 100), (300, 100), (300, 400), (100, 400)],
        real_width_m=10.0, real_length_m=20.0,
    )
    x, y = pm.to_world(100, 100)
    assert abs(x) < 1e-6 and abs(y) < 1e-6


def test_known_rectangle_distance():
    """Vertical span of the calibration rectangle equals its real length."""
    pm = PerspectiveMapper(
        src_points=[(100, 100), (300, 100), (300, 400), (100, 400)],
        real_width_m=10.0, real_length_m=20.0,
    )
    # top-left (100,100) to bottom-left (100,400) should be 20 m
    d = pm.world_distance((100, 100), (100, 400))
    assert abs(d - 20.0) < 0.01


def test_requires_four_points():
    with pytest.raises(ValueError):
        PerspectiveMapper([(0, 0), (1, 1)], 5.0, 5.0)
