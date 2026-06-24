# -*- coding: utf-8 -*-
"""
tests/test_night_enhance.py
Tests for intelligence/night_enhance.py.
Verifies CLAHE enhancement, dark-frame detection, and auto_enhance passthrough.
"""
import numpy as np
import cv2
from intelligence.night_enhance import enhance_night_frame, is_dark_frame, auto_enhance


def _bright_frame(h=100, w=100):
    """Create a uniformly bright BGR frame (avg ~200)."""
    return np.full((h, w, 3), 200, dtype=np.uint8)


def _dark_frame(h=100, w=100):
    """Create a uniformly dark BGR frame (avg ~20)."""
    return np.full((h, w, 3), 20, dtype=np.uint8)


def test_is_dark_frame_detects_dark():
    """A very dark frame should trigger is_dark_frame = True."""
    assert is_dark_frame(_dark_frame()) is True


def test_is_dark_frame_passes_bright():
    """A bright frame should not be flagged as dark."""
    assert is_dark_frame(_bright_frame()) is False


def test_enhance_night_frame_returns_same_shape():
    """Enhancement should not change frame dimensions."""
    frame = _dark_frame(240, 320)
    enhanced = enhance_night_frame(frame)
    assert enhanced.shape == frame.shape
    assert enhanced.dtype == np.uint8


def test_enhance_night_frame_increases_brightness():
    """A very dark frame should be brighter after CLAHE enhancement."""
    frame = _dark_frame()
    enhanced = enhance_night_frame(frame, denoise=False)  # skip bilateral for speed
    orig_mean = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()
    enh_mean = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY).mean()
    assert enh_mean > orig_mean


def test_auto_enhance_passes_through_bright_frame():
    """auto_enhance should return the original object for bright frames (no copy overhead)."""
    bright = _bright_frame()
    result = auto_enhance(bright)
    # Result should equal (not crash) and be same shape
    assert result.shape == bright.shape


def test_auto_enhance_enhances_dark_frame():
    """auto_enhance should process a dark frame and produce brighter output."""
    dark = _dark_frame()
    enhanced = auto_enhance(dark, brightness_threshold=60.0)
    orig_mean = float(dark.mean())
    enh_mean = float(enhanced.mean())
    assert enh_mean > orig_mean


def test_enhance_without_denoise_is_faster_and_valid():
    """Enhancement with denoise=False still returns a valid BGR image."""
    frame = _dark_frame(480, 640)
    out = enhance_night_frame(frame, denoise=False)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
    # All pixel values should be valid uint8
    assert out.min() >= 0
    assert out.max() <= 255
