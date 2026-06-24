# -*- coding: utf-8 -*-
"""Tests for helmet multi-frame confirmation (A11)."""
from intelligence.helmet_rule import HelmetConfirmer


def test_requires_consecutive_frames():
    hc = HelmetConfirmer(frames_to_confirm=3)
    assert hc.update({7}) == set()   # frame 1
    assert hc.update({7}) == set()   # frame 2
    assert hc.update({7}) == {7}     # frame 3 → confirmed


def test_streak_resets_when_absent():
    hc = HelmetConfirmer(frames_to_confirm=3)
    hc.update({7})
    hc.update(set())          # gap → streak resets
    assert hc.update({7}) == set()   # only 1 again, not confirmed


def test_fires_once_only():
    hc = HelmetConfirmer(frames_to_confirm=2)
    hc.update({5})
    assert hc.update({5}) == {5}     # confirmed
    assert hc.update({5}) == set()   # not re-fired
