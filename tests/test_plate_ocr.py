# -*- coding: utf-8 -*-
"""Tests for plate normalization and multi-read voting (A9 + voting)."""
from intelligence.plate_ocr import (
    normalize_plate_text, correct_segment, PlateCache,
)


def test_normalize_does_not_corrupt_letters():
    """A9: must not blindly turn O->0, I->1, S->5."""
    assert normalize_plate_text("KOSI BA") == "KOSI BA"


def test_normalize_strips_noise():
    assert normalize_plate_text("ba-12@pa#4567") == "BA 12 PA 4567"


def test_correct_segment_digit():
    assert correct_segment("I2O", "digit") == "120"


def test_correct_segment_letter():
    assert correct_segment("8A", "letter") == "BA"


def test_voting_consensus_wins():
    """Noisy minority read is outvoted by the consensus."""
    c = PlateCache(max_reads=5)
    for t in ["BA12PA4567", "BA12PA4567", "BA12PA4S67", "BA12PA4567"]:
        c.add_read(1, t, 0.8)
    assert c.get(1) == "BA12PA4567"


def test_voting_locks_after_max_reads():
    c = PlateCache(max_reads=3)
    for _ in range(3):
        c.add_read(2, "BA01KA0001", 0.9)
    assert c.needs_more_reads(2) is False


def test_low_confidence_ignored():
    c = PlateCache(max_reads=5)
    c.add_read(3, "GARBAGE", 0.05)   # below PLATE_MIN_CONFIDENCE
    assert c.get(3) is None


def test_positional_normalization():
    # 4-segment format: BA 12 PA 4567
    # Here, 12 has an 'I' instead of '1' and an 'O' instead of '0'.
    assert normalize_plate_text("BA I2 PA O456") == "BA 12 PA 0456"
    # 3-segment format: BA PA 12O4 (where O should be 0)
    assert normalize_plate_text("BA PA I2O4") == "BA PA 1204"
    # 2-segment format: BA I2O4
    assert normalize_plate_text("BA I2O4") == "BA 1204"

