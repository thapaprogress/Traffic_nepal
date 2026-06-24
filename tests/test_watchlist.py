# -*- coding: utf-8 -*-
"""
tests/test_watchlist.py
Tests for intelligence/watchlist.py — WatchlistSystem.
Uses an in-memory SQLite database so no files are created.
"""
import pytest
from intelligence.watchlist import WatchlistSystem


@pytest.fixture
def wl(tmp_path):
    """Provide a clean WatchlistSystem backed by a temp DB file."""
    db = str(tmp_path / "test_watchlist.db")
    return WatchlistSystem(db_path=db)


def test_add_and_check_stolen_plate(wl):
    """A plate added as STOLEN should be matched by check()."""
    wl.add_plate("BA 12 PA 1234", reason="STOLEN")
    result = wl.check("BA 12 PA 1234")
    assert result == "STOLEN"


def test_normalization_ignores_spaces_and_case(wl):
    """check() should normalize input — spaces, lowercase, dashes all OK."""
    wl.add_plate("BA12PA1234", reason="WANTED")
    # Different formatting, same plate
    assert wl.check("ba 12 pa 1234") == "WANTED"
    assert wl.check("BA-12-PA-1234") == "WANTED"


def test_unknown_plate_returns_none(wl):
    """Plates not in the watchlist should return None."""
    wl.add_plate("KO 01 CH 0001", reason="STOLEN")
    result = wl.check("KA 02 JA 5678")
    assert result is None


def test_remove_plate_clears_from_check(wl):
    """After remove_plate, check() should return None for that plate."""
    wl.add_plate("LU 03 KA 9999")
    wl.remove_plate("LU 03 KA 9999")
    assert wl.check("LU 03 KA 9999") is None


def test_count_reflects_entries(wl):
    """count property should match number of added unique plates."""
    assert wl.count == 0
    wl.add_plate("BA 12 PA 0001")
    wl.add_plate("BA 12 PA 0002")
    assert wl.count == 2


def test_get_all_returns_dict_list(wl):
    """get_all() returns list of dicts with plate/reason keys."""
    wl.add_plate("BA 12 PA 1111", reason="BLACKLISTED", notes="Test note")
    entries = wl.get_all()
    assert len(entries) == 1
    assert entries[0]["plate"] == "BA12PA1111"
    assert entries[0]["reason"] == "BLACKLISTED"


def test_short_plate_text_skipped(wl):
    """Plates shorter than 4 chars should not match (too ambiguous)."""
    wl.add_plate("ABC", reason="STOLEN")
    assert wl.check("ABC") is None


def test_duplicate_add_does_not_duplicate_entry(wl):
    """Adding the same plate twice should not double the count."""
    wl.add_plate("BA 12 PA 9999", reason="STOLEN")
    wl.add_plate("BA 12 PA 9999", reason="WANTED")  # update via INSERT OR REPLACE
    assert wl.count == 1
    # Most recent reason should win
    entries = wl.get_all()
    assert entries[0]["reason"] == "WANTED"
