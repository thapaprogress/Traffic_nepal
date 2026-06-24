# -*- coding: utf-8 -*-
"""
intelligence/watchlist.py
Stolen vehicle / wanted plate watchlist system.
Police can add plates via API; OCR results are checked against this list.
"""

import sqlite3
import re
import threading
from typing import List, Optional, Set
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH


@dataclass
class WatchlistMatch:
    plate_text:   str
    watch_reason: str   # STOLEN, WANTED, BLACKLISTED
    camera_id:    str
    timestamp:    float


class WatchlistSystem:
    """
    In-memory watchlist backed by SQLite for persistence.
    Thread-safe — can be checked from multiple inference threads.
    """

    def __init__(self, db_path: str = DB_PATH):
        self._lock    = threading.Lock()
        self._plates: Set[str] = set()
        self._reasons: dict = {}
        self._db_path = db_path
        self._ensure_table()
        self._load_from_db()

    def _ensure_table(self):
        conn = sqlite3.connect(self._db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                plate      TEXT UNIQUE,
                reason     TEXT DEFAULT 'STOLEN',
                added_at   REAL,
                notes      TEXT DEFAULT ''
            )
        """)
        conn.commit()
        conn.close()

    def _load_from_db(self):
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT plate, reason FROM watchlist").fetchall()
        conn.close()
        with self._lock:
            for plate, reason in rows:
                normalized = self._normalize(plate)
                self._plates.add(normalized)
                self._reasons[normalized] = reason

    @staticmethod
    def _normalize(plate: str) -> str:
        """Normalize plate for matching: uppercase, remove spaces/special chars."""
        return re.sub(r"[^A-Z0-9]", "", plate.upper())

    def add_plate(self, plate: str, reason: str = "STOLEN", notes: str = ""):
        """Add a plate to the watchlist."""
        import time
        normalized = self._normalize(plate)
        with self._lock:
            self._plates.add(normalized)
            self._reasons[normalized] = reason
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute(
                "INSERT OR REPLACE INTO watchlist (plate, reason, added_at, notes) VALUES (?,?,?,?)",
                (normalized, reason, time.time(), notes)
            )
            conn.commit()
        finally:
            conn.close()

    def remove_plate(self, plate: str):
        """Remove a plate from the watchlist."""
        normalized = self._normalize(plate)
        with self._lock:
            self._plates.discard(normalized)
            self._reasons.pop(normalized, None)
        conn = sqlite3.connect(self._db_path)
        conn.execute("DELETE FROM watchlist WHERE plate=?", (normalized,))
        conn.commit()
        conn.close()

    def check(self, plate_text: str) -> Optional[str]:
        """
        Check if a plate is on the watchlist.
        Returns the reason string if matched, None otherwise.
        """
        normalized = self._normalize(plate_text)
        if not normalized or len(normalized) < 4:
            return None
        with self._lock:
            if normalized in self._plates:
                return self._reasons.get(normalized, "WATCHLIST")
        return None

    def get_all(self) -> List[dict]:
        """Return all watchlist entries."""
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute("SELECT plate, reason, added_at, notes FROM watchlist").fetchall()
        conn.close()
        return [{"plate": r[0], "reason": r[1], "added_at": r[2], "notes": r[3]} for r in rows]

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._plates)


# Singleton
_watchlist_instance = None

def get_watchlist() -> WatchlistSystem:
    global _watchlist_instance
    if _watchlist_instance is None:
        _watchlist_instance = WatchlistSystem()
    return _watchlist_instance
