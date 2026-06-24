# -*- coding: utf-8 -*-
"""
alerts/alert_dispatcher.py
Unified alert dispatcher.
Writes violations to SQLite DB (MVP) and saves annotated snapshots.
Phase 2 will add PostgreSQL + WebSocket push + SMS.
"""

import sqlite3
import os
import cv2
import time
import json
import threading
from dataclasses import dataclass, asdict
from typing import Optional
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH, SNAPSHOT_DIR, ALERT_SAVE_SNAPSHOTS, ALERT_SNAPSHOT_QUALITY


@dataclass
class Alert:
    camera_id:      str
    violation_type: str          # HELMET | SPEED | CONGESTION
    vehicle_number: str = ""
    speed_kmh:      float = 0.0
    confidence:     float = 0.0
    location:       str = ""
    image_path:     str = ""
    track_id:       int = -1
    timestamp:      float = 0.0

    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


class AlertDispatcher:
    """
    Thread-safe alert dispatcher.
    - Saves snapshot image
    - Inserts record into SQLite
    - Maintains an in-memory log for the dashboard
    """

    def __init__(self, max_memory_log: int = 500):
        self._lock       = threading.Lock()
        self._log        = []          # in-memory circular log
        self._max_log    = max_memory_log
        self._db_conn    = None
        self._ensure_db()

    # ─── DB ───────────────────────────────────────────────────────────────────

    def _ensure_db(self):
        self._db_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # WAL mode = better concurrent read/write (fix A6)
        self._db_conn.execute("PRAGMA journal_mode=WAL;")
        self._db_conn.execute("PRAGMA synchronous=NORMAL;")
        cur = self._db_conn.cursor()
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS violations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id       TEXT,
            vehicle_number  TEXT,
            violation_type  TEXT,
            speed_kmh       REAL,
            confidence      REAL,
            location        TEXT,
            image_path      TEXT,
            track_id        INTEGER,
            detected_at     REAL
        );
        CREATE TABLE IF NOT EXISTS traffic_stats (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id        TEXT,
            vehicle_count    INTEGER,
            congestion_level TEXT,
            recorded_at      REAL
        );
        CREATE TABLE IF NOT EXISTS plate_reads (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id       TEXT,
            track_id        INTEGER,
            vehicle_number  TEXT,
            confidence      REAL,
            location        TEXT,
            detected_at     REAL,
            UNIQUE(camera_id, track_id, vehicle_number)
        );
        -- Indexes (fix A8) --
        CREATE INDEX IF NOT EXISTS idx_viol_type ON violations(violation_type);
        CREATE INDEX IF NOT EXISTS idx_viol_time ON violations(detected_at);
        CREATE INDEX IF NOT EXISTS idx_viol_cam  ON violations(camera_id);
        CREATE INDEX IF NOT EXISTS idx_stats_time ON traffic_stats(recorded_at);
        CREATE INDEX IF NOT EXISTS idx_plate_time ON plate_reads(detected_at);
        """)
        self._db_conn.commit()
        # Cached counts (fix A7 — avoid GROUP BY every frame)
        self._counts_cache = {}
        self._counts_cache_ts = 0.0

    def _insert_violation(self, alert: Alert):
        cur = self._db_conn.cursor()
        cur.execute("""
            INSERT INTO violations
            (camera_id, vehicle_number, violation_type, speed_kmh,
             confidence, location, image_path, track_id, detected_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (alert.camera_id, alert.vehicle_number, alert.violation_type,
              alert.speed_kmh, alert.confidence, alert.location,
              alert.image_path, alert.track_id, alert.timestamp))
        self._db_conn.commit()

    def log_traffic_stat(self, camera_id: str, vehicle_count: int, congestion_level: str):
        with self._lock:
            cur = self._db_conn.cursor()
            cur.execute("""
                INSERT INTO traffic_stats (camera_id, vehicle_count, congestion_level, recorded_at)
                VALUES (?,?,?,?)
            """, (camera_id, vehicle_count, congestion_level, time.time()))
            self._db_conn.commit()

    def log_plate_read(self, camera_id: str, track_id: int, plate: str,
                       confidence: float, location: str = "") -> bool:
        """
        Store a plate reading in the dedicated plate_reads table.
        UNIQUE constraint dedups (camera_id, track_id, plate) — fix A7.
        Returns True if a NEW row was inserted.
        """
        with self._lock:
            cur = self._db_conn.cursor()
            try:
                cur.execute("""
                    INSERT OR IGNORE INTO plate_reads
                    (camera_id, track_id, vehicle_number, confidence, location, detected_at)
                    VALUES (?,?,?,?,?,?)
                """, (camera_id, track_id, plate, confidence, location, time.time()))
                self._db_conn.commit()
                return cur.rowcount > 0
            except Exception as e:
                print(f"[AlertDispatcher] plate_read error: {e}")
                return False

    # ─── Public API ───────────────────────────────────────────────────────────

    def dispatch(self, alert: Alert, snapshot_frame=None):
        """
        Save snapshot → write DB → append to in-memory log.
        Thread-safe.
        """
        with self._lock:
            # Save snapshot
            if ALERT_SAVE_SNAPSHOTS and snapshot_frame is not None:
                ts = int(alert.timestamp * 1000)
                fname = f"{alert.violation_type}_{alert.track_id}_{ts}.jpg"
                fpath = os.path.join(SNAPSHOT_DIR, fname)
                cv2.imwrite(fpath, snapshot_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, ALERT_SNAPSHOT_QUALITY])
                alert.image_path = fpath

            # DB write
            try:
                self._insert_violation(alert)
            except Exception as e:
                print(f"[AlertDispatcher] DB write error: {e}")

            # Memory log
            self._log.append(asdict(alert))
            if len(self._log) > self._max_log:
                self._log.pop(0)

    @property
    def recent_alerts(self):
        """Last N alerts as list of dicts (for dashboard)."""
        with self._lock:
            return list(reversed(self._log[-50:]))

    def query_violations(self, limit: int = 100, violation_type: str = None):
        """Query DB for violations."""
        with self._lock:
            cur = self._db_conn.cursor()
            if violation_type:
                cur.execute("SELECT * FROM violations WHERE violation_type=? ORDER BY detected_at DESC LIMIT ?",
                            (violation_type, limit))
            else:
                cur.execute("SELECT * FROM violations ORDER BY detected_at DESC LIMIT ?", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def query_plate_reads(self, limit: int = 200):
        """Query the dedicated plate_reads table."""
        with self._lock:
            cur = self._db_conn.cursor()
            cur.execute("SELECT * FROM plate_reads ORDER BY detected_at DESC LIMIT ?", (limit,))
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def violation_counts(self, cache_seconds: float = 2.0):
        """
        Summary counts per violation type.
        Cached for `cache_seconds` to avoid a GROUP BY scan every frame (fix A7).
        """
        now = time.time()
        if now - self._counts_cache_ts < cache_seconds and self._counts_cache:
            return dict(self._counts_cache)
        with self._lock:
            cur = self._db_conn.cursor()
            cur.execute("SELECT violation_type, COUNT(*) FROM violations GROUP BY violation_type")
            counts = dict(cur.fetchall())
            # include plate reads count
            cur.execute("SELECT COUNT(*) FROM plate_reads")
            counts["PLATE_READ"] = cur.fetchone()[0]
        self._counts_cache = counts
        self._counts_cache_ts = now
        return dict(counts)


# Singleton
_dispatcher_instance = None

def get_dispatcher() -> AlertDispatcher:
    global _dispatcher_instance
    if _dispatcher_instance is None:
        _dispatcher_instance = AlertDispatcher()
    return _dispatcher_instance
