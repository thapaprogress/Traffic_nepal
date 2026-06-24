# -*- coding: utf-8 -*-
"""
api/database.py
SQLAlchemy models + session for Traffic Eye.
Uses SQLite for Phase 2 MVP, swap connection string for PostgreSQL.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float,
    Boolean, DateTime, func, UniqueConstraint
)
from sqlalchemy.orm import sessionmaker, declarative_base
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import DB_PATH

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


class Camera(Base):
    __tablename__ = "cameras_api"
    id        = Column(Integer, primary_key=True, index=True)
    camera_id = Column(String(50), unique=True, index=True)
    name      = Column(String(100))
    rtsp_url  = Column(String(500))
    location  = Column(String(200))
    latitude  = Column(Float, default=0.0)
    longitude = Column(Float, default=0.0)
    active    = Column(Boolean, default=True)


class Violation(Base):
    __tablename__ = "violations"
    id             = Column(Integer, primary_key=True, index=True)
    camera_id      = Column(String(50), index=True)
    vehicle_number = Column(String(50), default="")
    violation_type = Column(String(50), index=True)
    speed_kmh      = Column(Float, default=0.0)
    confidence     = Column(Float, default=0.0)
    location       = Column(String(200), default="")
    image_path     = Column(String(500), default="")
    track_id       = Column(Integer, default=-1)
    detected_at    = Column(Float, default=0.0)


class TrafficStat(Base):
    __tablename__ = "traffic_stats"
    id               = Column(Integer, primary_key=True, index=True)
    camera_id        = Column(String(50), index=True)
    vehicle_count    = Column(Integer, default=0)
    congestion_level = Column(String(10), default="LOW")
    recorded_at      = Column(Float, default=0.0)


class PlateRead(Base):
    __tablename__ = "plate_reads"
    id             = Column(Integer, primary_key=True, index=True)
    camera_id      = Column(String(50), index=True)
    track_id       = Column(Integer, index=True)
    vehicle_number = Column(String(50), default="")
    confidence     = Column(Float, default=0.0)
    location       = Column(String(200), default="")
    detected_at    = Column(Float, default=0.0)

    __table_args__ = (
        UniqueConstraint("camera_id", "track_id", "vehicle_number", name="uq_camera_track_plate"),
    )


class WatchlistEntry(Base):
    __tablename__ = "watchlist"
    id       = Column(Integer, primary_key=True, index=True)
    plate    = Column(String(50), unique=True, index=True)
    reason   = Column(String(50), default="STOLEN")
    added_at = Column(Float, default=0.0)
    notes    = Column(String(500), default="")



def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
