# -*- coding: utf-8 -*-
"""
api/models.py
Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


# ─── Camera ───────────────────────────────────────────────────────────────────
class CameraCreate(BaseModel):
    camera_id: str
    name: str
    rtsp_url: str
    location: str = ""
    latitude: float = 0.0
    longitude: float = 0.0


class CameraResponse(BaseModel):
    id: int
    camera_id: str
    name: str
    rtsp_url: str
    location: str
    latitude: float
    longitude: float
    active: bool

    class Config:
        from_attributes = True


# ─── Violation ────────────────────────────────────────────────────────────────
class ViolationResponse(BaseModel):
    id: int
    camera_id: str
    vehicle_number: str
    violation_type: str
    speed_kmh: float
    confidence: float
    location: str
    image_path: str
    track_id: int
    detected_at: float

    class Config:
        from_attributes = True


# ─── Stats ────────────────────────────────────────────────────────────────────
class StatsSummary(BaseModel):
    total_violations: int
    helmet_count: int
    speed_count: int
    wrong_lane_count: int
    total_cameras: int
    active_cameras: int


class TrafficStatResponse(BaseModel):
    id: int
    camera_id: str
    vehicle_count: int
    congestion_level: str
    recorded_at: float

    class Config:
        from_attributes = True
