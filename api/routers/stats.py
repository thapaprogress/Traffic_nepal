# -*- coding: utf-8 -*-
"""api/routers/stats.py — Statistics and analytics endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from api.database import get_db, Violation, TrafficStat, Camera
from api.models import StatsSummary, TrafficStatResponse

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("/summary", response_model=StatsSummary)
def get_summary(db: Session = Depends(get_db)):
    total = db.query(Violation).count()
    helmet = db.query(Violation).filter(Violation.violation_type == "HELMET").count()
    speed = db.query(Violation).filter(Violation.violation_type == "SPEED").count()
    wrong_lane = db.query(Violation).filter(Violation.violation_type == "WRONG_LANE").count()
    total_cams = db.query(Camera).count()
    active_cams = db.query(Camera).filter(Camera.active == True).count()
    return StatsSummary(
        total_violations=total,
        helmet_count=helmet,
        speed_count=speed,
        wrong_lane_count=wrong_lane,
        total_cameras=total_cams,
        active_cameras=active_cams,
    )


@router.get("/timeline", response_model=List[TrafficStatResponse])
def get_timeline(
    camera_id: str = None,
    limit: int = Query(200, le=2000),
    db: Session = Depends(get_db),
):
    q = db.query(TrafficStat)
    if camera_id:
        q = q.filter(TrafficStat.camera_id == camera_id)
    return q.order_by(TrafficStat.recorded_at.desc()).limit(limit).all()
