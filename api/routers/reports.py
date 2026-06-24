# -*- coding: utf-8 -*-
"""
api/routers/reports.py
PDF and CSV report generation endpoints.
"""

import io
import csv
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from api.database import get_db, Violation, TrafficStat

router = APIRouter(prefix="/reports", tags=["reports"])


@router.get("/violations/csv")
def export_violations_csv(
    hours: int = Query(24, description="Export last N hours"),
    violation_type: str = None,
    db: Session = Depends(get_db),
):
    """Export violations as CSV for the last N hours."""
    import time
    cutoff = time.time() - (hours * 3600)
    q = db.query(Violation).filter(Violation.detected_at >= cutoff)
    if violation_type:
        q = q.filter(Violation.violation_type == violation_type)
    rows = q.order_by(Violation.detected_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "ID", "Camera", "Type", "Plate", "Speed (km/h)",
        "Location", "Track ID", "Confidence", "Time"
    ])
    for r in rows:
        writer.writerow([
            r.id, r.camera_id, r.violation_type, r.vehicle_number,
            r.speed_kmh, r.location, r.track_id, r.confidence,
            datetime.fromtimestamp(r.detected_at).strftime("%Y-%m-%d %H:%M:%S")
        ])

    output.seek(0)
    filename = f"violations_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/traffic/csv")
def export_traffic_csv(
    hours: int = Query(24),
    camera_id: str = None,
    db: Session = Depends(get_db),
):
    """Export traffic stats as CSV."""
    import time
    cutoff = time.time() - (hours * 3600)
    q = db.query(TrafficStat).filter(TrafficStat.recorded_at >= cutoff)
    if camera_id:
        q = q.filter(TrafficStat.camera_id == camera_id)
    rows = q.order_by(TrafficStat.recorded_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Camera", "Vehicle Count", "Congestion Level", "Time"])
    for r in rows:
        writer.writerow([
            r.camera_id, r.vehicle_count, r.congestion_level,
            datetime.fromtimestamp(r.recorded_at).strftime("%Y-%m-%d %H:%M:%S")
        ])

    output.seek(0)
    filename = f"traffic_stats_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@router.get("/summary")
def get_daily_summary(db: Session = Depends(get_db)):
    """Get a daily summary (JSON) for dashboard or PDF generation."""
    import time
    now = time.time()
    today_start = now - (now % 86400)

    total_today = db.query(Violation).filter(
        Violation.detected_at >= today_start).count()
    helmet_today = db.query(Violation).filter(
        Violation.detected_at >= today_start,
        Violation.violation_type == "HELMET").count()
    speed_today = db.query(Violation).filter(
        Violation.detected_at >= today_start,
        Violation.violation_type == "SPEED").count()

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "total_violations": total_today,
        "helmet_violations": helmet_today,
        "speed_violations": speed_today,
        "generated_at": datetime.now().isoformat(),
    }
