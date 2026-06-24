# -*- coding: utf-8 -*-
"""api/routers/violations.py — Violation query endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from api.database import get_db, Violation
from api.models import ViolationResponse

router = APIRouter(prefix="/violations", tags=["violations"])


@router.get("/", response_model=List[ViolationResponse])
def get_violations(
    violation_type: Optional[str] = None,
    camera_id: Optional[str] = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
    db: Session = Depends(get_db),
):
    q = db.query(Violation)
    if violation_type:
        q = q.filter(Violation.violation_type == violation_type)
    if camera_id:
        q = q.filter(Violation.camera_id == camera_id)
    return q.order_by(Violation.detected_at.desc()).offset(offset).limit(limit).all()


@router.get("/{violation_id}", response_model=ViolationResponse)
def get_violation(violation_id: int, db: Session = Depends(get_db)):
    obj = db.query(Violation).filter(Violation.id == violation_id).first()
    if not obj:
        from fastapi import HTTPException
        raise HTTPException(404, "Violation not found")
    return obj
