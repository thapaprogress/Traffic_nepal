# -*- coding: utf-8 -*-
"""api/routers/cameras.py — Camera CRUD endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from api.database import get_db, Camera
from api.models import CameraCreate, CameraResponse

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("/", response_model=List[CameraResponse])
def list_cameras(db: Session = Depends(get_db)):
    return db.query(Camera).all()


@router.post("/", response_model=CameraResponse)
def add_camera(cam: CameraCreate, db: Session = Depends(get_db)):
    existing = db.query(Camera).filter(Camera.camera_id == cam.camera_id).first()
    if existing:
        raise HTTPException(400, "Camera ID already exists")
    obj = Camera(**cam.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


@router.delete("/{camera_id}")
def remove_camera(camera_id: str, db: Session = Depends(get_db)):
    obj = db.query(Camera).filter(Camera.camera_id == camera_id).first()
    if not obj:
        raise HTTPException(404, "Camera not found")
    db.delete(obj)
    db.commit()
    return {"status": "deleted", "camera_id": camera_id}
