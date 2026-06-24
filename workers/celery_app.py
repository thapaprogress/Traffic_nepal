# -*- coding: utf-8 -*-
"""
workers/celery_app.py
Celery application for distributed inference tasks.
Allows scaling to multiple cameras across multiple workers.

Usage:
    celery -A workers.celery_app worker --loglevel=info --pool=solo
    
Requires:
    pip install celery redis
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

try:
    from celery import Celery

    app = Celery(
        "traffic_eye",
        broker=REDIS_URL,
        backend=REDIS_URL,
    )

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        timezone="Asia/Kathmandu",
        enable_utc=True,
        task_track_started=True,
        worker_prefetch_multiplier=1,  # one task at a time per worker
        task_acks_late=True,
    )

    @app.task(name="process_frame_batch")
    def process_frame_batch(camera_id: str, frame_indices: list):
        """
        Process a batch of frames for a camera.
        In production, this would pull frames from a Redis queue
        and push results back.
        """
        from detection.yoloworld_engine import YOLOWorldEngine
        engine = YOLOWorldEngine.get_instance()
        # Placeholder — actual frame data would come from Redis/shared memory
        return {
            "camera_id": camera_id,
            "processed": len(frame_indices),
            "status": "complete",
        }

    @app.task(name="run_ocr_on_plate")
    def run_ocr_on_plate(plate_crop_b64: str, camera_id: str, track_id: int):
        """
        Async OCR task for license plate reading.
        Offloads OCR to a worker so inference pipeline isn't blocked.
        """
        import base64
        import numpy as np
        import cv2
        from intelligence.plate_ocr import preprocess_plate, _get_ocr, normalize_plate_text

        # Decode base64 crop
        img_bytes = base64.b64decode(plate_crop_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if crop is None:
            return {"plate": "", "confidence": 0}

        processed = preprocess_plate(crop)
        reader = _get_ocr()
        results = reader.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            paragraph=True,
        )
        raw_text = " ".join(r[1] for r in results) if results else ""
        plate_text = normalize_plate_text(raw_text)
        avg_conf = sum(r[2] for r in results) / len(results) if results else 0

        return {
            "plate": plate_text,
            "confidence": round(avg_conf, 2),
            "camera_id": camera_id,
            "track_id": track_id,
        }

    @app.task(name="check_watchlist")
    def check_watchlist(plate_text: str, camera_id: str, track_id: int):
        """Check plate against stolen vehicle watchlist."""
        from intelligence.watchlist import get_watchlist
        watchlist = get_watchlist()
        reason = watchlist.check(plate_text)
        return {
            "plate": plate_text,
            "match": reason is not None,
            "reason": reason or "",
            "camera_id": camera_id,
            "track_id": track_id,
        }

except ImportError:
    # Celery not installed — provide a dummy for imports
    class DummyCelery:
        def task(self, *a, **kw):
            def decorator(f):
                return f
            return decorator
    app = DummyCelery()
    print("[workers/celery_app] Celery not installed. Tasks run synchronously.")
