# -*- coding: utf-8 -*-
"""
alerts/snapshot_store.py
MinIO/S3 Snapshot storage backend.
Uploads snapshots to MinIO bucket or falls back to local storage if not configured.
"""

import os
import io
import cv2
import numpy as np

MINIO_URL    = os.environ.get("MINIO_URL", "")
MINIO_KEY    = os.environ.get("MINIO_KEY", "")
MINIO_SECRET = os.environ.get("MINIO_SECRET", "")
BUCKET       = "traffic-eye-snapshots"

client = None

if MINIO_URL and MINIO_KEY and MINIO_SECRET:
    try:
        from minio import Minio
        # Support secure connection option if url contains port 443 or ssl
        secure_conn = False
        if MINIO_URL.startswith("https://") or ":443" in MINIO_URL:
            secure_conn = True
        
        # Clean url if it contains http/https protocol prefix
        clean_url = MINIO_URL.replace("https://", "").replace("http://", "")
        
        client = Minio(
            clean_url,
            access_key=MINIO_KEY,
            secret_key=MINIO_SECRET,
            secure=secure_conn
        )
        # Ensure bucket exists
        if not client.bucket_exists(BUCKET):
            client.make_bucket(BUCKET)
    except Exception as e:
        print(f"[SnapshotStore] Failed to initialize MinIO client: {e}")

def upload_snapshot(frame: np.ndarray, filename: str) -> str:
    """
    Uploads snapshot to MinIO object storage.
    Falls back to local disk storage if MinIO is not configured.
    """
    if client is not None:
        try:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            data = io.BytesIO(buf.tobytes())
            client.put_object(
                BUCKET,
                filename,
                data,
                len(buf),
                content_type="image/jpeg"
            )
            # Return URL to the snapshot
            protocol = "https" if ":443" in MINIO_URL else "http"
            return f"{protocol}://{MINIO_URL}/{BUCKET}/{filename}"
        except Exception as e:
            print(f"[SnapshotStore] MinIO upload failed, using local fallback: {e}")
            
    # Fallback to local snapshot save
    from config.settings import SNAPSHOT_DIR, ALERT_SNAPSHOT_QUALITY
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    fpath = os.path.join(SNAPSHOT_DIR, filename)
    cv2.imwrite(fpath, frame, [cv2.IMWRITE_JPEG_QUALITY, ALERT_SNAPSHOT_QUALITY])
    return fpath
