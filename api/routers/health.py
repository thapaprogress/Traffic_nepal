# -*- coding: utf-8 -*-
"""api/routers/health.py — Health check and metrics endpoints (#7)."""

import time
import os
import sys
from fastapi import APIRouter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

router = APIRouter(tags=["health"])

_START_TIME = time.time()


@router.get("/health")
def health():
    """Liveness probe — always fast, no DB hit."""
    return {"status": "ok", "uptime_seconds": round(time.time() - _START_TIME, 1)}


@router.get("/metrics")
def metrics():
    """
    System metrics for monitoring:
      - DB row counts + size
      - GPU availability
      - violation counts
    """
    data = {"uptime_seconds": round(time.time() - _START_TIME, 1)}

    # DB stats
    try:
        from alerts.alert_dispatcher import get_dispatcher
        disp = get_dispatcher()
        data["database"] = disp.db_stats()
        data["violation_counts"] = disp.violation_counts(cache_seconds=5.0)
    except Exception as e:
        data["database_error"] = str(e)

    # GPU info
    try:
        import torch
        data["gpu"] = {
            "cuda_available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else "cpu"),
        }
    except Exception:
        data["gpu"] = {"cuda_available": False, "device_name": "unknown"}

    return data


@router.post("/maintenance/cleanup")
def run_cleanup(retention_days: float = 7.0):
    """Trigger data retention cleanup (delete old rows + orphan snapshots)."""
    try:
        from alerts.alert_dispatcher import get_dispatcher
        summary = get_dispatcher().cleanup(retention_days=retention_days)
        return {"status": "ok", "removed": summary}
    except Exception as e:
        return {"status": "error", "detail": str(e)}
