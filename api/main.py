# -*- coding: utf-8 -*-
"""
api/main.py
FastAPI backend for Traffic Eye Nepal.
Run: uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from api.database import init_db
from api.routers.cameras import router as cameras_router
from api.routers.violations import router as violations_router
from api.routers.stats import router as stats_router
from api.routers.stream import router as stream_router
from api.routers.reports import router as reports_router
from api.routers.auth import router as auth_router

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Traffic Eye Nepal API",
    version="2.0.0",
    description="AI-Powered Traffic Intelligence System",
)

# ─── CORS (allow React frontend) ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static files for snapshot images ─────────────────────────────────────────
SNAP_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)
app.mount("/snapshots", StaticFiles(directory=SNAP_DIR), name="snapshots")

# ─── Routers ──────────────────────────────────────────────────────────────────
app.include_router(cameras_router)
app.include_router(violations_router)
app.include_router(stats_router)
app.include_router(stream_router)
app.include_router(reports_router)
app.include_router(auth_router)

# ─── Events ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {
        "name": "Traffic Eye Nepal API",
        "version": "2.0.0",
        "endpoints": {
            "cameras":    "/cameras",
            "violations": "/violations",
            "stats":      "/stats/summary",
            "timeline":   "/stats/timeline",
            "snapshots":  "/snapshots/{filename}",
            "docs":       "/docs",
        }
    }
