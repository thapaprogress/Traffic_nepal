# -*- coding: utf-8 -*-
"""
config/settings.py
Central configuration for Traffic Eye Nepal.
Edit values here — all modules import from this file.
"""

import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SNAPSHOT_DIR  = os.path.join(BASE_DIR, "snapshots")
WEIGHTS_DIR   = os.path.join(BASE_DIR, "..", "weights")   # shared with YOLO_Projects
DB_PATH       = os.path.join(BASE_DIR, "traffic_eye.db")  # SQLite for MVP

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ─── YOLO-World Model ─────────────────────────────────────────────────────────
YOLO_WEIGHTS  = os.path.join(WEIGHTS_DIR, "yolov8s-worldv2.pt")
YOLO_IMGSZ    = 640
YOLO_CONF     = 0.30
YOLO_IOU      = 0.45
YOLO_MAX_DET  = 200
FRAME_SKIP    = 2          # process every Nth frame (1 = every frame)
DISPLAY_WIDTH = 960        # resize for display output

# ─── Nepal Traffic Detection Prompts ──────────────────────────────────────────
DETECTION_CLASSES = [
    "motorcycle",
    "car",
    "bus",
    "truck",
    "person",
    "helmet",
    "license plate",
    "traffic police",
    "bicycle",
    "auto rickshaw",
]

# Subset used for helmet violation check
HELMET_CLASSES    = {"motorcycle", "person", "helmet"}
VEHICLE_CLASSES   = {"car", "bus", "truck", "motorcycle", "bicycle", "auto rickshaw"}

# ─── Congestion Thresholds ────────────────────────────────────────────────────
CONGESTION_LOW    = 10
CONGESTION_MEDIUM = 30
# > CONGESTION_MEDIUM  → HIGH

# ─── Speed Detection ──────────────────────────────────────────────────────────
# Real-world distance between virtual speed lines (metres)
SPEED_REAL_DISTANCE_M = 20.0
# Speed limit before flagging violation
SPEED_LIMIT_KMH       = 60.0

# ─── Alerts ───────────────────────────────────────────────────────────────────
ALERT_SAVE_SNAPSHOTS  = True
ALERT_SNAPSHOT_QUALITY = 85   # JPEG quality 0-100

# ─── Camera Defaults ──────────────────────────────────────────────────────────
DEFAULT_CAMERA = {
    "id":       "cam_01",
    "name":     "Kalanki Chowk",
    "location": "Kalanki, Kathmandu",
    "rtsp_url": "0",   # "0" = webcam, replace with rtsp://... for real CCTV
}
