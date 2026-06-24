# -*- coding: utf-8 -*-
"""
intelligence/plate_ocr.py
Two-Stage Vehicle-Linked License Plate Recognition.

Stage 1: For each detected vehicle, check if a "license plate" bbox overlaps it.
Stage 2: If no plate bbox found, crop the lower portion of vehicle and run direct OCR.

Result: Every vehicle gets a plate reading attempt, cached by track_id.
"""

import cv2
import numpy as np
import re
import time
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.yoloworld_engine import Detection
from config.settings import VEHICLE_CLASSES

# ─── Config ───────────────────────────────────────────────────────────────────
PLATE_MIN_VEHICLE_WIDTH = 80    # skip vehicles narrower than this (too far away)
PLATE_MAX_OCR_PER_FRAME = 5    # limit OCR calls per frame for performance
PLATE_CACHE_TTL_SECONDS = 30   # how long to cache a plate reading per track_id
PLATE_VEHICLE_CROP_RATIO = 0.4 # crop bottom 40% of vehicle bbox for plate
PLATE_VOTE_READS = 5           # max OCR reads per vehicle before locking result
PLATE_MIN_CONFIDENCE = 0.30    # ignore OCR reads below this confidence

# ─── Lazy OCR loader ──────────────────────────────────────────────────────────
_ocr_reader = None


def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            # B1: use GPU for OCR if available
            try:
                import torch
                use_gpu = torch.cuda.is_available()
            except Exception:
                use_gpu = False
            _ocr_reader = easyocr.Reader(
                ["en"], gpu=use_gpu, verbose=False
            )
        except ImportError:
            raise ImportError("easyocr not installed. Run: pip install easyocr")
    return _ocr_reader


# ─── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class PlateResult:
    track_id:     int
    class_name:   str           # vehicle class (car, motorcycle, etc.)
    plate_text:   str
    confidence:   float
    plate_bbox:   Tuple[int, int, int, int]
    vehicle_bbox: Tuple[int, int, int, int]
    method:       str           # "detected" or "crop_ocr"
    plate_crop:   Optional[np.ndarray] = None


# ─── Plate Cache with Multi-Read Voting ───────────────────────────────────────
class PlateCache:
    """
    Accumulates multiple OCR reads per track_id and keeps the most-voted
    (confidence-weighted) plate text. This is far more accurate than trusting
    the first noisy read — the same plate is read across several frames and
    the consensus result wins.
    """

    def __init__(self, ttl: float = PLATE_CACHE_TTL_SECONDS,
                 max_reads: int = PLATE_VOTE_READS):
        # track_id → dict(votes={text: total_conf}, counts={text: n}, reads, locked, ts, best, best_conf)
        self._data: Dict[int, dict] = {}
        self._ttl = ttl
        self._max_reads = max_reads

    def _best(self, entry: dict) -> Tuple[str, float]:
        votes = entry["votes"]
        if not votes:
            return ("", 0.0)
        best_text = max(votes, key=votes.get)
        # average confidence for that text
        avg_conf = votes[best_text] / max(entry["counts"][best_text], 1)
        return (best_text, round(avg_conf, 2))

    def add_read(self, track_id: int, text: str, confidence: float):
        """Register one OCR read (a 'vote') for a vehicle."""
        if confidence < PLATE_MIN_CONFIDENCE or not text:
            return
        # Bound cache size
        if len(self._data) > 1000:
            oldest = min(self._data.items(), key=lambda kv: kv[1]["ts"])[0]
            del self._data[oldest]
        entry = self._data.get(track_id)
        if entry is None:
            entry = {"votes": {}, "counts": {}, "reads": 0,
                     "locked": False, "ts": time.time()}
            self._data[track_id] = entry
        entry["votes"][text] = entry["votes"].get(text, 0.0) + confidence
        entry["counts"][text] = entry["counts"].get(text, 0) + 1
        entry["reads"] += 1
        entry["ts"] = time.time()
        if entry["reads"] >= self._max_reads:
            entry["locked"] = True

    def get(self, track_id: int) -> Optional[str]:
        """Return current best plate text (or None if no valid reads / expired)."""
        entry = self._data.get(track_id)
        if entry is None:
            return None
        if time.time() - entry["ts"] > self._ttl:
            del self._data[track_id]
            return None
        text, _ = self._best(entry)
        return text or None

    def needs_more_reads(self, track_id: int) -> bool:
        """True if this vehicle should still be OCR'd (not locked yet)."""
        entry = self._data.get(track_id)
        if entry is None:
            return True
        return not entry["locked"]

    # Back-compat shim used by AsyncOCRWorker
    def put(self, track_id: int, text: str, confidence: float):
        self.add_read(track_id, text, confidence)

    def get_all(self) -> Dict[int, str]:
        """Return {track_id: best_plate_text} for all active entries."""
        now = time.time()
        active, stale = {}, []
        for tid, entry in self._data.items():
            if now - entry["ts"] < self._ttl:
                text, _ = self._best(entry)
                if text:
                    active[tid] = text
            else:
                stale.append(tid)
        for tid in stale:
            del self._data[tid]
        return active


# Global cache instance
_plate_cache = PlateCache()


def get_plate_cache() -> PlateCache:
    return _plate_cache


# ─── Preprocessing ────────────────────────────────────────────────────────────
def preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """Enhance plate crop for OCR: grayscale → CLAHE → upscale → denoise."""
    if crop is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    if w < 150:
        scale = 150 / w
        crop = cv2.resize(crop, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    return gray


def normalize_plate_text(text: str) -> str:
    """
    Clean OCR output without corrupting valid letters (fix A9).
    Does NOT blindly replace O/I/S — only strips noise and uppercases.
    Positional digit/letter correction is applied per-segment.
    """
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Char confusion maps for positional correction
_TO_DIGIT = {"O": "0", "I": "1", "Z": "2", "B": "8", "S": "5", "G": "6", "Q": "0"}
_TO_LETTER = {"0": "O", "1": "I", "8": "B", "5": "S", "6": "G", "2": "Z"}


def correct_segment(seg: str, expect: str) -> str:
    """
    Correct a plate segment given expected type.
    expect = 'digit' → convert letters to likely digits
    expect = 'letter' → convert digits to likely letters
    """
    out = []
    for ch in seg:
        if expect == "digit" and ch in _TO_DIGIT:
            out.append(_TO_DIGIT[ch])
        elif expect == "letter" and ch in _TO_LETTER:
            out.append(_TO_LETTER[ch])
        else:
            out.append(ch)
    return "".join(out)


def _run_ocr_on_crop(crop: np.ndarray) -> Tuple[str, float]:
    """Run EasyOCR on a preprocessed crop. Returns (text, confidence)."""
    reader = _get_ocr()
    processed = preprocess_plate(crop)
    try:
        results = reader.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
            paragraph=True,
        )
    except Exception:
        return ("", 0.0)
    if not results:
        return ("", 0.0)
    raw = " ".join(r[1] for r in results)
    text = normalize_plate_text(raw)
    conf = sum(r[2] for r in results) / len(results)
    return (text, conf)


# ─── Crop preparation for ASYNC OCR (B4 + B6) ─────────────────────────────────
def prepare_plate_crops(detections, frame):
    """
    For each vehicle, prepare a plate crop from the FULL-RESOLUTION frame (B6)
    WITHOUT running OCR. Returns list of (track_id, class_name, crop, vehicle_bbox).
    OCR is run asynchronously by AsyncOCRWorker (B4).

    Returns only vehicles that:
      - are large enough (readable)
      - are not already in the plate cache
    """
    h, w = frame.shape[:2]
    cache = get_plate_cache()

    vehicles = [d for d in detections
                if d.class_name in VEHICLE_CLASSES and d.track_id >= 0]
    plates   = [d for d in detections if d.class_name == "license plate"]

    jobs = []
    for veh in vehicles:
        if (veh.x2 - veh.x1) < PLATE_MIN_VEHICLE_WIDTH:
            continue
        if not cache.needs_more_reads(veh.track_id):   # voting locked — skip
            continue

        # Stage 1: overlapping plate detection
        crop = None
        bbox = veh.bbox
        for p in plates:
            px, py = p.center
            if veh.x1 <= px <= veh.x2 and veh.y1 <= py <= veh.y2:
                px1 = max(0, p.x1); py1 = max(0, p.y1)
                px2 = min(w, p.x2); py2 = min(h, p.y2)
                if (px2 - px1) > 20 and (py2 - py1) > 10:
                    crop = frame[py1:py2, px1:px2].copy()
                    bbox = (px1, py1, px2, py2)
                break

        # Stage 2: crop lower portion of vehicle from FULL-RES frame
        if crop is None:
            vx1 = max(0, veh.x1); vy1 = max(0, veh.y1)
            vx2 = min(w, veh.x2); vy2 = min(h, veh.y2)
            veh_h = vy2 - vy1
            if veh.class_name == "motorcycle":
                cy1 = vy1
            else:
                cy1 = vy1 + int(veh_h * (1.0 - PLATE_VEHICLE_CROP_RATIO))
            crop = frame[cy1:vy2, vx1:vx2].copy()
            bbox = (vx1, cy1, vx2, vy2)

        if crop is not None and crop.size > 0:
            jobs.append((veh.track_id, veh.class_name, crop, bbox))

    return jobs


def cached_plate_map():
    """Return {track_id: plate_text} from the cache for drawing/association."""
    return get_plate_cache().get_all()


# ─── Main Function: Vehicle-Linked Plate Extraction ───────────────────────────
def extract_plates_from_vehicles(
    detections: List[Detection],
    frame: np.ndarray,
) -> List[PlateResult]:
    """
    Two-stage plate extraction linked to each vehicle.

    For every vehicle detection:
      1. Check if a "license plate" detection overlaps it → use that crop
      2. If none found → crop lower 40% of vehicle bbox → direct OCR
      3. Cache result by track_id (don't re-OCR same vehicle every frame)

    Returns PlateResult for each vehicle where a plate was successfully read.
    """
    h, w = frame.shape[:2]
    cache = get_plate_cache()

    vehicles = [d for d in detections
                if d.class_name in VEHICLE_CLASSES and d.track_id >= 0]
    plates   = [d for d in detections if d.class_name == "license plate"]

    results: List[PlateResult] = []
    ocr_calls_this_frame = 0

    for veh in vehicles:
        # Skip vehicles too small (too far from camera)
        veh_width = veh.x2 - veh.x1
        if veh_width < PLATE_MIN_VEHICLE_WIDTH:
            continue

        # Check cache first
        cached = cache.get(veh.track_id)
        if cached:
            results.append(PlateResult(
                track_id    = veh.track_id,
                class_name  = veh.class_name,
                plate_text  = cached,
                confidence  = 1.0,
                plate_bbox  = (0, 0, 0, 0),
                vehicle_bbox = veh.bbox,
                method      = "cached",
            ))
            continue

        # Rate limit OCR calls per frame
        if ocr_calls_this_frame >= PLATE_MAX_OCR_PER_FRAME:
            continue

        # ── Stage 1: Find overlapping plate detection ──────────────────
        plate_crop = None
        plate_bbox = (0, 0, 0, 0)
        method = "crop_ocr"

        for p in plates:
            # Check if plate center is inside vehicle bbox
            px, py = p.center
            if veh.x1 <= px <= veh.x2 and veh.y1 <= py <= veh.y2:
                # Use this plate's bbox as crop
                px1 = max(0, p.x1); py1 = max(0, p.y1)
                px2 = min(w, p.x2); py2 = min(h, p.y2)
                if (px2 - px1) > 20 and (py2 - py1) > 10:
                    plate_crop = frame[py1:py2, px1:px2].copy()
                    plate_bbox = (px1, py1, px2, py2)
                    method = "detected"
                    break

        # ── Stage 2: No plate detected → crop lower portion of vehicle ─
        if plate_crop is None:
            vx1 = max(0, veh.x1); vy1 = max(0, veh.y1)
            vx2 = min(w, veh.x2); vy2 = min(h, veh.y2)
            veh_h = vy2 - vy1

            if veh.class_name == "motorcycle":
                # For motorcycles, plate can be anywhere — use full bbox
                crop_y1 = vy1
            else:
                # For cars/trucks/buses — plate is in bottom portion
                crop_y1 = vy1 + int(veh_h * (1.0 - PLATE_VEHICLE_CROP_RATIO))

            plate_crop = frame[crop_y1:vy2, vx1:vx2].copy()
            plate_bbox = (vx1, crop_y1, vx2, vy2)

        # ── Run OCR ────────────────────────────────────────────────────
        if plate_crop is not None and plate_crop.size > 0:
            ocr_calls_this_frame += 1
            text, conf = _run_ocr_on_crop(plate_crop)

            if text and len(text) >= 4:
                cache.put(veh.track_id, text, conf)
                results.append(PlateResult(
                    track_id    = veh.track_id,
                    class_name  = veh.class_name,
                    plate_text  = text,
                    confidence  = round(conf, 2),
                    plate_bbox  = plate_bbox,
                    vehicle_bbox = veh.bbox,
                    method      = method,
                    plate_crop  = plate_crop,
                ))

    return results


# ─── Legacy wrapper (backward compat) ─────────────────────────────────────────
def extract_plates(
    detections: List[Detection],
    frame: np.ndarray,
    min_plate_width: int = 30,
) -> List[PlateResult]:
    """Backward-compatible wrapper. Now uses vehicle-linked extraction."""
    return extract_plates_from_vehicles(detections, frame)
