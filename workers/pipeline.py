# -*- coding: utf-8 -*-
"""
workers/pipeline.py
Main inference pipeline for one camera.
Ties together: stream → detect → track → intelligence → alerts.
Designed to run in a thread; results stored in a shared state dict.
"""

import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import FRAME_SKIP, DISPLAY_WIDTH
from detection.yoloworld_engine import YOLOWorldEngine, Detection
from tracking.bytetrack_wrapper import ByteTracker
from intelligence.helmet_rule import check_helmet_violations, draw_violations
from intelligence.congestion_monitor import CongestionMonitor
from intelligence.speed_estimator import SpeedEstimator
from alerts.alert_dispatcher import Alert, get_dispatcher
from intelligence.plate_ocr import prepare_plate_crops, cached_plate_map
from intelligence.async_ocr import AsyncOCRWorker, OCRJob
from intelligence.wrong_lane import WrongLaneDetector
from intelligence.watchlist import get_watchlist
from intelligence.night_enhance import auto_enhance


def _resize(frame: np.ndarray, width: int = DISPLAY_WIDTH) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)), interpolation=cv2.INTER_LINEAR)


class CameraPipeline:
    """
    Full inference pipeline for one camera stream.

    shared_state dict is updated each frame so the Streamlit UI
    can read it without blocking inference.
    """

    def __init__(
        self,
        camera_id:    str,
        camera_name:  str,
        source,                         # int / str / RTSP URL
        roi_polygon=None,
        line_y_a:     int = 300,
        line_y_b:     int = 450,
        speed_limit:  float = 60.0,
        allowed_direction_deg: float = 180.0,   # wrong-lane: expected flow
        enable_wrong_lane: bool = True,
        enable_night_enhance: bool = True,
        enable_watchlist: bool = True,
    ):
        self.camera_id   = camera_id
        self.camera_name = camera_name
        self.source      = source
        self.enable_wrong_lane    = enable_wrong_lane
        self.enable_night_enhance = enable_night_enhance
        self.enable_watchlist     = enable_watchlist

        # Modules
        self.engine   = YOLOWorldEngine.get_instance()
        self.tracker  = ByteTracker()
        self.congestion = CongestionMonitor(camera_id=camera_id,
                                            roi_polygon=roi_polygon)
        self.speed    = SpeedEstimator(line_y_a=line_y_a, line_y_b=line_y_b,
                                       speed_limit_kmh=speed_limit)
        self.wrong_lane = WrongLaneDetector(allowed_direction_deg=allowed_direction_deg)
        self.watchlist  = get_watchlist()
        self._notified_watch = set()   # track_ids already alerted for watchlist
        from intelligence.helmet_rule import HelmetConfirmer
        self.helmet_confirmer = HelmetConfirmer(frames_to_confirm=3)
        self.alerts   = get_dispatcher()

        # B4: async OCR worker — writes plate reads to DB via callback
        def _on_plate(track_id, plate, conf, job: OCRJob):
            self.alerts.log_plate_read(
                camera_id  = job.camera_id,
                track_id   = track_id,
                plate      = plate,
                confidence = conf,
                location   = job.location,
            )
            # Watchlist check — fire alert if plate is flagged (stolen/wanted)
            if self.enable_watchlist and track_id not in self._notified_watch:
                reason = self.watchlist.check(plate)
                if reason:
                    self._notified_watch.add(track_id)
                    self.alerts.dispatch(Alert(
                        camera_id      = job.camera_id,
                        violation_type = f"WATCHLIST_{reason}",
                        vehicle_number = plate,
                        confidence     = conf,
                        track_id       = track_id,
                        location       = job.location,
                    ), snapshot_frame=None)
        self.ocr_worker = AsyncOCRWorker(on_result=_on_plate)

        # Shared state (thread-safe via GIL for simple dicts)
        self.state: Dict[str, Any] = {
            "frame_rgb":       None,
            "frame_num":       0,
            "fps":             0.0,
            "vehicle_count":   0,
            "congestion":      "LOW",
            "helmet_violations": 0,
            "speed_violations":  0,
            "wrong_lane_violations": 0,
            "watchlist_hits":    0,
            "total_detections":  0,
            "running":         False,
            "error":           None,
            "traceback":       None,
        }
        self._stop_event = threading.Event()

    def start_thread(self):
        """Run pipeline in background thread."""
        t = threading.Thread(target=self.run, daemon=True,
                             name=f"pipeline-{self.camera_id}")
        t.start()
        return t

    def stop(self):
        self._stop_event.set()

    def run(self):
        """Main loop — blocking. Call start_thread() for async use."""
        from ingest.stream_reader import StreamReader, MockStreamReader

        # Use MockStreamReader for file paths, StreamReader for webcam/RTSP
        source_str = str(self.source)
        if os.path.isfile(source_str):
            reader = MockStreamReader(source_str, name=self.camera_id)
        else:
            reader = StreamReader(self.source, name=self.camera_id)
        reader.start()
        self.ocr_worker.start()   # B4: start async OCR thread

        # Sync speed estimator with real source FPS (fix A1)
        try:
            self.speed.set_fps(reader.fps if reader.fps else 25.0)
        except Exception:
            pass

        self.state["running"] = True
        frame_idx  = 0
        t_start    = time.time()

        try:
            while not self._stop_event.is_set():
                frame = reader.read_blocking(timeout=2.0)
                if frame is None:
                    continue

                frame_idx += 1
                if frame_idx % FRAME_SKIP != 0:
                    continue

                # ── Night enhancement (auto, only on dark frames) ──────────
                if self.enable_night_enhance:
                    frame = auto_enhance(frame)

                # ── Detection ──────────────────────────────────────────────
                detections = self.engine.predict(frame)

                # ── Tracking ───────────────────────────────────────────────
                detections = self.tracker.update(detections)

                # ── License Plate OCR (ASYNC — never blocks feed, B4) ──────
                # Submit full-res crops to background OCR worker (B6)
                for tid, cname, crop, vbbox in prepare_plate_crops(detections, frame):
                    self.ocr_worker.submit(OCRJob(
                        track_id   = tid,
                        class_name = cname,
                        crop       = crop,
                        camera_id  = self.camera_id,
                        location   = self.camera_name,
                    ))
                # Read whatever plates have been resolved so far (from cache)
                plate_map = cached_plate_map()

                # ── Helmet violations (multi-frame confirmed) ──────────────
                helmet_viols = check_helmet_violations(detections, frame, frame_idx)
                # Map candidate track_ids → violation object
                viol_by_id = {v.track_id: v for v in helmet_viols}
                confirmed_ids = self.helmet_confirmer.update(set(viol_by_id.keys()))
                for tid in confirmed_ids:
                    v = viol_by_id[tid]
                    alert = Alert(
                        camera_id      = self.camera_id,
                        violation_type = "HELMET",
                        track_id       = v.track_id,
                        vehicle_number = plate_map.get(v.track_id, ""),
                        location       = self.camera_name,
                    )
                    snap = draw_violations(frame.copy(), [v], in_place=True)
                    self.alerts.dispatch(alert, snapshot_frame=snap)

                # ── Speed estimation ───────────────────────────────────────
                speed_recs = self.speed.update(detections, frame_idx)
                for rec in speed_recs:
                    if rec.is_violation:
                        alert = Alert(
                            camera_id      = self.camera_id,
                            violation_type = "SPEED",
                            track_id       = rec.track_id,
                            speed_kmh      = rec.speed_kmh,
                            vehicle_number = plate_map.get(rec.track_id, ""),
                            location       = self.camera_name,
                        )
                        self.alerts.dispatch(alert, snapshot_frame=frame.copy())

                # ── Wrong-lane / wrong-direction detection ─────────────────
                wrong_lane_viols = []
                if self.enable_wrong_lane:
                    wrong_lane_viols = self.wrong_lane.check(self.tracker)
                    for wl in wrong_lane_viols:
                        alert = Alert(
                            camera_id      = self.camera_id,
                            violation_type = "WRONG_LANE",
                            track_id       = wl.track_id,
                            vehicle_number = plate_map.get(wl.track_id, ""),
                            location       = self.camera_name,
                        )
                        self.alerts.dispatch(alert, snapshot_frame=frame.copy())

                # ── Congestion ─────────────────────────────────────────────
                cong_state = self.congestion.update(detections)
                # Log to DB every ~30 frames
                if frame_idx % 30 == 0:
                    self.alerts.log_traffic_stat(
                        self.camera_id,
                        cong_state.vehicle_count,
                        cong_state.congestion_level
                    )
                    # Redis publish stats (Task 3.12)
                    try:
                        from workers.redis_broker import publish_event
                        publish_event("traffic:stats", {
                            "camera_id": self.camera_id,
                            "vehicle_count": cong_state.vehicle_count,
                            "congestion": cong_state.congestion_level,
                        })
                    except Exception:
                        pass

                # ── Annotate frame (single copy, rest in-place — fix A13) ──
                annotated = self.engine.annotate(frame, detections)
                draw_violations(annotated, helmet_viols, in_place=True)
                self.congestion.draw_roi(annotated, cong_state, in_place=True)
                self.speed.draw_lines(annotated, speed_recs)

                # Draw wrong-lane markers
                for wl in wrong_lane_viols:
                    cx, cy = wl.center
                    cv2.putText(annotated, f"WRONG WAY #{wl.track_id}",
                                (cx - 60, cy - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.circle(annotated, (cx, cy), 8, (0, 0, 255), -1)

                # Draw plate text on vehicles (from cache, filled by async OCR)
                for d in detections:
                    if d.track_id in plate_map and d.class_name in self.engine.classes:
                        plate_txt = plate_map[d.track_id]
                        vx1, vy2 = d.x1, d.y2
                        label = f"[{plate_txt}]"
                        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(annotated, (vx1, vy2 - th - 8), (vx1 + tw + 6, vy2), (0, 180, 255), -1)
                        cv2.putText(annotated, label, (vx1 + 3, vy2 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

                # FPS overlay
                elapsed = time.time() - t_start
                fps     = frame_idx / max(elapsed, 0.001) * (1 / FRAME_SKIP)
                cv2.putText(annotated, f"FPS:{fps:.1f}  Frame:{frame_idx}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (200, 200, 200), 2, cv2.LINE_AA)
                cv2.putText(annotated, self.camera_name,
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255), 2, cv2.LINE_AA)

                # ── Update shared state ────────────────────────────────────
                display = _resize(annotated)
                rgb     = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                counts  = self.alerts.violation_counts()

                self.state.update({
                    "frame_rgb":         rgb,
                    "frame_num":         frame_idx,
                    "fps":               round(fps, 1),
                    "vehicle_count":     cong_state.vehicle_count,
                    "congestion":        cong_state.congestion_level,
                    "helmet_violations": counts.get("HELMET", 0),
                    "speed_violations":  counts.get("SPEED", 0),
                    "wrong_lane_violations": counts.get("WRONG_LANE", 0),
                    "watchlist_hits":    sum(v for k, v in counts.items()
                                             if k.startswith("WATCHLIST")),
                    "plates_read":       counts.get("PLATE_READ", 0),
                    "total_detections":  len(detections),
                    "running":           True,
                    "error":             None,
                })

        except Exception as e:
            import traceback
            self.state["error"]   = f"{e}"
            self.state["traceback"] = traceback.format_exc()
            self.state["running"] = False
            try:
                reader.stop()
                self.ocr_worker.stop()
            except Exception:
                pass
            return   # don't re-raise in daemon thread (fix A12)

        reader.stop()
        self.ocr_worker.stop()
        self.state["running"] = False
