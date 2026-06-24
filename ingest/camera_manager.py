# -*- coding: utf-8 -*-
"""
ingest/camera_manager.py
Multi-camera pipeline manager.
Manages N CameraPipeline instances in background threads.
"""

import threading
from typing import Dict, List, Any, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from workers.pipeline import CameraPipeline


class CameraManager:
    """
    Thread-safe manager for multiple camera inference pipelines.

    Usage:
        mgr = CameraManager()
        mgr.add_camera("cam_01", "Kalanki", "rtsp://...")
        mgr.start_all()
        states = mgr.get_all_states()
        mgr.stop_all()
    """

    def __init__(self):
        self._cameras: Dict[str, CameraPipeline] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._lock = threading.Lock()

    def add_camera(
        self,
        camera_id: str,
        camera_name: str,
        source,
        roi_polygon=None,
        line_y_a: int = 300,
        line_y_b: int = 450,
        speed_limit: float = 60.0,
    ) -> bool:
        """Register a new camera pipeline."""
        with self._lock:
            if camera_id in self._cameras:
                return False
            pipe = CameraPipeline(
                camera_id=camera_id,
                camera_name=camera_name,
                source=source,
                roi_polygon=roi_polygon,
                line_y_a=line_y_a,
                line_y_b=line_y_b,
                speed_limit=speed_limit,
            )
            self._cameras[camera_id] = pipe
            return True

    def remove_camera(self, camera_id: str) -> bool:
        """Stop and remove a camera."""
        with self._lock:
            if camera_id not in self._cameras:
                return False
            self._cameras[camera_id].stop()
            del self._cameras[camera_id]
            if camera_id in self._threads:
                del self._threads[camera_id]
            return True

    def start_camera(self, camera_id: str) -> bool:
        """Start a single camera pipeline."""
        with self._lock:
            if camera_id not in self._cameras:
                return False
            pipe = self._cameras[camera_id]
            t = pipe.start_thread()
            self._threads[camera_id] = t
            return True

    def start_all(self):
        """Start all registered cameras."""
        with self._lock:
            for cid, pipe in self._cameras.items():
                if cid not in self._threads or not self._threads[cid].is_alive():
                    t = pipe.start_thread()
                    self._threads[cid] = t

    def stop_camera(self, camera_id: str):
        with self._lock:
            if camera_id in self._cameras:
                self._cameras[camera_id].stop()

    def stop_all(self):
        with self._lock:
            for pipe in self._cameras.values():
                pipe.stop()

    def get_state(self, camera_id: str) -> Optional[Dict[str, Any]]:
        pipe = self._cameras.get(camera_id)
        return pipe.state if pipe else None

    def get_all_states(self) -> Dict[str, Dict[str, Any]]:
        return {cid: pipe.state for cid, pipe in self._cameras.items()}

    @property
    def camera_ids(self) -> List[str]:
        return list(self._cameras.keys())

    @property
    def active_count(self) -> int:
        return sum(1 for p in self._cameras.values()
                   if p.state.get("running"))
