# -*- coding: utf-8 -*-
"""
intelligence/async_ocr.py
Background OCR worker (B4).

The detection pipeline submits (track_id, vehicle_class, crop) jobs to a queue
and immediately continues — OCR runs on a separate thread so the live feed
never freezes. Completed reads are written into the shared plate cache and
optionally a callback fires (e.g. to log to DB).
"""

import threading
import queue
import time
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from intelligence.plate_ocr import (
    _run_ocr_on_crop, get_plate_cache,
)


@dataclass
class OCRJob:
    track_id:    int
    class_name:  str
    crop:        np.ndarray
    camera_id:   str
    location:    str


class AsyncOCRWorker:
    """
    Single background thread that drains an OCR job queue.
    Drops jobs when the queue is full so the pipeline never blocks.
    """

    def __init__(self, on_result: Optional[Callable] = None,
                 max_queue: int = 32):
        self._queue: "queue.Queue[OCRJob]" = queue.Queue(maxsize=max_queue)
        self._on_result = on_result    # callback(track_id, plate, conf, job)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._inflight: set = set()    # track_ids currently queued/processing
        self._lock = threading.Lock()

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="async-ocr")
        self._thread.start()

    def stop(self):
        self._running = False

    def submit(self, job: OCRJob) -> bool:
        """
        Submit a job. Returns False if dropped (queue full or already in-flight).
        De-dups by track_id — won't queue the same vehicle twice.
        """
        cache = get_plate_cache()
        if cache.get(job.track_id):          # already read
            return False
        with self._lock:
            if job.track_id in self._inflight:
                return False
            try:
                self._queue.put_nowait(job)
                self._inflight.add(job.track_id)
                return True
            except queue.Full:
                return False

    def _loop(self):
        while self._running:
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                text, conf = _run_ocr_on_crop(job.crop)
                if text and len(text) >= 4:
                    get_plate_cache().put(job.track_id, text, conf)
                    if self._on_result:
                        try:
                            self._on_result(job.track_id, text, conf, job)
                        except Exception as e:
                            print(f"[AsyncOCR] callback error: {e}")
            except Exception as e:
                print(f"[AsyncOCR] OCR error: {e}")
            finally:
                with self._lock:
                    self._inflight.discard(job.track_id)

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()
