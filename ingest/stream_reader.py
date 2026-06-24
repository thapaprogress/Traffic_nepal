# -*- coding: utf-8 -*-
"""
ingest/stream_reader.py
Thread-safe RTSP / file / webcam stream reader with auto-reconnect.
"""

import cv2
import threading
import time
import queue
import logging
from typing import Optional, Union

logger = logging.getLogger(__name__)


class StreamReader:
    """
    Non-blocking video reader that runs a background thread.
    Supports: webcam index (0), file path, RTSP URL.

    Usage:
        reader = StreamReader("rtsp://admin:pass@192.168.1.10/live")
        reader.start()
        while True:
            frame = reader.read()
            if frame is not None:
                process(frame)
        reader.stop()
    """

    def __init__(
        self,
        source: Union[str, int] = 0,
        buffer_size: int = 4,
        reconnect_delay: float = 3.0,
        name: str = "cam_01",
    ):
        self.source          = int(source) if str(source).isdigit() else source
        self.buffer_size     = buffer_size
        self.reconnect_delay = reconnect_delay
        self.name            = name

        self._cap:    Optional[cv2.VideoCapture] = None
        self._queue:  queue.Queue = queue.Queue(maxsize=buffer_size)
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock    = threading.Lock()

        # Meta
        self.width  = 0
        self.height = 0
        self.fps    = 25.0
        self.frame_count = 0

    # ─── Public API ───────────────────────────────────────────────────────────

    def start(self):
        """Start the background capture thread."""
        self._running = True
        self._connect()
        self._thread = threading.Thread(target=self._capture_loop,
                                        daemon=True, name=f"stream-{self.name}")
        self._thread.start()
        logger.info(f"[StreamReader:{self.name}] started  source={self.source}")
        return self

    def read(self) -> Optional[object]:
        """Get the latest frame (non-blocking). Returns None if no frame yet."""
        try:
            return self._queue.get_nowait()
        except queue.Empty:
            return None

    def read_blocking(self, timeout: float = 2.0):
        """Block until a frame is available or timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self._running = False
        if self._cap:
            self._cap.release()
        logger.info(f"[StreamReader:{self.name}] stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _connect(self) -> bool:
        with self._lock:
            if self._cap:
                self._cap.release()
            self._cap = cv2.VideoCapture(self.source)
            if not self._cap.isOpened():
                logger.warning(f"[StreamReader:{self.name}] failed to open {self.source}")
                return False
            self.width  = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps    = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
            logger.info(f"[StreamReader:{self.name}] connected {self.width}x{self.height} @ {self.fps:.1f}fps")
            return True

    def _capture_loop(self):
        while self._running:
            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    logger.warning(f"[StreamReader:{self.name}] reconnecting in {self.reconnect_delay}s…")
                    time.sleep(self.reconnect_delay)
                    self._connect()
                    continue
                ret, frame = self._cap.read()

            if not ret:
                logger.warning(f"[StreamReader:{self.name}] read failed — reconnecting…")
                time.sleep(self.reconnect_delay)
                self._connect()
                continue

            self.frame_count += 1
            # Drop oldest frame if buffer is full to avoid lag
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)


class MockStreamReader(StreamReader):
    """
    Loops a local video file; useful for development without a real RTSP camera.
    When the file ends it loops back to the start.
    """

    def __init__(self, video_path: str, **kwargs):
        super().__init__(source=video_path, **kwargs)
        self._video_path = video_path

    def _capture_loop(self):
        while self._running:
            with self._lock:
                if self._cap is None or not self._cap.isOpened():
                    self._connect()
                ret, frame = self._cap.read()

            if not ret:
                # Loop the file
                with self._lock:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self.frame_count += 1
            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            self._queue.put(frame)
            time.sleep(1.0 / max(self.fps, 1))
