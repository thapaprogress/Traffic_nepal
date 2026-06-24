# -*- coding: utf-8 -*-
"""
tests/test_async_ocr.py
Tests for intelligence/async_ocr.py — AsyncOCRWorker.
Validates queue de-duplication, job submission, and result callbacks.
"""
import time
import numpy as np
from intelligence.async_ocr import AsyncOCRWorker, OCRJob


def _make_job(track_id: int, text: str = "test") -> OCRJob:
    """Create a minimal OCRJob with a blank crop."""
    return OCRJob(
        track_id=track_id,
        class_name="motorcycle",
        crop=np.zeros((32, 128, 3), dtype=np.uint8),
        camera_id="cam_test",
        location="Test Location",
    )


def test_worker_starts_and_stops():
    """Worker thread starts and stops cleanly."""
    worker = AsyncOCRWorker()
    worker.start()
    assert worker._running is True
    worker.stop()
    # Give thread time to exit
    time.sleep(0.1)


def test_submit_returns_true_for_new_job(monkeypatch):
    """submit() accepts a new job and returns True."""
    from intelligence.plate_ocr import PlateCache

    # Mock the cache to always say more reads are needed
    mock_cache = PlateCache()
    monkeypatch.setattr(
        "intelligence.async_ocr.get_plate_cache", lambda: mock_cache
    )

    worker = AsyncOCRWorker()
    job = _make_job(track_id=101)
    result = worker.submit(job)
    assert result is True
    assert 101 in worker._inflight


def test_submit_deduplicates_inflight_jobs(monkeypatch):
    """submit() returns False if the same track_id is already in-flight."""
    from intelligence.plate_ocr import PlateCache

    mock_cache = PlateCache()
    monkeypatch.setattr(
        "intelligence.async_ocr.get_plate_cache", lambda: mock_cache
    )

    worker = AsyncOCRWorker()
    job1 = _make_job(track_id=202)
    job2 = _make_job(track_id=202)

    r1 = worker.submit(job1)
    r2 = worker.submit(job2)  # duplicate

    assert r1 is True
    assert r2 is False   # rejected — already inflight


def test_queue_size_reflects_submitted_jobs(monkeypatch):
    """queue_size increases with submitted jobs."""
    from intelligence.plate_ocr import PlateCache

    mock_cache = PlateCache()
    monkeypatch.setattr(
        "intelligence.async_ocr.get_plate_cache", lambda: mock_cache
    )

    worker = AsyncOCRWorker(max_queue=10)
    for tid in range(5):
        worker.submit(_make_job(track_id=300 + tid))

    assert worker.queue_size == 5


def test_callback_fires_on_completion(monkeypatch, tmp_path):
    """
    When a real OCR result is produced, the on_result callback fires.
    Monkeypatches _run_ocr_on_crop to avoid needing EasyOCR installed.
    """
    results = []

    def _fake_ocr(crop):
        return ("BA 12 PA 1234", 0.92)

    monkeypatch.setattr("intelligence.async_ocr._run_ocr_on_crop", _fake_ocr)

    from intelligence.plate_ocr import PlateCache
    mock_cache = PlateCache()
    monkeypatch.setattr(
        "intelligence.async_ocr.get_plate_cache", lambda: mock_cache
    )

    def on_result(track_id, plate, conf, job):
        results.append((track_id, plate, conf))

    worker = AsyncOCRWorker(on_result=on_result)
    worker.start()
    worker.submit(_make_job(track_id=999))

    # Wait for the worker thread to process
    deadline = time.time() + 3.0
    while not results and time.time() < deadline:
        time.sleep(0.05)

    worker.stop()
    assert len(results) == 1
    assert results[0][0] == 999
    assert results[0][1] == "BA 12 PA 1234"
