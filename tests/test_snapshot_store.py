# -*- coding: utf-8 -*-
"""
tests/test_snapshot_store.py
Tests for alerts/snapshot_store.py (Task 3.8).
Verifies local fallback when MinIO is not configured.
"""
import os
import numpy as np
import cv2


def test_upload_snapshot_local_fallback(tmp_path, monkeypatch):
    """
    When MINIO_URL is empty (not configured), upload_snapshot should
    save to local SNAPSHOT_DIR and return a valid file path.
    """
    # Ensure env vars are unset so MinIO client is None
    monkeypatch.delenv("MINIO_URL", raising=False)
    monkeypatch.delenv("MINIO_KEY", raising=False)
    monkeypatch.delenv("MINIO_SECRET", raising=False)

    # Override SNAPSHOT_DIR to use the tmp_path fixture
    import config.settings as settings
    monkeypatch.setattr(settings, "SNAPSHOT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "ALERT_SNAPSHOT_QUALITY", 85)

    # Force a reload with no MinIO client
    import importlib
    import sys
    if "alerts.snapshot_store" in sys.modules:
        del sys.modules["alerts.snapshot_store"]
    import alerts.snapshot_store as store
    monkeypatch.setattr(store, "client", None)

    # Create a dummy BGR test frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(frame, (10, 10), (90, 90), (0, 200, 100), -1)

    result_path = store.upload_snapshot(frame, "test_snapshot.jpg")

    # Must return a local path (not a URL)
    assert os.path.isfile(result_path), f"Expected a local file, got: {result_path}"
    assert result_path.endswith(".jpg")

    # Verify the file is readable as an image
    loaded = cv2.imread(result_path)
    assert loaded is not None, "Saved snapshot is not a valid image"


def test_upload_snapshot_with_mock_minio_client(monkeypatch, tmp_path):
    """
    When a mock MinIO client is injected, upload_snapshot should call
    put_object and return a URL.
    """
    import alerts.snapshot_store as store
    import importlib, sys

    uploads = []

    class MockBucket:
        pass

    class MockMinioClient:
        def put_object(self, bucket, key, data, length, content_type=None):
            uploads.append((bucket, key))

    monkeypatch.setattr(store, "client", MockMinioClient())
    monkeypatch.setattr(store, "MINIO_URL", "localhost:9000")
    monkeypatch.setattr(store, "BUCKET", "traffic-eye-snapshots")

    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = store.upload_snapshot(frame, "cam_shot.jpg")

    assert len(uploads) == 1
    assert uploads[0][1] == "cam_shot.jpg"
    assert "cam_shot.jpg" in result
