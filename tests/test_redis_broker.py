# -*- coding: utf-8 -*-
"""
tests/test_redis_broker.py
Tests for the Redis pub/sub broker (Task 3.12).
Verifies graceful fallback when Redis is unavailable.
"""
import importlib, sys


def test_publish_event_no_redis_returns_false(monkeypatch):
    """When Redis is not available, publish_event returns False without raising."""
    # Force the module to reload with no redis client
    if "workers.redis_broker" in sys.modules:
        del sys.modules["workers.redis_broker"]

    # Patch redis import to fail
    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "redis":
            raise ImportError("redis not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    # Re-import the module
    if "workers.redis_broker" in sys.modules:
        del sys.modules["workers.redis_broker"]

    import workers.redis_broker as broker
    importlib.reload(broker)

    # Should return False (no-op) not raise
    result = broker.publish_event("traffic:violations", {"type": "test"})
    assert result is False


def test_publish_event_with_client_calls_publish(monkeypatch):
    """When a mock redis client is injected, publish_event calls .publish()."""
    import workers.redis_broker as broker

    published = []

    class MockRedis:
        def publish(self, channel, data):
            published.append((channel, data))

    monkeypatch.setattr(broker, "_redis_client", MockRedis())

    result = broker.publish_event("traffic:stats", {"camera_id": "cam_01", "count": 5})
    assert result is True
    assert len(published) == 1
    assert published[0][0] == "traffic:stats"
    assert "cam_01" in published[0][1]
