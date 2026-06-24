# -*- coding: utf-8 -*-
"""
workers/redis_broker.py
Redis Pub/Sub Event Broker.
Publishes detection/violation and traffic stats events, with graceful fallback.
"""

import os
import json

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

_redis_client = None

try:
    import redis
    _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    # Validate connection
    _redis_client.ping()
except Exception as e:
    print(f"[RedisBroker] Redis not available at {REDIS_URL}: {e}. Event publishing will run in fallback/no-op mode.")
    _redis_client = None

def publish_event(channel: str, data: dict) -> bool:
    """
    Publish an event to a Redis channel.
    Returns True if successfully published, False otherwise.
    """
    if _redis_client is not None:
        try:
            payload = json.dumps(data)
            _redis_client.publish(channel, payload)
            return True
        except Exception as e:
            print(f"[RedisBroker] Failed to publish event to {channel}: {e}")
    return False

def subscribe_events(channel: str):
    """
    Subscribe to a Redis channel. Generator yielding deserialized message dicts.
    """
    if _redis_client is not None:
        try:
            pubsub = _redis_client.pubsub()
            pubsub.subscribe(channel)
            for message in pubsub.listen():
                if message and message["type"] == "message":
                    yield json.loads(message["data"])
        except Exception as e:
            print(f"[RedisBroker] Subscription error on channel {channel}: {e}")
