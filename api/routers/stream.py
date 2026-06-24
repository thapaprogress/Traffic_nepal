# -*- coding: utf-8 -*-
"""
api/routers/stream.py
WebSocket endpoint for live annotated frame streaming (MJPEG over WS).
"""

import asyncio
import cv2
import base64
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

router = APIRouter(tags=["stream"])

# Global reference to camera manager (set at startup)
_camera_manager = None


def set_camera_manager(mgr):
    global _camera_manager
    _camera_manager = mgr


@router.websocket("/ws/live/{camera_id}")
async def live_stream(websocket: WebSocket, camera_id: str):
    """
    Stream live annotated frames as base64 JPEG over WebSocket.
    Client receives JSON: {"frame": "<base64>", "stats": {...}}
    """
    await websocket.accept()
    try:
        while True:
            if _camera_manager is None:
                await websocket.send_json({"error": "No camera manager"})
                await asyncio.sleep(1)
                continue

            state = _camera_manager.get_state(camera_id)
            if state is None:
                await websocket.send_json({"error": f"Camera {camera_id} not found"})
                await asyncio.sleep(1)
                continue

            frame_rgb = state.get("frame_rgb")
            if frame_rgb is not None:
                # Encode as JPEG
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", frame_bgr,
                                      [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode("utf-8")

                payload = {
                    "frame": b64,
                    "stats": {
                        "fps": state.get("fps", 0),
                        "vehicle_count": state.get("vehicle_count", 0),
                        "congestion": state.get("congestion", "LOW"),
                        "helmet_violations": state.get("helmet_violations", 0),
                        "speed_violations": state.get("speed_violations", 0),
                        "frame_num": state.get("frame_num", 0),
                    }
                }
                await websocket.send_json(payload)

            await asyncio.sleep(0.04)  # ~25 fps

    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()


@router.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    """
    Push real-time violation alerts to connected clients.
    Client receives JSON: {"type": "HELMET", "track_id": 5, ...}
    """
    await websocket.accept()
    last_count = 0
    try:
        while True:
            from alerts.alert_dispatcher import get_dispatcher
            dispatcher = get_dispatcher()
            alerts = dispatcher.recent_alerts
            current_count = len(alerts)

            if current_count > last_count:
                new_alerts = alerts[:current_count - last_count]
                for alert in new_alerts:
                    await websocket.send_json(alert)
                last_count = current_count

            await asyncio.sleep(0.5)

    except WebSocketDisconnect:
        pass
    except Exception:
        await websocket.close()
