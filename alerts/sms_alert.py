# -*- coding: utf-8 -*-
"""
alerts/sms_alert.py
SMS alert dispatcher via Twilio (or console mock for dev).
Set environment variables: TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM, ALERT_PHONE
"""

import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Config from environment
TWILIO_SID   = os.environ.get("TWILIO_SID", "")
TWILIO_TOKEN = os.environ.get("TWILIO_TOKEN", "")
TWILIO_FROM  = os.environ.get("TWILIO_FROM", "+10000000000")
ALERT_PHONE  = os.environ.get("ALERT_PHONE", "+9779800000000")


def send_sms(violation_type: str, location: str, plate: str = "",
             speed: float = 0, track_id: int = -1) -> bool:
    """
    Send SMS alert for a violation.
    If Twilio is not configured, prints to console (mock mode).
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    body = (
        f"[Traffic Eye] {violation_type} VIOLATION\n"
        f"Location: {location}\n"
        f"Time: {timestamp}\n"
    )
    if plate:
        body += f"Plate: {plate}\n"
    if speed > 0:
        body += f"Speed: {speed} km/h\n"
    body += f"Track ID: #{track_id}"

    # If Twilio configured, send real SMS
    if TWILIO_SID and TWILIO_TOKEN:
        try:
            from twilio.rest import Client
            client = Client(TWILIO_SID, TWILIO_TOKEN)
            msg = client.messages.create(
                body=body,
                from_=TWILIO_FROM,
                to=ALERT_PHONE,
            )
            logger.info(f"[SMS] Sent: {msg.sid}")
            return True
        except Exception as e:
            logger.error(f"[SMS] Failed: {e}")
            return False
    else:
        # Mock mode — print to console
        print(f"\n{'='*50}")
        print(f"📱 SMS ALERT (mock — Twilio not configured)")
        print(f"{'='*50}")
        print(body)
        print(f"{'='*50}\n")
        return True
