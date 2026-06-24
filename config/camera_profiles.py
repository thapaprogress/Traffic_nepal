# -*- coding: utf-8 -*-
"""
config/camera_profiles.py
Persist per-camera calibration (ROI polygon, speed lines, direction, limits)
to a JSON file so you don't re-enter settings on every run.
"""

import json
import os
from typing import Dict, Any, Optional

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILES_PATH = os.path.join(_BASE, "config", "camera_profiles.json")


def _load_all() -> Dict[str, Any]:
    if os.path.isfile(PROFILES_PATH):
        try:
            with open(PROFILES_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_all(data: Dict[str, Any]):
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_profile(camera_id: str, profile: Dict[str, Any]):
    """
    Save a camera's calibration profile.
    Expected keys: name, location, line_y_a, line_y_b, speed_limit,
                   allowed_direction_deg, roi_polygon, source.
    """
    data = _load_all()
    data[camera_id] = profile
    _save_all(data)


def load_profile(camera_id: str) -> Optional[Dict[str, Any]]:
    """Load a single camera profile, or None if not found."""
    return _load_all().get(camera_id)


def list_profiles() -> Dict[str, Any]:
    """Return all saved profiles."""
    return _load_all()


def delete_profile(camera_id: str) -> bool:
    data = _load_all()
    if camera_id in data:
        del data[camera_id]
        _save_all(data)
        return True
    return False
