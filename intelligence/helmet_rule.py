# -*- coding: utf-8 -*-
"""
intelligence/helmet_rule.py
Helmet violation detection logic.

Rule:
    IF motorcycle is detected
    AND a person bbox overlaps or is near the motorcycle
    AND no helmet is detected in the person's head region
    THEN → HELMET VIOLATION
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection.yoloworld_engine import Detection


@dataclass
class HelmetViolation:
    track_id:   int
    person_bbox: Tuple[int, int, int, int]
    bike_bbox:   Tuple[int, int, int, int]
    frame_num:  int
    head_crop:  Optional[np.ndarray] = None   # cropped head image for OCR/snapshot


def _expand_box(x1, y1, x2, y2, factor: float, frame_h: int, frame_w: int):
    """Expand bounding box by a factor, clamped to frame dimensions."""
    cx, cy   = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh   = (x2 - x1) / 2 * factor, (y2 - y1) / 2 * factor
    return (max(0, int(cx - hw)), max(0, int(cy - hh)),
            min(frame_w, int(cx + hw)), min(frame_h, int(cy + hh)))


def _boxes_nearby(bike: Detection, person: Detection, proximity: float = 0.5) -> bool:
    """Check if person bbox is near (or overlapping) the motorcycle bbox."""
    # Expand bike bbox to capture riders slightly outside the bike outline
    bike_x1, bike_y1, bike_x2, bike_y2 = bike.bbox
    expand = max(bike.x2 - bike.x1, bike.y2 - bike.y1) * proximity
    ebx1 = bike_x1 - expand; eby1 = bike_y1 - expand
    ebx2 = bike_x2 + expand; eby2 = bike_y2 + expand
    # Check if person center is inside expanded bike box
    px, py = person.center
    return ebx1 <= px <= ebx2 and eby1 <= py <= eby2


def _is_rider(bike: Detection, person: Detection) -> bool:
    """
    Stronger check than proximity (fix A11): the person must actually be
    riding the bike — horizontally aligned and vertically above/overlapping
    the bike, not just standing nearby.
    """
    if not _boxes_nearby(bike, person, proximity=0.3):
        return False
    px, _ = person.center
    # Person's horizontal center should sit within the bike's x-span (widened)
    bw = bike.x2 - bike.x1
    if not (bike.x1 - bw * 0.3 <= px <= bike.x2 + bw * 0.3):
        return False
    # Rider's body should be above the bike's bottom (sitting on it)
    if person.y2 < bike.y1:        # person entirely above the bike → not a rider
        return False
    if person.y1 > bike.y2:        # person entirely below the bike → not a rider
        return False
    return True


class HelmetConfirmer:
    """
    Requires N consecutive frames of 'no helmet' for the same track_id
    before confirming a violation (fix A11 — reduces false positives).
    """
    def __init__(self, frames_to_confirm: int = 3):
        self.frames_to_confirm = frames_to_confirm
        self._streak = {}        # track_id → consecutive no-helmet count
        self._confirmed = set()  # already-fired track_ids

    def update(self, candidate_ids: set) -> set:
        """Return the set of track_ids whose violation is newly confirmed."""
        newly = set()
        # Increment streaks for current candidates
        for tid in candidate_ids:
            self._streak[tid] = self._streak.get(tid, 0) + 1
            if (self._streak[tid] >= self.frames_to_confirm
                    and tid not in self._confirmed):
                self._confirmed.add(tid)
                newly.add(tid)
        # Reset streaks for tracks no longer violating
        for tid in list(self._streak.keys()):
            if tid not in candidate_ids:
                self._streak.pop(tid, None)
        return newly


def check_helmet_violations(
    detections: List[Detection],
    frame: np.ndarray,
    frame_num: int,
    overlap_threshold: float = 0.1,
) -> List[HelmetViolation]:
    """
    Given a list of detections for a single frame, return all helmet violations.
    """
    h, w = frame.shape[:2]

    motorcycles = [d for d in detections if d.class_name == "motorcycle"]
    persons     = [d for d in detections if d.class_name == "person"]
    helmets     = [d for d in detections if d.class_name == "helmet"]

    violations: List[HelmetViolation] = []

    for bike in motorcycles:
        # Find persons actually riding this bike (stronger than proximity)
        riders = [p for p in persons if _is_rider(bike, p)]

        for rider in riders:
            # Head region = top 35% of the person bbox
            hx1, hy1, hx2, hy2 = rider.bbox
            head_h = int((hy2 - hy1) * 0.35)
            head_box = (hx1, hy1, hx2, hy1 + head_h)

            # Check if any helmet overlaps the head region
            helmet_found = False
            for helmet in helmets:
                # helmet center inside head box?
                hcx, hcy = helmet.center
                if (head_box[0] <= hcx <= head_box[2] and
                        head_box[1] <= hcy <= head_box[3]):
                    helmet_found = True
                    break
                # or meaningful IoU overlap
                if rider.iou(helmet) > overlap_threshold:
                    helmet_found = True
                    break

            if not helmet_found:
                # Crop head region for snapshot
                x1c, y1c, x2c, y2c = head_box
                head_crop = frame[y1c:y2c, x1c:x2c].copy() if y2c > y1c and x2c > x1c else None
                violations.append(HelmetViolation(
                    track_id    = rider.track_id,
                    person_bbox = rider.bbox,
                    bike_bbox   = bike.bbox,
                    frame_num   = frame_num,
                    head_crop   = head_crop,
                ))

    return violations


def draw_violations(frame: np.ndarray, violations: List[HelmetViolation],
                    in_place: bool = False) -> np.ndarray:
    """Draw red overlay on violators. Set in_place=True to avoid a copy (fix A13)."""
    out = frame if in_place else frame.copy()
    for v in violations:
        x1, y1, x2, y2 = v.person_bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"NO HELMET #{v.track_id}"
        cv2.rectangle(out, (x1, y1 - 26), (x1 + len(label) * 11, y1), (0, 0, 200), -1)
        cv2.putText(out, label, (x1 + 2, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        # Red cross on bike
        bx1, by1, bx2, by2 = v.bike_bbox
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 0, 180), 2)
    return out
