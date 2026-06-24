# -*- coding: utf-8 -*-
"""
detection/yoloworld_engine.py
YOLO-World detection engine wrapper.
Loads once, re-used across all modules.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import (
    YOLO_WEIGHTS, YOLO_IMGSZ, YOLO_CONF, YOLO_IOU, YOLO_MAX_DET,
    DETECTION_CLASSES
)


@dataclass
class Detection:
    """Single detected object."""
    track_id:   int
    class_name: str
    confidence: float
    x1: int; y1: int; x2: int; y2: int

    @property
    def bbox(self):
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def center(self):
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def iou(self, other: "Detection") -> float:
        """Intersection over Union with another Detection."""
        ix1 = max(self.x1, other.x1); iy1 = max(self.y1, other.y1)
        ix2 = min(self.x2, other.x2); iy2 = min(self.y2, other.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlaps(self, other: "Detection", threshold: float = 0.1) -> bool:
        return self.iou(other) >= threshold

    def contains_point(self, x: int, y: int) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2


class YOLOWorldEngine:
    """
    Thin wrapper around ultralytics YOLOWorld.
    Handles model loading, class setting, and inference.
    """

    _instance: Optional["YOLOWorldEngine"] = None  # singleton

    def __init__(self, weights: str = YOLO_WEIGHTS,
                 classes: List[str] = None,
                 conf: float = YOLO_CONF,
                 iou: float = YOLO_IOU,
                 imgsz: int = YOLO_IMGSZ,
                 max_det: int = YOLO_MAX_DET,
                 device: str = None):
        from ultralytics import YOLOWorld
        # B1: auto-detect GPU — uses CUDA the moment it's available, else CPU
        if device is None:
            try:
                import torch
                device = "0" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        self.device  = device
        self.model  = YOLOWorld(weights)
        self.classes = classes or DETECTION_CLASSES
        self.conf    = conf
        self.iou     = iou
        self.imgsz   = imgsz
        self.max_det = max_det
        self.model.set_classes(self.classes)
        # Move model to the chosen device
        try:
            self.model.to(device)
        except Exception:
            pass
        print(f"[Engine] Loaded {weights} | device={device} | classes: {self.classes}")

    @classmethod
    def get_instance(cls) -> "YOLOWorldEngine":
        """Singleton — load model once per process."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def set_classes(self, classes: List[str]):
        self.classes = classes
        self.model.set_classes(classes)

    def predict(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a BGR frame.
        Returns list of Detection objects (track_id=-1 until tracker assigns).
        """
        results = self.model.predict(
            frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device=self.device,
            verbose=False,
        )
        detections: List[Detection] = []
        r = results[0]
        if r.boxes is None:
            return detections
        for box in r.boxes:
            cls_id     = int(box.cls[0])
            class_name = self.classes[cls_id] if cls_id < len(self.classes) else "unknown"
            conf_val   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(Detection(
                track_id=-1,
                class_name=class_name,
                confidence=conf_val,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))
        return detections

    def annotate(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels onto a copy of frame."""
        out = frame.copy()
        COLORS = {
            "motorcycle": (0, 165, 255),
            "car":        (0, 255, 0),
            "bus":        (255, 100, 0),
            "truck":      (255, 50, 50),
            "person":     (0, 200, 255),
            "helmet":     (0, 255, 180),
            "license plate": (255, 255, 0),
            "default":    (180, 180, 180),
        }
        for d in detections:
            color = COLORS.get(d.class_name, COLORS["default"])
            cv2.rectangle(out, (d.x1, d.y1), (d.x2, d.y2), color, 2)
            label = f"{'#'+str(d.track_id)+' ' if d.track_id >= 0 else ''}{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (d.x1, d.y1 - th - 6), (d.x1 + tw + 4, d.y1), color, -1)
            cv2.putText(out, label, (d.x1 + 2, d.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return out
