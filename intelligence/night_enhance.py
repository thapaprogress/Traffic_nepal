# -*- coding: utf-8 -*-
"""
intelligence/night_enhance.py
Night-time frame enhancement for low-light CCTV footage.
Applies CLAHE + denoising to improve detection in darkness.
"""

import cv2
import numpy as np


def enhance_night_frame(frame: np.ndarray, clip_limit: float = 3.0,
                        tile_size: int = 8, denoise: bool = True) -> np.ndarray:
    """
    Enhance a dark/night frame using CLAHE on the L channel of LAB colorspace.
    
    Args:
        frame: BGR input frame
        clip_limit: CLAHE contrast limit (higher = more contrast)
        tile_size: CLAHE grid tile size
        denoise: apply bilateral filter for noise reduction
    
    Returns:
        Enhanced BGR frame
    """
    # Convert to LAB colorspace
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to L channel (luminance)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    l_enhanced = clahe.apply(l)

    # Merge and convert back
    enhanced_lab = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Optional denoising
    if denoise:
        enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)

    return enhanced


def is_dark_frame(frame: np.ndarray, threshold: float = 60.0) -> bool:
    """
    Heuristic to detect if a frame is too dark and needs enhancement.
    Returns True if average brightness is below threshold.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(gray.mean()) < threshold


def auto_enhance(frame: np.ndarray, brightness_threshold: float = 60.0) -> np.ndarray:
    """
    Automatically enhance frame only if it's dark.
    Pass-through for daytime frames (no processing overhead).
    """
    if is_dark_frame(frame, brightness_threshold):
        return enhance_night_frame(frame)
    return frame
