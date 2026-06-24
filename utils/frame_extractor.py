# -*- coding: utf-8 -*-
"""
utils/frame_extractor.py
Helper utility to extract unique, non-redundant frames from traffic video files
or RTSP streams to compile the Nepal Traffic dataset for YOLO-World fine-tuning.
"""

import os
import cv2
import time
import argparse
from datetime import datetime

def extract_frames(source, output_dir, interval_sec=10.0, max_frames=500, prefix="frame"):
    """
    Extracts frames from a video file or RTSP stream at a given time interval.
    
    Args:
        source (str/int): Path to video file, RTSP stream URL, or webcam ID (0).
        output_dir (str): Directory where extracted frames will be saved.
        interval_sec (float): Minimum seconds between saved frames.
        max_frames (int): Maximum number of frames to extract.
        prefix (str): Prefix for saved image filenames.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Try converting source to int if it looks like a webcam ID
    try:
        source = int(source)
    except ValueError:
        pass

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Default to 30 if FPS cannot be read (common for some RTSP streams)
        fps = 30.0

    print(f"[Info] Opened video source: {source}")
    print(f"[Info] Video FPS: {fps:.2f}")
    print(f"[Info] Frame extraction interval: {interval_sec} seconds")
    print(f"[Info] Saving to directory: {output_dir}")

    saved_count = 0
    frame_idx = 0
    last_saved_time = -interval_sec  # Force saving the first frame

    # If parsing a file, we can map frame index to time
    is_file = isinstance(source, str) and os.path.isfile(source)

    start_time = time.time()

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            print("[Info] End of video stream or file reached.")
            break

        # Calculate timestamp of current frame in seconds
        if is_file:
            current_time_sec = frame_idx / fps
        else:
            current_time_sec = time.time() - start_time

        # Check if interval has elapsed
        if current_time_sec - last_saved_time >= interval_sec:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{prefix}_{timestamp_str}_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            # Save frame
            cv2.imwrite(filepath, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            print(f"[Saved] {filename} (at time {current_time_sec:.1f}s)")
            
            last_saved_time = current_time_sec
            saved_count += 1

        frame_idx += 1

    cap.release()
    print(f"\n[Completed] Extracted {saved_count} frames successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract non-redundant training frames from traffic video / RTSP streams.")
    parser.add_argument("--source", type=str, default="0", help="Video file path or RTSP URL or 0 for webcam.")
    parser.add_argument("--output", type=str, default="data/nepal_traffic/images/train", help="Output directory for frames.")
    parser.add_argument("--interval", type=float, default=10.0, help="Interval in seconds between saved frames.")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum number of frames to extract.")
    parser.add_argument("--prefix", type=str, default="nepal_traffic", help="Filename prefix.")
    
    args = parser.parse_args()
    
    extract_frames(
        source=args.source,
        output_dir=args.output,
        interval_sec=args.interval,
        max_frames=args.max_frames,
        prefix=args.prefix
    )
