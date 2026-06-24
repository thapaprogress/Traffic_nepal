# -*- coding: utf-8 -*-
"""
deploy/export_engine.py
Export YOLO-World to ONNX and TensorRT for maximum inference speed.

Usage:
    python deploy/export_engine.py --format onnx
    python deploy/export_engine.py --format tensorrt --half
    python deploy/export_engine.py --format all --half

Requires:
    - NVIDIA GPU + CUDA for TensorRT
    - pip install onnx onnxruntime tensorrt (for TRT)
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import YOLO_WEIGHTS, YOLO_IMGSZ, DETECTION_CLASSES


def export_model(format: str = "onnx", half: bool = False, imgsz: int = 640):
    from ultralytics import YOLOWorld

    weights = YOLO_WEIGHTS
    if not os.path.isfile(weights):
        weights = "yolov8s-worldv2.pt"

    print(f"\n{'='*60}")
    print(f"  Traffic Eye — Model Export")
    print(f"{'='*60}")
    print(f"  Weights:  {weights}")
    print(f"  Format:   {format}")
    print(f"  FP16:     {half}")
    print(f"  ImgSize:  {imgsz}")
    print(f"  Classes:  {DETECTION_CLASSES}")
    print(f"{'='*60}\n")

    # Load model and set classes
    model = YOLOWorld(weights)
    model.set_classes(DETECTION_CLASSES)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "weights")
    os.makedirs(output_dir, exist_ok=True)

    results = {}

    if format in ("onnx", "all"):
        print("[1] Exporting to ONNX...")
        t0 = time.time()
        onnx_path = model.export(format="onnx", imgsz=imgsz, simplify=True)
        elapsed = time.time() - t0
        results["onnx"] = onnx_path
        print(f"    ✅ ONNX saved: {onnx_path} ({elapsed:.1f}s)")

    if format in ("tensorrt", "engine", "all"):
        print("[2] Exporting to TensorRT Engine...")
        t0 = time.time()
        try:
            engine_path = model.export(
                format="engine",
                imgsz=imgsz,
                half=half,
                device=0,
            )
            elapsed = time.time() - t0
            results["tensorrt"] = engine_path
            print(f"    ✅ TensorRT saved: {engine_path} ({elapsed:.1f}s)")
        except Exception as e:
            print(f"    ❌ TensorRT export failed: {e}")
            print("    Make sure you have NVIDIA GPU + TensorRT installed.")
            results["tensorrt"] = None

    if format == "openvino":
        print("[3] Exporting to OpenVINO...")
        t0 = time.time()
        ov_path = model.export(format="openvino", imgsz=imgsz, half=half)
        elapsed = time.time() - t0
        results["openvino"] = ov_path
        print(f"    ✅ OpenVINO saved: {ov_path} ({elapsed:.1f}s)")

    # Benchmark
    print(f"\n{'='*60}")
    print("  Export complete! Benchmark the exported model with:")
    for fmt, path in results.items():
        if path:
            print(f"    yolo benchmark model={path} imgsz={imgsz}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO-World model")
    parser.add_argument("--format", default="onnx",
                        choices=["onnx", "tensorrt", "engine", "openvino", "all"])
    parser.add_argument("--half", action="store_true", help="FP16 quantization")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    export_model(format=args.format, half=args.half, imgsz=args.imgsz)
