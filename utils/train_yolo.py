# -*- coding: utf-8 -*-
"""
utils/train_yolo.py
Fine-tuning execution script for training YOLO-World on the Nepal Traffic dataset.
Run this script on a GPU-enabled environment.
"""

import os
import argparse
from ultralytics import YOLO

def run_training(epochs=80, batch_size=16, imgsz=640, model_weights="yolov8s-worldv2.pt", data_config="data/nepal_traffic.yaml"):
    """
    Executes the fine-tuning process for YOLO-World on the custom Nepal Traffic dataset.
    """
    print("=== Traffic Eye Nepal YOLO-World Fine-Tuning ===")
    
    # 1. Verify that dataset config exists
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Dataset config YAML not found at: {data_config}")
    
    print(f"[1/4] Loading model: {model_weights}")
    model = YOLO(model_weights)
    
    print(f"[2/4] Initializing training: epochs={epochs}, batch={batch_size}, imgsz={imgsz}")
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=0,                   # Train on GPU device 0
        project="runs/nepal_traffic",
        name="finetune_v1",
        amp=True,                   # Automatic Mixed Precision
        val=True                    # Validate during training
    )
    
    print("\n[3/4] Running final validation on validation set...")
    val_results = model.val(data=data_config)
    print(f"Validation mAP50: {val_results.results_dict.get('metrics/mAP50(B)', 0.0):.4f}")
    print(f"Validation mAP50-95: {val_results.results_dict.get('metrics/mAP50-95(B)', 0.0):.4f}")
    
    print("\n[4/4] Exporting model formats (TensorRT, ONNX)...")
    # Export model to standard ONNX and TensorRT formats
    try:
        onnx_path = model.export(format="onnx")
        print(f"Exported ONNX model to: {onnx_path}")
    except Exception as e:
        print(f"ONNX Export failed: {e}")
        
    try:
        engine_path = model.export(format="engine", half=True, device=0)
        print(f"Exported TensorRT model to: {engine_path}")
    except Exception as e:
        print(f"TensorRT Export failed (requires TensorRT runtime): {e}")

    print("\n=== Training Process Finished Successfully ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO-World model on Nepal Traffic dataset.")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (reduce if VRAM is low).")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--model", type=str, default="yolov8s-worldv2.pt", help="Base model weights.")
    parser.add_argument("--data", type=str, default="data/nepal_traffic.yaml", help="Dataset configuration path.")
    
    args = parser.parse_args()
    
    run_training(
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        model_weights=args.model,
        data_config=args.data
    )
