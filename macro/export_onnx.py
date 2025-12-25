#!/usr/bin/env python3
import os
import argparse
import torch
import torch.onnx
import sys

### How to use it
# python export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --output meg2ang_final.onnx

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.model import XECRegressor
except ImportError:
    print("Error: Could not import 'XECRegressor'.")
    print("Please ensure 'model.py' is in the current directory or python path.")
    sys.exit(1)

def load_checkpoint_weights(checkpoint_path, prefer_ema=True):
    """
    Loads weights from a checkpoint, preferring EMA weights if available.
    Returns the state_dict and a status string.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    state_dict = None
    source = ""

    # 1. Try Loading EMA
    if prefer_ema and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        print("[INFO] Found EMA state dict. Preparing to load smoothed weights.")
        raw_dict = checkpoint["ema_state_dict"]
        source = "EMA"
    # 2. Fallback to Standard
    elif "model_state_dict" in checkpoint:
        print("[INFO] No EMA state found (or EMA disabled). Loading standard model weights.")
        raw_dict = checkpoint["model_state_dict"]
        source = "Standard"
    else:
        raise KeyError("Checkpoint does not contain 'ema_state_dict' or 'model_state_dict'.")

    # 3. Key Sanitization (Remove 'module.' prefix from AveragedModel or DDP)
    clean_dict = {}
    for k, v in raw_dict.items():
        # AveragedModel often saves keys as "module.layer.weight"
        if k.startswith("module."):
            clean_k = k[7:] # strip "module."
        else:
            clean_k = k
        clean_dict[clean_k] = v
        
    return clean_dict, source

def main():
    parser = argparse.ArgumentParser(description="Export PyTorch Checkpoint to ONNX")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output .onnx filename")
    parser.add_argument("--no-ema", action="store_true", help="Force using standard weights even if EMA exists")
    
    # Model Architecture Args (Must match training!)
    parser.add_argument("--outer_mode", type=str, default="finegrid", choices=["finegrid", "split"])
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="Should be 0 for export/inference")
    
    args = parser.parse_args()

    # 1. Initialize Model
    print("[INFO] Initializing Model...")
    # Note: We set drop_path_rate to 0 for export to remove stochastic behavior
    model = XECRegressor(
        outer_mode=args.outer_mode,
        outer_fine_pool=(3,3),
        drop_path_rate=0.0 
    )
    model.eval()

    # 2. Load Weights
    state_dict, source = load_checkpoint_weights(args.checkpoint, prefer_ema=not args.no_ema)
    
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[SUCCESS] Loaded {source} weights successfully.")
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed. Attempting non-strict (keys might differ).")
        print(f"Error details: {e}")
        model.load_state_dict(state_dict, strict=False)

    # 3. Export
    dummy_input = torch.randn(1, 4760, 2)
    
    # Ensure directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"[INFO] Exporting to {args.output}...")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"[SUCCESS] Model exported to {args.output}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()