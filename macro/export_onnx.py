#!/usr/bin/env python3
"""
Export PyTorch checkpoint to ONNX format.

Usage:
    # Auto-detect tasks from checkpoint
    python macro/export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --output model.onnx

    # Specify tasks explicitly
    python macro/export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --tasks angle energy --output model.onnx
"""
import os
import argparse
import torch
import torch.onnx
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.models import XECEncoder, XECMultiHeadModel


def load_checkpoint_weights(checkpoint_path, prefer_ema=True):
    """
    Loads weights from a checkpoint, preferring EMA weights if available.
    Returns the state_dict, a status string, and optional config metadata.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_meta = {}

    # Extract config metadata if available
    if "config" in checkpoint:
        config_meta.update(checkpoint["config"])
    if "active_tasks" in checkpoint:
        config_meta["active_tasks"] = checkpoint["active_tasks"]
        print(f"[INFO] Found active_tasks in checkpoint: {checkpoint['active_tasks']}")
    if "model_config" in checkpoint:
        config_meta["model_config"] = checkpoint["model_config"]

    # 1. Try Loading EMA
    if prefer_ema and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        print("[INFO] Found EMA state dict. Loading smoothed weights.")
        raw_dict = checkpoint["ema_state_dict"]
        source = "EMA"
    # 2. Fallback to Standard
    elif "model_state_dict" in checkpoint:
        print("[INFO] Loading standard model weights.")
        raw_dict = checkpoint["model_state_dict"]
        source = "Standard"
    else:
        raise KeyError("Checkpoint does not contain 'ema_state_dict' or 'model_state_dict'.")

    # 3. Key Sanitization (Remove 'module.' prefix from AveragedModel or DDP)
    clean_dict = {}
    for k, v in raw_dict.items():
        if k.startswith("module."):
            clean_k = k[7:]
        else:
            clean_k = k
        clean_dict[clean_k] = v

    return clean_dict, source, config_meta


def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch Checkpoint to ONNX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output .onnx filename")
    parser.add_argument("--no-ema", action="store_true", help="Force using standard weights even if EMA exists")

    # Model Architecture Args (Must match training!)
    parser.add_argument("--outer_mode", type=str, default="finegrid", choices=["finegrid", "split"])

    # Task specification
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Active tasks (auto-detected from checkpoint if not specified)")

    args = parser.parse_args()

    # 1. Load Weights first to get config metadata
    state_dict, source, config_meta = load_checkpoint_weights(args.checkpoint, prefer_ema=not args.no_ema)

    # 2. Determine active tasks
    active_tasks = args.tasks

    # Auto-detect from checkpoint if available
    if active_tasks is None and "active_tasks" in config_meta:
        active_tasks = config_meta["active_tasks"]
        print(f"[INFO] Auto-detected tasks from checkpoint: {active_tasks}")

    # Try to infer from state_dict keys if not in metadata
    if active_tasks is None:
        multi_task_keys = [k for k in state_dict.keys() if k.startswith("heads.")]
        if multi_task_keys:
            inferred_tasks = set()
            for key in multi_task_keys:
                parts = key.split(".")
                if len(parts) >= 2:
                    inferred_tasks.add(parts[1])
            if inferred_tasks:
                active_tasks = sorted(list(inferred_tasks))
                print(f"[INFO] Inferred tasks from state_dict keys: {active_tasks}")

    # Default to angle-only if nothing detected
    if active_tasks is None:
        active_tasks = ["angle"]
        print("[INFO] No task info found, defaulting to ['angle']")

    # 3. Initialize Model
    print(f"[INFO] Creating XECMultiHeadModel with tasks: {active_tasks}")

    backbone = XECEncoder(
        outer_mode=args.outer_mode,
        outer_fine_pool=(3, 3),
        drop_path_rate=0.0,  # Always 0 for export
        encoder_dim=config_meta.get('encoder_dim', 1024),
        dim_feedforward=config_meta.get('dim_feedforward', None),
        num_fusion_layers=config_meta.get('num_fusion_layers', 2),
        sentinel_time=config_meta.get('sentinel_time', -1.0),
    )

    model = XECMultiHeadModel(
        backbone=backbone,
        active_tasks=active_tasks
    )
    model.eval()

    # 4. Load Weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[SUCCESS] Loaded {source} weights successfully.")
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed: {e}")
        print("[INFO] Attempting non-strict loading...")
        model.load_state_dict(state_dict, strict=False)

    # 5. Export
    dummy_input = torch.randn(1, 4760, 2)

    # Ensure directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Build output names
    output_names = [f"output_{task}" for task in active_tasks]
    dynamic_axes = {'input': {0: 'batch_size'}}
    for name in output_names:
        dynamic_axes[name] = {0: 'batch_size'}

    print(f"[INFO] Exporting to {args.output}...")
    print(f"[INFO] Output names: {output_names}")
    try:
        torch.onnx.export(
            model,
            dummy_input,
            args.output,
            export_params=True,
            opset_version=20,
            do_constant_folding=True,
            input_names=['input'],
            output_names=output_names,
            dynamic_axes=dynamic_axes
        )
        print(f"[SUCCESS] Model exported to {args.output}")
    except Exception as e:
        print(f"[ERROR] Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
