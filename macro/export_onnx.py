#!/usr/bin/env python3
# Usage:
# python macro/export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --output model.onnx
import os
import argparse
import torch
import torch.onnx
import sys

### How to use it
# Single-task (legacy):
#   python export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --output meg2ang_final.onnx
# Multi-task (auto-detect from checkpoint):
#   python export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --multi-task --output model.onnx
# Multi-task (specify tasks):
#   python export_onnx.py artifacts/<RUN_NAME>/checkpoint_best.pth --multi-task --tasks angle energy --output model.onnx

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.model_regressor import XECEncoder, XECMultiHeadModel
except ImportError:
    print("Error: Could not import 'XECEncoder' or 'XECMultiHeadModel'.")
    print("Please ensure 'model_regressor.py' is in the current directory or python path.")
    sys.exit(1)

def load_checkpoint_weights(checkpoint_path, prefer_ema=True):
    """
    Loads weights from a checkpoint, preferring EMA weights if available.
    Returns the state_dict, a status string, and optional config metadata.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = None
    source = ""
    config_meta = {}

    # Extract config metadata if available
    if "active_tasks" in checkpoint:
        config_meta["active_tasks"] = checkpoint["active_tasks"]
        print(f"[INFO] Found active_tasks in checkpoint: {checkpoint['active_tasks']}")
    if "model_config" in checkpoint:
        config_meta["model_config"] = checkpoint["model_config"]

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

    return clean_dict, source, config_meta

def main():
    parser = argparse.ArgumentParser(
        description="Export PyTorch Checkpoint to ONNX",
        epilog="""
Examples:
  # Export single-task angle model (legacy XECEncoder)
  python export_onnx.py artifacts/run/checkpoint_best.pth --output model.onnx

  # Export multi-task model (auto-detect tasks from checkpoint)
  python export_onnx.py artifacts/run/checkpoint_best.pth --multi-task --output model.onnx

  # Export multi-task model with specific tasks
  python export_onnx.py artifacts/run/checkpoint_best.pth --multi-task --tasks angle energy --output model.onnx
        """
    )
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx", help="Output .onnx filename")
    parser.add_argument("--no-ema", action="store_true", help="Force using standard weights even if EMA exists")

    # Model Architecture Args (Must match training!)
    parser.add_argument("--outer_mode", type=str, default="finegrid", choices=["finegrid", "split"])
    parser.add_argument("--drop_path_rate", type=float, default=0.0, help="Should be 0 for export/inference")

    # Multi-task support
    parser.add_argument("--multi-task", action="store_true", help="Use XECMultiHeadModel instead of XECEncoder")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Active tasks for multi-task model (auto-detected from checkpoint if not specified)")

    args = parser.parse_args()

    # 1. Load Weights first to get config metadata
    state_dict, source, config_meta = load_checkpoint_weights(args.checkpoint, prefer_ema=not args.no_ema)

    # 2. Determine model type and tasks
    active_tasks = args.tasks
    use_multi_task = args.multi_task

    # Auto-detect from checkpoint if available
    if active_tasks is None and "active_tasks" in config_meta:
        active_tasks = config_meta["active_tasks"]
        if len(active_tasks) > 1 or (len(active_tasks) == 1 and active_tasks[0] != "angle"):
            use_multi_task = True
            print(f"[INFO] Auto-detected multi-task model with tasks: {active_tasks}")

    # Try to infer multi-task from state_dict keys if not explicitly set
    if active_tasks is None and not use_multi_task:
        # Check for multi-head model keys in state_dict
        multi_task_keys = [k for k in state_dict.keys() if k.startswith("heads.")]
        if multi_task_keys:
            # Infer tasks from head names (e.g., "heads.angle.0.weight" -> "angle")
            inferred_tasks = set()
            for key in multi_task_keys:
                parts = key.split(".")
                if len(parts) >= 2:
                    inferred_tasks.add(parts[1])
            if inferred_tasks:
                active_tasks = sorted(list(inferred_tasks))
                use_multi_task = True
                print(f"[INFO] Inferred multi-task model from state_dict keys: {active_tasks}")
                print("[WARN] 'active_tasks' not found in checkpoint metadata. "
                      "Consider re-saving checkpoint with updated training code.")

    # Default to angle-only if not specified
    if active_tasks is None:
        active_tasks = ["angle"]
        if not args.multi_task:
            print("[INFO] No task info in checkpoint, defaulting to angle-only. "
                  "Use --multi-task --tasks to specify if this is incorrect.")

    # 3. Initialize Model
    print("[INFO] Initializing Model...")
    if use_multi_task:
        print(f"[INFO] Creating XECMultiHeadModel with tasks: {active_tasks}")

        # First create the backbone (XECEncoder)
        backbone = XECEncoder(
            outer_mode=args.outer_mode,
            outer_fine_pool=(3, 3),
            drop_path_rate=0.0  # Always 0 for export
        )

        # Then create the multi-head model with the backbone
        model = XECMultiHeadModel(
            backbone=backbone,
            active_tasks=active_tasks
        )
    else:
        print("[INFO] Creating XECEncoder (single-task angle model)")
        model = XECEncoder(
            outer_mode=args.outer_mode,
            outer_fine_pool=(3, 3),
            drop_path_rate=0.0
        )
    model.eval()

    # 4. Load Weights
    try:
        model.load_state_dict(state_dict, strict=True)
        print(f"[SUCCESS] Loaded {source} weights successfully.")
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed. Attempting non-strict (keys might differ).")
        print(f"Error details: {e}")
        model.load_state_dict(state_dict, strict=False)

    # 5. Export
    dummy_input = torch.randn(1, 4760, 2)

    # Ensure directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Build output names for multi-task
    if use_multi_task:
        output_names = [f"output_{task}" for task in active_tasks]
        dynamic_axes = {'input': {0: 'batch_size'}}
        for name in output_names:
            dynamic_axes[name] = {0: 'batch_size'}
    else:
        output_names = ['output']
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }

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
