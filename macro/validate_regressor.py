#!/usr/bin/env python3
"""
Run validation-only with an existing checkpoint.

This script loads a checkpoint and runs validation to generate predictions
and resolution plots without training.

Usage:
    python macro/validate_regressor.py checkpoint.pth --val_path data/val/ --config config.yaml
    python macro/validate_regressor.py checkpoint.pth --val_path data/val/ --tasks energy uvwFI
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.models import XECEncoder, XECMultiHeadModel
from lib.dataset import get_dataloader
from lib.engines import run_epoch_stream
from lib.config import load_config
from lib.train_regressor import save_validation_artifacts
from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run validation-only with an existing checkpoint",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--config", type=str, default=None, help="Config file (optional, for normalization params)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Tasks to evaluate (default: from checkpoint)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get config from checkpoint or file
    if "config" in checkpoint:
        cfg = checkpoint["config"]
        print("[INFO] Using config from checkpoint")
    elif args.config:
        cfg = load_config(args.config)
        print(f"[INFO] Using config from: {args.config}")
    else:
        cfg = None
        print("[INFO] No config found, using defaults")

    # Determine active tasks
    if args.tasks:
        active_tasks = args.tasks
    elif cfg and hasattr(cfg, 'tasks'):
        active_tasks = [t for t, tc in cfg.tasks.items() if tc.enabled]
    else:
        # Try to infer from checkpoint state dict
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        active_tasks = []
        if any("angle" in k for k in state_dict.keys()):
            active_tasks.append("angle")
        if any("energy" in k for k in state_dict.keys()):
            active_tasks.append("energy")
        if any("timing" in k for k in state_dict.keys()):
            active_tasks.append("timing")
        if any("uvwFI" in k or "position" in k for k in state_dict.keys()):
            active_tasks.append("uvwFI")

    print(f"[INFO] Active tasks: {active_tasks}")

    # Get normalization params
    if cfg:
        npho_scale = cfg.normalization.npho_scale
        npho_scale2 = cfg.normalization.npho_scale2
        time_scale = cfg.normalization.time_scale
        time_shift = cfg.normalization.time_shift
        sentinel_time = cfg.normalization.sentinel_time
    else:
        npho_scale = DEFAULT_NPHO_SCALE
        npho_scale2 = DEFAULT_NPHO_SCALE2
        time_scale = DEFAULT_TIME_SCALE
        time_shift = DEFAULT_TIME_SHIFT
        sentinel_time = DEFAULT_SENTINEL_TIME

    # Get model params
    if cfg:
        outer_mode = cfg.model.outer_mode
        outer_fine_pool = tuple(cfg.model.outer_fine_pool) if cfg.model.outer_fine_pool else None
        drop_path_rate = cfg.model.drop_path_rate
        encoder_dim = cfg.model.encoder_dim
        dim_feedforward = cfg.model.dim_feedforward
        num_fusion_layers = cfg.model.num_fusion_layers
    else:
        outer_mode = "finegrid"
        outer_fine_pool = (3, 3)
        drop_path_rate = 0.0
        encoder_dim = 1024
        dim_feedforward = None
        num_fusion_layers = 2

    # Build model
    print(f"[INFO] Building model: outer_mode={outer_mode}, outer_fine_pool={outer_fine_pool}, encoder_dim={encoder_dim}")
    base_regressor = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool,
        drop_path_rate=drop_path_rate,
        encoder_dim=encoder_dim,
        dim_feedforward=dim_feedforward,
        num_fusion_layers=num_fusion_layers,
        sentinel_time=sentinel_time,
    )
    model = XECMultiHeadModel(base_regressor, active_tasks=active_tasks)
    model.to(device)

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[INFO] Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataloader
    print(f"[INFO] Loading validation data from: {args.val_path}")
    val_loader = get_dataloader(
        args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        npho_scale=npho_scale,
        npho_scale2=npho_scale2,
        time_scale=time_scale,
        time_shift=time_shift,
        sentinel_time=sentinel_time,
        load_truth_branches=True,
    )

    # Build task_weights dict
    task_weights = {task: 1.0 for task in active_tasks}

    # Run validation
    print("[INFO] Running validation...")
    t0 = time.time()
    val_metrics, angle_pred, angle_true, extra_info, val_stats = run_epoch_stream(
        model, None, device, val_loader,
        scaler=None,
        train=False,
        amp=False,
        task_weights=task_weights,
        reweighter=None,
        channel_dropout_rate=0.0,
        grad_clip=0.0,
    )
    elapsed = time.time() - t0
    print(f"[INFO] Validation completed in {elapsed:.1f}s")

    # Print metrics
    print("\n=== Validation Metrics ===")
    for k, v in sorted(val_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # Get root_data
    root_data = extra_info.get("root_data", {}) if extra_info else {}

    # Check uvw data
    print("\n=== Data Collection Check ===")
    for key in ["true_u", "true_v", "true_w", "pred_u", "pred_v", "pred_w"]:
        arr = root_data.get(key, np.array([]))
        print(f"  {key}: {arr.shape if hasattr(arr, 'shape') else len(arr)} entries")

    # Output directory
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)

    # Save artifacts
    print(f"\n[INFO] Saving artifacts to: {output_dir}")
    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_validation_artifacts(
        model=model,
        angle_pred=angle_pred,
        angle_true=angle_true,
        root_data=root_data,
        active_tasks=active_tasks,
        artifact_dir=output_dir,
        run_name=run_name,
        epoch=None,
        worst_events=extra_info.get("worst_events", []) if extra_info else [],
    )

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
