#!/usr/bin/env python3
"""
Run validation-only with an existing checkpoint (.pth) or ONNX model (.onnx).

This script loads a model and runs validation to generate predictions
and resolution plots without training.

Usage:
    # PyTorch checkpoint (config embedded or from file)
    python macro/validate_regressor.py checkpoint.pth --val_path data/val/

    # ONNX model (requires --config for normalization params)
    python macro/validate_regressor.py model.onnx --val_path data/cex/ --config config/reg/scan/step3b_model_large.yaml --tasks energy

    # Specify tasks explicitly
    python macro/validate_regressor.py checkpoint.pth --val_path data/val/ --tasks energy uvwFI
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import get_dataloader
from lib.config import load_config
from lib.train_regressor import save_validation_artifacts
from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME,
)


def _run_pytorch(args, cfg, active_tasks, norm_params):
    """Run validation with a PyTorch checkpoint."""
    from lib.models import XECEncoder, XECMultiHeadModel
    from lib.engines import run_epoch_stream

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

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
        sentinel_time=norm_params["sentinel_time"],
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
        **norm_params,
        load_truth_branches=True,
    )

    # Run validation
    task_weights = {task: 1.0 for task in active_tasks}
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

    root_data = extra_info.get("root_data", {}) if extra_info else {}
    worst_events = extra_info.get("worst_events", []) if extra_info else []

    return model, root_data, angle_pred, angle_true, worst_events


def _run_onnx(args, cfg, active_tasks, norm_params):
    """Run validation with an ONNX model using onnxruntime."""
    import onnxruntime as ort

    print(f"[INFO] Loading ONNX model: {args.checkpoint}")
    sess = ort.InferenceSession(args.checkpoint)

    # Inspect model I/O
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()
    print(f"[INFO] ONNX inputs:  {[(i.name, i.shape) for i in input_info]}")
    print(f"[INFO] ONNX outputs: {[(o.name, o.shape) for o in output_info]}")

    input_name = input_info[0].name
    output_names = [o.name for o in output_info]

    # Create dataloader
    print(f"[INFO] Loading validation data from: {args.val_path}")
    val_loader = get_dataloader(
        args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        **norm_params,
        load_truth_branches=True,
    )

    # Inference loop
    root_data = {
        "run_id": [], "event_id": [],
        "pred_theta": [], "pred_phi": [],
        "true_theta": [], "true_phi": [],
        "opening_angle": [],
        "pred_energy": [], "true_energy": [],
        "pred_timing": [], "true_timing": [],
        "pred_u": [], "pred_v": [], "pred_w": [],
        "true_u": [], "true_v": [], "true_w": [],
        "pos_angle": [],
        "x_truth": [], "y_truth": [], "z_truth": [],
        "x_vtx": [], "y_vtx": [], "z_vtx": [],
    }

    print("[INFO] Running ONNX inference...")
    t0 = time.time()
    n_events = 0
    n_batches = 0

    for input_tensor, target_dict in val_loader:
        # input_tensor: (batch, 4760, 2)
        x_np = input_tensor.numpy().astype(np.float32)
        outputs = sess.run(output_names, {input_name: x_np})

        # Map outputs to tasks
        output_map = dict(zip(output_names, outputs))

        batch_size = x_np.shape[0]
        n_events += batch_size
        n_batches += 1

        if n_batches % 50 == 1:
            print(f"  Batch {n_batches}: {n_events} events processed")

        # Collect run/event metadata
        if "run" in target_dict:
            root_data["run_id"].append(target_dict["run"].numpy())
        if "event" in target_dict:
            root_data["event_id"].append(target_dict["event"].numpy())

        # Energy task
        if "energy" in active_tasks:
            out_key = "output_energy"
            if out_key in output_map:
                pred = output_map[out_key].flatten()
                root_data["pred_energy"].append(pred)
            true_e = target_dict.get("energy")
            if true_e is not None:
                root_data["true_energy"].append(true_e.numpy().flatten())

        # Angle task
        if "angle" in active_tasks:
            out_key = "output_angle"
            if out_key in output_map:
                pred = output_map[out_key]
                root_data["pred_theta"].append(pred[:, 0])
                root_data["pred_phi"].append(pred[:, 1])
            true_a = target_dict.get("angle")
            if true_a is not None:
                a = true_a.numpy()
                root_data["true_theta"].append(a[:, 0])
                root_data["true_phi"].append(a[:, 1])

        # Timing task
        if "timing" in active_tasks:
            out_key = "output_timing"
            if out_key in output_map:
                pred = output_map[out_key].flatten()
                root_data["pred_timing"].append(pred)
            true_t = target_dict.get("timing")
            if true_t is not None:
                root_data["true_timing"].append(true_t.numpy().flatten())

        # Position task (uvwFI)
        if "uvwFI" in active_tasks:
            out_key = "output_uvwFI"
            if out_key in output_map:
                pred = output_map[out_key]
                root_data["pred_u"].append(pred[:, 0])
                root_data["pred_v"].append(pred[:, 1])
                root_data["pred_w"].append(pred[:, 2])
            true_uvw = target_dict.get("uvwFI")
            if true_uvw is not None:
                u = true_uvw.numpy()
                root_data["true_u"].append(u[:, 0])
                root_data["true_v"].append(u[:, 1])
                root_data["true_w"].append(u[:, 2])

    elapsed = time.time() - t0
    print(f"[INFO] ONNX inference completed: {n_events} events in {elapsed:.1f}s "
          f"({n_events/elapsed:.0f} events/s)")

    # Concatenate all arrays
    for k, v_list in root_data.items():
        if v_list:
            root_data[k] = np.concatenate(v_list, axis=0)
        else:
            root_data[k] = np.array([])

    # Print basic metrics for energy task
    if "energy" in active_tasks:
        pred_e = root_data.get("pred_energy", np.array([]))
        true_e = root_data.get("true_energy", np.array([]))
        if pred_e.size > 0 and true_e.size > 0:
            # Filter out sentinel truth values (1e10 = unavailable)
            valid = true_e < 1e9
            if valid.sum() > 0:
                err = pred_e[valid] - true_e[valid]
                print(f"\n=== Energy Metrics ({valid.sum()} events with truth) ===")
                print(f"  MAE:  {np.mean(np.abs(err))*1e3:.2f} MeV")
                print(f"  RMSE: {np.sqrt(np.mean(err**2))*1e3:.2f} MeV")
                print(f"  Bias: {np.mean(err)*1e3:+.2f} MeV")
            else:
                print(f"\n=== Energy: {pred_e.size} predictions, no truth available ===")
                print(f"  Pred mean: {np.mean(pred_e)*1e3:.2f} MeV")
                print(f"  Pred std:  {np.std(pred_e)*1e3:.2f} MeV")

    # Build angle arrays for save_validation_artifacts
    angle_pred = None
    angle_true = None
    if "angle" in active_tasks:
        pt = root_data.get("pred_theta", np.array([]))
        pp = root_data.get("pred_phi", np.array([]))
        if pt.size > 0 and pp.size > 0:
            angle_pred = np.stack([pt, pp], axis=1)
            angle_true = np.stack([
                root_data.get("true_theta", np.zeros_like(pt)),
                root_data.get("true_phi", np.zeros_like(pp))
            ], axis=1)

    return None, root_data, angle_pred, angle_true, []


def main():
    parser = argparse.ArgumentParser(
        description="Run validation-only with an existing checkpoint or ONNX model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint (.pth) or ONNX model (.onnx)")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation data")
    parser.add_argument("--config", type=str, default=None, help="Config file (required for ONNX, optional for .pth)")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: same as checkpoint)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Tasks to evaluate (default: from checkpoint)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    # Expand tilde in checkpoint path
    args.checkpoint = os.path.expanduser(args.checkpoint)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model not found: {args.checkpoint}")
        sys.exit(1)

    is_onnx = args.checkpoint.endswith(".onnx")
    device = torch.device(args.device)
    print(f"[INFO] Model format: {'ONNX' if is_onnx else 'PyTorch'}")
    print(f"[INFO] Using device: {'cpu (ONNX)' if is_onnx else device}")

    # --- Load config ---
    cfg = None
    if not is_onnx:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            print("[INFO] Using config from checkpoint")
        elif args.config:
            cfg = load_config(args.config)
            print(f"[INFO] Using config from: {args.config}")
        else:
            print("[INFO] No config found, using defaults")
    else:
        if args.config:
            cfg = load_config(args.config)
            print(f"[INFO] Using config from: {args.config}")
        else:
            print("[WARN] No --config provided for ONNX model. Using default normalization.")
            print("       This is likely wrong! Specify --config for correct normalization.")

    # --- Determine active tasks ---
    if args.tasks:
        active_tasks = args.tasks
    elif cfg and hasattr(cfg, 'tasks'):
        active_tasks = [t for t, tc in cfg.tasks.items() if tc.enabled]
    elif not is_onnx:
        # Infer from PyTorch checkpoint keys
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
    else:
        active_tasks = ["energy"]
        print("[INFO] No task info, defaulting to ['energy']")

    print(f"[INFO] Active tasks: {active_tasks}")

    # --- Normalization params ---
    if cfg:
        norm_params = {
            "npho_scale": cfg.normalization.npho_scale,
            "npho_scale2": cfg.normalization.npho_scale2,
            "time_scale": cfg.normalization.time_scale,
            "time_shift": cfg.normalization.time_shift,
            "sentinel_time": cfg.normalization.sentinel_time,
        }
        if hasattr(cfg.normalization, 'npho_scheme'):
            norm_params["npho_scheme"] = cfg.normalization.npho_scheme
        if hasattr(cfg.normalization, 'sentinel_npho'):
            norm_params["sentinel_npho"] = cfg.normalization.sentinel_npho
    else:
        norm_params = {
            "npho_scale": DEFAULT_NPHO_SCALE,
            "npho_scale2": DEFAULT_NPHO_SCALE2,
            "time_scale": DEFAULT_TIME_SCALE,
            "time_shift": DEFAULT_TIME_SHIFT,
            "sentinel_time": DEFAULT_SENTINEL_TIME,
        }

    print(f"[INFO] Normalization: npho_scheme={norm_params.get('npho_scheme', 'log1p')}, "
          f"npho_scale={norm_params.get('npho_scale')}, "
          f"npho_scale2={norm_params.get('npho_scale2')}")

    # --- Run inference ---
    if is_onnx:
        model, root_data, angle_pred, angle_true, worst_events = \
            _run_onnx(args, cfg, active_tasks, norm_params)
    else:
        model, root_data, angle_pred, angle_true, worst_events = \
            _run_pytorch(args, cfg, active_tasks, norm_params)

    # --- Check collected data ---
    print("\n=== Data Collection Check ===")
    for key in ["true_energy", "pred_energy", "true_u", "true_v", "true_w", "pred_u", "pred_v", "pred_w"]:
        arr = root_data.get(key, np.array([]))
        print(f"  {key}: {arr.shape if hasattr(arr, 'shape') else len(arr)} entries")

    # --- Save artifacts ---
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)

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
        worst_events=worst_events,
    )

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
