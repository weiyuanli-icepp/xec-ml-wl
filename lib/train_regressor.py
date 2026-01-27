#!/usr/bin/env python3
"""
XEC Regressor Training Script

Config-based training for XEC multi-task regression model.
Usage:
    python -m lib.train_regressor --config config/train_config.yaml
"""

import os
import time
import tempfile
import numpy as np
import pandas as pd
import warnings
import yaml
import mlflow
import mlflow.pytorch
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.tracking._tracking_service.utils")
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel

from .models import XECEncoder, XECMultiHeadModel, AutomaticLossScaler
from .engines import run_epoch_stream
from .dataset import get_dataloader, expand_path
from .reweighting import create_reweighter_from_config
from .config import load_config, get_active_tasks, get_task_weights
from .utils import (
    count_model_params,
    log_system_metrics_to_mlflow,
    validate_data_paths,
    check_artifact_directory
)
from .plotting import (
    plot_resolution_profile,
    plot_energy_resolution_profile,
    plot_timing_resolution_profile,
    plot_position_resolution_profile
)

# ------------------------------------------------------------
# Enable TensorFloat32
torch.set_float32_matmul_precision('high')

# Disable Debugging/Profiling overhead
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)


# ------------------------------------------------------------
#  Config-based Training Entry (Main)
# ------------------------------------------------------------
def train_with_config(config_path: str, profile: bool = None):
    """
    Train XEC regressor using YAML config file.

    Args:
        config_path: Path to YAML configuration file.
        profile: Enable training profiler (overrides config if set).
    """
    # Load configuration
    cfg = load_config(config_path)

    # Profile: CLI override takes precedence, otherwise use config
    enable_profile = profile if profile is not None else getattr(cfg.training, 'profile', False)
    if enable_profile:
        print("[INFO] Training profiler enabled - timing breakdown will be shown per epoch.")

    # Get active tasks and their weights
    active_tasks = get_active_tasks(cfg)
    task_weights = get_task_weights(cfg)

    if not active_tasks:
        raise ValueError("No tasks enabled in config. Enable at least one task.")

    print(f"[INFO] Active tasks: {active_tasks}")
    print(f"[INFO] Task weights: {task_weights}")

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Validate data paths before creating data loaders (use expand_path for consistency)
    validate_data_paths(cfg.data.train_path, cfg.data.val_path, expand_func=expand_path)

    # Check artifact directory for existing files
    run_name = cfg.mlflow.run_name or time.strftime("run_cv2_%Y%m%d_%H%M%S")
    artifact_dir = os.path.abspath(os.path.join(cfg.checkpoint.save_dir, run_name))
    check_artifact_directory(artifact_dir)

    # --- Data Loaders ---
    norm_kwargs = {
        "npho_scale": cfg.normalization.npho_scale,
        "npho_scale2": cfg.normalization.npho_scale2,
        "time_scale": cfg.normalization.time_scale,
        "time_shift": cfg.normalization.time_shift,
        "sentinel_value": cfg.normalization.sentinel_value,
        "step_size": cfg.data.chunksize,
        "npho_branch": getattr(cfg.data, "npho_branch", "relative_npho"),
        "time_branch": getattr(cfg.data, "time_branch", "relative_time"),
    }

    train_loader = get_dataloader(
        cfg.data.train_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_threads=cfg.data.num_threads,
        **norm_kwargs
    )

    val_loader = get_dataloader(
        cfg.data.val_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_threads=cfg.data.num_threads,
        **norm_kwargs
    )

    # --- Model ---
    outer_fine_pool = tuple(cfg.model.outer_fine_pool) if cfg.model.outer_fine_pool else None
    base_regressor = XECEncoder(
        outer_mode=cfg.model.outer_mode,
        outer_fine_pool=outer_fine_pool,
        drop_path_rate=cfg.model.drop_path_rate
    )
    model = XECMultiHeadModel(
        base_regressor,
        active_tasks=active_tasks,
        hidden_dim=cfg.model.hidden_dim
    ).to(device)
    total_params, trainable_params = count_model_params(model)
    print("[INFO] Regressor created:")
    print(f"  - Total params: {total_params:,}")
    print(f"  - Trainable params: {trainable_params:,}")
    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)

    # --- Optimizer ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or "bn" in n.lower() or "norm" in n.lower() else decay).append(p)

    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": cfg.training.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.training.lr,
        fused=True
    )

    # --- EMA ---
    ema_model = None
    ema_decay = cfg.training.ema_decay
    if ema_decay > 0.0:
        print(f"[INFO] Using EMA with decay={ema_decay}")

        def robust_ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return ema_decay * averaged_model_parameter + (1.0 - ema_decay) * model_parameter

        ema_model = AveragedModel(model, avg_fn=robust_ema_avg, use_buffers=True)
        ema_model.to(device)

    # --- Loss Scaler (Auto Balance) ---
    loss_scaler = None
    if cfg.loss_balance == "auto":
        print(f"[INFO] Using Automatic Loss Balancing.")
        loss_scaler = AutomaticLossScaler(active_tasks).to(device)
        optimizer.add_param_group({"params": loss_scaler.parameters()})

    # --- AMP Scaler ---
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.amp and torch.cuda.is_available()))

    # --- Scheduler ---
    scheduler = None
    if cfg.training.use_scheduler:
        print(f"[INFO] Using Cosine Annealing with {cfg.training.warmup_epochs} warmup epochs.")
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs - cfg.training.warmup_epochs,
            eta_min=1e-6
        )
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=cfg.training.warmup_epochs
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[cfg.training.warmup_epochs]
        )

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val = float("inf")
    best_ema_state = None  # Track best EMA state for final evaluation
    mlflow_run_id = None

    if cfg.checkpoint.resume_from and os.path.exists(cfg.checkpoint.resume_from):
        print(f"[INFO] Loading checkpoint: {cfg.checkpoint.resume_from}")
        checkpoint = torch.load(cfg.checkpoint.resume_from, map_location=device, weights_only=False)

        # Determine checkpoint type
        is_full_regressor_checkpoint = False
        is_mae_checkpoint = False
        is_raw_encoder_checkpoint = False

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            has_heads = any(k.startswith("heads.") for k in state_dict.keys())
            has_encoder_prefix = any(k.startswith("encoder.") for k in state_dict.keys())

            if has_heads:
                is_full_regressor_checkpoint = True
            elif has_encoder_prefix:
                is_mae_checkpoint = True
            else:
                is_full_regressor_checkpoint = "optimizer_state_dict" in checkpoint
        else:
            is_raw_encoder_checkpoint = True

        if is_full_regressor_checkpoint:
            print(f"[INFO] Detected full regressor checkpoint. Resuming training state.")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint.get("best_val", float("inf"))
            mlflow_run_id = checkpoint.get("mlflow_run_id", None)

            # Load scheduler state if available
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print(f"[INFO] Restored scheduler state.")

            if ema_model is not None and "ema_state_dict" in checkpoint:
                ema_model.load_state_dict(checkpoint["ema_state_dict"])

            # Restore best EMA state if this was the best checkpoint
            if "best_ema_state" in checkpoint:
                best_ema_state = checkpoint["best_ema_state"]

        elif is_mae_checkpoint:
            print(f"[INFO] Detected MAE checkpoint. Loading encoder for fine-tuning.")
            mae_state = checkpoint["model_state_dict"]

            if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
                print(f"[INFO] Using EMA weights from MAE checkpoint.")
                mae_state = checkpoint["ema_state_dict"]
                mae_state = {k.replace("module.", ""): v for k, v in mae_state.items()}

            encoder_state = {}
            for key, value in mae_state.items():
                if key.startswith("encoder."):
                    new_key = "backbone." + key[8:]
                    encoder_state[new_key] = value

            if encoder_state:
                missing, unexpected = model.load_state_dict(encoder_state, strict=False)
                print(f"[INFO] Loaded {len(encoder_state)} encoder weights from MAE.")
                print(f"[INFO] Missing keys (expected for heads): {len(missing)}")

            if ema_model is not None:
                ema_model.module.load_state_dict(model.state_dict())
                if hasattr(ema_model, 'n_averaged'):
                    ema_model.n_averaged.zero_()

        elif is_raw_encoder_checkpoint:
            print(f"[INFO] Detected raw encoder checkpoint. Loading for fine-tuning.")
            encoder_state = {}
            raw_state = checkpoint if not isinstance(checkpoint, dict) else checkpoint
            for key, value in raw_state.items():
                new_key = "backbone." + key
                encoder_state[new_key] = value

            missing, unexpected = model.load_state_dict(encoder_state, strict=False)
            print(f"[INFO] Loaded {len(encoder_state)} encoder weights.")
            print(f"[INFO] Missing keys (expected for heads): {len(missing)}")

            if ema_model is not None:
                ema_model.module.load_state_dict(model.state_dict())
                if hasattr(ema_model, 'n_averaged'):
                    ema_model.n_averaged.zero_()

    # --- MLflow Setup ---
    mlflow.set_experiment(cfg.mlflow.experiment)
    run_name = cfg.mlflow.run_name or time.strftime("run_%Y%m%d_%H%M%S")

    # --- Reweighting ---
    reweighter = None
    if hasattr(cfg, 'reweighting'):
        # Convert config dataclass to dict for create_reweighter_from_config
        rw_dict = {
            "angle": {
                "enabled": cfg.reweighting.angle.enabled,
                "nbins_2d": list(cfg.reweighting.angle.nbins_2d),
            },
            "energy": {
                "enabled": cfg.reweighting.energy.enabled,
                "nbins": cfg.reweighting.energy.nbins,
            },
            "timing": {
                "enabled": cfg.reweighting.timing.enabled,
                "nbins": cfg.reweighting.timing.nbins,
            },
            "uvwFI": {
                "enabled": cfg.reweighting.uvwFI.enabled,
                "nbins_2d": list(cfg.reweighting.uvwFI.nbins_2d),
            },
        }
        reweighter = create_reweighter_from_config(rw_dict)

        if reweighter.is_enabled:
            # Get training file list for fitting
            train_files = expand_path(cfg.data.train_path)
            reweighter.fit(train_files, cfg.data.tree_name, step_size=cfg.data.chunksize)
        else:
            print("[INFO] No reweighting enabled.")

    # --- Training Loop ---
    with mlflow.start_run(run_id=mlflow_run_id, run_name=run_name if not mlflow_run_id else None) as run:
        mlflow_run_id = run.info.run_id
        artifact_dir = os.path.abspath(os.path.join(cfg.checkpoint.save_dir, run_name))
        os.makedirs(artifact_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

        # Log config
        if start_epoch == 1:
            mlflow.log_params({
                "active_tasks": ",".join(active_tasks),
                "batch_size": cfg.data.batch_size,
                "lr": cfg.training.lr,
                "epochs": cfg.training.epochs,
                "outer_mode": cfg.model.outer_mode,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "loss_balance": cfg.loss_balance,
            })

        best_state = None

        for ep in range(start_epoch, cfg.training.epochs + 1):
            t0 = time.time()

            # === TRAIN ===
            tr_metrics, _, _, _, _ = run_epoch_stream(
                model, optimizer, device, train_loader,
                scaler=scaler,
                train=True,
                amp=cfg.training.amp,
                task_weights=task_weights,
                loss_scaler=loss_scaler,
                reweighter=reweighter,
                channel_dropout_rate=cfg.training.channel_dropout_rate,
                scheduler=scheduler,
                ema_model=ema_model,
                grad_clip=cfg.training.grad_clip,
                grad_accum_steps=getattr(cfg.training, 'grad_accum_steps', 1),
                profile=enable_profile,
            )

            # === VALIDATION ===
            val_model = ema_model if ema_model is not None else model
            val_model.eval()
            val_metrics, pred_val, true_val, extra_info, val_stats = run_epoch_stream(
                val_model, optimizer, device, val_loader,
                scaler=None,
                train=False,
                amp=False,
                task_weights=task_weights,
                reweighter=None,
                channel_dropout_rate=0.0,
                grad_clip=0.0,
            )

            sec = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']

            tr_loss = tr_metrics["total_opt"]
            val_loss = val_metrics["total_opt"]

            print(f"[{ep:03d}] tr_loss {tr_loss:.5f} val_loss {val_loss:.5f} lr {current_lr:.2e} time {sec:.1f}s")

            # --- Logging ---
            log_system_metrics_to_mlflow(
                step=ep,
                device=device,
                epoch_time_sec=sec,
                lr=current_lr,
            )

            log_dict = {
                "train_loss": tr_loss,
                "val_loss": val_loss,
            }

            for metric_key in ["smooth_l1", "l1", "mse", "cos"]:
                if metric_key in val_metrics:
                    log_dict[f"val_{metric_key}"] = val_metrics[metric_key]

            if val_stats:
                log_dict.update(val_stats)

            mlflow.log_metrics(log_dict, step=ep)

            if loss_scaler is not None:
                for task, log_var in loss_scaler.log_vars.items():
                    weight = (0.5 * torch.exp(-log_var)).item()
                    mlflow.log_metrics({
                        f"task/{task}_log_var": log_var.item(),
                        f"task/{task}_weight": weight,
                    }, step=ep)

            writer.add_scalar("loss/train", tr_loss, ep)
            writer.add_scalar("loss/val", val_loss, ep)
            writer.add_scalar("lr", current_lr, ep)

            # --- Checkpointing ---
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
                # Save best EMA state for final evaluation
                if ema_model is not None:
                    best_ema_state = {k: v.detach().clone().cpu() for k, v in ema_model.state_dict().items()}

                checkpoint_data = {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_val": best_val,
                    "mlflow_run_id": mlflow_run_id,
                }
                torch.save(checkpoint_data, os.path.join(artifact_dir, "checkpoint_best.pth"))
                print(f"   [info] New best val_loss: {best_val:.6f}")

            # Save last checkpoint
            checkpoint_data = {
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if ema_model else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val": best_val,
                "mlflow_run_id": mlflow_run_id,
            }
            torch.save(checkpoint_data, os.path.join(artifact_dir, "checkpoint_last.pth"))

        # --- Final Evaluation & Artifacts ---
        # Use best model state for final evaluation and export
        if ema_model is not None and best_ema_state is not None:
            print("[INFO] Loading best EMA state for final evaluation.")
            ema_model.load_state_dict(best_ema_state)
            final_model = ema_model
        elif best_state is not None:
            print("[INFO] Loading best model state for final evaluation.")
            model.load_state_dict(best_state)
            final_model = model
        else:
            final_model = ema_model if ema_model is not None else model

        # Run final validation with best model
        final_model.eval()
        _, angle_pred, angle_true, extra_info, _ = run_epoch_stream(
            final_model, optimizer, device, val_loader,
            scaler=None,
            train=False,
            amp=False,
            task_weights=task_weights,
            reweighter=None,
            channel_dropout_rate=0.0,
            grad_clip=0.0,
        )

        # Get collected validation data
        root_data = extra_info.get("root_data", {}) if extra_info else {}

        # === Save predictions and plots for each active task ===

        # --- Angle Task ---
        if "angle" in active_tasks and angle_pred is not None:
            csv_path = os.path.join(artifact_dir, f"predictions_angle_{run_name}.csv")
            pd.DataFrame({
                "true_theta": angle_true[:, 0], "true_phi": angle_true[:, 1],
                "pred_theta": angle_pred[:, 0], "pred_phi": angle_pred[:, 1]
            }).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

            res_pdf = os.path.join(artifact_dir, f"resolution_angle_{run_name}.pdf")
            plot_resolution_profile(angle_pred, angle_true, outfile=res_pdf)
            mlflow.log_artifact(res_pdf)
            print(f"[INFO] Angle predictions saved to {csv_path}")

        # --- Energy Task ---
        if "energy" in active_tasks:
            pred_energy = root_data.get("pred_energy", np.array([]))
            true_energy = root_data.get("true_energy", np.array([]))
            if pred_energy.size > 0 and true_energy.size > 0:
                csv_path = os.path.join(artifact_dir, f"predictions_energy_{run_name}.csv")
                pd.DataFrame({
                    "true_energy": true_energy,
                    "pred_energy": pred_energy
                }).to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path)

                res_pdf = os.path.join(artifact_dir, f"resolution_energy_{run_name}.pdf")
                plot_energy_resolution_profile(pred_energy, true_energy, outfile=res_pdf)
                mlflow.log_artifact(res_pdf)
                print(f"[INFO] Energy predictions saved to {csv_path}")

        # --- Timing Task ---
        if "timing" in active_tasks:
            pred_timing = root_data.get("pred_timing", np.array([]))
            true_timing = root_data.get("true_timing", np.array([]))
            if pred_timing.size > 0 and true_timing.size > 0:
                csv_path = os.path.join(artifact_dir, f"predictions_timing_{run_name}.csv")
                pd.DataFrame({
                    "true_timing": true_timing,
                    "pred_timing": pred_timing
                }).to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path)

                res_pdf = os.path.join(artifact_dir, f"resolution_timing_{run_name}.pdf")
                plot_timing_resolution_profile(pred_timing, true_timing, outfile=res_pdf)
                mlflow.log_artifact(res_pdf)
                print(f"[INFO] Timing predictions saved to {csv_path}")

        # --- Position Task (uvwFI) ---
        if "uvwFI" in active_tasks:
            pred_u = root_data.get("pred_u", np.array([]))
            pred_v = root_data.get("pred_v", np.array([]))
            pred_w = root_data.get("pred_w", np.array([]))
            true_u = root_data.get("true_u", np.array([]))
            true_v = root_data.get("true_v", np.array([]))
            true_w = root_data.get("true_w", np.array([]))
            if pred_u.size > 0 and true_u.size > 0:
                csv_path = os.path.join(artifact_dir, f"predictions_uvwFI_{run_name}.csv")
                pd.DataFrame({
                    "true_u": true_u, "true_v": true_v, "true_w": true_w,
                    "pred_u": pred_u, "pred_v": pred_v, "pred_w": pred_w
                }).to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path)

                pred_uvw = np.stack([pred_u, pred_v, pred_w], axis=1)
                true_uvw = np.stack([true_u, true_v, true_w], axis=1)
                res_pdf = os.path.join(artifact_dir, f"resolution_uvwFI_{run_name}.pdf")
                plot_position_resolution_profile(pred_uvw, true_uvw, outfile=res_pdf)
                mlflow.log_artifact(res_pdf)
                print(f"[INFO] Position predictions saved to {csv_path}")

        # ONNX export
        if cfg.export.onnx:
            onnx_path = os.path.join(artifact_dir, cfg.export.onnx)
            final_model.eval()
            dummy_input = torch.randn(1, 4760, 2, device=device)
            try:
                torch.onnx.export(
                    final_model, dummy_input, onnx_path,
                    export_params=True, opset_version=20,
                    do_constant_folding=True,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                mlflow.log_artifact(onnx_path)
                print(f"[INFO] ONNX exported to {onnx_path}")
            except Exception as e:
                print(f"[WARN] ONNX export failed: {e}")

        writer.close()

    print(f"[INFO] Training complete. Best val_loss: {best_val:.6f}")
    return best_val


# ------------------------------------------------------------
#  CLI Argument Parser
# ------------------------------------------------------------
def get_parser():
    """Build argument parser with config file + CLI overrides."""
    import argparse
    parser = argparse.ArgumentParser(
        description="Train XEC Multi-Task Regressor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use config file only
  python -m lib.train_regressor --config config/train_config.yaml

  # Override specific values
  python -m lib.train_regressor --config config/train_config.yaml --lr 1e-4 --epochs 30

  # Quick test
  python -m lib.train_regressor --config config/train_config.yaml --train_path /path/train --val_path /path/val
"""
    )

    # Config file (required)
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")

    # Data paths (override config)
    parser.add_argument("--train_path", type=str, default=None, help="Training data path")
    parser.add_argument("--val_path", type=str, default=None, help="Validation data path")
    parser.add_argument("--tree", type=str, default=None, help="ROOT tree name")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--chunksize", type=int, default=None, help="Chunk size for streaming")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--num_threads", type=int, default=None, help="CPU preprocessing threads")

    # Normalization (override config)
    parser.add_argument("--npho_scale", type=float, default=None)
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    # Model (override config)
    parser.add_argument("--outer_mode", type=str, default=None, choices=["finegrid", "split"])
    parser.add_argument("--outer_fine_pool", type=int, nargs=2, default=None, help="Pooling kernel [h, w]")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--drop_path_rate", type=float, default=None)

    # Training (override config)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--channel_dropout_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)

    # Tasks (override config)
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Enable specific tasks")
    parser.add_argument("--loss_balance", type=str, default=None, choices=["manual", "auto"])

    # Checkpoint (override config)
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save artifacts")

    # MLflow (override config)
    parser.add_argument("--mlflow_experiment", type=str, default=None, help="MLflow experiment name")
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")

    # Export (override config)
    parser.add_argument("--onnx", type=str, default=None, help="ONNX filename (or 'null' to disable)")

    # Runtime
    parser.add_argument("--profile", action="store_true", help="Enable training profiler")

    return parser


def apply_cli_overrides(cfg, args):
    """Apply CLI argument overrides to config object."""
    # Data
    if args.train_path is not None:
        cfg.data.train_path = args.train_path
    if args.val_path is not None:
        cfg.data.val_path = args.val_path
    if args.tree is not None:
        cfg.data.tree_name = args.tree
    if args.batch_size is not None:
        cfg.data.batch_size = args.batch_size
    if args.chunksize is not None:
        cfg.data.chunksize = args.chunksize
    if args.num_workers is not None:
        cfg.data.num_workers = args.num_workers
    if args.num_threads is not None:
        cfg.data.num_threads = args.num_threads

    # Normalization
    if args.npho_scale is not None:
        cfg.normalization.npho_scale = args.npho_scale
    if args.npho_scale2 is not None:
        cfg.normalization.npho_scale2 = args.npho_scale2
    if args.time_scale is not None:
        cfg.normalization.time_scale = args.time_scale
    if args.time_shift is not None:
        cfg.normalization.time_shift = args.time_shift
    if args.sentinel_value is not None:
        cfg.normalization.sentinel_value = args.sentinel_value

    # Model
    if args.outer_mode is not None:
        cfg.model.outer_mode = args.outer_mode
    if args.outer_fine_pool is not None:
        cfg.model.outer_fine_pool = list(args.outer_fine_pool)
    if args.hidden_dim is not None:
        cfg.model.hidden_dim = args.hidden_dim
    if args.drop_path_rate is not None:
        cfg.model.drop_path_rate = args.drop_path_rate

    # Training
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.lr = args.lr
    if args.weight_decay is not None:
        cfg.training.weight_decay = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.training.warmup_epochs = args.warmup_epochs
    if args.ema_decay is not None:
        cfg.training.ema_decay = args.ema_decay
    if args.channel_dropout_rate is not None:
        cfg.training.channel_dropout_rate = args.channel_dropout_rate
    if args.grad_clip is not None:
        cfg.training.grad_clip = args.grad_clip
    if args.grad_accum_steps is not None:
        cfg.training.grad_accum_steps = args.grad_accum_steps

    # Tasks
    if args.tasks is not None:
        # Disable all tasks first, then enable specified ones
        from .config import TaskConfig
        for task_name in ["angle", "energy", "timing", "uvwFI"]:
            if task_name not in cfg.tasks:
                cfg.tasks[task_name] = TaskConfig(enabled=False)
            else:
                cfg.tasks[task_name].enabled = False
        for task_name in args.tasks:
            if task_name not in cfg.tasks:
                cfg.tasks[task_name] = TaskConfig(enabled=True)
            else:
                cfg.tasks[task_name].enabled = True

    if args.loss_balance is not None:
        cfg.loss_balance = args.loss_balance

    # Checkpoint
    if args.resume_from is not None:
        cfg.checkpoint.resume_from = args.resume_from
    if args.save_dir is not None:
        cfg.checkpoint.save_dir = args.save_dir

    # MLflow
    if args.mlflow_experiment is not None:
        cfg.mlflow.experiment = args.mlflow_experiment
    if args.run_name is not None:
        cfg.mlflow.run_name = args.run_name

    # Export
    if args.onnx is not None:
        cfg.export.onnx = None if args.onnx.lower() == "null" else args.onnx

    return cfg


# ------------------------------------------------------------
#  CLI Entry Point
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Load config and apply CLI overrides
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Save merged config to temp file for train_with_config
    def config_to_dict(cfg):
        """Convert config dataclass to dict for YAML serialization."""
        result = {}
        for section in ["data", "normalization", "model", "training", "checkpoint", "mlflow", "export"]:
            if hasattr(cfg, section):
                obj = getattr(cfg, section)
                result[section] = {k: v for k, v in obj.__dict__.items()}
        result["loss_balance"] = cfg.loss_balance
        # Tasks
        result["tasks"] = {}
        for task_name, task_cfg in cfg.tasks.items():
            result["tasks"][task_name] = {k: v for k, v in task_cfg.__dict__.items()}
        # Reweighting
        if hasattr(cfg, "reweighting"):
            result["reweighting"] = {}
            for task_name in ["angle", "energy", "timing", "uvwFI"]:
                task_rw = getattr(cfg.reweighting, task_name)
                result["reweighting"][task_name] = {k: v for k, v in task_rw.__dict__.items()}
        return result

    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_to_dict(cfg), f, default_flow_style=False)
        temp_config_path = f.name

    try:
        train_with_config(temp_config_path, profile=args.profile)
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
