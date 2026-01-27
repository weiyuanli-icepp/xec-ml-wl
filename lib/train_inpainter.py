"""
Training script for dead channel inpainting.

Usage:
    # Config mode
    python -m lib.train_inpainter --config config/inpainter_config.yaml

    # CLI mode
    python -m lib.train_inpainter --train_root /path/train.root --val_root /path/val.root \
        --mae_checkpoint artifacts/mae/checkpoint_best.pth --save_path artifacts/inpainter
"""

import os
import warnings
import logging

# Set MLflow tracking URI before importing mlflow
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"

logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.store.db.utils").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)
logging.getLogger("alembic.runtime.migration").setLevel(logging.WARNING)

import torch
import argparse
import time
import glob
import mlflow
import numpy as np

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .model import XECEncoder
from .model_inpainter import XEC_Inpainter
from .engine_inpainter import (
    run_epoch_inpainter,
    run_eval_inpainter,
    save_predictions_to_root,
    RootPredictionWriter,
)
from .utils import log_system_metrics_to_mlflow, validate_data_paths, check_artifact_directory
from .geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE
)
from .config import load_inpainter_config

# Enable TensorFloat32
torch.set_float32_matmul_precision('high')

# Disable debugging overhead
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)


def load_mae_encoder(checkpoint_path: str, device: torch.device, outer_mode: str = "finegrid", outer_fine_pool=None):
    """
    Load encoder weights from MAE checkpoint.

    Args:
        checkpoint_path: path to MAE checkpoint
        device: torch device
        outer_mode: encoder outer mode
        outer_fine_pool: encoder outer fine pool config

    Returns:
        encoder: XECEncoder with loaded weights
    """
    print(f"[INFO] Loading MAE encoder from {checkpoint_path}")

    # Create encoder
    outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
    encoder = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool_tuple
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        # Full MAE checkpoint
        if "model_state_dict" in checkpoint:
            mae_state = checkpoint["model_state_dict"]
        elif "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
            mae_state = checkpoint["ema_state_dict"]
            # Remove "module." prefix if present (from EMA wrapper)
            mae_state = {k.replace("module.", ""): v for k, v in mae_state.items()}
        else:
            # Assume raw state dict
            mae_state = checkpoint
    else:
        mae_state = checkpoint

    # Extract encoder weights (they start with "encoder.")
    encoder_state = {}
    for key, value in mae_state.items():
        if key.startswith("encoder."):
            encoder_state[key[8:]] = value  # Remove "encoder." prefix

    if not encoder_state:
        # Maybe it's a raw encoder checkpoint
        encoder_state = mae_state

    # Load weights
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"[WARN] Missing keys in encoder: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in encoder: {unexpected}")

    encoder = encoder.to(device)
    print(f"[INFO] Encoder loaded successfully ({sum(p.numel() for p in encoder.parameters())} params)")

    return encoder


def main():
    parser = argparse.ArgumentParser(
        description="Train dead channel inpainting model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Config mode
  python -m lib.train_inpainter --config config/inpainter_config.yaml

  # CLI mode
  python -m lib.train_inpainter --train_root /path/train.root --val_root /path/val.root \\
      --mae_checkpoint artifacts/mae/checkpoint_best.pth --save_path artifacts/inpainter

  # Config + CLI override
  python -m lib.train_inpainter --config config/inpainter_config.yaml --epochs 50 --lr 5e-4
        """
    )

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data paths
    parser.add_argument("--train_root", type=str, default=None, help="Path to training ROOT file(s)")
    parser.add_argument("--val_root", type=str, default=None, help="Path to validation ROOT file(s)")
    parser.add_argument("--mae_checkpoint", type=str, default=None, help="Path to MAE checkpoint for encoder")
    parser.add_argument("--save_path", type=str, default=None, help="Directory to save checkpoints")

    # Training params
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=None)
    parser.add_argument("--npho_branch", type=str, default=None, help="Input branch for photon counts")
    parser.add_argument("--time_branch", type=str, default=None, help="Input branch for timing")

    # Normalization
    parser.add_argument("--npho_scale", type=float, default=None)
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    # Model
    parser.add_argument("--outer_mode", type=str, default=None, choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool", type=int, nargs=2, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None, help="Mask ratio for training (default 0.05)")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder (default)")
    parser.add_argument("--finetune_encoder", action="store_true", help="Fine-tune encoder (not frozen)")

    # Optimizer
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    # Loss
    parser.add_argument("--loss_fn", type=str, default=None, choices=["smooth_l1", "mse", "l1", "huber"])
    parser.add_argument("--npho_weight", type=float, default=None)
    parser.add_argument("--time_weight", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--disable_mae_rmse_metrics", action="store_true", help="Skip MAE/RMSE metric computation for speed")
    parser.add_argument("--grad_accum_steps", type=int, default=None, help="Number of gradient accumulation steps")
    parser.add_argument("--time_mask_ratio_scale", type=float, default=None, help="Scale factor for masking valid-time sensors (1.0=uniform)")
    parser.add_argument("--npho_threshold", type=float, default=None, help="Npho threshold for conditional time loss (raw scale)")
    parser.add_argument("--use_npho_time_weight", action="store_true", help="Weight time loss by sqrt(npho)")
    parser.add_argument("--no_npho_time_weight", action="store_true", help="Disable npho time weighting")
    parser.add_argument("--profile", action="store_true", help="Enable training profiler to identify bottlenecks")

    # MLflow
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    # Checkpoint
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from inpainter checkpoint")
    parser.add_argument("--save_interval", type=int, default=None)

    args = parser.parse_args()

    # Load config or use CLI defaults
    if args.config:
        cfg = load_inpainter_config(args.config)

        # Config values with CLI overrides
        train_root = args.train_root or cfg.data.train_path
        val_root = args.val_root or cfg.data.val_path
        mae_checkpoint = args.mae_checkpoint or cfg.training.mae_checkpoint
        save_path = args.save_path or cfg.checkpoint.save_dir

        epochs = args.epochs if args.epochs is not None else cfg.training.epochs
        batch_size = args.batch_size if args.batch_size is not None else cfg.data.batch_size
        chunksize = args.chunksize if args.chunksize is not None else cfg.data.chunksize
        num_workers = args.num_workers if args.num_workers is not None else cfg.data.num_workers
        num_threads = args.num_threads if args.num_threads is not None else cfg.data.num_threads
        npho_branch = args.npho_branch or getattr(cfg.data, "npho_branch", "relative_npho")
        time_branch = args.time_branch or getattr(cfg.data, "time_branch", "relative_time")

        npho_scale = float(args.npho_scale if args.npho_scale is not None else cfg.normalization.npho_scale)
        npho_scale2 = float(args.npho_scale2 if args.npho_scale2 is not None else cfg.normalization.npho_scale2)
        time_scale = float(args.time_scale if args.time_scale is not None else cfg.normalization.time_scale)
        time_shift = float(args.time_shift if args.time_shift is not None else cfg.normalization.time_shift)
        sentinel_value = float(args.sentinel_value if args.sentinel_value is not None else cfg.normalization.sentinel_value)

        outer_mode = args.outer_mode or cfg.model.outer_mode
        outer_fine_pool = args.outer_fine_pool or cfg.model.outer_fine_pool
        mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.model.mask_ratio
        time_mask_ratio_scale = args.time_mask_ratio_scale if args.time_mask_ratio_scale is not None else getattr(cfg.model, "time_mask_ratio_scale", 1.0)
        freeze_encoder = cfg.model.freeze_encoder if not args.finetune_encoder else False

        lr = args.lr if args.lr is not None else cfg.training.lr
        lr_scheduler = args.lr_scheduler or getattr(cfg.training, "lr_scheduler", None)
        lr_min = args.lr_min if args.lr_min is not None else getattr(cfg.training, "lr_min", 1e-6)
        warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else getattr(cfg.training, "warmup_epochs", 0)
        weight_decay = args.weight_decay if args.weight_decay is not None else cfg.training.weight_decay

        loss_fn = args.loss_fn or cfg.training.loss_fn
        npho_weight = args.npho_weight if args.npho_weight is not None else cfg.training.npho_weight
        time_weight = args.time_weight if args.time_weight is not None else cfg.training.time_weight
        npho_threshold = args.npho_threshold if args.npho_threshold is not None else getattr(cfg.training, "npho_threshold", None)
        use_npho_time_weight = not args.no_npho_time_weight and getattr(cfg.training, "use_npho_time_weight", True)
        grad_clip = args.grad_clip if args.grad_clip is not None else cfg.training.grad_clip
        # If --disable_mae_rmse_metrics flag is passed, disable; otherwise use config value
        track_mae_rmse = False if args.disable_mae_rmse_metrics else getattr(cfg.training, "track_mae_rmse", True)
        # Read save_predictions from checkpoint (new location) or training (old location) for backward compat
        save_root_predictions = getattr(cfg.checkpoint, "save_predictions", None)
        if save_root_predictions is None:
            save_root_predictions = getattr(cfg.training, "save_root_predictions", True)
        grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else getattr(cfg.training, "grad_accum_steps", 1)
        track_train_metrics = getattr(cfg.training, "track_train_metrics", True)
        profile = args.profile

        mlflow_experiment = args.mlflow_experiment or cfg.mlflow.experiment
        mlflow_run_name = args.mlflow_run_name or cfg.mlflow.run_name
        resume_from = args.resume_from or cfg.checkpoint.resume_from
        save_interval = args.save_interval if args.save_interval is not None else cfg.checkpoint.save_interval

    else:
        # CLI defaults (no config file)
        train_root = args.train_root
        val_root = args.val_root
        mae_checkpoint = args.mae_checkpoint
        save_path = args.save_path or "artifacts/inpainter"

        if not train_root:
            raise ValueError("--train_root is required (or use --config)")

        epochs = args.epochs or 50
        batch_size = args.batch_size or 1024
        chunksize = args.chunksize or 256000
        num_workers = args.num_workers or 1
        num_threads = args.num_threads or 4
        npho_branch = args.npho_branch or "relative_npho"
        time_branch = args.time_branch or "relative_time"

        npho_scale = args.npho_scale or DEFAULT_NPHO_SCALE
        npho_scale2 = args.npho_scale2 or DEFAULT_NPHO_SCALE2
        time_scale = args.time_scale or DEFAULT_TIME_SCALE
        time_shift = args.time_shift or DEFAULT_TIME_SHIFT
        sentinel_value = args.sentinel_value or DEFAULT_SENTINEL_VALUE

        outer_mode = args.outer_mode or "finegrid"
        outer_fine_pool = args.outer_fine_pool or [3, 3]
        mask_ratio = args.mask_ratio or 0.05
        time_mask_ratio_scale = args.time_mask_ratio_scale or 1.0
        freeze_encoder = not args.finetune_encoder  # Default: frozen

        lr = args.lr or 1e-4
        lr_scheduler = args.lr_scheduler
        lr_min = args.lr_min or 1e-6
        warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else 0
        weight_decay = args.weight_decay or 1e-4

        loss_fn = args.loss_fn or "smooth_l1"
        npho_weight = args.npho_weight or 1.0
        time_weight = args.time_weight or 1.0
        npho_threshold = args.npho_threshold  # None uses DEFAULT_NPHO_THRESHOLD
        use_npho_time_weight = not args.no_npho_time_weight
        grad_clip = args.grad_clip or 1.0
        track_mae_rmse = not bool(args.disable_mae_rmse_metrics)
        save_root_predictions = True
        grad_accum_steps = args.grad_accum_steps or 1
        track_train_metrics = True
        profile = args.profile

        mlflow_experiment = args.mlflow_experiment or "inpainting"
        mlflow_run_name = args.mlflow_run_name
        resume_from = args.resume_from
        save_interval = args.save_interval or 10

    if lr_scheduler == "none":
        lr_scheduler = None

    # Validate required params
    if not train_root:
        raise ValueError("train_path must be specified in config or --train_root")

    # Expand paths
    def expand_path(p):
        path = os.path.expanduser(p)
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.root"))
            if not files:
                raise ValueError(f"No ROOT files found in directory {path}")
            return sorted(files)
        return [path]

    train_files = expand_path(train_root)
    val_files = expand_path(val_root) if val_root else []

    print(f"[INFO] Training files: {len(train_files)}")
    print(f"[INFO] Validation files: {len(val_files)}")

    # Validate data paths exist
    validate_data_paths(train_root, val_root, expand_func=expand_path)

    # Check artifact directory for existing files
    check_artifact_directory(save_path)

    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
    print(f"[INFO] Using device: {device}")

    # Load encoder from MAE checkpoint or create from scratch
    if mae_checkpoint:
        encoder = load_mae_encoder(
            mae_checkpoint, device,
            outer_mode=outer_mode,
            outer_fine_pool=outer_fine_pool
        )
    else:
        print("[INFO] No MAE checkpoint provided, initializing encoder from scratch")
        outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
        encoder = XECEncoder(
            outer_mode=outer_mode,
            outer_fine_pool=outer_fine_pool_tuple
        ).to(device)

    # Create inpainter model
    model = XEC_Inpainter(
        encoder, freeze_encoder=freeze_encoder, sentinel_value=sentinel_value,
        time_mask_ratio_scale=time_mask_ratio_scale
    ).to(device)

    print(f"[INFO] Inpainter created:")
    print(f"  - Total params: {model.get_num_total_params():,}")
    print(f"  - Trainable params: {model.get_num_trainable_params():,}")
    print(f"  - Encoder frozen: {freeze_encoder}")

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    # Scheduler
    scheduler = None
    if lr_scheduler == "cosine":
        if warmup_epochs >= epochs:
            print(f"[WARN] warmup_epochs ({warmup_epochs}) >= epochs ({epochs}); disabling warmup.")
            warmup_epochs = 0
        if warmup_epochs > 0:
            main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min)
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs],
            )
            print(f"[INFO] Using CosineAnnealingLR with {warmup_epochs} warmup epochs (eta_min={lr_min})")
        else:
            scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
            print(f"[INFO] Using CosineAnnealingLR with eta_min={lr_min}")

    # AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    mlflow_run_id = None
    if resume_from and os.path.exists(resume_from):
        print(f"[INFO] Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Note: scheduler state is intentionally NOT restored to allow
            # configuring new epochs/lr_max/lr_min on resume
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            mlflow_run_id = checkpoint.get('mlflow_run_id', None)
            print(f"[INFO] Resumed from epoch {start_epoch}")
        else:
            model.load_state_dict(checkpoint, strict=False)

    # MLflow setup
    mlflow.set_experiment(mlflow_experiment)
    os.makedirs(save_path, exist_ok=True)

    print(f"[INFO] Starting inpainter training")
    print(f"  - Experiment: {mlflow_experiment}")
    print(f"  - Run name: {mlflow_run_name}")
    print(f"  - Mask ratio: {mask_ratio}")

    with mlflow.start_run(run_id=mlflow_run_id, run_name=mlflow_run_name if not mlflow_run_id else None) as run:
        mlflow_run_id = run.info.run_id
        # Log parameters
        mlflow.log_params({
            "train_root": train_root,
            "val_root": val_root,
            "mae_checkpoint": mae_checkpoint,
            "save_path": save_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "chunksize": chunksize,
            "mask_ratio": mask_ratio,
            "freeze_encoder": freeze_encoder,
            "lr": lr,
            "lr_scheduler": lr_scheduler,
            "warmup_epochs": warmup_epochs,
            "weight_decay": weight_decay,
            "loss_fn": loss_fn,
            "npho_weight": npho_weight,
            "time_weight": time_weight,
            "grad_clip": grad_clip,
            "outer_mode": outer_mode,
            "trainable_params": model.get_num_trainable_params(),
            "total_params": model.get_num_total_params(),
        })

        # Training loop
        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            # Train
            train_metrics = run_epoch_inpainter(
                model, optimizer, device,
                train_files, "tree",
                batch_size=batch_size,
                step_size=chunksize,
                mask_ratio=mask_ratio,
                npho_branch=npho_branch,
                time_branch=time_branch,
                npho_scale=float(npho_scale),
                npho_scale2=float(npho_scale2),
                time_scale=float(time_scale),
                time_shift=float(time_shift),
                sentinel_value=float(sentinel_value),
                loss_fn=loss_fn,
                npho_weight=npho_weight,
                time_weight=time_weight,
                grad_clip=grad_clip,
                scaler=scaler,
                track_mae_rmse=track_mae_rmse,
                dataloader_workers=num_workers,
                dataset_workers=num_threads,
                grad_accum_steps=grad_accum_steps,
                track_metrics=track_train_metrics,
                npho_threshold=npho_threshold,
                use_npho_time_weight=use_npho_time_weight,
                profile=profile,
            )

            # Validation
            val_metrics = {}
            if val_files:
                val_metrics = run_eval_inpainter(
                    model, device,
                    val_files, "tree",
                    batch_size=batch_size,
                    step_size=chunksize,
                    mask_ratio=mask_ratio,
                    npho_branch=npho_branch,
                    time_branch=time_branch,
                    npho_scale=float(npho_scale),
                    npho_scale2=float(npho_scale2),
                    time_scale=float(time_scale),
                    time_shift=float(time_shift),
                    sentinel_value=float(sentinel_value),
                    loss_fn=loss_fn,
                    npho_weight=npho_weight,
                    time_weight=time_weight,
                    track_mae_rmse=track_mae_rmse,
                    dataloader_workers=num_workers,
                    dataset_workers=num_threads,
                    npho_threshold=npho_threshold,
                    use_npho_time_weight=use_npho_time_weight,
                    profile=profile,
                )

            dt = time.time() - t0

            # Log metrics
            train_loss = train_metrics.get("total_loss", 0.0)
            val_loss = val_metrics.get("total_loss", 0.0) if val_metrics else 0.0

            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"Time: {dt:.1f}s")

            # MLflow logging
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train/{key}", value, step=epoch)

            if val_metrics:
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val/{key}", value, step=epoch)

            # Learning rate
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            # System metrics (standardized)
            log_system_metrics_to_mlflow(
                step=epoch,
                device=device,
                epoch_time_sec=dt,
                lr=current_lr,
            )

            # Check best model
            is_best = val_loss < best_val_loss if val_metrics else False
            if is_best:
                best_val_loss = val_loss

            # Save checkpoint
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs or is_best:
                t_ckpt_start = time.time()
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'mlflow_run_id': mlflow_run_id,
                    'config': {
                        'outer_mode': outer_mode,
                        'outer_fine_pool': outer_fine_pool,
                        'mask_ratio': mask_ratio,
                        'freeze_encoder': freeze_encoder,
                    }
                }
                if scheduler is not None:
                    checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()

                # Save last
                ckpt_path = os.path.join(save_path, "inpainter_checkpoint_last.pth")
                torch.save(checkpoint_dict, ckpt_path)
                t_ckpt_elapsed = time.time() - t_ckpt_start
                print(f"  Saved checkpoint to {ckpt_path} ({t_ckpt_elapsed:.1f}s)")

                # Save best
                if is_best:
                    t_best_start = time.time()
                    best_path = os.path.join(save_path, "inpainter_checkpoint_best.pth")
                    torch.save(checkpoint_dict, best_path)
                    t_best_elapsed = time.time() - t_best_start
                    print(f"  Saved best checkpoint to {best_path} ({t_best_elapsed:.1f}s)")

            # Save ROOT predictions every 10 epochs (and at end)
            root_save_interval = 10
            if save_root_predictions and val_files and ((epoch + 1) % root_save_interval == 0 or (epoch + 1) == epochs):
                t_root_start = time.time()
                print(f"  Collecting predictions for ROOT output...")
                with RootPredictionWriter(
                    save_path, epoch + 1, run_id=mlflow_run_id,
                    npho_scale=float(npho_scale),
                    npho_scale2=float(npho_scale2),
                    time_scale=float(time_scale),
                    time_shift=float(time_shift),
                    sentinel_value=float(sentinel_value),
                ) as writer:
                    val_metrics_with_pred, _ = run_eval_inpainter(
                        model, device,
                        val_files, "tree",
                        batch_size=batch_size,
                        step_size=chunksize,
                        mask_ratio=mask_ratio,
                        npho_branch=npho_branch,
                        time_branch=time_branch,
                        npho_scale=float(npho_scale),
                        npho_scale2=float(npho_scale2),
                        time_scale=float(time_scale),
                        time_shift=float(time_shift),
                        sentinel_value=float(sentinel_value),
                        loss_fn=loss_fn,
                        npho_weight=npho_weight,
                        time_weight=time_weight,
                        collect_predictions=True,
                        prediction_writer=writer.write,
                        track_mae_rmse=track_mae_rmse,
                        dataloader_workers=num_workers,
                        dataset_workers=num_threads,
                        npho_threshold=npho_threshold,
                        use_npho_time_weight=use_npho_time_weight,
                        profile=profile,
                    )
                root_path = writer.filepath if writer.count > 0 else None
                t_root_elapsed = time.time() - t_root_start
                if root_path:
                    print(f"  Saved predictions to {root_path} ({t_root_elapsed:.1f}s)")
                    mlflow.log_artifact(root_path)

        print(f"\n[INFO] Training complete!")
        print(f"  Best validation loss: {best_val_loss:.6f}")
        print(f"  Checkpoints saved to: {save_path}")


if __name__ == "__main__":
    main()
