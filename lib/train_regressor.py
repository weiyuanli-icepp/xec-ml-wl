#!/usr/bin/env python3
"""
XEC Regressor Training Script

Config-based training for XEC multi-task regression model.
Usage:
    python -m lib.train_regressor --config config/train_config.yaml
"""

# Suppress Triton autotuning verbose output (must be set BEFORE importing torch)
import os
os.environ.setdefault("TORCHINDUCTOR_LOG_LEVEL", "WARNING")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")
import time
import tempfile
from contextlib import nullcontext
import numpy as np
import pandas as pd
import warnings
import yaml
import mlflow
import mlflow.pytorch
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.tracking._tracking_service.utils")
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, LinearLR, SequentialLR,
    OneCycleLR, ReduceLROnPlateau
)
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
    plot_position_resolution_profile,
    plot_face_weights,
)
from .event_display import save_worst_case_events
from .distributed import (
    setup_ddp, cleanup_ddp, is_main_process,
    reduce_metrics, wrap_ddp,
)

# ------------------------------------------------------------
# Suppress torch.compile / Triton autotuning verbose output
import logging
import warnings
logging.getLogger("torch._inductor").setLevel(logging.WARNING)
logging.getLogger("torch._dynamo").setLevel(logging.WARNING)
# Suppress torch.compile UserWarnings about tensor construction
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*", category=UserWarning)
warnings.filterwarnings("ignore", module="torch._dynamo.*")
warnings.filterwarnings("ignore", module="torch.fx.*")
# Suppress scheduler warnings (harmless, PyTorch internal)
warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler.step.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*", category=UserWarning)

# ------------------------------------------------------------
# Enable TensorFloat32
torch.set_float32_matmul_precision('high')


# ------------------------------------------------------------
#  Artifact Saving Helper
# ------------------------------------------------------------
def _safe_log_artifact(path):
    """Log artifact to MLflow only if there's an active run."""
    try:
        if mlflow.active_run() is not None:
            mlflow.log_artifact(path)
    except Exception:
        pass  # Silently skip if MLflow logging fails


def save_validation_artifacts(
    model,
    angle_pred, angle_true,
    root_data,
    active_tasks,
    artifact_dir,
    run_name,
    epoch=None,
    worst_events=None,
):
    """
    Save validation artifacts (plots and CSVs) for all active tasks.

    Args:
        model: The model (for face weights plot)
        angle_pred: Angle predictions array (N, 2) or None
        angle_true: Angle truth array (N, 2) or None
        root_data: Dict with task predictions from validation
        active_tasks: List of active task names
        artifact_dir: Directory to save artifacts
        run_name: Run name for file naming
        epoch: Optional epoch number (for intermediate saves)
        worst_events: List of worst case event dicts (optional)
    """
    suffix = f"_ep{epoch}" if epoch is not None else ""

    # --- Face Weights (model-level, not task-specific) ---
    try:
        face_pdf = os.path.join(artifact_dir, f"face_weights_{run_name}{suffix}.pdf")
        # For multi-head model, get the backbone
        if hasattr(model, 'module'):
            backbone = model.module.backbone if hasattr(model.module, 'backbone') else model.module
        elif hasattr(model, 'backbone'):
            backbone = model.backbone
        else:
            backbone = model
        plot_face_weights(backbone, outfile=face_pdf)
        _safe_log_artifact(face_pdf)
    except Exception as e:
        print(f"[WARN] Could not save face weights plot: {e}")

    # --- Angle Task ---
    if "angle" in active_tasks and angle_pred is not None and len(angle_pred) > 0:
        csv_path = os.path.join(artifact_dir, f"predictions_angle_{run_name}{suffix}.csv")
        pd.DataFrame({
            "true_theta": angle_true[:, 0], "true_phi": angle_true[:, 1],
            "pred_theta": angle_pred[:, 0], "pred_phi": angle_pred[:, 1]
        }).to_csv(csv_path, index=False)
        _safe_log_artifact(csv_path)

        res_pdf = os.path.join(artifact_dir, f"resolution_angle_{run_name}{suffix}.pdf")
        plot_resolution_profile(angle_pred, angle_true, outfile=res_pdf)
        _safe_log_artifact(res_pdf)

    # --- Energy Task ---
    if "energy" in active_tasks:
        pred_energy = root_data.get("pred_energy", np.array([]))
        true_energy = root_data.get("true_energy", np.array([]))
        if pred_energy.size > 0 and true_energy.size > 0:
            csv_path = os.path.join(artifact_dir, f"predictions_energy_{run_name}{suffix}.csv")
            # Include uvw truth for position-profiled resolution plots
            energy_data = {
                "true_energy": true_energy,
                "pred_energy": pred_energy
            }
            # Add position truth if available (for resolution vs position plots)
            for key in ["true_u", "true_v", "true_w"]:
                if key in root_data and len(root_data[key]) == len(true_energy):
                    energy_data[key] = root_data[key]
            pd.DataFrame(energy_data).to_csv(csv_path, index=False)
            _safe_log_artifact(csv_path)

            res_pdf = os.path.join(artifact_dir, f"resolution_energy_{run_name}{suffix}.pdf")
            plot_energy_resolution_profile(pred_energy, true_energy, root_data, outfile=res_pdf)
            _safe_log_artifact(res_pdf)

    # --- Timing Task ---
    if "timing" in active_tasks:
        pred_timing = root_data.get("pred_timing", np.array([]))
        true_timing = root_data.get("true_timing", np.array([]))
        if pred_timing.size > 0 and true_timing.size > 0:
            csv_path = os.path.join(artifact_dir, f"predictions_timing_{run_name}{suffix}.csv")
            pd.DataFrame({
                "true_timing": true_timing,
                "pred_timing": pred_timing
            }).to_csv(csv_path, index=False)
            _safe_log_artifact(csv_path)

            res_pdf = os.path.join(artifact_dir, f"resolution_timing_{run_name}{suffix}.pdf")
            plot_timing_resolution_profile(pred_timing, true_timing, outfile=res_pdf)
            _safe_log_artifact(res_pdf)

    # --- Position Task (uvwFI) ---
    if "uvwFI" in active_tasks:
        pred_u = root_data.get("pred_u", np.array([]))
        pred_v = root_data.get("pred_v", np.array([]))
        pred_w = root_data.get("pred_w", np.array([]))
        true_u = root_data.get("true_u", np.array([]))
        true_v = root_data.get("true_v", np.array([]))
        true_w = root_data.get("true_w", np.array([]))
        if pred_u.size > 0 and true_u.size > 0:
            csv_path = os.path.join(artifact_dir, f"predictions_uvwFI_{run_name}{suffix}.csv")
            pd.DataFrame({
                "true_u": true_u, "true_v": true_v, "true_w": true_w,
                "pred_u": pred_u, "pred_v": pred_v, "pred_w": pred_w
            }).to_csv(csv_path, index=False)
            _safe_log_artifact(csv_path)

            pred_uvw = np.stack([pred_u, pred_v, pred_w], axis=1)
            true_uvw = np.stack([true_u, true_v, true_w], axis=1)
            res_pdf = os.path.join(artifact_dir, f"resolution_uvwFI_{run_name}{suffix}.pdf")
            plot_position_resolution_profile(pred_uvw, true_uvw, outfile=res_pdf)
            _safe_log_artifact(res_pdf)

    # --- Worst Case Events ---
    if worst_events:
        try:
            save_worst_case_events(worst_events, artifact_dir, run_name, epoch=epoch, max_events=5)
        except Exception as e:
            print(f"[WARN] Could not save worst case events: {e}")


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

    # Device / DDP (must be before any is_main_process() calls)
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # Profile: CLI override takes precedence, otherwise use config
    enable_profile = profile if profile is not None else getattr(cfg.training, 'profile', False)
    if enable_profile and is_main_process():
        print("[INFO] Training profiler enabled - timing breakdown will be shown per epoch.")

    # Get active tasks and their weights
    active_tasks = get_active_tasks(cfg)
    task_weights = get_task_weights(cfg)

    if not active_tasks:
        raise ValueError("No tasks enabled in config. Enable at least one task.")

    if is_main_process():
        print(f"[INFO] Active tasks: {active_tasks}")
        print(f"[INFO] Task weights: {task_weights}")
    if is_main_process():
        print(f"[INFO] Using device: {device}" + (f" (world_size={world_size})" if world_size > 1 else ""))

    # Validate data paths before creating data loaders (use expand_path for consistency)
    validate_data_paths(cfg.data.train_path, cfg.data.val_path, expand_func=expand_path)

    # Check artifact directory for existing files
    run_name = cfg.mlflow.run_name or time.strftime("run_cv2_%Y%m%d_%H%M%S")
    artifact_dir = os.path.abspath(os.path.join(cfg.checkpoint.save_dir, run_name))
    check_artifact_directory(artifact_dir)

    # --- Fiducial Volume Cut ---
    fiducial = cfg.data.fiducial if cfg.data.fiducial.enabled else None
    if fiducial is not None and is_main_process():
        print(f"[INFO] Fiducial volume cut enabled:")
        print(f"  |u| < {fiducial.u_max} cm, |v| < {fiducial.v_max} cm, w >= {fiducial.w_min} cm"
              + (f", w <= {fiducial.w_max} cm" if fiducial.w_max is not None else ""))

    # --- Data Loaders ---
    norm_kwargs = {
        "npho_scale": cfg.normalization.npho_scale,
        "npho_scale2": cfg.normalization.npho_scale2,
        "time_scale": cfg.normalization.time_scale,
        "time_shift": cfg.normalization.time_shift,
        "sentinel_time": cfg.normalization.sentinel_time,
        "sentinel_npho": cfg.normalization.sentinel_npho,
        "step_size": cfg.data.chunksize,
        "npho_branch": getattr(cfg.data, "npho_branch", "relative_npho"),
        "time_branch": getattr(cfg.data, "time_branch", "relative_time"),
        "log_invalid_npho": getattr(cfg.data, "log_invalid_npho", True),
        "npho_scheme": getattr(cfg.normalization, "npho_scheme", "log1p"),
        "fiducial": fiducial,
    }

    train_loader = get_dataloader(
        cfg.data.train_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_threads=cfg.data.num_threads,
        prefetch_factor=getattr(cfg.data, 'prefetch_factor', 2),
        rank=rank, world_size=world_size,
        **norm_kwargs
    )

    val_loader = get_dataloader(
        cfg.data.val_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        num_threads=cfg.data.num_threads,
        prefetch_factor=getattr(cfg.data, 'prefetch_factor', 2),
        rank=rank, world_size=world_size,
        **norm_kwargs
    )

    # Full (unsharded) val loader for final evaluation artifacts (rank 0 only)
    if world_size > 1:
        val_loader_full = get_dataloader(
            cfg.data.val_path,
            batch_size=cfg.data.batch_size,
            num_workers=cfg.data.num_workers,
            num_threads=cfg.data.num_threads,
            prefetch_factor=getattr(cfg.data, 'prefetch_factor', 2),
            **norm_kwargs
        )
    else:
        val_loader_full = val_loader

    # --- Model ---
    outer_fine_pool = tuple(cfg.model.outer_fine_pool) if cfg.model.outer_fine_pool else None
    base_regressor = XECEncoder(
        outer_mode=cfg.model.outer_mode,
        outer_fine_pool=outer_fine_pool,
        drop_path_rate=cfg.model.drop_path_rate,
        sentinel_time=cfg.normalization.sentinel_time,
        encoder_dim=cfg.model.encoder_dim,
        dim_feedforward=cfg.model.dim_feedforward,
        num_fusion_layers=cfg.model.num_fusion_layers,
    )
    model = XECMultiHeadModel(
        base_regressor,
        active_tasks=active_tasks,
        hidden_dim=cfg.model.hidden_dim
    ).to(device)
    total_params, trainable_params = count_model_params(model)
    if is_main_process():
        print("[INFO] Regressor created:")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Trainable params: {trainable_params:,}")

    # Keep unwrapped reference for checkpointing and EMA
    model_without_ddp = model

    # Wrap with DDP (before compile, after .to(device))
    model = wrap_ddp(model, local_rank)
    model_ddp = model  # Save DDP reference before compile (for .no_sync access)

    # Optionally compile model (can be disabled or use different modes to reduce memory)
    # Auto-detect ARM architecture (GH nodes) and disable compile (Triton not supported)
    import platform
    is_arm = platform.machine() in ('aarch64', 'arm64')
    compile_mode = getattr(cfg.training, 'compile', 'max-autotune')
    compile_fullgraph = getattr(cfg.training, 'compile_fullgraph', False)

    if is_arm and compile_mode and compile_mode not in ('false', 'none'):
        if is_main_process():
            print(f"[INFO] ARM architecture detected - disabling torch.compile (Triton not supported)")
        compile_mode = 'none'

    if compile_mode and compile_mode != 'false' and compile_mode != 'none':
        # Increase dynamo cache limit to avoid CacheLimitExceeded errors
        torch._dynamo.config.cache_size_limit = 64
        fg_str = "fullgraph" if compile_fullgraph else "partial"
        if is_main_process():
            print(f"[INFO] Compiling model with mode='{compile_mode}', {fg_str} (this may take a few minutes...)")
        model = torch.compile(model, mode=compile_mode, fullgraph=compile_fullgraph, dynamic=False)
    else:
        if is_main_process():
            print("[INFO] Model compilation disabled (eager mode)")

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
        if is_main_process():
            print(f"[INFO] Using EMA with decay={ema_decay}")

        def robust_ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            return ema_decay * averaged_model_parameter + (1.0 - ema_decay) * model_parameter

        # Build EMA from unwrapped model (no DDP wrapper)
        ema_model = AveragedModel(model_without_ddp, avg_fn=robust_ema_avg, use_buffers=True)
        ema_model.to(device)

    # --- Loss Scaler (Auto Balance) ---
    loss_scaler = None
    if cfg.loss_balance == "auto":
        if is_main_process():
            print(f"[INFO] Using Automatic Loss Balancing.")
        loss_scaler = AutomaticLossScaler(active_tasks).to(device)
        optimizer.add_param_group({"params": loss_scaler.parameters()})

    # --- AMP Scaler ---
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.training.amp and torch.cuda.is_available()))

    # --- Detect resume to auto-disable warmup ---
    # When resuming from a checkpoint, warmup is not needed since the model
    # is already past the initial training phase. We detect this before
    # creating the scheduler to avoid unnecessary warmup.
    warmup_epochs = cfg.training.warmup_epochs
    is_resuming = False
    if cfg.checkpoint.resume_from and os.path.exists(cfg.checkpoint.resume_from):
        try:
            ckpt_probe = torch.load(cfg.checkpoint.resume_from, map_location="cpu", weights_only=False)
            if isinstance(ckpt_probe, dict) and "epoch" in ckpt_probe and ckpt_probe.get("epoch", 0) > 0:
                is_resuming = True
                if warmup_epochs > 0:
                    if is_main_process():
                        print(f"[INFO] Resuming from checkpoint - disabling warmup (was {warmup_epochs} epochs)")
                    warmup_epochs = 0
            del ckpt_probe
        except Exception:
            pass  # Will be handled later in the full resume logic

    # --- Scheduler ---
    # Support both naming conventions: lr_scheduler (new) or use_scheduler+scheduler (legacy)
    scheduler = None
    lr_scheduler_cfg = getattr(cfg.training, 'lr_scheduler', None)
    if lr_scheduler_cfg is not None:
        # New style: lr_scheduler directly specifies the type (or null/none to disable)
        scheduler_type = lr_scheduler_cfg if lr_scheduler_cfg not in (None, 'none', 'null') else 'none'
    else:
        # Legacy style: use_scheduler bool + scheduler type
        scheduler_type = getattr(cfg.training, 'scheduler', 'cosine') if getattr(cfg.training, 'use_scheduler', True) else 'none'

    if scheduler_type == 'cosine':
        if is_main_process():
            print(f"[INFO] Using Cosine Annealing with {warmup_epochs} warmup epochs.")
        main_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs - warmup_epochs,
            eta_min=getattr(cfg.training, 'lr_min', 1e-6)
        )
        if warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, main_scheduler],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = main_scheduler

    elif scheduler_type == 'onecycle':
        # OneCycleLR: needs estimated steps per epoch
        # For streaming datasets, estimate based on expected iterations
        # User can specify max_lr, otherwise defaults to lr
        max_lr = getattr(cfg.training, 'max_lr', None) or cfg.training.lr
        pct_start = getattr(cfg.training, 'pct_start', 0.3)
        # Estimate steps per epoch (will be updated after first epoch if needed)
        # For now, use a placeholder - OneCycleLR will adjust
        if is_main_process():
            print(f"[INFO] Using OneCycleLR with max_lr={max_lr}, pct_start={pct_start}")
            print(f"       Note: OneCycleLR requires knowing total steps. Using epoch-based stepping.")
        # We'll step per epoch, so total_steps = epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=cfg.training.epochs,
            steps_per_epoch=1,  # Step once per epoch
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=25.0,  # initial_lr = max_lr / div_factor
            final_div_factor=1e4,  # final_lr = initial_lr / final_div_factor
        )

    elif scheduler_type == 'plateau':
        patience = getattr(cfg.training, 'lr_patience', 5)
        factor = getattr(cfg.training, 'lr_factor', 0.5)
        min_lr = getattr(cfg.training, 'lr_min', 1e-7)
        if is_main_process():
            print(f"[INFO] Using ReduceLROnPlateau with patience={patience}, factor={factor}, min_lr={min_lr}")
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True
        )

    elif scheduler_type in ('none', None):
        if is_main_process():
            print("[INFO] No learning rate scheduler enabled.")
        scheduler = None

    else:
        if is_main_process():
            print(f"[WARN] Unknown scheduler type '{scheduler_type}', using no scheduler.")
        scheduler = None

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val = float("inf")
    best_ema_state = None  # Track best EMA state for final evaluation
    mlflow_run_id = None

    if cfg.checkpoint.resume_from and os.path.exists(cfg.checkpoint.resume_from):
        if is_main_process():
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
            if is_main_process():
                print(f"[INFO] Detected full regressor checkpoint. Resuming training state.")
            model_without_ddp.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            checkpoint_epoch = checkpoint["epoch"]
            best_val = checkpoint.get("best_val", float("inf"))
            mlflow_run_id = checkpoint.get("mlflow_run_id", None)

            # Handle reset_epoch: start from epoch 1 (only load weights)
            reset_epoch = getattr(cfg.checkpoint, 'reset_epoch', False)
            if reset_epoch:
                start_epoch = 1
                if is_main_process():
                    print(f"[INFO] reset_epoch=True: Starting from epoch 1 (weights loaded from epoch {checkpoint_epoch})")
            else:
                start_epoch = checkpoint_epoch + 1

            # Handle refresh_lr: recreate scheduler for remaining epochs
            refresh_lr = getattr(cfg.checkpoint, 'refresh_lr', False)
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                if refresh_lr:
                    # Recreate scheduler for remaining epochs
                    remaining_epochs = cfg.training.epochs - start_epoch + 1
                    if is_main_process():
                        print(f"[INFO] refresh_lr=True: Creating fresh scheduler with lr={cfg.training.lr}, "
                              f"T_max={remaining_epochs} (epochs {start_epoch}-{cfg.training.epochs})")
                    if scheduler_type == 'cosine':
                        scheduler = CosineAnnealingLR(
                            optimizer,
                            T_max=remaining_epochs,
                            eta_min=getattr(cfg.training, 'lr_min', 1e-6)
                        )
                    # Note: OneCycleLR and Plateau don't need recreation as they adapt
                else:
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    if is_main_process():
                        print(f"[INFO] Restored scheduler state.")

            if ema_model is not None and "ema_state_dict" in checkpoint:
                ema_model.load_state_dict(checkpoint["ema_state_dict"])

            # Restore best EMA state if this was the best checkpoint
            if "best_ema_state" in checkpoint:
                best_ema_state = checkpoint["best_ema_state"]

        elif is_mae_checkpoint:
            if is_main_process():
                print(f"[INFO] Detected MAE checkpoint. Loading encoder for fine-tuning.")
            mae_state = checkpoint["model_state_dict"]

            if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
                if is_main_process():
                    print(f"[INFO] Using EMA weights from MAE checkpoint.")
                mae_state = checkpoint["ema_state_dict"]
                mae_state = {k.replace("module.", ""): v for k, v in mae_state.items()}

            encoder_state = {}
            for key, value in mae_state.items():
                if key.startswith("encoder."):
                    new_key = "backbone." + key[8:]
                    encoder_state[new_key] = value

            if encoder_state:
                missing, unexpected = model_without_ddp.load_state_dict(encoder_state, strict=False)
                if is_main_process():
                    print(f"[INFO] Loaded {len(encoder_state)} encoder weights from MAE.")
                    print(f"[INFO] Missing keys (expected for heads): {len(missing)}")

            if ema_model is not None:
                ema_model.module.load_state_dict(model_without_ddp.state_dict())
                if hasattr(ema_model, 'n_averaged'):
                    ema_model.n_averaged.zero_()

        elif is_raw_encoder_checkpoint:
            if is_main_process():
                print(f"[INFO] Detected raw encoder checkpoint. Loading for fine-tuning.")
            encoder_state = {}
            raw_state = checkpoint if not isinstance(checkpoint, dict) else checkpoint
            for key, value in raw_state.items():
                new_key = "backbone." + key
                encoder_state[new_key] = value

            missing, unexpected = model_without_ddp.load_state_dict(encoder_state, strict=False)
            if is_main_process():
                print(f"[INFO] Loaded {len(encoder_state)} encoder weights.")
                print(f"[INFO] Missing keys (expected for heads): {len(missing)}")

            if ema_model is not None:
                ema_model.module.load_state_dict(model_without_ddp.state_dict())
                if hasattr(ema_model, 'n_averaged'):
                    ema_model.n_averaged.zero_()

    # --- Check for valid epoch range ---
    if start_epoch >= cfg.training.epochs:
        print("\n" + "=" * 70)
        print("[ERROR] No epochs to train!")
        print(f"  Resumed from epoch {start_epoch - 1}, but config has epochs={cfg.training.epochs}")
        print(f"  The training loop range({start_epoch}, {cfg.training.epochs}) is empty.")
        print(f"\n  To continue training, set 'epochs' higher than {start_epoch - 1}.")
        print(f"  For example, to train 10 more epochs, set epochs={start_epoch + 9}")
        print("=" * 70 + "\n")
        raise ValueError(f"start_epoch ({start_epoch}) >= epochs ({cfg.training.epochs}). "
                        f"Set epochs > {start_epoch - 1} to continue training.")

    # --- Force new MLflow run if requested ---
    if getattr(cfg.checkpoint, 'new_mlflow_run', False) and mlflow_run_id is not None:
        if is_main_process():
            print(f"[INFO] new_mlflow_run=True: Starting fresh MLflow run (ignoring run_id from checkpoint)")
        mlflow_run_id = None

    # --- MLflow Setup (rank 0 only) ---
    run_name = cfg.mlflow.run_name or time.strftime("run_%Y%m%d_%H%M%S")
    if is_main_process():
        # Default to SQLite backend if MLFLOW_TRACKING_URI is not set
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            default_uri = f"sqlite:///{os.getcwd()}/mlruns.db"
            mlflow.set_tracking_uri(default_uri)
            print(f"[INFO] MLflow tracking URI: {default_uri}")
        mlflow.set_experiment(cfg.mlflow.experiment)

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
            if is_main_process():
                print("[INFO] No reweighting enabled.")
            reweighter = None  # Set to None so it won't be passed to run_epoch_stream

    # --- Training Loop ---
    # Disable MLflow's automatic system metrics (uses wall clock time)
    # We log our own system metrics with step=epoch for consistent x-axis
    # Only rank 0 interacts with MLflow
    mlflow_ctx = (
        mlflow.start_run(run_id=mlflow_run_id, run_name=run_name if not mlflow_run_id else None,
                         log_system_metrics=False)
        if is_main_process()
        else nullcontext()
    )
    with mlflow_ctx as run:
        if is_main_process():
            mlflow_run_id = run.info.run_id
        artifact_dir = os.path.abspath(os.path.join(cfg.checkpoint.save_dir, run_name))
        os.makedirs(artifact_dir, exist_ok=True)

        # Determine no_sync context for gradient accumulation (skip AllReduce on intermediate steps)
        no_sync_ctx = model_ddp.no_sync if world_size > 1 else None

        # Log config
        if start_epoch == 1 and is_main_process():
            log_params = {
                "active_tasks": ",".join(active_tasks),
                "batch_size": cfg.data.batch_size,
                "lr": cfg.training.lr,
                "epochs": cfg.training.epochs,
                "outer_mode": cfg.model.outer_mode,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "loss_balance": cfg.loss_balance,
                "npho_scheme": getattr(cfg.normalization, "npho_scheme", "log1p"),
                "world_size": world_size,
                "fiducial_enabled": cfg.data.fiducial.enabled,
            }
            if cfg.data.fiducial.enabled:
                log_params.update({
                    "fiducial_u_max": cfg.data.fiducial.u_max,
                    "fiducial_v_max": cfg.data.fiducial.v_max,
                    "fiducial_w_min": cfg.data.fiducial.w_min,
                    "fiducial_w_max": cfg.data.fiducial.w_max,
                })
            mlflow.log_params(log_params)

        best_state = None

        for ep in range(start_epoch, cfg.training.epochs + 1):
            t0 = time.time()

            # === TRAIN ===
            # For ReduceLROnPlateau, step after validation with val_loss, not during training
            is_plateau_scheduler = isinstance(scheduler, ReduceLROnPlateau) if scheduler else False
            epoch_scheduler = None if is_plateau_scheduler else scheduler

            tr_metrics, _, _, _, _ = run_epoch_stream(
                model, optimizer, device, train_loader,
                scaler=scaler,
                train=True,
                amp=cfg.training.amp,
                task_weights=task_weights,
                loss_scaler=loss_scaler,
                reweighter=reweighter,
                channel_dropout_rate=cfg.training.channel_dropout_rate,
                scheduler=epoch_scheduler,
                ema_model=ema_model,
                grad_clip=cfg.training.grad_clip,
                grad_accum_steps=getattr(cfg.training, 'grad_accum_steps', 1),
                profile=enable_profile and is_main_process(),
                no_sync_ctx=no_sync_ctx,
            )
            tr_metrics = reduce_metrics(tr_metrics, device)

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
            val_metrics = reduce_metrics(val_metrics, device)

            sec = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']

            tr_loss = tr_metrics["total_opt"]
            val_loss = val_metrics["total_opt"]

            # Step ReduceLROnPlateau scheduler with validation loss
            if is_plateau_scheduler:
                scheduler.step(val_loss)

            if is_main_process():
                print(f"[{ep:03d}] tr_loss {tr_loss:.2e} val_loss {val_loss:.2e} lr {current_lr:.2e} time {sec:.1f}s")

            # --- Logging (rank 0 only) ---
            if not is_main_process():
                continue

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

            # Add gradient norm from training metrics
            if "system/grad_norm_max" in tr_metrics:
                log_dict["grad_norm_max"] = tr_metrics["system/grad_norm_max"]

            for metric_key in ["smooth_l1", "l1", "mse"]:
                if metric_key in val_metrics:
                    log_dict[f"val_{metric_key}"] = val_metrics[metric_key]
            # Only log cosine loss if angle task is active
            if "angle" in active_tasks and "cos" in val_metrics:
                log_dict["val_cos"] = val_metrics["cos"]
            # Log position cosine loss if position task is active
            if "uvwFI" in active_tasks and "cos_pos" in val_metrics:
                log_dict["val_cos_pos"] = val_metrics["cos_pos"]

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

            # --- Checkpointing ---
            # Get validation data for artifact saving
            root_data = extra_info.get("root_data", {}) if extra_info else {}

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().clone().cpu() for k, v in model_without_ddp.state_dict().items()}
                # Save best EMA state for final evaluation
                if ema_model is not None:
                    best_ema_state = {k: v.detach().clone().cpu() for k, v in ema_model.state_dict().items()}

                checkpoint_data = {
                    "epoch": ep,
                    "model_state_dict": model_without_ddp.state_dict(),
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                    "best_val": best_val,
                    "mlflow_run_id": mlflow_run_id,
                    "config": {
                        "outer_mode": cfg.model.outer_mode,
                        "outer_fine_pool": list(outer_fine_pool) if outer_fine_pool else None,
                        "npho_scale": float(cfg.normalization.npho_scale),
                        "npho_scale2": float(cfg.normalization.npho_scale2),
                        "time_scale": float(cfg.normalization.time_scale),
                        "time_shift": float(cfg.normalization.time_shift),
                        "sentinel_time": float(cfg.normalization.sentinel_time),
                        "sentinel_npho": float(cfg.normalization.sentinel_npho),
                        "npho_scheme": getattr(cfg.normalization, "npho_scheme", "log1p"),
                        "encoder_dim": cfg.model.encoder_dim,
                        "dim_feedforward": cfg.model.dim_feedforward,
                        "num_fusion_layers": cfg.model.num_fusion_layers,
                        "active_tasks": active_tasks,
                    },
                }
                torch.save(checkpoint_data, os.path.join(artifact_dir, "checkpoint_best.pth"))
                print(f"   [info] New best val_loss: {best_val:.2e}")

            # Loss spike detection: warn if val_loss suddenly increases significantly
            elif val_loss > best_val * 5.0:
                print(f"   [WARN] Loss spike detected! val_loss ({val_loss:.2e}) > 5x best ({best_val:.2e})")
                print(f"   [WARN] Consider: (1) reducing lr, (2) reducing grad_clip, (3) resuming from checkpoint_best.pth")

                # Save validation artifacts (plots and CSVs) for best checkpoint
                # Note: worst_events are only saved at end of training to reduce time
                if getattr(cfg.checkpoint, 'save_artifacts', True):
                    save_validation_artifacts(
                        model=val_model,
                        angle_pred=pred_val,
                        angle_true=true_val,
                        root_data=root_data,
                        active_tasks=active_tasks,
                        artifact_dir=artifact_dir,
                        run_name=run_name,
                        epoch=ep,
                        worst_events=None,  # Skip during training, only save at end
                    )

            # Save last checkpoint
            checkpoint_data = {
                "epoch": ep,
                "model_state_dict": model_without_ddp.state_dict(),
                "ema_state_dict": ema_model.state_dict() if ema_model else None,
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "best_val": best_val,
                "mlflow_run_id": mlflow_run_id,
                "config": {
                    "outer_mode": cfg.model.outer_mode,
                    "outer_fine_pool": list(outer_fine_pool) if outer_fine_pool else None,
                    "npho_scale": float(cfg.normalization.npho_scale),
                    "npho_scale2": float(cfg.normalization.npho_scale2),
                    "time_scale": float(cfg.normalization.time_scale),
                    "time_shift": float(cfg.normalization.time_shift),
                    "sentinel_time": float(cfg.normalization.sentinel_time),
                    "sentinel_npho": float(cfg.normalization.sentinel_npho),
                    "npho_scheme": getattr(cfg.normalization, "npho_scheme", "log1p"),
                    "encoder_dim": cfg.model.encoder_dim,
                    "dim_feedforward": cfg.model.dim_feedforward,
                    "num_fusion_layers": cfg.model.num_fusion_layers,
                    "active_tasks": active_tasks,
                },
            }
            torch.save(checkpoint_data, os.path.join(artifact_dir, "checkpoint_last.pth"))

        # --- Final Evaluation & Artifacts (rank 0 only) ---
        if is_main_process():
            # Use best model state for final evaluation and export
            if ema_model is not None and best_ema_state is not None:
                print("[INFO] Loading best EMA state for final evaluation.")
                ema_model.load_state_dict(best_ema_state)
                final_model = ema_model
            elif best_state is not None:
                print("[INFO] Loading best model state for final evaluation.")
                model_without_ddp.load_state_dict(best_state)
                final_model = model_without_ddp
            else:
                final_model = ema_model if ema_model is not None else model_without_ddp

            # Run final validation with best model (use full val data, not sharded)
            final_model.eval()
            _, angle_pred, angle_true, extra_info, _ = run_epoch_stream(
                final_model, optimizer, device, val_loader_full,
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

            # Save final validation artifacts (without epoch suffix)
            if getattr(cfg.checkpoint, 'save_artifacts', True):
                print("[INFO] Saving final validation artifacts...")
                worst_events = extra_info.get("worst_events", []) if extra_info else []
                save_validation_artifacts(
                    model=final_model,
                    angle_pred=angle_pred,
                    angle_true=angle_true,
                    root_data=root_data,
                    active_tasks=active_tasks,
                    artifact_dir=artifact_dir,
                    run_name=run_name,
                    epoch=None,  # No epoch suffix for final artifacts
                    worst_events=worst_events,
                )
            else:
                print("[INFO] Skipping artifact saving (save_artifacts=false)")

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

    if is_main_process():
        print(f"[INFO] Training complete. Best val_loss: {best_val:.2e}")
    cleanup_ddp()
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
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch factor")

    # Normalization (override config)
    parser.add_argument("--npho_scale", type=float, default=None)
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_time", type=float, default=None)
    parser.add_argument("--npho_scheme", type=str, default=None,
                        choices=["log1p", "anscombe", "sqrt", "linear"],
                        help="Normalization scheme for npho (default: from config)")

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
    parser.add_argument("--compile", type=str, default=None,
                        choices=["max-autotune", "reduce-overhead", "default", "false", "none"],
                        help="torch.compile mode (default: max-autotune, use 'false' to disable)")

    # Tasks (override config)
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Enable specific tasks")
    parser.add_argument("--loss_balance", type=str, default=None, choices=["manual", "auto"])

    # Checkpoint (override config)
    parser.add_argument("--resume_from", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save artifacts")
    parser.add_argument("--refresh_lr", action="store_true", help="Reset LR scheduler when resuming (schedule for remaining epochs)")
    parser.add_argument("--reset_epoch", action="store_true", help="Start from epoch 1 when resuming (only load model weights)")
    parser.add_argument("--new_mlflow_run", action="store_true", help="Force new MLflow run when resuming")

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
    # Auto-limit num_workers on ARM/GH nodes (multiprocessing issues)
    import platform
    if platform.machine() in ("aarch64", "arm64") and cfg.data.num_workers > 1:
        print(f"[INFO] ARM/GH node detected - limiting num_workers from {cfg.data.num_workers} to 1")
        cfg.data.num_workers = 1
    if args.num_threads is not None:
        cfg.data.num_threads = args.num_threads
    if args.prefetch_factor is not None:
        cfg.data.prefetch_factor = args.prefetch_factor

    # Normalization
    if args.npho_scale is not None:
        cfg.normalization.npho_scale = args.npho_scale
    if args.npho_scale2 is not None:
        cfg.normalization.npho_scale2 = args.npho_scale2
    if args.time_scale is not None:
        cfg.normalization.time_scale = args.time_scale
    if args.time_shift is not None:
        cfg.normalization.time_shift = args.time_shift
    if args.sentinel_time is not None:
        cfg.normalization.sentinel_time = args.sentinel_time
    if args.npho_scheme is not None:
        cfg.normalization.npho_scheme = args.npho_scheme

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
    if args.compile is not None:
        cfg.training.compile = args.compile

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
    if getattr(args, 'refresh_lr', False):
        cfg.checkpoint.refresh_lr = True
    if getattr(args, 'reset_epoch', False):
        cfg.checkpoint.reset_epoch = True
    if getattr(args, 'new_mlflow_run', False):
        cfg.checkpoint.new_mlflow_run = True

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
def collect_cli_overrides(args):
    """Collect non-None CLI arguments as overrides dict for display."""
    overrides = {}
    # Map of arg names to display names
    override_args = [
        "train_path", "val_path", "tree", "batch_size", "chunksize",
        "num_workers", "num_threads", "npho_scale", "npho_scale2",
        "time_scale", "time_shift", "sentinel_time", "npho_scheme",
        "outer_mode", "outer_fine_pool", "hidden_dim", "drop_path_rate",
        "epochs", "lr", "weight_decay", "warmup_epochs", "ema_decay",
        "channel_dropout_rate", "grad_clip", "grad_accum_steps", "compile",
        "tasks", "loss_balance", "resume_from", "save_dir",
        "mlflow_experiment", "run_name", "onnx"
    ]
    # Boolean flags (action="store_true") - only include if True
    bool_flags = ["profile", "refresh_lr", "reset_epoch", "new_mlflow_run"]
    for arg_name in override_args:
        val = getattr(args, arg_name, None)
        if val is not None:
            overrides[arg_name] = val
    for arg_name in bool_flags:
        val = getattr(args, arg_name, False)
        if val:  # Only include if explicitly set to True
            overrides[arg_name] = val
    return overrides


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # Show config file and collect CLI overrides
    print(f"[INFO] Config file: {os.path.abspath(args.config)}")
    cli_overrides = collect_cli_overrides(args)
    if cli_overrides:
        print(f"[INFO] CLI overrides: {cli_overrides}")

    # Load config and apply CLI overrides
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Save merged config to temp file for train_with_config
    def config_to_dict(cfg):
        """Convert config dataclass to dict for YAML serialization."""
        import dataclasses
        def _to_dict(obj):
            if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
                return {k: _to_dict(v) for k, v in obj.__dict__.items()}
            return obj
        result = {}
        for section in ["data", "normalization", "model", "training", "checkpoint", "mlflow", "export"]:
            if hasattr(cfg, section):
                obj = getattr(cfg, section)
                result[section] = _to_dict(obj)
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
        # Only pass profile if explicitly set via CLI (--profile), otherwise let config decide
        train_with_config(temp_config_path, profile=True if args.profile else None)
    finally:
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)
