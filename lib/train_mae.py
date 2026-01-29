import os
import warnings
import logging

# Set MLflow tracking URI before importing mlflow to avoid deprecation warning
if "MLFLOW_TRACKING_URI" not in os.environ:
    os.environ["MLFLOW_TRACKING_URI"] = "sqlite:///mlruns.db"

# Suppress MLflow/Alembic INFO messages about database initialization
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)

# Suppress torch.compile/dynamo warnings about tensor construction
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*")
warnings.filterwarnings("ignore", message=".*skipping cudagraphs.*")

# Suppress torch dynamo verbose output and Triton autotuning
os.environ["TORCH_LOGS"] = "-all"
os.environ["TORCHDYNAMO_VERBOSE"] = "0"
os.environ.setdefault("TORCHINDUCTOR_LOG_LEVEL", "WARNING")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

import torch
import argparse
import time
import glob
import platform
import mlflow
import uproot
import numpy as np

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .models import XECEncoder, XEC_MAE
from .engines import run_epoch_mae, run_eval_mae
from .utils import count_model_params, log_system_metrics_to_mlflow, validate_data_paths, check_artifact_directory
from .geom_defs import DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2, DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE
from .config import load_mae_config

# Usage
# CLI Mode: python -m lib.train_mae --train_root path/to/data.root --save_path mae_pretrained.pth --epochs 20
# Config Mode: python -m lib.train_mae --config config/mae_config.yaml
# Config + CLI Override: python -m lib.train_mae --config config/mae_config.yaml --epochs 50 --lr 5e-5

# Enable TensorFloat32
torch.set_float32_matmul_precision('high')

# Disable Debugging/Profiling overhead
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)


def save_predictions_to_root(predictions, save_path, epoch, run_id=None):
    """
    Save MAE predictions to ROOT file for analysis.

    Args:
        predictions: dict with keys: truth_npho, truth_time, pred_npho, pred_time, mask, x_masked
        save_path: directory to save the file
        epoch: current epoch number
    """
    root_path = os.path.join(save_path, f"mae_predictions_epoch_{epoch+1}.root")

    n_events = len(predictions["truth_npho"])
    if n_events == 0:
        print(f"[WARN] No predictions to save")
        return

    # Prepare branch data
    branch_data = {
        "event_id": np.arange(n_events, dtype=np.int32),
        "truth_npho": predictions["truth_npho"].astype(np.float32),
        "truth_time": predictions["truth_time"].astype(np.float32),
        "mask": predictions["mask"].astype(np.float32),
    }

    if run_id:
        run_id_str = str(run_id)
        branch_data["run_id"] = np.array([run_id_str] * n_events)

    # Add masked input if available
    if "x_masked" in predictions and len(predictions["x_masked"]) > 0:
        branch_data["masked_npho"] = predictions["x_masked"][:, :, 0].astype(np.float32)
        branch_data["masked_time"] = predictions["x_masked"][:, :, 1].astype(np.float32)

    if "pred_npho" in predictions and len(predictions["pred_npho"]) > 0:
        pred_npho = predictions["pred_npho"].astype(np.float32)
        pred_time = predictions["pred_time"].astype(np.float32)
        branch_data["pred_npho"] = pred_npho
        branch_data["pred_time"] = pred_time
        branch_data["err_npho"] = (pred_npho - predictions["truth_npho"]).astype(np.float32)
        branch_data["err_time"] = (pred_time - predictions["truth_time"]).astype(np.float32)

    # Use explicit type specification to avoid awkward import issues
    branch_types = {k: v.dtype for k, v in branch_data.items()}

    with uproot.recreate(root_path) as f:
        f.mktree("tree", branch_types)
        f["tree"].extend(branch_data)

    print(f"[INFO] Saved {n_events} events to {root_path}")
    return root_path


def main():
    parser = argparse.ArgumentParser(
        description="MAE Pre-training for XEC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # CLI mode (legacy)
  python -m lib.train_mae --train_root /path/to/train --epochs 20

  # Config mode
  python -m lib.train_mae --config config/mae_config.yaml

  # Config + CLI override
  python -m lib.train_mae --config config/mae_config.yaml --epochs 50 --lr 5e-5
        """
    )

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data paths (CLI or config override)
    parser.add_argument("--train_root", type=str, default=None, help="Path to Training ROOT file(s)")
    parser.add_argument("--val_root",   type=str, default=None, help="Path to Validation ROOT file(s)")
    parser.add_argument("--save_path",  type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--chunksize",  type=int, default=None, help="Number of events per read")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--num_threads", type=int, default=None, help="CPU preprocessing threads")
    parser.add_argument("--npho_branch", type=str, default=None, help="Input branch for photon counts")
    parser.add_argument("--time_branch", type=str, default=None, help="Input branch for timing")

    parser.add_argument("--npho_scale",     type=float, default=None)
    parser.add_argument("--npho_scale2",    type=float, default=None)
    parser.add_argument("--time_scale",     type=float, default=None)
    parser.add_argument("--time_shift",     type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    parser.add_argument("--outer_mode",           type=str, default=None, choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool",      type=int, nargs=2, default=None)
    parser.add_argument("--mask_ratio",           type=float, default=None)
    parser.add_argument("--lr",                   type=float, default=None)
    parser.add_argument("--lr_scheduler",         type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--lr_min",               type=float, default=None, help="Minimum lr for cosine scheduler")
    parser.add_argument("--warmup_epochs",        type=int, default=None, help="Warmup epochs for cosine scheduler")
    parser.add_argument("--weight_decay",         type=float, default=None)
    parser.add_argument("--loss_fn",              type=str, default=None, choices=["smooth_l1", "mse", "l1", "huber"])
    parser.add_argument("--npho_weight",          type=float, default=None)
    parser.add_argument("--time_weight",          type=float, default=None)
    parser.add_argument("--auto_channel_weight",  action="store_true", help="Enable homoscedastic channel weighting")
    parser.add_argument("--channel_dropout_rate", type=float, default=None)
    parser.add_argument("--grad_clip",            type=float, default=None)
    parser.add_argument("--grad_accum_steps",     type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--ema_decay",            type=float, default=None, help="EMA decay (None to disable)")
    parser.add_argument("--time_mask_ratio_scale", type=float, default=None, help="Scale factor for masking valid-time sensors (1.0=uniform)")
    parser.add_argument("--npho_threshold", type=float, default=None, help="Npho threshold for conditional time loss (raw scale)")
    parser.add_argument("--use_npho_time_weight", action="store_true", help="Weight time loss by sqrt(npho)")
    parser.add_argument("--no_npho_time_weight", action="store_true", help="Disable npho time weighting")
    parser.add_argument("--track_mae_rmse", action="store_true", help="Enable MAE/RMSE metric tracking (slower)")
    parser.add_argument("--no_track_mae_rmse", action="store_true", help="Disable MAE/RMSE metric tracking (faster)")
    parser.add_argument("--track_train_metrics", action="store_true", help="Enable per-face train metrics tracking")
    parser.add_argument("--no_track_train_metrics", action="store_true", help="Disable per-face train metrics (faster)")
    parser.add_argument("--profile", action="store_true", help="Enable training profiler to identify bottlenecks")

    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name",   type=str, default=None)
    parser.add_argument("--resume_from",       type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--save_predictions",  action="store_true", help="Save sensor-level predictions to ROOT")

    args = parser.parse_args()

    # Load config if provided, otherwise use CLI defaults
    if args.config:
        print(f"[INFO] Loading config from: {args.config}")
        cfg = load_mae_config(args.config)

        # Apply CLI overrides
        train_root = args.train_root or cfg.data.train_path
        val_root = args.val_root or cfg.data.val_path or None
        save_path = args.save_path or cfg.checkpoint.save_dir
        epochs = args.epochs if args.epochs is not None else cfg.training.epochs
        batch_size = args.batch_size if args.batch_size is not None else cfg.data.batch_size
        chunksize = args.chunksize if args.chunksize is not None else cfg.data.chunksize
        num_workers = args.num_workers if args.num_workers is not None else cfg.data.num_workers
        num_threads = args.num_threads if args.num_threads is not None else getattr(cfg.data, 'num_threads', 4)
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
        track_mae_rmse = not args.no_track_mae_rmse and getattr(cfg.training, "track_mae_rmse", False)
        track_train_metrics = not args.no_track_train_metrics and getattr(cfg.training, "track_train_metrics", False)
        profile = args.profile
        auto_channel_weight = args.auto_channel_weight or cfg.training.auto_channel_weight
        channel_dropout_rate = args.channel_dropout_rate if args.channel_dropout_rate is not None else cfg.training.channel_dropout_rate
        grad_clip = args.grad_clip if args.grad_clip is not None else getattr(cfg.training, 'grad_clip', 1.0)
        grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else getattr(cfg.training, 'grad_accum_steps', 1)
        ema_decay = args.ema_decay if args.ema_decay is not None else getattr(cfg.training, 'ema_decay', None)
        mlflow_experiment = args.mlflow_experiment or cfg.mlflow.experiment
        mlflow_run_name = args.mlflow_run_name or cfg.mlflow.run_name
        mlflow_new_run = getattr(cfg.checkpoint, 'new_mlflow_run', False)
        resume_from = args.resume_from or cfg.checkpoint.resume_from
        save_predictions = args.save_predictions or getattr(cfg.checkpoint, 'save_predictions', False)
        save_interval = getattr(cfg.checkpoint, 'save_interval', 10)
        use_compile = getattr(cfg.training, 'compile', True)  # Default True for backward compat
    else:
        # Pure CLI mode (legacy) - require train_root
        if not args.train_root:
            parser.error("--train_root is required when not using --config")

        train_root = args.train_root
        val_root = args.val_root
        save_path = args.save_path or "artifacts/mae_run"
        epochs = args.epochs or 20
        batch_size = args.batch_size or 1024
        chunksize = args.chunksize or 256000
        num_workers = args.num_workers or 8
        num_threads = args.num_threads or 4
        npho_branch = args.npho_branch or "relative_npho"
        time_branch = args.time_branch or "relative_time"
        npho_scale = args.npho_scale or DEFAULT_NPHO_SCALE
        npho_scale2 = args.npho_scale2 or DEFAULT_NPHO_SCALE2
        time_scale = args.time_scale or DEFAULT_TIME_SCALE
        time_shift = args.time_shift or DEFAULT_TIME_SHIFT
        sentinel_value = args.sentinel_value or DEFAULT_SENTINEL_VALUE
        outer_mode = args.outer_mode or "finegrid"
        outer_fine_pool = args.outer_fine_pool
        mask_ratio = args.mask_ratio or 0.6
        time_mask_ratio_scale = args.time_mask_ratio_scale or 1.0
        lr = args.lr or 1e-4
        lr_scheduler = args.lr_scheduler
        lr_min = args.lr_min if args.lr_min is not None else 1e-6
        warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else 0
        weight_decay = args.weight_decay or 1e-4
        loss_fn = args.loss_fn or "smooth_l1"
        npho_weight = args.npho_weight or 1.0
        time_weight = args.time_weight or 1.0
        npho_threshold = args.npho_threshold  # None uses DEFAULT_NPHO_THRESHOLD
        use_npho_time_weight = not args.no_npho_time_weight
        track_mae_rmse = args.track_mae_rmse and not args.no_track_mae_rmse
        track_train_metrics = args.track_train_metrics and not args.no_track_train_metrics
        profile = args.profile
        auto_channel_weight = args.auto_channel_weight
        channel_dropout_rate = args.channel_dropout_rate or 0.1
        grad_clip = args.grad_clip or 1.0
        grad_accum_steps = args.grad_accum_steps or 1
        ema_decay = args.ema_decay  # None by default
        mlflow_experiment = args.mlflow_experiment or "mae_pretraining"
        mlflow_run_name = args.mlflow_run_name
        mlflow_new_run = False  # No config file, default to False
        resume_from = args.resume_from
        save_predictions = args.save_predictions
        save_interval = 10
        use_compile = True  # Default for CLI mode

    if lr_scheduler == "none":
        lr_scheduler = None

    def expand_path(p):
        path = os.path.expanduser(p)
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.root"))
            if not files:
                raise ValueError(f"No ROOT files found in directory {path}")
            return sorted(files)
        return [path]

    train_files = expand_path(train_root)
    print(f"[INFO] Training Data: {len(train_files)} files")

    val_files = None
    if val_root:
        val_files = expand_path(val_root)
        print(f"[INFO] Validation Data: {len(val_files)} files")

    # Validate data paths exist
    validate_data_paths(train_root, val_root, expand_func=expand_path)

    # Check artifact directory for existing files
    check_artifact_directory(save_path)

    if torch.cuda.is_available():
        try:
            torch.cuda.get_device_name(0)
            device = torch.device("cuda")
        except Exception:
            print("[WARN] CUDA driver issue detected. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize Model
    outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
    encoder = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool_tuple
    ).to(device)

    model = XEC_MAE(
        encoder, mask_ratio=mask_ratio, learn_channel_logvars=auto_channel_weight,
        sentinel_value=sentinel_value, time_mask_ratio_scale=time_mask_ratio_scale
    ).to(device)
    total_params, trainable_params = count_model_params(model)
    print("[INFO] MAE created:")
    print(f"  - Total params: {total_params:,}")
    print(f"  - Trainable params: {trainable_params:,}")

    # torch.compile requires triton, which is only available on x86_64
    # Can be disabled via config to avoid LLVM/multiprocessing conflicts
    is_arm = platform.machine() in ("aarch64", "arm64")
    if not use_compile:
        print("[INFO] torch.compile disabled via config.")
    elif device.type == "cuda" and not is_arm:
        try:
            import triton  # Check if triton is available
            # Suppress verbose Triton autotuning logs
            import logging
            logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
            print("[INFO] Attempting torch.compile...")
            # Use "reduce-overhead" mode for less verbose output (vs "max-autotune")
            model = torch.compile(model, mode="reduce-overhead", dynamic=False)
        except ImportError:
            print("[INFO] Triton not available, skipping torch.compile.")
        except Exception as e:
            print(f"[WARN] torch.compile failed with error: {e}.")
            print("[INFO] Proceeding with standard Eager mode.")
    elif is_arm:
        print("[INFO] ARM architecture detected: torch.compile disabled (triton not supported).")
    else:
        print("[INFO] Running on CPU: torch.compile is disabled for stability.")
            
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=(device.type == "cuda"),
        weight_decay=weight_decay
    )

    # Initialize EMA model if enabled
    ema_model = None
    if ema_decay is not None and ema_decay > 0:
        print(f"[INFO] EMA enabled with decay={ema_decay}")
        ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))

    # Detect resume to auto-disable warmup
    # When resuming from a checkpoint, warmup is not needed since the model
    # is already past the initial training phase.
    if resume_from and os.path.exists(resume_from) and warmup_epochs > 0:
        try:
            ckpt_probe = torch.load(resume_from, map_location="cpu", weights_only=False)
            if isinstance(ckpt_probe, dict) and "epoch" in ckpt_probe and ckpt_probe.get("epoch", 0) > 0:
                print(f"[INFO] Resuming from checkpoint - disabling warmup (was {warmup_epochs} epochs)")
                warmup_epochs = 0
            del ckpt_probe
        except Exception:
            pass  # Will be handled later in the full resume logic

    scheduler = None
    if lr_scheduler:
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
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == "cuda"))

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    mlflow_run_id = None
    if resume_from and os.path.exists(resume_from):
        print(f"[INFO] Resuming MAE from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if "ema_state_dict" in checkpoint and ema_model is not None:
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
            # Note: scheduler state is intentionally NOT restored to allow
            # configuring new epochs/lr_max/lr_min on resume
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            mlflow_run_id = checkpoint.get('mlflow_run_id', None)
            print(f"[INFO] Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        else:
            print("[WARN] Loaded raw weights. Starting from Epoch 1 (Optimizer reset).")
            model.load_state_dict(checkpoint, strict=False)
            start_epoch = 0

    # Check for valid epoch range
    if start_epoch >= epochs:
        print("\n" + "=" * 70)
        print("[ERROR] No epochs to train!")
        print(f"  Resumed from epoch {start_epoch - 1}, but config has epochs={epochs}")
        print(f"  The training loop range({start_epoch}, {epochs}) is empty.")
        print(f"\n  To continue training, set 'epochs' higher than {start_epoch - 1}.")
        print(f"  For example, to train 10 more epochs, set epochs={start_epoch + 9}")
        print("=" * 70 + "\n")
        raise ValueError(f"start_epoch ({start_epoch}) >= epochs ({epochs}). "
                        f"Set epochs > {start_epoch - 1} to continue training.")

    # Force new MLflow run if requested
    if mlflow_new_run and mlflow_run_id is not None:
        print(f"[INFO] mlflow.new_run=true: Starting fresh MLflow run (ignoring run_id from checkpoint)")
        mlflow_run_id = None

    # MLflow Setup
    # Default to SQLite backend if MLFLOW_TRACKING_URI is not set
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        default_uri = f"sqlite:///{os.getcwd()}/mlruns.db"
        mlflow.set_tracking_uri(default_uri)
        print(f"[INFO] MLflow tracking URI: {default_uri}")
    mlflow.set_experiment(mlflow_experiment)
    os.makedirs(save_path, exist_ok=True)
    print(f"[INFO] Starting MAE Pre-training")
    print(f"  - Experiment: {mlflow_experiment}")
    print(f"  - Run name: {mlflow_run_name}")
    print(f"  - Mask ratio: {mask_ratio}")

    # Disable MLflow's automatic system metrics (uses wall clock time)
    # We log our own system metrics with step=epoch for consistent x-axis
    with mlflow.start_run(run_id=mlflow_run_id, run_name=mlflow_run_name if not mlflow_run_id else None,
                          log_system_metrics=False) as run:
        mlflow_run_id = run.info.run_id
        outer_mode_label = outer_mode
        if outer_mode == "finegrid" and outer_fine_pool:
            pool_str = ",".join(str(x) for x in outer_fine_pool)
            outer_mode_label = f"{outer_mode}, pool [{pool_str}]"

        if auto_channel_weight:
            channel_weights_label = "auto"
        else:
            channel_weights_label = f"npho {npho_weight}, time {time_weight}"

        scheduler_name = lr_scheduler
        if scheduler_name is None and args.config:
            scheduler_name = (
                getattr(cfg.training, "lr_scheduler", None)
                or getattr(cfg.training, "scheduler", None)
            )
        lr_label = f"scheduler:{scheduler_name}" if scheduler_name else lr

        resume_state = "no"
        if resume_from:
            resume_state = f"yes: {resume_from}" if os.path.exists(resume_from) else f"missing: {resume_from}"

        # Log parameters
        mlflow.log_params({
            "train_root": train_root,
            "val_root": val_root,
            "save_path": save_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "chunksize": chunksize,
            "num_workers": num_workers,
            "num_threads": num_threads,
            "npho_scale": npho_scale,
            "npho_scale2": npho_scale2,
            "time_scale": time_scale,
            "time_shift": time_shift,
            "sentinel_value": sentinel_value,
            "outer_mode": outer_mode_label,
            "mask_ratio": mask_ratio,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "lr": lr_label,
            "warmup_epochs": warmup_epochs,
            "weight_decay": weight_decay,
            "loss_fn": loss_fn,
            "channel_weights": channel_weights_label,
            "channel_dropout_rate": channel_dropout_rate,
            "grad_clip": grad_clip,
            "ema_decay": ema_decay,
            "resume_state": resume_state,
        })

        # Training Loop
        print("Starting MAE epoch loop...")
        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            # --- TRAIN ---
            train_metrics = run_epoch_mae(
                model, optimizer, device, train_files, "tree",
                batch_size=batch_size,
                step_size=chunksize,
                npho_branch=npho_branch,
                time_branch=time_branch,
                NphoScale=npho_scale,
                NphoScale2=npho_scale2,
                time_scale=time_scale,
                time_shift=time_shift,
                sentinel_value=sentinel_value,
                loss_fn=loss_fn,
                npho_weight=npho_weight,
                time_weight=time_weight,
                auto_channel_weight=auto_channel_weight,
                channel_dropout_rate=channel_dropout_rate,
                grad_clip=grad_clip,
                grad_accum_steps=grad_accum_steps,
                scaler=scaler,
                num_workers=num_workers,
                npho_threshold=npho_threshold,
                use_npho_time_weight=use_npho_time_weight,
                track_mae_rmse=track_mae_rmse,
                track_train_metrics=track_train_metrics,
                profile=profile,
            )

            # Update EMA model
            if ema_model is not None:
                ema_model.update_parameters(model)

            # --- VALIDATION ---
            val_metrics = {}
            predictions = None
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()

            if val_files:
                collect_preds = save_predictions and ((epoch + 1) % save_interval == 0 or (epoch + 1) == epochs)

                result = run_eval_mae(
                    eval_model, device, val_files, "tree",
                    batch_size=batch_size,
                    step_size=chunksize,
                    npho_branch=npho_branch,
                    time_branch=time_branch,
                    NphoScale=npho_scale,
                    NphoScale2=npho_scale2,
                    time_scale=time_scale,
                    time_shift=time_shift,
                    sentinel_value=sentinel_value,
                    loss_fn=loss_fn,
                    npho_weight=npho_weight,
                    time_weight=time_weight,
                    auto_channel_weight=auto_channel_weight,
                    collect_predictions=collect_preds,
                    max_events=1000,
                    num_workers=num_workers,
                    npho_threshold=npho_threshold,
                    use_npho_time_weight=use_npho_time_weight,
                    track_mae_rmse=track_mae_rmse,
                    profile=profile,
                )

                if collect_preds:
                    val_metrics, predictions = result
                else:
                    val_metrics = result

            dt = time.time() - t0

            # Epoch summary
            train_loss = train_metrics.get("total_loss", 0.0)
            val_loss = val_metrics.get("total_loss", 0.0) if val_metrics else 0.0
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {dt:.1f}s")

            # Log MLflow
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train/{key}", value, step=epoch)

            # Log validation metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val/{key}", value, step=epoch)

            # Log learned channel weights (homoscedastic uncertainty)
            if auto_channel_weight and hasattr(model, "channel_log_vars") and model.channel_log_vars is not None:
                log_vars = model.channel_log_vars.detach()
                # log_var = log(sigma^2), so sigma = exp(log_var / 2)
                # weight = 1 / (2 * sigma^2) = 0.5 * exp(-log_var)
                mlflow.log_metrics({
                    "channel/npho_log_var": log_vars[0].item(),
                    "channel/time_log_var": log_vars[1].item(),
                    "channel/npho_weight": (0.5 * torch.exp(-log_vars[0])).item(),
                    "channel/time_weight": (0.5 * torch.exp(-log_vars[1])).item(),
                }, step=epoch)

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

            # Save predictions
            if predictions is not None:
                try:
                    root_path = save_predictions_to_root(predictions, save_path, epoch, run_id=run.info.run_id)
                    if root_path:
                        mlflow.log_artifact(root_path)
                except Exception as e:
                    print(f"[WARN] Could not save predictions to ROOT: {e}")

            # Check if this is the best model
            is_best = val_loss < best_val_loss if val_metrics else False
            if is_best:
                best_val_loss = val_loss

            # Save model checkpoint
            if (epoch + 1) % save_interval == 0 or (epoch + 1) == epochs or is_best:
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
                    }
                }
                if ema_model is not None:
                    checkpoint_dict['ema_state_dict'] = ema_model.state_dict()
                if scheduler is not None:
                    checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict()

                # Save last checkpoint
                full_ckpt_path = os.path.join(save_path, "mae_checkpoint_last.pth")
                torch.save(checkpoint_dict, full_ckpt_path)
                print(f"Saved MAE checkpoint to {full_ckpt_path}")

                # Save best checkpoint
                if is_best:
                    best_ckpt_path = os.path.join(save_path, "mae_checkpoint_best.pth")
                    torch.save(checkpoint_dict, best_ckpt_path)
                    print(f"Saved best MAE checkpoint to {best_ckpt_path}")

                # Save encoder weights for transfer learning
                encoder_path = os.path.join(save_path, f"mae_encoder_epoch_{epoch+1}.pth")
                encoder_to_save = ema_model.module.encoder if ema_model is not None else model.encoder
                torch.save(encoder_to_save.state_dict(), encoder_path)
                print(f"Saved encoder weights to {encoder_path}")
                mlflow.log_artifact(encoder_path)

        print("[INFO] MAE Pre-training complete!")


if __name__ == "__main__":
    main()
