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
# Suppress LR scheduler warnings (harmless, PyTorch internal)
warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler.step.*")
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*")

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
from contextlib import nullcontext

from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from .models import XECEncoder, XEC_MAE
from .engines import run_epoch_mae, run_eval_mae
from .utils import count_model_params, log_system_metrics_to_mlflow, validate_data_paths, check_artifact_directory
from .geom_defs import DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2, DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME
from .config import load_mae_config
from .reweighting import create_intensity_reweighter_from_config
from .distributed import (
    setup_ddp, cleanup_ddp, is_main_process,
    shard_file_list, reduce_metrics, wrap_ddp, barrier,
)

# Usage
# CLI Mode: python -m lib.train_mae --train_root path/to/data.root --save_path mae_pretrained.pth --epochs 20
# Config Mode: python -m lib.train_mae --config config/mae_config.yaml
# Config + CLI Override: python -m lib.train_mae --config config/mae_config.yaml --epochs 50 --lr 5e-5

# Enable TensorFloat32
torch.set_float32_matmul_precision('high')

# Disable Debugging/Profiling overhead
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)


def save_predictions_to_root(predictions, save_path, epoch, run_id=None,
                             predict_channels=None,
                             npho_scale=None, npho_scale2=None,
                             time_scale=None, time_shift=None,
                             npho_scheme=None):
    """
    Save MAE predictions to ROOT file for analysis.

    Args:
        predictions: dict with keys: truth_npho, truth_time, pred_npho, pred_time, mask, x_masked
        save_path: directory to save the file
        epoch: current epoch number
        predict_channels: List of predicted channels (['npho'] or ['npho', 'time'])
        npho_scale, npho_scale2, time_scale, time_shift: Normalization parameters for metadata
        npho_scheme: Normalization scheme for npho ('log1p', 'anscombe', 'sqrt', 'linear')
    """
    from lib.geom_defs import (
        DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
        DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT
    )

    root_path = os.path.join(save_path, f"mae_predictions_epoch_{epoch+1}.root")

    if predict_channels is None:
        predict_channels = ['npho', 'time']
    predict_time = 'time' in predict_channels

    # Set defaults for normalization params
    if npho_scale is None:
        npho_scale = DEFAULT_NPHO_SCALE
    if npho_scale2 is None:
        npho_scale2 = DEFAULT_NPHO_SCALE2
    if time_scale is None:
        time_scale = DEFAULT_TIME_SCALE
    if time_shift is None:
        time_shift = DEFAULT_TIME_SHIFT
    if npho_scheme is None:
        npho_scheme = "log1p"

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

    # Note: run_id is not stored in ROOT (uproot can't write Unicode strings)
    # The run_id is available in the filename and MLflow artifact metadata

    # Add masked input if available
    if "x_masked" in predictions and len(predictions["x_masked"]) > 0:
        branch_data["masked_npho"] = predictions["x_masked"][:, :, 0].astype(np.float32)
        branch_data["masked_time"] = predictions["x_masked"][:, :, 1].astype(np.float32)

    if "pred_npho" in predictions and len(predictions["pred_npho"]) > 0:
        pred_npho = predictions["pred_npho"].astype(np.float32)
        branch_data["pred_npho"] = pred_npho
        branch_data["err_npho"] = (pred_npho - predictions["truth_npho"]).astype(np.float32)

        # Only add time predictions if time was predicted
        if predict_time and "pred_time" in predictions and len(predictions["pred_time"]) > 0:
            pred_time = predictions["pred_time"].astype(np.float32)
            branch_data["pred_time"] = pred_time
            branch_data["err_time"] = (pred_time - predictions["truth_time"]).astype(np.float32)

    # Use explicit type specification to avoid awkward import issues
    branch_types = {k: v.dtype for k, v in branch_data.items()}

    # Metadata for downstream analysis scripts
    metadata = {
        'predict_channels': np.array([','.join(predict_channels)], dtype='U32'),
        'npho_scale': np.array([npho_scale], dtype=np.float64),
        'npho_scale2': np.array([npho_scale2], dtype=np.float64),
        'time_scale': np.array([time_scale], dtype=np.float64),
        'time_shift': np.array([time_shift], dtype=np.float64),
        'npho_scheme': np.array([npho_scheme], dtype='U32'),
    }

    with uproot.recreate(root_path) as f:
        f.mktree("tree", branch_types)
        f["tree"].extend(branch_data)
        f.mktree('metadata', metadata)

    print(f"[INFO] Saved {n_events} events to {root_path} (predict_channels={predict_channels})")
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
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch factor")
    parser.add_argument("--npho_branch", type=str, default=None, help="Input branch for photon counts")
    parser.add_argument("--time_branch", type=str, default=None, help="Input branch for timing")

    parser.add_argument("--npho_scale",     type=float, default=None)
    parser.add_argument("--npho_scale2",    type=float, default=None)
    parser.add_argument("--time_scale",     type=float, default=None)
    parser.add_argument("--time_shift",     type=float, default=None)
    parser.add_argument("--sentinel_time", type=float, default=None)

    parser.add_argument("--outer_mode",           type=str, default=None, choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool",      type=int, nargs=2, default=None)
    parser.add_argument("--mask_ratio",           type=float, default=None)
    parser.add_argument("--decoder_dim",         type=int, default=None, help="Decoder hidden dimension (default: 128)")
    parser.add_argument("--lr",                   type=float, default=None)
    parser.add_argument("--lr_scheduler",         type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--lr_min",               type=float, default=None, help="Minimum lr for cosine scheduler")
    parser.add_argument("--warmup_epochs",        type=int, default=None, help="Warmup epochs for cosine scheduler")
    parser.add_argument("--weight_decay",         type=float, default=None)
    parser.add_argument("--loss_fn",              type=str, default=None, choices=["smooth_l1", "mse", "l1", "huber"])
    parser.add_argument("--loss_beta",            type=float, default=None, help="Beta for smooth_l1/huber loss")
    parser.add_argument("--npho_weight",          type=float, default=None)
    parser.add_argument("--time_weight",          type=float, default=None)
    parser.add_argument("--auto_channel_weight",  action="store_true", help="Enable homoscedastic channel weighting")
    parser.add_argument("--grad_clip",            type=float, default=None)
    parser.add_argument("--grad_accum_steps",     type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--ema_decay",            type=float, default=None, help="EMA decay (None to disable)")
    parser.add_argument("--time_mask_ratio_scale", type=float, default=None, help="Scale factor for masking valid-time sensors (1.0=uniform)")
    parser.add_argument("--npho_threshold", type=float, default=None, help="Npho threshold for conditional time loss (raw scale)")
    parser.add_argument("--use_npho_time_weight", action="store_true", help="Weight time loss by sqrt(npho)")
    parser.add_argument("--no_npho_time_weight", action="store_true", help="Disable npho time weighting")
    parser.add_argument("--track_mae_rmse", action="store_true", help="Enable MAE/RMSE metric tracking (slower)")
    parser.add_argument("--no_track_mae_rmse", action="store_true", help="Disable MAE/RMSE metric tracking (faster)")
    parser.add_argument("--track_metrics", action="store_true", help="Enable per-face train metrics tracking")
    parser.add_argument("--no_track_metrics", action="store_true", help="Disable per-face train metrics (faster)")
    parser.add_argument("--profile", action="store_true", help="Enable training profiler to identify bottlenecks")
    parser.add_argument("--npho_scheme", type=str, default=None, choices=["log1p", "anscombe", "sqrt", "linear"],
                        help="Normalization scheme for npho (default: log1p)")
    parser.add_argument("--npho_loss_weight_enabled", action="store_true", help="Enable npho loss weighting by intensity")
    parser.add_argument("--npho_loss_weight_alpha", type=float, default=None, help="Exponent for npho loss weighting (default: 0.5)")
    parser.add_argument("--intensity_reweighting_enabled", action="store_true", help="Enable intensity-based sample reweighting")
    parser.add_argument("--intensity_reweighting_nbins", type=int, default=None, help="Number of bins for intensity reweighting")
    parser.add_argument("--intensity_reweighting_target", type=str, default=None, help="Target distribution for intensity reweighting")
    parser.add_argument("--compile", type=str, default=None,
                        choices=["max-autotune", "reduce-overhead", "default", "false", "none"],
                        help="torch.compile mode (default: reduce-overhead, use 'false' to disable)")

    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name",   type=str, default=None)
    parser.add_argument("--resume_from",       type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--reset_epoch",       action="store_true", help="Start from epoch 0 when resuming (only load model weights)")
    parser.add_argument("--refresh_lr",        action="store_true", help="Reset LR scheduler when resuming (schedule for remaining epochs)")
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
        # Auto-limit num_workers on ARM/GH nodes (multiprocessing issues)
        if platform.machine() in ("aarch64", "arm64") and num_workers > 1:
            print(f"[INFO] ARM/GH node detected - limiting num_workers from {num_workers} to 1")
            num_workers = 1
        num_threads = args.num_threads if args.num_threads is not None else getattr(cfg.data, 'num_threads', 4)
        prefetch_factor = args.prefetch_factor if args.prefetch_factor is not None else int(getattr(cfg.data, 'prefetch_factor', 2))
        npho_branch = args.npho_branch or getattr(cfg.data, "npho_branch", "npho")
        time_branch = args.time_branch or getattr(cfg.data, "time_branch", "relative_time")
        log_invalid_npho = getattr(cfg.data, "log_invalid_npho", True)
        npho_scale = float(args.npho_scale if args.npho_scale is not None else cfg.normalization.npho_scale)
        npho_scale2 = float(args.npho_scale2 if args.npho_scale2 is not None else cfg.normalization.npho_scale2)
        time_scale = float(args.time_scale if args.time_scale is not None else cfg.normalization.time_scale)
        time_shift = float(args.time_shift if args.time_shift is not None else cfg.normalization.time_shift)
        sentinel_time = float(args.sentinel_time if args.sentinel_time is not None else cfg.normalization.sentinel_time)
        sentinel_npho = float(getattr(cfg.normalization, 'sentinel_npho', -1.0))
        outer_mode = args.outer_mode or cfg.model.outer_mode
        outer_fine_pool = args.outer_fine_pool or cfg.model.outer_fine_pool
        mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.model.mask_ratio
        # time_mask_ratio_scale moved to nested time config
        time_mask_ratio_scale = args.time_mask_ratio_scale if args.time_mask_ratio_scale is not None else cfg.training.time.mask_ratio_scale
        lr = float(args.lr if args.lr is not None else cfg.training.lr)
        lr_scheduler = args.lr_scheduler or getattr(cfg.training, "lr_scheduler", None)
        lr_min = float(args.lr_min if args.lr_min is not None else getattr(cfg.training, "lr_min", 1e-6))
        warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else getattr(cfg.training, "warmup_epochs", 0)
        weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg.training.weight_decay)
        loss_fn = args.loss_fn or cfg.training.loss_fn
        loss_beta = args.loss_beta if args.loss_beta is not None else getattr(cfg.training, 'loss_beta', 1.0)
        npho_weight = args.npho_weight if args.npho_weight is not None else cfg.training.npho_weight
        # Get time config from nested structure
        time_weight = args.time_weight if args.time_weight is not None else cfg.training.time.weight
        npho_threshold = args.npho_threshold if args.npho_threshold is not None else cfg.training.time.npho_threshold
        use_npho_time_weight = not args.no_npho_time_weight and cfg.training.time.use_npho_weight
        # predict_channels controls output channels (npho-only or npho+time)
        predict_channels = cfg.model.predict_channels
        decoder_dim = args.decoder_dim if args.decoder_dim is not None else getattr(cfg.model, 'decoder_dim', 128)
        encoder_dim = cfg.model.encoder_dim
        encoder_dim_feedforward = cfg.model.dim_feedforward
        encoder_num_fusion_layers = cfg.model.num_fusion_layers
        track_mae_rmse = not args.no_track_mae_rmse and getattr(cfg.training, "track_mae_rmse", False)
        track_metrics = not args.no_track_metrics and getattr(cfg.training, "track_metrics", False)
        profile = args.profile or getattr(cfg.training, 'profile', False)
        auto_channel_weight = args.auto_channel_weight or cfg.training.auto_channel_weight
        grad_clip = args.grad_clip if args.grad_clip is not None else getattr(cfg.training, 'grad_clip', 1.0)
        grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else getattr(cfg.training, 'grad_accum_steps', 1)
        ema_decay = args.ema_decay if args.ema_decay is not None else getattr(cfg.training, 'ema_decay', None)
        amp = getattr(cfg.training, 'amp', True)
        mlflow_experiment = args.mlflow_experiment or cfg.mlflow.experiment
        mlflow_run_name = args.mlflow_run_name or cfg.mlflow.run_name
        mlflow_new_run = getattr(cfg.checkpoint, 'new_mlflow_run', False)
        resume_from = os.path.expanduser(args.resume_from or cfg.checkpoint.resume_from or "")  or None
        reset_epoch = args.reset_epoch or getattr(cfg.checkpoint, 'reset_epoch', False)
        refresh_lr = args.refresh_lr or getattr(cfg.checkpoint, 'refresh_lr', False)
        save_predictions = args.save_predictions or getattr(cfg.checkpoint, 'save_predictions', False)
        save_interval = getattr(cfg.checkpoint, 'save_interval', 10)
        root_save_interval = getattr(cfg.checkpoint, 'root_save_interval', 10)
        # Handle compile option: can be string mode or boolean (for backward compat)
        compile_cfg = getattr(cfg.training, 'compile', 'reduce-overhead')
        if isinstance(compile_cfg, bool):
            compile_mode = 'reduce-overhead' if compile_cfg else 'none'
        else:
            compile_mode = compile_cfg if compile_cfg else 'reduce-overhead'
        if args.compile is not None:
            compile_mode = args.compile
        compile_fullgraph = getattr(cfg.training, 'compile_fullgraph', False)
        # New normalization and loss weighting options
        npho_scheme = args.npho_scheme or getattr(cfg.normalization, 'npho_scheme', 'log1p')
        npho_loss_weight_enabled = args.npho_loss_weight_enabled or cfg.training.npho_loss_weight.enabled
        npho_loss_weight_alpha = args.npho_loss_weight_alpha if args.npho_loss_weight_alpha is not None else cfg.training.npho_loss_weight.alpha
        intensity_reweighting_enabled = args.intensity_reweighting_enabled or cfg.training.intensity_reweighting.enabled
        intensity_reweighting_nbins = args.intensity_reweighting_nbins if args.intensity_reweighting_nbins is not None else cfg.training.intensity_reweighting.nbins
        intensity_reweighting_target = args.intensity_reweighting_target or cfg.training.intensity_reweighting.target
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
        # Auto-limit num_workers on ARM/GH nodes (multiprocessing issues)
        if platform.machine() in ("aarch64", "arm64") and num_workers > 1:
            print(f"[INFO] ARM/GH node detected - limiting num_workers from {num_workers} to 1")
            num_workers = 1
        num_threads = args.num_threads or 4
        prefetch_factor = args.prefetch_factor or 2
        npho_branch = args.npho_branch or "npho"
        time_branch = args.time_branch or "relative_time"
        log_invalid_npho = True  # Default: enabled
        npho_scale = args.npho_scale or DEFAULT_NPHO_SCALE
        npho_scale2 = args.npho_scale2 or DEFAULT_NPHO_SCALE2
        time_scale = args.time_scale or DEFAULT_TIME_SCALE
        time_shift = args.time_shift or DEFAULT_TIME_SHIFT
        sentinel_time = args.sentinel_time or DEFAULT_SENTINEL_TIME
        sentinel_npho = -1.0
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
        loss_beta = args.loss_beta if args.loss_beta is not None else 1.0
        npho_weight = args.npho_weight or 1.0
        time_weight = args.time_weight or 1.0
        npho_threshold = args.npho_threshold  # None uses DEFAULT_NPHO_THRESHOLD
        use_npho_time_weight = not args.no_npho_time_weight
        predict_channels = ["npho", "time"]  # Default: predict both channels
        decoder_dim = args.decoder_dim if args.decoder_dim is not None else 128
        encoder_dim = 1024
        encoder_dim_feedforward = None
        encoder_num_fusion_layers = 2
        track_mae_rmse = args.track_mae_rmse and not args.no_track_mae_rmse
        track_metrics = args.track_metrics and not args.no_track_metrics
        profile = args.profile
        auto_channel_weight = args.auto_channel_weight
        grad_clip = args.grad_clip or 1.0
        grad_accum_steps = args.grad_accum_steps or 1
        ema_decay = args.ema_decay  # None by default
        amp = True  # Always enabled in CLI mode
        mlflow_experiment = args.mlflow_experiment or "mae_pretraining"
        mlflow_run_name = args.mlflow_run_name
        mlflow_new_run = False  # No config file, default to False
        resume_from = os.path.expanduser(args.resume_from) if args.resume_from else None
        reset_epoch = args.reset_epoch
        refresh_lr = args.refresh_lr
        save_predictions = args.save_predictions
        save_interval = 10
        root_save_interval = 10
        compile_mode = args.compile if args.compile is not None else 'reduce-overhead'
        compile_fullgraph = False  # Default for CLI mode
        # New normalization and loss weighting options (CLI defaults)
        npho_scheme = args.npho_scheme or "log1p"
        npho_loss_weight_enabled = args.npho_loss_weight_enabled
        npho_loss_weight_alpha = args.npho_loss_weight_alpha if args.npho_loss_weight_alpha is not None else 0.5
        intensity_reweighting_enabled = args.intensity_reweighting_enabled
        intensity_reweighting_nbins = args.intensity_reweighting_nbins if args.intensity_reweighting_nbins is not None else 5
        intensity_reweighting_target = args.intensity_reweighting_target or "uniform"

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

    # DDP setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        print(f"[INFO] Using device: {device}" + (f" (world_size={world_size})" if world_size > 1 else ""))

    train_files = expand_path(train_root)
    val_files = expand_path(val_root) if val_root else None
    all_val_files = val_files  # Keep full list for prediction saving (rank 0)

    # Shard file lists across ranks
    if world_size > 1:
        train_files = shard_file_list(train_files, rank, world_size)
        if val_files:
            val_files = shard_file_list(val_files, rank, world_size)

    if is_main_process():
        print(f"[INFO] Training Data: {len(train_files)} files" + (f" (per rank)" if world_size > 1 else ""))
        if val_files:
            print(f"[INFO] Validation Data: {len(val_files)} files" + (f" (per rank)" if world_size > 1 else ""))

    # Validate data paths exist
    validate_data_paths(train_root, val_root, expand_func=expand_path)

    # Check artifact directory for existing files
    check_artifact_directory(save_path)

    # Initialize Model
    outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
    encoder = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool_tuple,
        sentinel_time=sentinel_time,
        encoder_dim=encoder_dim,
        dim_feedforward=encoder_dim_feedforward,
        num_fusion_layers=encoder_num_fusion_layers,
    ).to(device)

    model = XEC_MAE(
        encoder, mask_ratio=mask_ratio, learn_channel_logvars=auto_channel_weight,
        sentinel_time=sentinel_time, time_mask_ratio_scale=time_mask_ratio_scale,
        predict_channels=predict_channels, decoder_dim=decoder_dim,
        sentinel_npho=sentinel_npho
    ).to(device)
    total_params, trainable_params = count_model_params(model)
    if is_main_process():
        print("[INFO] MAE created:")
        print(f"  - Total params: {total_params:,}")
        print(f"  - Trainable params: {trainable_params:,}")
        print(f"  - Predict channels: {predict_channels}")
        print(f"  - Decoder dim: {decoder_dim}")

    # Keep unwrapped reference for checkpointing and EMA
    model_without_ddp = model

    # Wrap with DDP (before compile, after .to(device))
    model = wrap_ddp(model, local_rank)
    model_ddp = model  # Save DDP reference before compile (for .no_sync access)

    # torch.compile - auto-detect ARM architecture and disable (Triton not supported)
    is_arm = platform.machine() in ("aarch64", "arm64")
    if is_arm and compile_mode and compile_mode not in ('false', 'none'):
        if is_main_process():
            print(f"[INFO] ARM architecture detected - disabling torch.compile (Triton not supported)")
        compile_mode = 'none'

    if compile_mode and compile_mode not in ('false', 'none'):
        if device.type == "cuda":
            try:
                import triton  # Check if triton is available
                # Suppress verbose Triton autotuning logs
                logging.getLogger("torch._inductor.autotune_process").setLevel(logging.WARNING)
                # Increase dynamo cache limit to avoid CacheLimitExceeded errors
                # when batch sizes vary (e.g., last batch of epoch)
                torch._dynamo.config.cache_size_limit = 64
                fg_str = "fullgraph" if compile_fullgraph else "partial"
                if is_main_process():
                    print(f"[INFO] Compiling model with mode='{compile_mode}', {fg_str} (this may take a few minutes...)")
                model = torch.compile(model, mode=compile_mode, fullgraph=compile_fullgraph, dynamic=False)
            except ImportError:
                if is_main_process():
                    print("[INFO] Triton not available, skipping torch.compile.")
            except Exception as e:
                if is_main_process():
                    print(f"[WARN] torch.compile failed with error: {e}.")
                    print("[INFO] Proceeding with standard Eager mode.")
        else:
            if is_main_process():
                print("[INFO] Running on CPU: torch.compile is disabled for stability.")
    else:
        if is_main_process():
            print("[INFO] torch.compile disabled via config.")
            
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=(device.type == "cuda"),
        weight_decay=weight_decay
    )

    # Initialize EMA model if enabled (from unwrapped model)
    ema_model = None
    if ema_decay is not None and ema_decay > 0:
        if is_main_process():
            print(f"[INFO] EMA enabled with decay={ema_decay}")
        ema_model = AveragedModel(model_without_ddp, multi_avg_fn=get_ema_multi_avg_fn(ema_decay))
        ema_model.to(device)

    # Detect resume to auto-disable warmup
    # When resuming from a checkpoint, warmup is not needed since the model
    # is already past the initial training phase.
    if resume_from and not os.path.exists(resume_from):
        raise FileNotFoundError(
            f"resume_from checkpoint not found: {resume_from}"
        )
    if resume_from and os.path.exists(resume_from) and warmup_epochs > 0:
        try:
            ckpt_probe = torch.load(resume_from, map_location="cpu", weights_only=False)
            if isinstance(ckpt_probe, dict) and "epoch" in ckpt_probe and ckpt_probe.get("epoch", 0) > 0:
                if is_main_process():
                    print(f"[INFO] Resuming from checkpoint - disabling warmup (was {warmup_epochs} epochs)")
                warmup_epochs = 0
            del ckpt_probe
        except Exception:
            pass  # Will be handled later in the full resume logic

    scheduler = None
    if lr_scheduler:
        if lr_scheduler == "cosine":
            if warmup_epochs >= epochs:
                if is_main_process():
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
                if is_main_process():
                    print(f"[INFO] Using CosineAnnealingLR with {warmup_epochs} warmup epochs (eta_min={lr_min})")
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
                if is_main_process():
                    print(f"[INFO] Using CosineAnnealingLR with eta_min={lr_min}")
        else:
            raise ValueError(f"Unsupported lr_scheduler: {lr_scheduler}")

    # Initialize GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=(amp and device.type == "cuda"))

    # Intensity-based sample reweighter
    intensity_reweighter = None
    if intensity_reweighting_enabled:
        intensity_reweighter = create_intensity_reweighter_from_config({
            'enabled': True,
            'nbins': intensity_reweighting_nbins,
            'target': intensity_reweighting_target,
        })
        if intensity_reweighter is not None:
            intensity_reweighter.fit(
                train_files, "tree",
                npho_branch=npho_branch,
                step_size=chunksize,
                npho_scale=npho_scale,
                npho_scale2=npho_scale2,
                npho_scheme=npho_scheme,
            )
        if intensity_reweighter is None or not intensity_reweighter.is_enabled:
            if is_main_process():
                print("[INFO] Intensity reweighting disabled or failed to fit.")
            intensity_reweighter = None

    # Resume from checkpoint if provided
    start_epoch = 0
    best_val_loss = float('inf')
    mlflow_run_id = None
    if resume_from and os.path.exists(resume_from):
        if is_main_process():
            print(f"[INFO] Resuming MAE from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
            if "optimizer_state_dict" in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                except ValueError as e:
                    if is_main_process():
                        print(f"[WARN] Could not load optimizer state (parameter groups changed): {e}")
                        print("[WARN] Using fresh optimizer")
            if "ema_state_dict" in checkpoint and ema_model is not None:
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

            checkpoint_epoch = checkpoint.get('epoch', 0)
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            mlflow_run_id = checkpoint.get('mlflow_run_id', None)

            # Handle reset_epoch: start from epoch 0 (only load weights)
            if reset_epoch:
                start_epoch = 0
                if is_main_process():
                    print(f"[INFO] reset_epoch=True: Starting from epoch 0 (weights loaded from epoch {checkpoint_epoch})")
            else:
                start_epoch = checkpoint_epoch + 1

            # Handle refresh_lr: recreate scheduler for remaining epochs
            if scheduler is not None and "scheduler_state_dict" in checkpoint:
                if refresh_lr:
                    remaining_epochs = epochs - start_epoch
                    if is_main_process():
                        print(f"[INFO] refresh_lr=True: Creating fresh scheduler with lr={lr}, "
                              f"T_max={remaining_epochs} (epochs {start_epoch}-{epochs - 1})")
                    if lr_scheduler == 'cosine':
                        scheduler = CosineAnnealingLR(optimizer, T_max=remaining_epochs, eta_min=lr_min)
                else:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    if is_main_process():
                        print(f"[INFO] Restored scheduler state.")

            if is_main_process():
                print(f"[INFO] Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.6f}")
        else:
            if is_main_process():
                print("[WARN] Loaded raw weights. Starting from Epoch 0 (Optimizer reset).")
            model_without_ddp.load_state_dict(checkpoint, strict=False)
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
        if is_main_process():
            print(f"[INFO] mlflow.new_run=true: Starting fresh MLflow run (ignoring run_id from checkpoint)")
        mlflow_run_id = None

    # MLflow Setup (rank 0 only)
    os.makedirs(save_path, exist_ok=True)
    if is_main_process():
        # Default to SQLite backend if MLFLOW_TRACKING_URI is not set
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            default_uri = f"sqlite:///{os.getcwd()}/mlruns.db"
            mlflow.set_tracking_uri(default_uri)
            print(f"[INFO] MLflow tracking URI: {default_uri}")
        mlflow.set_experiment(mlflow_experiment)
        print(f"[INFO] Starting MAE Pre-training")
        print(f"  - Experiment: {mlflow_experiment}")
        print(f"  - Run name: {mlflow_run_name}")
        print(f"  - Mask ratio: {mask_ratio}")

    # Disable MLflow's automatic system metrics (uses wall clock time)
    # We log our own system metrics with step=epoch for consistent x-axis
    # Only rank 0 interacts with MLflow
    _is_fresh_mlflow_run = (mlflow_run_id is None)  # Before start_run reassigns it
    mlflow_ctx = (
        mlflow.start_run(run_id=mlflow_run_id, run_name=mlflow_run_name if not mlflow_run_id else None,
                         log_system_metrics=False)
        if is_main_process()
        else nullcontext()
    )
    with mlflow_ctx as run:
        if is_main_process():
            mlflow_run_id = run.info.run_id

        # Determine no_sync context for gradient accumulation
        no_sync_ctx = model_ddp.no_sync if world_size > 1 else None

        if is_main_process() and _is_fresh_mlflow_run:
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

            # Log parameters (only on fresh runs — MLflow params are immutable)
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
                "sentinel_time": sentinel_time,
                "sentinel_npho": sentinel_npho,
                "npho_scheme": npho_scheme,
                "outer_mode": outer_mode_label,
                "mask_ratio": mask_ratio,
                "decoder_dim": decoder_dim,
                "predict_channels": ",".join(predict_channels),
                "encoder_dim": encoder_dim,
                "dim_feedforward": encoder_dim_feedforward,
                "num_fusion_layers": encoder_num_fusion_layers,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "lr": lr_label,
                "warmup_epochs": warmup_epochs,
                "weight_decay": weight_decay,
                "loss_fn": loss_fn,
                "loss_beta": loss_beta,
                "channel_weights": channel_weights_label,
                "grad_clip": grad_clip,
                "grad_accum_steps": grad_accum_steps,
                "compile": compile_mode,
                "ema_decay": ema_decay,
                "resume_state": resume_state,
                "npho_loss_weight_enabled": npho_loss_weight_enabled,
                "npho_loss_weight_alpha": npho_loss_weight_alpha,
                "intensity_reweighting_enabled": intensity_reweighting_enabled,
                "world_size": world_size,
            })

        # Training Loop
        if is_main_process():
            print("Starting MAE epoch loop...")
        for epoch in range(start_epoch, epochs):
            t0 = time.time()

            # --- TRAIN ---
            train_metrics = run_epoch_mae(
                model, optimizer, device, train_files, "tree",
                batch_size=batch_size,
                step_size=chunksize,
                amp=amp,
                npho_branch=npho_branch,
                time_branch=time_branch,
                npho_scale=npho_scale,
                npho_scale2=npho_scale2,
                time_scale=time_scale,
                time_shift=time_shift,
                sentinel_time=sentinel_time,
                loss_fn=loss_fn,
                loss_beta=loss_beta,
                npho_weight=npho_weight,
                time_weight=time_weight,
                auto_channel_weight=auto_channel_weight,
                grad_clip=grad_clip,
                grad_accum_steps=grad_accum_steps,
                scaler=scaler,
                dataloader_workers=0,  # Dataset handles batching internally
                dataset_workers=num_threads,
                prefetch_factor=prefetch_factor,
                npho_threshold=npho_threshold,
                use_npho_time_weight=use_npho_time_weight,
                track_mae_rmse=track_mae_rmse,
                track_metrics=track_metrics,
                profile=profile and is_main_process(),
                log_invalid_npho=log_invalid_npho,
                npho_scheme=npho_scheme,
                npho_loss_weight_enabled=npho_loss_weight_enabled,
                npho_loss_weight_alpha=npho_loss_weight_alpha,
                intensity_reweighter=intensity_reweighter,
                no_sync_ctx=no_sync_ctx,
                sentinel_npho=sentinel_npho,
                ema_model=ema_model,
            )
            train_metrics = reduce_metrics(train_metrics, device)

            # --- VALIDATION ---
            val_metrics = {}
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()

            if val_files:
                val_metrics = run_eval_mae(
                    eval_model, device, val_files, "tree",
                    batch_size=batch_size,
                    step_size=chunksize,
                    amp=amp,
                    npho_branch=npho_branch,
                    time_branch=time_branch,
                    npho_scale=npho_scale,
                    npho_scale2=npho_scale2,
                    time_scale=time_scale,
                    time_shift=time_shift,
                    sentinel_time=sentinel_time,
                    loss_fn=loss_fn,
                    loss_beta=loss_beta,
                    npho_weight=npho_weight,
                    time_weight=time_weight,
                    auto_channel_weight=auto_channel_weight,
                    collect_predictions=False,
                    max_events=1000,
                    dataloader_workers=0,  # Dataset handles batching internally
                    dataset_workers=num_threads,
                    prefetch_factor=prefetch_factor,
                    npho_threshold=npho_threshold,
                    use_npho_time_weight=use_npho_time_weight,
                    track_mae_rmse=track_mae_rmse,
                    profile=profile and is_main_process(),
                    log_invalid_npho=log_invalid_npho,
                    npho_scheme=npho_scheme,
                    npho_loss_weight_enabled=npho_loss_weight_enabled,
                    npho_loss_weight_alpha=npho_loss_weight_alpha,
                    sentinel_npho=sentinel_npho,
                )

            if val_metrics:
                val_metrics = reduce_metrics(val_metrics, device)

            dt = time.time() - t0

            # Epoch summary
            train_loss = train_metrics.get("total_loss", 0.0)
            val_loss = val_metrics.get("total_loss", 0.0) if val_metrics else 0.0

            # Learning rate
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process():
                print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {dt:.1f}s")

            # --- Logging & checkpointing (rank 0 only) ---
            if is_main_process():
                # Log MLflow
                for key, value in train_metrics.items():
                    mlflow.log_metric(f"train/{key}", value, step=epoch)

                # Log validation metrics
                if val_metrics:
                    for key, value in val_metrics.items():
                        mlflow.log_metric(f"val/{key}", value, step=epoch)

                # Log learned channel weights (homoscedastic uncertainty)
                if auto_channel_weight and hasattr(model_without_ddp, "channel_log_vars") and model_without_ddp.channel_log_vars is not None:
                    log_vars = model_without_ddp.channel_log_vars.detach()
                    mlflow.log_metrics({
                        "channel/npho_log_var": log_vars[0].item(),
                        "channel/time_log_var": log_vars[1].item(),
                        "channel/npho_weight": (0.5 * torch.exp(-log_vars[0])).item(),
                        "channel/time_weight": (0.5 * torch.exp(-log_vars[1])).item(),
                    }, step=epoch)

                # System metrics (standardized)
                log_system_metrics_to_mlflow(
                    step=epoch,
                    device=device,
                    epoch_time_sec=dt,
                    lr=current_lr,
                )

                # Save predictions (use full val files, not sharded)
                if save_predictions and all_val_files and ((epoch + 1) % root_save_interval == 0 or (epoch + 1) == epochs):
                    try:
                        _, predictions = run_eval_mae(
                            eval_model, device, all_val_files, "tree",
                            batch_size=batch_size,
                            step_size=chunksize,
                            amp=amp,
                            npho_branch=npho_branch,
                            time_branch=time_branch,
                            npho_scale=npho_scale,
                            npho_scale2=npho_scale2,
                            time_scale=time_scale,
                            time_shift=time_shift,
                            sentinel_time=sentinel_time,
                            loss_fn=loss_fn,
                            loss_beta=loss_beta,
                            npho_weight=npho_weight,
                            time_weight=time_weight,
                            auto_channel_weight=auto_channel_weight,
                            collect_predictions=True,
                            max_events=1000,
                            dataloader_workers=0,
                            dataset_workers=num_threads,
                            prefetch_factor=prefetch_factor,
                            npho_threshold=npho_threshold,
                            use_npho_time_weight=use_npho_time_weight,
                            track_mae_rmse=track_mae_rmse,
                            profile=False,
                            log_invalid_npho=log_invalid_npho,
                            npho_scheme=npho_scheme,
                            npho_loss_weight_enabled=npho_loss_weight_enabled,
                            npho_loss_weight_alpha=npho_loss_weight_alpha,
                            sentinel_npho=sentinel_npho,
                        )
                        root_path = save_predictions_to_root(
                            predictions, save_path, epoch, run_id=mlflow_run_id,
                            predict_channels=predict_channels,
                            npho_scale=npho_scale, npho_scale2=npho_scale2,
                            time_scale=time_scale, time_shift=time_shift,
                            npho_scheme=npho_scheme
                        )
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
                        'model_state_dict': model_without_ddp.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                        'best_val_loss': best_val_loss,
                        'mlflow_run_id': mlflow_run_id,
                        'config': {
                            'outer_mode': outer_mode,
                            'outer_fine_pool': outer_fine_pool,
                            'mask_ratio': mask_ratio,
                            'decoder_dim': decoder_dim,
                            'predict_channels': list(predict_channels),
                            'sentinel_time': float(sentinel_time),
                            'sentinel_npho': float(sentinel_npho),
                            'npho_scale': float(npho_scale),
                            'npho_scale2': float(npho_scale2),
                            'time_scale': float(time_scale),
                            'time_shift': float(time_shift),
                            'npho_scheme': npho_scheme,
                            'encoder_dim': encoder_dim,
                            'dim_feedforward': encoder_dim_feedforward,
                            'num_fusion_layers': encoder_num_fusion_layers,
                            'npho_branch': npho_branch,
                            'time_branch': time_branch,
                        }
                    }
                    checkpoint_dict['ema_state_dict'] = ema_model.state_dict() if ema_model is not None else None
                    checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict() if scheduler is not None else None

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
                    encoder_to_save = ema_model.module.encoder if ema_model is not None else model_without_ddp.encoder
                    torch.save(encoder_to_save.state_dict(), encoder_path)
                    print(f"Saved encoder weights to {encoder_path}")
                    mlflow.log_artifact(encoder_path)

            # Barrier: all ranks wait for rank 0 to finish checkpointing/ROOT saving
            barrier()

    if is_main_process():
        print("[INFO] MAE Pre-training complete!")
    cleanup_ddp()


if __name__ == "__main__":
    main()
