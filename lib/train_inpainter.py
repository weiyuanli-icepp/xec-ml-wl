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

# Suppress Triton autotuning verbose output (must be set BEFORE importing torch)
os.environ.setdefault("TORCHINDUCTOR_LOG_LEVEL", "WARNING")
os.environ.setdefault("TRITON_PRINT_AUTOTUNING", "0")

import torch
import argparse
import time
import glob
import platform
import mlflow
from contextlib import nullcontext

from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .models import XECEncoder, XEC_Inpainter
from .engines.inpainter import (
    run_epoch_inpainter,
    run_eval_inpainter,
    save_predictions_to_root,
    RootPredictionWriter,
)
from .utils import log_system_metrics_to_mlflow, validate_data_paths, check_artifact_directory
from .reweighting import create_intensity_reweighter_from_config
from .geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME
)
from .config import load_inpainter_config
from .distributed import (
    setup_ddp, cleanup_ddp, is_main_process,
    shard_file_list, reduce_metrics, wrap_ddp,
)

# Suppress common harmless warnings
warnings.filterwarnings("ignore", message=".*To copy construct from a tensor.*", category=UserWarning)
warnings.filterwarnings("ignore", module="torch._dynamo.*")
warnings.filterwarnings("ignore", module="torch.fx.*")
# Suppress scheduler warnings (harmless, PyTorch internal)
warnings.filterwarnings("ignore", message=".*epoch parameter in.*scheduler.step.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*lr_scheduler.step.*before.*optimizer.step.*", category=UserWarning)

# Enable TensorFloat32
torch.set_float32_matmul_precision('high')

# Anomaly detection (enable for debugging NaN issues: torch.autograd.set_detect_anomaly(True))
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.emit_nvtx(False)


def load_mae_encoder(checkpoint_path: str, device: torch.device, outer_mode: str = "finegrid", outer_fine_pool=None,
                     encoder_dim: int = 1024, dim_feedforward=None, num_fusion_layers: int = 2,
                     sentinel_time=None):
    """
    Load encoder weights from MAE checkpoint.

    Args:
        checkpoint_path: path to MAE checkpoint
        device: torch device
        outer_mode: encoder outer mode
        outer_fine_pool: encoder outer fine pool config
        encoder_dim: d_model for fusion transformer
        dim_feedforward: FFN dim (default = encoder_dim * 4)
        num_fusion_layers: number of transformer layers
        sentinel_time: sentinel value for time channel

    Returns:
        encoder: XECEncoder with loaded weights
    """
    print(f"[INFO] Loading MAE encoder from {checkpoint_path}")

    # Create encoder
    outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
    encoder = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool_tuple,
        sentinel_time=sentinel_time,
        encoder_dim=encoder_dim,
        dim_feedforward=dim_feedforward,
        num_fusion_layers=num_fusion_layers,
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
    parser.add_argument("--prefetch_factor", type=int, default=None, help="DataLoader prefetch factor")
    parser.add_argument("--npho_branch", type=str, default=None, help="Input branch for photon counts")
    parser.add_argument("--time_branch", type=str, default=None, help="Input branch for timing")

    # Normalization
    parser.add_argument("--npho_scale", type=float, default=None)
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_time", type=float, default=None)

    # Model
    parser.add_argument("--outer_mode", type=str, default=None, choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool", type=int, nargs=2, default=None)
    parser.add_argument("--mask_ratio", type=float, default=None, help="Mask ratio for training (default 0.05)")
    parser.add_argument("--freeze_encoder", action="store_true", help="Freeze encoder (default)")
    parser.add_argument("--finetune_encoder", action="store_true", help="Fine-tune encoder (not frozen)")
    parser.add_argument("--global_only", action="store_true",
                        help="Disable local context in inpainting heads (ablation: global latent only, like MAE decoder)")

    # Optimizer
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)

    # Loss
    parser.add_argument("--loss_fn", type=str, default=None, choices=["smooth_l1", "mse", "l1", "huber"])
    parser.add_argument("--loss_beta", type=float, default=None, help="Beta for smooth_l1/huber loss (default 1.0)")
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
    parser.add_argument("--compile", type=str, default=None,
                        choices=["max-autotune", "reduce-overhead", "default", "false", "none"],
                        help="torch.compile mode (default: reduce-overhead, use 'false' to disable)")
    parser.add_argument("--ema_decay", type=float, default=None, help="EMA decay rate (None to disable, 0.999 typical)")
    parser.add_argument("--npho_scheme", type=str, default=None, choices=["log1p", "anscombe", "sqrt", "linear"],
                        help="Normalization scheme for npho (default: log1p)")
    parser.add_argument("--npho_loss_weight_enabled", action="store_true", help="Enable npho loss weighting by intensity")
    parser.add_argument("--npho_loss_weight_alpha", type=float, default=None, help="Exponent for npho loss weighting (default: 0.5)")
    parser.add_argument("--intensity_reweighting_enabled", action="store_true", help="Enable intensity-based sample reweighting")
    parser.add_argument("--intensity_reweighting_nbins", type=int, default=None, help="Number of bins for intensity reweighting")
    parser.add_argument("--intensity_reweighting_target", type=str, default=None, help="Target distribution for intensity reweighting")

    # MLflow
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    # Checkpoint
    parser.add_argument("--resume_from", type=str, default=None, help="Resume from inpainter checkpoint")
    parser.add_argument("--reset_epoch", action="store_true", help="Start from epoch 0 when resuming (only load model weights)")
    parser.add_argument("--refresh_lr", action="store_true", help="Reset LR scheduler when resuming (schedule for remaining epochs)")
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
        # Auto-limit num_workers on ARM/GH nodes (multiprocessing issues)
        if platform.machine() in ("aarch64", "arm64") and num_workers > 1:
            print(f"[INFO] ARM/GH node detected - limiting num_workers from {num_workers} to 1")
            num_workers = 1
        num_threads = args.num_threads if args.num_threads is not None else cfg.data.num_threads
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
        freeze_encoder = cfg.model.freeze_encoder if not args.finetune_encoder else False
        # predict_channels controls output channels (npho-only or npho+time)
        predict_channels = cfg.model.predict_channels
        # --global_only disables local context (ablation study)
        use_local_context = not args.global_only and getattr(cfg.model, "use_local_context", True)
        use_masked_attention = getattr(cfg.model, "use_masked_attention", False)
        head_type = getattr(cfg.model, "head_type", "per_face")
        sensor_positions_file = getattr(cfg.model, "sensor_positions_file", None)
        cross_attn_k = getattr(cfg.model, "cross_attn_k", 16)
        cross_attn_hidden = getattr(cfg.model, "cross_attn_hidden", 64)
        cross_attn_latent_dim = getattr(cfg.model, "cross_attn_latent_dim", 128)
        cross_attn_pos_dim = getattr(cfg.model, "cross_attn_pos_dim", 96)
        encoder_dim = cfg.model.encoder_dim
        encoder_dim_feedforward = cfg.model.dim_feedforward
        encoder_num_fusion_layers = cfg.model.num_fusion_layers

        lr = float(args.lr if args.lr is not None else cfg.training.lr)
        lr_scheduler = args.lr_scheduler or getattr(cfg.training, "lr_scheduler", None)
        lr_min = float(args.lr_min if args.lr_min is not None else getattr(cfg.training, "lr_min", 1e-6))
        warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else getattr(cfg.training, "warmup_epochs", 0)
        weight_decay = float(args.weight_decay if args.weight_decay is not None else cfg.training.weight_decay)

        loss_fn = args.loss_fn or cfg.training.loss_fn
        loss_beta = args.loss_beta if args.loss_beta is not None else getattr(cfg.training, "loss_beta", 1.0)
        npho_weight = args.npho_weight if args.npho_weight is not None else cfg.training.npho_weight
        # Get time config from nested structure
        time_weight = args.time_weight if args.time_weight is not None else cfg.training.time.weight
        npho_threshold = args.npho_threshold if args.npho_threshold is not None else cfg.training.time.npho_threshold
        use_npho_time_weight = not args.no_npho_time_weight and cfg.training.time.use_npho_weight
        grad_clip = args.grad_clip if args.grad_clip is not None else cfg.training.grad_clip
        # If --disable_mae_rmse_metrics flag is passed, disable; otherwise use config value
        track_mae_rmse = False if args.disable_mae_rmse_metrics else getattr(cfg.training, "track_mae_rmse", True)
        save_root_predictions = cfg.checkpoint.save_predictions
        root_save_interval = cfg.checkpoint.root_save_interval
        grad_accum_steps = args.grad_accum_steps if args.grad_accum_steps is not None else getattr(cfg.training, "grad_accum_steps", 1)
        track_metrics = getattr(cfg.training, "track_metrics", True)
        profile = args.profile or getattr(cfg.training, 'profile', False)
        # Handle compile option: can be string mode or boolean (for backward compat)
        compile_cfg = getattr(cfg.training, 'compile', 'reduce-overhead')
        if isinstance(compile_cfg, bool):
            compile_mode = 'reduce-overhead' if compile_cfg else 'none'
        else:
            compile_mode = compile_cfg if compile_cfg else 'reduce-overhead'
        if args.compile is not None:
            compile_mode = args.compile
        compile_fullgraph = getattr(cfg.training, 'compile_fullgraph', False)
        ema_decay = args.ema_decay if args.ema_decay is not None else getattr(cfg.training, 'ema_decay', None)
        amp = getattr(cfg.training, 'amp', True)
        # New normalization and loss weighting options
        npho_scheme = args.npho_scheme or getattr(cfg.normalization, 'npho_scheme', 'log1p')
        npho_loss_weight_enabled = args.npho_loss_weight_enabled or cfg.training.npho_loss_weight.enabled
        npho_loss_weight_alpha = args.npho_loss_weight_alpha if args.npho_loss_weight_alpha is not None else cfg.training.npho_loss_weight.alpha
        intensity_reweighting_enabled = args.intensity_reweighting_enabled or cfg.training.intensity_reweighting.enabled
        intensity_reweighting_nbins = args.intensity_reweighting_nbins if args.intensity_reweighting_nbins is not None else cfg.training.intensity_reweighting.nbins
        intensity_reweighting_target = args.intensity_reweighting_target or cfg.training.intensity_reweighting.target

        mlflow_experiment = args.mlflow_experiment or cfg.mlflow.experiment
        mlflow_run_name = args.mlflow_run_name or cfg.mlflow.run_name
        mlflow_new_run = getattr(cfg.checkpoint, 'new_mlflow_run', False)
        resume_from = args.resume_from or cfg.checkpoint.resume_from
        reset_epoch = args.reset_epoch or getattr(cfg.checkpoint, 'reset_epoch', False)
        refresh_lr = args.refresh_lr or getattr(cfg.checkpoint, 'refresh_lr', False)
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
        outer_fine_pool = args.outer_fine_pool  # None means no pooling
        mask_ratio = args.mask_ratio or 0.05
        time_mask_ratio_scale = args.time_mask_ratio_scale or 1.0
        freeze_encoder = not args.finetune_encoder  # Default: frozen
        use_local_context = not args.global_only  # Default: True (use local context)
        predict_channels = ["npho", "time"]  # Default: predict both channels
        use_masked_attention = False
        head_type = "per_face"
        sensor_positions_file = None
        cross_attn_k = 16
        cross_attn_hidden = 64
        cross_attn_latent_dim = 128
        cross_attn_pos_dim = 96
        encoder_dim = 1024
        encoder_dim_feedforward = None
        encoder_num_fusion_layers = 2

        lr = args.lr or 1e-4
        lr_scheduler = args.lr_scheduler
        lr_min = args.lr_min or 1e-6
        warmup_epochs = args.warmup_epochs if args.warmup_epochs is not None else 0
        weight_decay = args.weight_decay or 1e-4

        loss_fn = args.loss_fn or "smooth_l1"
        loss_beta = args.loss_beta if args.loss_beta is not None else 1.0
        npho_weight = args.npho_weight or 1.0
        time_weight = args.time_weight or 1.0
        npho_threshold = args.npho_threshold  # None uses DEFAULT_NPHO_THRESHOLD
        use_npho_time_weight = not args.no_npho_time_weight
        grad_clip = args.grad_clip or 1.0
        track_mae_rmse = not bool(args.disable_mae_rmse_metrics)
        save_root_predictions = True
        root_save_interval = 10
        grad_accum_steps = args.grad_accum_steps or 1
        track_metrics = True
        profile = args.profile
        compile_mode = args.compile if args.compile is not None else 'reduce-overhead'
        compile_fullgraph = False  # Default for CLI mode
        ema_decay = args.ema_decay  # None by default
        amp = True  # Always enabled in CLI mode
        # New normalization and loss weighting options (CLI defaults)
        npho_scheme = args.npho_scheme or "log1p"
        npho_loss_weight_enabled = args.npho_loss_weight_enabled
        npho_loss_weight_alpha = args.npho_loss_weight_alpha if args.npho_loss_weight_alpha is not None else 0.5
        intensity_reweighting_enabled = args.intensity_reweighting_enabled
        intensity_reweighting_nbins = args.intensity_reweighting_nbins if args.intensity_reweighting_nbins is not None else 5
        intensity_reweighting_target = args.intensity_reweighting_target or "uniform"

        mlflow_experiment = args.mlflow_experiment or "inpainting"
        mlflow_run_name = args.mlflow_run_name
        mlflow_new_run = False  # No config file, default to False
        resume_from = args.resume_from
        reset_epoch = args.reset_epoch
        refresh_lr = args.refresh_lr
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

    # DDP setup
    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if is_main_process():
        print(f"[INFO] Using device: {device}" + (f" (world_size={world_size})" if world_size > 1 else ""))

    train_files = expand_path(train_root)
    val_files = expand_path(val_root) if val_root else []
    all_val_files = val_files  # Keep full list for prediction saving (rank 0)

    # Shard file lists across ranks
    if world_size > 1:
        train_files = shard_file_list(train_files, rank, world_size)
        if val_files:
            val_files = shard_file_list(val_files, rank, world_size)

    if is_main_process():
        print(f"[INFO] Training files: {len(train_files)}" + (f" (per rank)" if world_size > 1 else ""))
        print(f"[INFO] Validation files: {len(val_files)}" + (f" (per rank)" if world_size > 1 else ""))

    # Validate data paths exist
    validate_data_paths(train_root, val_root, expand_func=expand_path)

    # Check artifact directory for existing files
    check_artifact_directory(save_path)

    # Load encoder from MAE checkpoint or create from scratch
    if mae_checkpoint:
        encoder = load_mae_encoder(
            mae_checkpoint, device,
            outer_mode=outer_mode,
            outer_fine_pool=outer_fine_pool,
            encoder_dim=encoder_dim,
            dim_feedforward=encoder_dim_feedforward,
            num_fusion_layers=encoder_num_fusion_layers,
            sentinel_time=sentinel_time,
        )
    else:
        print("[INFO] No MAE checkpoint provided, initializing encoder from scratch")
        outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
        encoder = XECEncoder(
            outer_mode=outer_mode,
            outer_fine_pool=outer_fine_pool_tuple,
            sentinel_time=sentinel_time,
            encoder_dim=encoder_dim,
            dim_feedforward=encoder_dim_feedforward,
            num_fusion_layers=encoder_num_fusion_layers,
        ).to(device)

    # Create inpainter model
    model = XEC_Inpainter(
        encoder, freeze_encoder=freeze_encoder, sentinel_time=sentinel_time,
        time_mask_ratio_scale=time_mask_ratio_scale,
        use_local_context=use_local_context,
        predict_channels=predict_channels,
        use_masked_attention=use_masked_attention,
        head_type=head_type,
        sensor_positions_file=sensor_positions_file,
        cross_attn_k=cross_attn_k,
        cross_attn_hidden=cross_attn_hidden,
        cross_attn_latent_dim=cross_attn_latent_dim,
        cross_attn_pos_dim=cross_attn_pos_dim,
        sentinel_npho=sentinel_npho,
    ).to(device)

    if is_main_process():
        print(f"[INFO] Inpainter created:")
        print(f"  - Total params: {model.get_num_total_params():,}")
        print(f"  - Trainable params: {model.get_num_trainable_params():,}")
        print(f"  - Encoder frozen: {freeze_encoder}")
        print(f"  - Use local context: {use_local_context}")
        print(f"  - Predict channels: {predict_channels}")
        print(f"  - Head type: {head_type}")
        if head_type == "cross_attention":
            print(f"  - Sensor positions: {sensor_positions_file}")
            print(f"  - Cross-attn K={cross_attn_k}, hidden={cross_attn_hidden}, "
                  f"latent_dim={cross_attn_latent_dim}, pos_dim={cross_attn_pos_dim}")
        else:
            print(f"  - Masked attention heads: {use_masked_attention}")

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

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay, fused=(device.type == "cuda"))

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

    # Scheduler
    scheduler = None
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

    # AMP scaler (None when amp disabled — engine uses scaler presence to toggle autocast)
    scaler = torch.amp.GradScaler('cuda', enabled=True) if (amp and device.type == "cuda") else None

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

    # Resume from checkpoint
    start_epoch = 0
    best_val_loss = float('inf')
    mlflow_run_id = None
    if resume_from and os.path.exists(resume_from):
        if is_main_process():
            print(f"[INFO] Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            if "ema_state_dict" in checkpoint and ema_model is not None:
                ema_model.load_state_dict(checkpoint['ema_state_dict'])
                if is_main_process():
                    print("[INFO] Loaded EMA state from checkpoint")

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
                print(f"[INFO] Resumed from epoch {start_epoch}")
        else:
            model_without_ddp.load_state_dict(checkpoint, strict=False)

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

    # MLflow setup (rank 0 only)
    os.makedirs(save_path, exist_ok=True)
    if is_main_process():
        # Default to SQLite backend if MLFLOW_TRACKING_URI is not set
        if not os.environ.get("MLFLOW_TRACKING_URI"):
            default_uri = f"sqlite:///{os.getcwd()}/mlruns.db"
            mlflow.set_tracking_uri(default_uri)
            print(f"[INFO] MLflow tracking URI: {default_uri}")
        mlflow.set_experiment(mlflow_experiment)
        print(f"[INFO] Starting inpainter training")
        print(f"  - Experiment: {mlflow_experiment}")
        print(f"  - Run name: {mlflow_run_name}")
        print(f"  - Mask ratio: {mask_ratio}")

    # Disable MLflow's automatic system metrics (uses wall clock time)
    # We log our own system metrics with step=epoch for consistent x-axis
    # Only rank 0 interacts with MLflow
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

        # Log parameters (rank 0 only)
        if is_main_process():
            resume_state = "no"
            if resume_from:
                resume_state = f"yes: {resume_from}" if os.path.exists(resume_from) else f"missing: {resume_from}"
            mlflow.log_params({
                "train_root": train_root,
                "val_root": val_root,
                "mae_checkpoint": mae_checkpoint,
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
                "mask_ratio": mask_ratio,
                "freeze_encoder": freeze_encoder,
                "predict_channels": ",".join(predict_channels),
                "lr": lr,
                "lr_scheduler": lr_scheduler,
                "warmup_epochs": warmup_epochs,
                "weight_decay": weight_decay,
                "loss_fn": loss_fn,
                "loss_beta": loss_beta,
                "npho_weight": npho_weight,
                "time_weight": time_weight,
                "grad_clip": grad_clip,
                "grad_accum_steps": grad_accum_steps,
                "amp": amp,
                "compile": compile_mode,
                "ema_decay": ema_decay,
                "outer_mode": outer_mode,
                "encoder_dim": encoder_dim,
                "dim_feedforward": encoder_dim_feedforward,
                "num_fusion_layers": encoder_num_fusion_layers,
                "trainable_params": model_without_ddp.get_num_trainable_params(),
                "total_params": model_without_ddp.get_num_total_params(),
                "npho_loss_weight_enabled": npho_loss_weight_enabled,
                "npho_loss_weight_alpha": npho_loss_weight_alpha,
                "intensity_reweighting_enabled": intensity_reweighting_enabled,
                "use_masked_attention": use_masked_attention,
                "head_type": head_type,
                "resume_state": resume_state,
                "world_size": world_size,
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
                sentinel_time=float(sentinel_time),
                loss_fn=loss_fn,
                loss_beta=loss_beta,
                npho_weight=npho_weight,
                time_weight=time_weight,
                grad_clip=grad_clip,
                scaler=scaler,
                track_mae_rmse=track_mae_rmse,
                dataloader_workers=num_workers,
                dataset_workers=num_threads,
                prefetch_factor=prefetch_factor,
                grad_accum_steps=grad_accum_steps,
                track_metrics=track_metrics,
                npho_threshold=npho_threshold,
                use_npho_time_weight=use_npho_time_weight,
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

            # Validation (use EMA model if available)
            eval_model = ema_model if ema_model is not None else model
            eval_model.eval()
            val_metrics = {}
            if val_files:
                val_metrics = run_eval_inpainter(
                    eval_model, device,
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
                    sentinel_time=float(sentinel_time),
                    loss_fn=loss_fn,
                    loss_beta=loss_beta,
                    npho_weight=npho_weight,
                    time_weight=time_weight,
                    track_mae_rmse=track_mae_rmse,
                    dataloader_workers=num_workers,
                    dataset_workers=num_threads,
                    prefetch_factor=prefetch_factor,
                    npho_threshold=npho_threshold,
                    use_npho_time_weight=use_npho_time_weight,
                    profile=profile and is_main_process(),
                    log_invalid_npho=log_invalid_npho,
                    amp=amp,
                    npho_scheme=npho_scheme,
                    npho_loss_weight_enabled=npho_loss_weight_enabled,
                    npho_loss_weight_alpha=npho_loss_weight_alpha,
                    sentinel_npho=sentinel_npho,
                )

            if val_metrics:
                val_metrics = reduce_metrics(val_metrics, device)

            dt = time.time() - t0

            # Log metrics
            train_loss = train_metrics.get("total_loss", 0.0)
            val_loss = val_metrics.get("total_loss", 0.0) if val_metrics else 0.0

            # Learning rate
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]

            if is_main_process():
                print(f"Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Time: {dt:.1f}s")

            # --- Logging & checkpointing (rank 0 only) ---
            if not is_main_process():
                continue

            # MLflow logging
            for key, value in train_metrics.items():
                mlflow.log_metric(f"train/{key}", value, step=epoch)

            if val_metrics:
                for key, value in val_metrics.items():
                    mlflow.log_metric(f"val/{key}", value, step=epoch)

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
                    'model_state_dict': model_without_ddp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'mlflow_run_id': mlflow_run_id,
                    'config': {
                        'outer_mode': outer_mode,
                        'outer_fine_pool': outer_fine_pool,
                        'mask_ratio': mask_ratio,
                        'freeze_encoder': freeze_encoder,
                        'predict_channels': list(predict_channels),
                        'use_masked_attention': use_masked_attention,
                        'head_type': head_type,
                        'sensor_positions_file': sensor_positions_file,
                        'cross_attn_k': cross_attn_k,
                        'cross_attn_hidden': cross_attn_hidden,
                        'cross_attn_latent_dim': cross_attn_latent_dim,
                        'cross_attn_pos_dim': cross_attn_pos_dim,
                        # Normalization parameters (critical for inference)
                        'npho_scale': float(npho_scale),
                        'npho_scale2': float(npho_scale2),
                        'time_scale': float(time_scale),
                        'time_shift': float(time_shift),
                        'sentinel_time': float(sentinel_time),
                        'npho_branch': npho_branch,
                        'time_branch': time_branch,
                        'sentinel_npho': float(sentinel_npho),
                        'npho_scheme': npho_scheme,
                        'encoder_dim': encoder_dim,
                        'dim_feedforward': encoder_dim_feedforward,
                        'num_fusion_layers': encoder_num_fusion_layers,
                    }
                }
                checkpoint_dict['scheduler_state_dict'] = scheduler.state_dict() if scheduler is not None else None
                checkpoint_dict['ema_state_dict'] = ema_model.state_dict() if ema_model is not None else None

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

            # Save ROOT predictions periodically (and at end)
            if save_root_predictions and all_val_files and ((epoch + 1) % root_save_interval == 0 or (epoch + 1) == epochs):
                t_root_start = time.time()
                print(f"  Collecting predictions for ROOT output...")
                with RootPredictionWriter(
                    save_path, epoch + 1, run_id=mlflow_run_id,
                    npho_scale=float(npho_scale),
                    npho_scale2=float(npho_scale2),
                    time_scale=float(time_scale),
                    time_shift=float(time_shift),
                    sentinel_time=float(sentinel_time),
                    predict_channels=list(predict_channels),
                    npho_scheme=npho_scheme,
                ) as writer:
                    # Use EMA model for predictions if available
                    pred_model = ema_model if ema_model is not None else model_without_ddp
                    val_metrics_with_pred, _ = run_eval_inpainter(
                        pred_model, device,
                        all_val_files, "tree",
                        batch_size=batch_size,
                        step_size=chunksize,
                        mask_ratio=mask_ratio,
                        npho_branch=npho_branch,
                        time_branch=time_branch,
                        npho_scale=float(npho_scale),
                        npho_scale2=float(npho_scale2),
                        time_scale=float(time_scale),
                        time_shift=float(time_shift),
                        sentinel_time=float(sentinel_time),
                        loss_fn=loss_fn,
                        loss_beta=loss_beta,
                        npho_weight=npho_weight,
                        time_weight=time_weight,
                        collect_predictions=True,
                        prediction_writer=writer.write,
                        track_mae_rmse=track_mae_rmse,
                        dataloader_workers=num_workers,
                        dataset_workers=num_threads,
                        prefetch_factor=prefetch_factor,
                        npho_threshold=npho_threshold,
                        use_npho_time_weight=use_npho_time_weight,
                        profile=False,
                        log_invalid_npho=log_invalid_npho,
                        amp=amp,
                        npho_scheme=npho_scheme,
                        npho_loss_weight_enabled=npho_loss_weight_enabled,
                        npho_loss_weight_alpha=npho_loss_weight_alpha,
                        sentinel_npho=sentinel_npho,
                    )
                root_path = writer.filepath if writer.count > 0 else None
                t_root_elapsed = time.time() - t_root_start
                if root_path:
                    print(f"  Saved predictions to {root_path} ({t_root_elapsed:.1f}s)")
                    mlflow.log_artifact(root_path)

    if is_main_process():
        print(f"\n[INFO] Training complete!")
        print(f"  Best validation loss: {best_val_loss:.2e}")
        print(f"  Checkpoints saved to: {save_path}")
    cleanup_ddp()


if __name__ == "__main__":
    main()
