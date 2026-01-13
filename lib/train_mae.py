import torch
import argparse
import time
import os
import glob
import psutil
import mlflow

from .model import XECRegressor
from .model_mae import XEC_MAE
from .engine_mae import run_epoch_mae, run_eval_mae
from .utils import get_gpu_memory_stats
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

    # Config file (new)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data paths (CLI or config override)
    parser.add_argument("--train_root", type=str, default=None, help="Path to Training ROOT file(s)")
    parser.add_argument("--val_root",   type=str, default=None, help="Path to Validation ROOT file(s)")
    parser.add_argument("--save_path",  type=str, default=None, help="Directory to save checkpoints")
    parser.add_argument("--epochs",     type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--chunksize",  type=int, default=None, help="Number of events per read")

    parser.add_argument("--npho_scale",     type=float, default=None)
    parser.add_argument("--npho_scale2",    type=float, default=None)
    parser.add_argument("--time_scale",     type=float, default=None)
    parser.add_argument("--time_shift",     type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    parser.add_argument("--outer_mode",           type=str, default=None, choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool",      type=int, nargs=2, default=None)
    parser.add_argument("--mask_ratio",           type=float, default=None)
    parser.add_argument("--lr",                   type=float, default=None)
    parser.add_argument("--weight_decay",         type=float, default=None)
    parser.add_argument("--channel_dropout_rate", type=float, default=None)

    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name",   type=str, default=None)
    parser.add_argument("--resume_from",       type=str, default=None, help="Path to checkpoint to resume from")

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
        npho_scale = args.npho_scale if args.npho_scale is not None else cfg.normalization.npho_scale
        npho_scale2 = args.npho_scale2 if args.npho_scale2 is not None else cfg.normalization.npho_scale2
        time_scale = args.time_scale if args.time_scale is not None else cfg.normalization.time_scale
        time_shift = args.time_shift if args.time_shift is not None else cfg.normalization.time_shift
        sentinel_value = args.sentinel_value if args.sentinel_value is not None else cfg.normalization.sentinel_value
        outer_mode = args.outer_mode or cfg.model.outer_mode
        outer_fine_pool = args.outer_fine_pool or cfg.model.outer_fine_pool
        mask_ratio = args.mask_ratio if args.mask_ratio is not None else cfg.model.mask_ratio
        lr = args.lr if args.lr is not None else cfg.training.lr
        weight_decay = args.weight_decay if args.weight_decay is not None else cfg.training.weight_decay
        channel_dropout_rate = args.channel_dropout_rate if args.channel_dropout_rate is not None else cfg.training.channel_dropout_rate
        mlflow_experiment = args.mlflow_experiment or cfg.mlflow.experiment
        mlflow_run_name = args.mlflow_run_name or cfg.mlflow.run_name
        resume_from = args.resume_from or cfg.checkpoint.resume_from
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
        npho_scale = args.npho_scale or DEFAULT_NPHO_SCALE
        npho_scale2 = args.npho_scale2 or DEFAULT_NPHO_SCALE2
        time_scale = args.time_scale or DEFAULT_TIME_SCALE
        time_shift = args.time_shift or DEFAULT_TIME_SHIFT
        sentinel_value = args.sentinel_value or DEFAULT_SENTINEL_VALUE
        outer_mode = args.outer_mode or "finegrid"
        outer_fine_pool = args.outer_fine_pool
        mask_ratio = args.mask_ratio or 0.6
        lr = args.lr or 1e-4
        weight_decay = args.weight_decay or 1e-4
        channel_dropout_rate = args.channel_dropout_rate or 0.1
        mlflow_experiment = args.mlflow_experiment or "mae_pretraining"
        mlflow_run_name = args.mlflow_run_name
        resume_from = args.resume_from
    
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    outer_fine_pool_tuple = tuple(outer_fine_pool) if outer_fine_pool else None
    encoder = XECRegressor(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool_tuple
    ).to(device)

    model = XEC_MAE(encoder, mask_ratio=mask_ratio).to(device)
    model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        fused=True,
        weight_decay=weight_decay
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        print(f"[INFO] Resuming MAE from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if "optimizer_state_dict" in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f"[INFO] Resumed from epoch {start_epoch}")
        else:
            print("[WARN] Loaded raw weights. Starting from Epoch 1 (Optimizer reset).")
            model.load_state_dict(checkpoint, strict=False)
            start_epoch = 0
    
    # MLflow Setup
    mlflow.set_experiment(mlflow_experiment)
    print(f"Starting MAE Pre-training in experiment: {mlflow_experiment}, run name: {mlflow_run_name}")
    os.makedirs(save_path, exist_ok=True)
    with mlflow.start_run(run_name=mlflow_run_name) as run:
        # Log parameters
        mlflow.log_params({
            "train_root": train_root,
            "val_root": val_root,
            "save_path": save_path,
            "epochs": epochs,
            "batch_size": batch_size,
            "chunksize": chunksize,
            "npho_scale": npho_scale,
            "npho_scale2": npho_scale2,
            "time_scale": time_scale,
            "time_shift": time_shift,
            "sentinel_value": sentinel_value,
            "outer_mode": outer_mode,
            "outer_fine_pool": outer_fine_pool,
            "mask_ratio": mask_ratio,
            "lr": lr,
            "weight_decay": weight_decay,
            "channel_dropout_rate": channel_dropout_rate,
        })
    
        # 5. Training Loop
        print("Starting MAE epoch loop...")
        for epoch in range(epochs):
            t0 = time.time()

            # --- TRAIN ---
            train_loss = run_epoch_mae(
                model, optimizer, device, train_files, "tree",
                batch_size=batch_size,
                step_size=chunksize,
                npho_branch="relative_npho",
                time_branch="relative_time",
                NphoScale=npho_scale,
                NphoScale2=npho_scale2,
                time_scale=time_scale,
                time_shift=time_shift,
                sentinel_value=sentinel_value,
                channel_dropout_rate=channel_dropout_rate,
            )

            # --- VALIDATION ---
            val_loss = 0.0
            if val_files:
                val_loss = run_eval_mae(
                    model, device, val_files, "tree",
                    batch_size=batch_size,
                    step_size=chunksize,
                    npho_branch="relative_npho",
                    time_branch="relative_time",
                    NphoScale=npho_scale,
                    NphoScale2=npho_scale2,
                    time_scale=time_scale,
                    time_shift=time_shift,
                    sentinel_value=sentinel_value,
                )

            dt = time.time() - t0

            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {dt:.1f}s")
            
            # Log training metrics
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            if val_files:
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("epoch_time_sec", dt, step=epoch)
            
            # GPU stats
            if device.type == "cuda":
                stats = get_gpu_memory_stats(device)
                if stats:
                    total_mem = torch.cuda.get_device_properties(device).total_memory
                    vram_util = stats['allocated'] / total_mem
                    mlflow.log_metrics({
                        "system/memory_allocated_GB": stats['allocated'] / 1e9,
                        "system/memory_reserved_GB": stats['reserved'] / 1e9,
                        "system/vram_utilization": vram_util,
                    }, step=epoch)

            # CPU RAM Stats
            try:
                ram = psutil.virtual_memory()
                process = psutil.Process()
                mem_info = process.memory_info()
                mlflow.log_metrics({
                    "system/ram_total_GB": ram.total / 1e9,
                    "system/ram_used_GB": ram.used / 1e9,
                    "system/ram_percent": ram.percent,
                    "system/process_rss_GB": mem_info.rss / 1e9,
                    "system/process_vms_GB": mem_info.vms / 1e9,
                }, step=epoch)
            except Exception as e:
                print(f"[WARNING] Could not log CPU RAM stats: {e}")
            
            # Save model checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
                full_ckpt_path = os.path.join(save_path, "mae_checkpoint_epoch_last.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, full_ckpt_path)
                print(f"Saved full MAE checkpoint to {full_ckpt_path}")

                encoder_path = os.path.join(save_path, f"mae_encoder_epoch_{epoch+1}.pth")
                torch.save(model.encoder.state_dict(), encoder_path)
                print(f"Saved encoder weights to {encoder_path}")
                mlflow.log_artifact(encoder_path)
                
if __name__ == "__main__":
    main()
