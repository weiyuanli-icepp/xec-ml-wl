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

# Usage
# Run Pre-training: python lib/train_mae.py --root path/to/data.root --save_path mae_pretrained.pth --epochs 20 --batch_size 1024
# Usage in regression: --resume_from mae_weights.pth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_root", type=str, required=True, help="Path to Training ROOT file(s)")
    parser.add_argument("--val_root",   type=str, default=None, help="Path to Validation ROOT file(s)")
    parser.add_argument("--save_path",  type=str, default="artifacts/mae_run", help="Directory to save checkpoints")
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--chunksize",  type=int, default=256000, help="Number of events per read")
    
    parser.add_argument("--npho_scale",     type=float, default=1.0)
    parser.add_argument("--npho_scale2",    type=float, default=1.0)
    parser.add_argument("--time_scale",     type=float, default=2.32e6)
    parser.add_argument("--time_shift",     type=float, default=0.0)
    parser.add_argument("--sentinel_value", type=float, default=-5.0)
    
    parser.add_argument("--outer_mode",       type=str, default="finegrid", choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool",  type=int, nargs=2, default=None)
    parser.add_argument("--mask_ratio",       type=float, default=0.6)
    parser.add_argument("--lr",               type=float, default=1e-4)
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    
    parser.add_argument("--mlflow_experiment", type=str, default="mae_pretraining")
    parser.add_argument("--mlflow_run_name",   type=str, default=None)
    parser.add_argument("--resume_from",       type=str, default=None, help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    def expand_path(p):
        path = os.path.expanduser(p)
        if os.path.isdir(path):
            files = glob.glob(os.path.join(path, "*.root"))
            if not files:
                raise ValueError(f"No ROOT files found in directory {path}")
            return sorted(files)
        return [path]
    
    train_files = expand_path(args.train_root)
    print(f"[INFO] Training Data: {len(train_files)} files")
    
    val_files = None
    if args.val_root:
        val_files = expand_path(args.val_root)
        print(f"[INFO] Validation Data: {len(val_files)} files")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize Model
    outer_fine_pool = tuple(args.outer_fine_pool) if args.outer_fine_pool else None
    encoder = XECRegressor(
        outer_mode=args.outer_mode,
        outer_fine_pool=outer_fine_pool
    ).to(device)
    
    model = XEC_MAE(encoder, mask_ratio=args.mask_ratio).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"[INFO] Resuming MAE from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        
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
    mlflow.set_experiment(args.mlflow_experiment)
    print(f"Starting MAE Pre-training in experiment: {args.mlflow_experiment}, run name: {args.mlflow_run_name}")
    os.makedirs(args.save_path, exist_ok=True)
    with mlflow.start_run(run_name=args.mlflow_run_name) as run:
        mlflow.log_params(vars(args))
    
        # 5. Training Loop
        print("Starting MAE epoch loop...")
        for epoch in range(args.epochs):
            t0 = time.time()
            
            # --- TRAIN ---
            train_loss = run_epoch_mae(
                model, optimizer, device, train_files, "tree", 
                batch_size=args.batch_size,
                step_size=args.chunksize,
                npho_branch="relative_npho",
                time_branch="relative_time",
                NphoScale=args.npho_scale,
                NphoScale2=args.npho_scale2,
                time_scale=args.time_scale,
                time_shift=args.time_shift,
                sentinel_value=args.sentinel_value,
            )
            
            # --- VALIDATION ---
            val_loss = 0.0
            if val_files:
                val_loss = run_eval_mae(
                    model, device, val_files, "tree", 
                    batch_size=args.batch_size,
                    step_size=args.chunksize,
                    npho_branch="relative_npho",
                    time_branch="relative_time",
                    NphoScale=args.npho_scale,
                    NphoScale2=args.npho_scale2,
                    time_scale=args.time_scale,
                    time_shift=args.time_shift,
                    sentinel_value=args.sentinel_value,
                )
                
            dt = time.time() - t0
            
            print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Time: {dt:.1f}s")
            
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
            if (epoch + 1) % 10 == 0 or (epoch + 1) == args.epochs:
                full_ckpt_path = os.path.join(args.save_path, "mae_checkpoint_epoch_last.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, full_ckpt_path)
                print(f"Saved full MAE checkpoint to {full_ckpt_path}")
                
                encoder_path = os.path.join(args.save_path, f"mae_encoder_epoch_{epoch+1}.pth")
                torch.save(model.encoder.state_dict(), encoder_path)
                print(f"Saved encoder weights to {encoder_path}")
                mlflow.log_artifact(encoder_path)
                
if __name__ == "__main__":
    main()
