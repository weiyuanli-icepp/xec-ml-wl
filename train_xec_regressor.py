#!/usr/bin/env python3

import os, time
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import mlflow
import mlflow.pytorch
import uproot
import psutil
import glob
import random

warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.tracking._tracking_service.utils")

from lib.model import XECRegressor
from lib.event_display import plot_event_faces, plot_event_time
from lib.angle_reweighting import scan_angle_hist_1d, scan_angle_hist_2d
from lib.engine import run_epoch_stream
from lib.utils import (
    get_gpu_memory_stats,
    iterate_chunks,
    compute_face_saliency
)
from lib.plotting import (
    plot_resolution_profile,
    plot_face_weights,
    plot_profile,
    plot_cos_residuals,
    plot_pred_truth_scatter,
    plot_saliency_profile
)
# ------------------------------------------------------------

# ------------------------------------------------------------
#  Main training entry
# ------------------------------------------------------------
def main_xec_regressor_with_args(
    root,
    tree="tree",
    epochs=20,
    batch=256,
    chunksize=4000,
    lr=3e-4,
    weight_decay=1e-4,
    drop_path_rate=0.0,
    time_shift=-0.29,
    time_scale=2.32e6,
    sentinel_value=-5.0,
    use_scheduler=-1,
    warmup_epochs=2,
    amp=True,
    max_chunks=None,
    npho_branch="relative_npho",             
    time_branch="relative_time",
    NphoScale=1e5,
    NphoScale2=13,
    onnx="meg2ang_convnextv2.onnx",
    mlflow_experiment="gamma_angle",
    run_name=None,
    outer_mode="finegrid", 
    outer_fine_pool=(3,3),           
    reweight_mode="none",
    nbins_theta=50,
    nbins_phi=50,
    loss_type="smooth_l1",
    loss_beta=1.0,
    resume_from=None, 
    ema_decay=0.999,
):

    root = os.path.expanduser(root)
    if os.path.isdir(root):
        print(f"[INFO] Input is a directory. Scanning for *.root files in: {root}")
        all_files = sorted(glob.glob(os.path.join(root, "*.root")))
    else:
        print(f"[INFO] Input is a path pattern. Resolving glob: {root}")
        all_files = sorted(glob.glob(root))
        
    if len(all_files) <= 10:
        raise ValueError(f"[ERROR] Not enough ROOT files found ({len(all_files)}). Please check the input path.")
    
    print(f"[INFO] Found {len(all_files)} ROOT files. First: {os.path.basename(all_files[0])}, Last: {os.path.basename(all_files[-1])}")

    random.Random(42).shuffle(all_files)
    split_idx = int(0.9 * len(all_files))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]
    print(f"[INFO] Data Split: {len(train_files)} Training files / {len(val_files)} Validation files")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Pass drop_path_rate to model ---
    model = XECRegressor(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool,
        drop_path_rate=drop_path_rate
    ).to(device)
    # ------------------------------------

    # --- Optimizer ---
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or "bn" in n.lower() or "norm" in n.lower() else decay).append(p)
    optimizer = optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )
    ema_model = None
    if ema_decay > 0.0:
        print(f"[INFO] Using EMA with decay={ema_decay}")
        
        def robust_ema_avg(averaged_model_parameter, model_parameter, num_averaged):
            decay = ema_decay
            return decay * averaged_model_parameter + (1.0 - decay) * model_parameter        

        ema_model  = AveragedModel(model, avg_fn=robust_ema_avg, use_buffers=True)
        ema_model.to(device)
    else:
        print(f"[INFO] EMA is DISABLED.")
    # ------------------
    
    # --- Initialize Scaler ---
    scaler = torch.amp.GradScaler("cuda", enabled=(amp and torch.cuda.is_available()))
    # -------------------------

    # --- Scheduler setup ---
    scheduler = None
    if use_scheduler == -1:
        print(f"[INFO] Using Cosine Annealing with {warmup_epochs} warmup epochs.")
        main_scheduler = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
        
        scheduler = SequentialLR(
            optimizer, 
            schedulers=[warmup_scheduler, main_scheduler], 
            milestones=[warmup_epochs]
        )
    else:
        print("[INFO] Using Constant LR (no scheduler).")
    # ------------------------

    # --- Resume from checkpoint ---
    start_epoch = 1
    best_val = float("inf")
    run_id = None
    
    if resume_from and os.path.exists(resume_from):
        print(f"[INFO] Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        
        # Resume training
        if "model_state_dict" in checkpoint:
            print(f"[INFO] Detected full checkpoint. Resuming training state.")
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            
            start_epoch = checkpoint["epoch"] + 1
            best_val = checkpoint.get("best_val", float("inf"))
            run_id = checkpoint.get("mlflow_run_id", None)
            if ema_model is not None:
                if "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
                    print(f"[INFO] Loading EMA state model from checkpoint.")
                    ema_model.load_state_dict(checkpoint["ema_state_dict"])
                else:
                    print(f"[INFO] No EMA state in checkpoint. Re-syncing EMA.")
                    ema_model.module.load_state_dict(model.state_dict())
                    if hasattr(ema_model, 'n_averaged'):
                        ema_model.n_averaged.zero_()
            else:
                print(f"[INFO] EMA is disabled. Skipping EMA state loading.")
            
        # MAE Fine-tuning
        else: 
            print(f"[INFO] Detected raw weights (MAE). Loading encoder for fine-tuning.")
            missing, unexpected = model.load_state_dict(checkpoint, strict=False)
            print(f"[INFO] Missing keys (expected for head): {len(missing)}")
            print(f"[INFO] Unexpected keys: {len(unexpected)}")
            
            start_epoch = 1
            best_val = float("inf")
            run_id = None
        
            if ema_model is not None:
                print(f"[INFO] Initializing EMA from MAE weights.")
                ema_model.module.load_state_dict(model.state_dict())
                if hasattr(ema_model, 'n_averaged'):
                    ema_model.n_averaged.zero_()
            else:
                print(f"[INFO] EMA is disabled. Skipping EMA state loading.")
    # ------------------------------
        
    # --- MLflow Setup ---
    mlflow.set_experiment(mlflow_experiment)
    if run_name is None:
        run_name = time.strftime("run_cv2_%Y%m%d_%H%M%S")
    # --------------------
    
    # --- Training Loop ---
    with mlflow.start_run(run_id=run_id, run_name=run_name if not run_id else None) as run:
        run_id = run.info.run_id 
        artifact_dir = os.path.abspath(os.path.join("artifacts", run_name))
        os.makedirs(artifact_dir, exist_ok=True)
        
        writer = SummaryWriter(log_dir=os.path.join("runs", run_name))
        
        if start_epoch == 1: 
            mlflow.log_params({
                "root_file": root,
                "tree": tree,
                "epochs": epochs,
                "batch": batch,
                "chunksize": chunksize,
                "weight_decay": weight_decay,
                "drop_path_rate": drop_path_rate,
                "time_shift": time_shift,
                "time_scale": time_scale,
                "sentinel_value": sentinel_value,
                "scheduler": "Cosine+Warmup" if use_scheduler == -1 else "Constant",
                "warmup_epochs": warmup_epochs,
                "onnx_export": onnx,
                "amp": amp,                
                "npho_branch": npho_branch,
                "time_branch": time_branch,
                "NphoScale": NphoScale,
                "NphoScale2": NphoScale2,
                "reweight": reweight_mode,
                "loss_type": loss_type,
                "loss_beta": loss_beta,
                "outer_mode": outer_mode,
                "outer_fine_pool": outer_fine_pool,
                "ema_decay": ema_decay,
            })
        
        # Reweighting histograms
        edges_theta = weights_theta = None
        edges_phi = weights_phi = None
        edges2_theta = edges2_phi = weights_2d = None

        if reweight_mode == "theta":
            edges_theta, weights_theta = scan_angle_hist_1d(root, tree=tree, comp=0, nbins=nbins_theta, step_size=chunksize)
        elif reweight_mode == "phi":
            edges_phi, weights_phi = scan_angle_hist_1d(root, tree=tree, comp=1, nbins=nbins_phi, step_size=chunksize)
        elif reweight_mode == "theta_phi":
            edges2_theta, edges2_phi, weights_2d = scan_angle_hist_2d(root, tree=tree, nbins_theta=nbins_theta, nbins_phi=nbins_phi, step_size=chunksize)
        
        best_state = None

        for ep in range(start_epoch, epochs+1):
            t0 = time.time()
            
            # TRAIN
            tr_metrics, _, _, _, _ = run_epoch_stream( 
                model, optimizer, device, 
                scaler=scaler,
                root=train_files,
                tree=tree,
                step_size=chunksize, 
                batch_size=batch,
                train=True, 
                amp=amp,
                max_chunks=max_chunks,
                npho_branch=npho_branch, 
                time_branch=time_branch,
                NphoScale=NphoScale, 
                NphoScale2=NphoScale2, 
                time_shift=time_shift, 
                time_scale=time_scale,
                sentinel_value=sentinel_value,
                reweight_mode=reweight_mode,
                edges_theta=edges_theta, 
                weights_theta=weights_theta,
                edges_phi=edges_phi,   
                weights_phi=weights_phi,
                edges2_theta=edges2_theta, 
                edges2_phi=edges2_phi, 
                weights_2d=weights_2d,
                loss_type=loss_type,
                loss_beta=loss_beta,
                scheduler=scheduler,
                ema_model=ema_model
            )

            # VAL
            val_model_to_use = ema_model if ema_model is not None else model
            val_metrics, pred_val, true_val, _, val_stats = run_epoch_stream( 
                val_model_to_use, optimizer, device, 
                scaler=None,
                root=val_files,
                tree=tree,
                step_size=chunksize, 
                batch_size=max(batch,256),
                train=False, 
                amp=False,
                max_chunks=max_chunks,
                npho_branch=npho_branch, 
                time_branch=time_branch,
                NphoScale=NphoScale, 
                NphoScale2=NphoScale2,
                time_shift=time_shift, 
                time_scale=time_scale,
                sentinel_value=sentinel_value,
                reweight_mode=reweight_mode,
                edges_theta=edges_theta, 
                weights_theta=weights_theta,
                edges_phi=edges_phi, 
                weights_phi=weights_phi,
                edges2_theta=edges2_theta, 
                edges2_phi=edges2_phi, 
                weights_2d=weights_2d,
                loss_type=loss_type,
                loss_beta=loss_beta,
                scheduler=scheduler,
                ema_model=ema_model
            )

            sec = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']
            
            # Primary loss for early stopping
            tr_loss = tr_metrics["total_opt"]
            val_loss = val_metrics["total_opt"]

            print(f"[{ep:03d}] tr_loss {tr_loss:.5f} val_loss {val_loss:.5f} lr {current_lr:.2e} time {sec:.1f}s")
            
            # --- LOGGING ---
            # 1. System Metrics (GPU & RAM)
            if device.type == "cuda":
                stats = get_gpu_memory_stats(device)
                if stats:
                    total_mem = torch.cuda.get_device_properties(device).total_memory
                    vram_util = stats["allocated"] / total_mem                    
                    frag = (stats["reserved"] - stats["allocated"]) / max(1, stats["reserved"])
                    mlflow.log_metrics({
                        "system/memory_allocated_GB": stats["allocated"] / 1e9,
                        "system/memory_reserved_GB": stats["reserved"] / 1e9,
                        "system/memory_peak_GB": stats["peak"] / 1e9,
                        "system/gpu_utilization_pct": vram_util,
                        "system/memory_fragmentation": frag,
                    }, step=ep)
            
            # 2. System RAM (CPU)
            process = psutil.Process()
            mem_info = process.memory_info()
            ram_gb = mem_info.rss / 1e9
            ram = psutil.virtual_memory()
            mlflow.log_metrics({
                "system/ram_used_gb": ram.used / 1e9,
                "system/ram_percent": ram.percent,
                "system/process_ram_gb": ram_gb
            }, step=ep)
            
            # 3. Throughput (Speed)
            mlflow.log_metric("system/epoch_duration_sec", sec, step=ep)
            
            # Main Metrics
            log_dict = {
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_smooth_l1": val_metrics["smooth_l1"],
                "val_l1": val_metrics["l1"],
                "val_mse": val_metrics["mse"],
                "val_cos_res": val_metrics["cos"],
                "lr": current_lr,
                **val_stats
            }
            
            perf_keys = [
                "system/throughput_events_per_sec", 
                "system/avg_data_load_sec", 
                "system/avg_batch_process_sec", 
                "system/compute_efficiency"
            ]
            for k in perf_keys:
                if k in tr_metrics:
                    log_dict[k] = tr_metrics[k]
            
            mlflow.log_metrics(log_dict, step=ep)
            
            writer.add_scalar("loss/train", tr_loss, ep)
            writer.add_scalar("loss/val", val_loss, ep)
            writer.add_scalar("lr", current_lr, ep)

            # Save Checkpoint for best result
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    "epoch": ep, 
                    "model_state_dict": model.state_dict(),
                    "ema_state_dict": ema_model.state_dict() if ema_model else None,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val": best_val,
                    "mlflow_run_id": run_id,
                }, os.path.join(artifact_dir, "checkpoint_best.pth"))
                print(f"   [info] New best val_loss: {best_val:.6f}")
            
            # Save Last Checkpoint
            torch.save({
                "epoch": ep, 
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema_model.state_dict() if ema_model else None,
                "optimizer_state_dict": optimizer.state_dict(), 
                "best_val": best_val,
                "mlflow_run_id": run_id,
            }, os.path.join(artifact_dir, "checkpoint_last.pth"))

        # --- ARTIFACTS ---
        final_model = ema_model if ema_model is not None else model
        if ema_model is None and best_state:
            model.load_state_dict(best_state)
            print("[INFO] Loaded best model state for final evaluation.")

        _, pred_all, true_all, extra_info, _ = run_epoch_stream(
            final_model, optimizer, device, 
            scaler=None,
            root=val_files,
            tree=tree,
            step_size=chunksize, 
            batch_size=max(batch,256), 
            train=False, 
            amp=False,
            max_chunks=max_chunks, 
            npho_branch=npho_branch, 
            time_branch=time_branch,
            NphoScale=NphoScale, 
            NphoScale2=NphoScale2,
            time_shift=time_shift, 
            time_scale=time_scale,
            sentinel_value=sentinel_value,
            loss_type=loss_type,
            loss_beta=loss_beta,
            reweight_mode=reweight_mode,
            edges_theta=edges_theta, 
            weights_theta=weights_theta,
            edges_phi=edges_phi,   
            weights_phi=weights_phi,
            edges2_theta=edges2_theta, 
            edges2_phi=edges2_phi, 
            weights_2d=weights_2d,
            scheduler=scheduler
        )

        if pred_all is not None:
            # --- SAVE PREDICTIONS CSV ---
            csv_path = os.path.join(artifact_dir, f"predictions_{run_name}.csv")
            pd.DataFrame({
                "true_theta": true_all[:,0], "true_phi": true_all[:,1],
                "pred_theta": pred_all[:,0], "pred_phi": pred_all[:,1]
            }).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

            # --- WORST EVENTS ---
            worst_events = extra_info.get("worst_events", [])
            for i, (err, raw_n, raw_t, p, t, xyz, vtx, energy, run_id, event_id) in enumerate(worst_events):
                energy = energy * 1e3  # GeV to MeV
                vtx_str = f"({vtx[0]:.1f}, {vtx[1]:.1f}, {vtx[2]:.1f})"
                xyz_str = f"({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f})"
                base_title = (f"Worst #{i+1} | Run {run_id} Event {event_id} | Loss={err:.4f}\n"
                              f"Truth E={energy:.2f} MeV | VTX={vtx_str}, XYZ={xyz_str}\n"
                              f"Truth: θ={t[0]:.2f}, φ={t[1]:.2f} | Pred: θ={p[0]:.2f}, φ={p[1]:.2f}")

                path_npho = os.path.join(artifact_dir, f"worst_event_{i}_{run_name}_npho.pdf")
                plot_event_faces(
                    raw_n, 
                    title=f"{base_title}\n(Photon Distribution)", 
                    savepath=path_npho, 
                    outer_mode=outer_mode
                )
                mlflow.log_artifact(path_npho)
                
                time_disp = raw_t / 1e-7 
                path_time = os.path.join(artifact_dir, f"worst_event_{i}_{run_name}_time.pdf")
                plot_event_time(
                    raw_n,
                    time_disp,
                    title=f"{base_title}\n(Time Distribution [1e-7s])", 
                    savepath=path_time
                )
                mlflow.log_artifact(path_time)

            # --- Metric Plots ---
            res_pdf = os.path.join(artifact_dir, f"resolution_profile_{run_name}.pdf")
            plot_resolution_profile(pred_all, true_all, outfile=res_pdf)
            mlflow.log_artifact(res_pdf)

            # --- Face Weights Plot ---
            weight_pdf = os.path.join(artifact_dir, f"face_weights_{run_name}.pdf")
            model_to_plot_weights = final_model.module if hasattr(final_model, "module") else final_model
            plot_face_weights(model_to_plot_weights, outfile=weight_pdf)
            mlflow.log_artifact(weight_pdf)

            plot_profile(pred_all[:,0], true_all[:,0], label="Theta", 
                         outfile=os.path.join(artifact_dir, "profile_theta.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "profile_theta.pdf"))
            
            plot_profile(pred_all[:,1], true_all[:,1], label="Phi", 
                         outfile=os.path.join(artifact_dir, "profile_phi.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "profile_phi.pdf"))
            
            plot_cos_residuals(pred_all, true_all, outfile=os.path.join(artifact_dir, "cos_res.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "cos_res.pdf"))
            
            plot_pred_truth_scatter(pred_all, true_all, outfile=os.path.join(artifact_dir, "scatter.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "scatter.pdf"))
            
            # --- Sensity Analysis for each face (potential bug in masking logic) ---
            try:                
                for arr in iterate_chunks(root, tree, [npho_branch, time_branch], step_size=256):
                    Npho = arr[npho_branch].astype("float32")
                    Time = arr[time_branch].astype("float32")
                    Npho = np.maximum(Npho, 0.0)
                    mask_garbage = (np.abs(Time) > 1.0) | np.isnan(Time)
                    mask_invalid = (Npho <= 0.0) | mask_garbage
                    Time[mask_invalid] = sentinel_value
                    
                    # Log/Scale
                    Time_norm = (Time - time_shift) / time_scale
                    Npho_log = np.log1p(Npho / NphoScale).astype("float32")
                    X_np = np.stack([Npho_log, Time_norm], axis=-1)
                    
                    X_tensor = torch.from_numpy(X_np).to(device)
                    break
                
                print("[INFO] Computing Physics Saliency...")
                saliency = compute_face_saliency(final_model, X_tensor, device)
                
                # Plot
                sal_pdf = os.path.join(artifact_dir, f"saliency_profile_{run_name}.pdf")
                plot_saliency_profile(saliency, outfile=sal_pdf)
                mlflow.log_artifact(sal_pdf)
                
            except Exception as e:
                print(f"[WARN] Saliency calculation failed: {e}")
            
            # --- VALIDATION ROOT FILE ---
            root_data = extra_info.get("root_data", None)
            if root_data is not None:
                val_root_path = os.path.join(artifact_dir, f"validation_results_{run_name}.root")
                print(f"[INFO] Saving validation ROOT file to {val_root_path}...")
                
                with uproot.recreate(val_root_path) as f:
                    branch_types = {k: v.dtype for k, v in root_data.items()}
                    f.mktree("val_tree", branch_types)
                    f["val_tree"].extend(root_data)
                mlflow.log_artifact(val_root_path)

        # --- EXPORT ONNX ---
        if onnx:
            onnx_path = os.path.join(artifact_dir, onnx)
            print(f"[INFO] Exporting ONNX model to {onnx_path}...")
            
            final_model.eval()
            dummy_input = torch.randn(1, 4760, 2, device=device)
            try:
                torch.onnx.export(
                    final_model, 
                    dummy_input,
                    onnx_path,
                    export_params=True,
                    opset_version=13,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                if os.path.exists(onnx_path):
                    mlflow.log_artifact(onnx_path)
                    print(f"[INFO] ONNX model exported and logged.")
            except Exception as e:
                 print(f"[WARN] Failed to export ONNX: {e}")

        writer.close()
    # ----------------------