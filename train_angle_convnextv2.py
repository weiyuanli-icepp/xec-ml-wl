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
import mlflow
import mlflow.pytorch

# Silence warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow.tracking._tracking_service.utils")

# --- CUSTOM MODULES (Imported from angle_lib) ---
# Note: AngleRegressorSharedFaces is now in angle_lib.model
#       plot_event_faces is now in angle_lib.event_display
from angle_lib.model import AngleRegressorSharedFaces
from angle_lib.event_display import plot_event_faces, plot_event_time
from angle_lib.angle_utils import get_gpu_memory_stats
from angle_lib.angle_reweighting import scan_angle_hist_1d, scan_angle_hist_2d
from angle_lib.angle_engine import run_epoch_stream
from angle_lib.angle_plotting import (
    plot_resolution_profile,
    plot_face_weights,
    plot_profile,
    plot_cos_residuals,
    plot_pred_truth_scatter
)

# ------------------------------------------------------------
#  Main training entry
# ------------------------------------------------------------
def main_angle_convnextv2_with_args(
    root,
    tree="tree",
    epochs=20,
    batch=256,
    chunksize=4000,
    lr=3e-4,
    weight_decay=1e-4,
    drop_path_rate=0.0,
    time_shift=0.0,
    time_scale=1e-7,
    use_scheduler=-1,
    warmup_epochs=2,
    amp=True,
    max_chunks=None,
    npho_branch="relative_npho",             
    time_branch="relative_time",
    NphoScale=2e5,
    onnx="meg2ang_convnextv2.onnx",
    mlflow_experiment="gamma_angle",
    run_name=None,
    outer_mode="finegrid", 
    outer_fine_pool=(3,3),           
    reweight_mode="none",
    nbins_theta=50,
    nbins_phi=50,
    loss_type="smooth_l1",
    resume_from=None, 
):
    root = os.path.expanduser(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pass drop_path_rate to model
    model = AngleRegressorSharedFaces(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool,
        drop_path_rate=drop_path_rate
    ).to(device)

    # Optimizer
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

    # Scheduler Logic
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

    # Resume Logic
    start_epoch = 1
    best_val = float("inf")
    run_id = None
    
    if resume_from and os.path.exists(resume_from):
        print(f"[INFO] Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val = checkpoint.get("best_val", float("inf"))
        run_id = checkpoint.get("mlflow_run_id", None)
        
    # MLflow Setup
    mlflow.set_experiment(mlflow_experiment)
    if run_name is None:
        run_name = time.strftime("run_cv2_%Y%m%d_%H%M%S")
    
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
                "scheduler": "Cosine+Warmup" if use_scheduler == -1 else "Constant",
                "warmup_epochs": warmup_epochs,
                "onnx_export": onnx,
                "amp": amp,                
                "npho_branch": npho_branch,
                "time_branch": time_branch,
                "NphoScale": NphoScale,
                "reweight": reweight_mode,
                "loss_type": loss_type,
                "outer_mode": outer_mode,
                "outer_fine_pool": outer_fine_pool,
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
                model, optimizer, device, root, tree,
                step_size=chunksize, batch_size=batch,
                train=True, amp=amp,
                max_chunks=max_chunks,
                npho_branch=npho_branch, time_branch=time_branch,
                NphoScale=NphoScale, time_shift=time_shift, time_scale=time_scale,
                reweight_mode=reweight_mode,
                edges_theta=edges_theta, weights_theta=weights_theta,
                edges_phi=edges_phi,   weights_phi=weights_phi,
                edges2_theta=edges2_theta, edges2_phi=edges2_phi, weights_2d=weights_2d,
                loss_type=loss_type,
                scheduler=scheduler
            )

            # VAL
            val_metrics, pred_val, true_val, _, val_stats = run_epoch_stream( 
                model, optimizer, device, root, tree,
                step_size=chunksize, batch_size=max(batch,256),
                train=False, amp=False,
                max_chunks=max_chunks,
                npho_branch=npho_branch, time_branch=time_branch,
                NphoScale=NphoScale, time_shift=time_shift, time_scale=time_scale,
                reweight_mode=reweight_mode,
                edges_theta=edges_theta, weights_theta=weights_theta,
                edges_phi=edges_phi,   weights_phi=weights_phi,
                edges2_theta=edges2_theta, edges2_phi=edges2_phi, weights_2d=weights_2d,
                loss_type=loss_type,
            )

            sec = time.time() - t0
            current_lr = optimizer.param_groups[0]['lr']
            
            # Primary loss for early stopping
            tr_loss = tr_metrics["total_opt"]
            val_loss = val_metrics["total_opt"]

            print(f"[{ep:03d}] tr_loss {tr_loss:.5f} val_loss {val_loss:.5f} lr {current_lr:.2e} time {sec:.1f}s")
            
            # --- LOGGING ---
            # System Metrics
            if device.type == "cuda":
                stats = get_gpu_memory_stats(device)
                if stats:
                    mlflow.log_metrics({"system/gpu_alloc_gb": stats["allocated"]/1e9}, step=ep)
                    mlflow.log_metrics({
                        "memory_allocated_GB": stats["allocated"] / 1e9,
                        "memory_peak_GB": stats["peak"] / 1e9
                    }, step=ep)
            
            # Main Metrics
            log_dict = {
                "epoch_time_sec": sec,
                "train_loss": tr_loss,
                "train_smooth_l1": tr_metrics["smooth_l1"],
                "train_l1": tr_metrics["l1"],
                "train_mse": tr_metrics["mse"],
                "train_cos_res": tr_metrics["cos"],
                "val_loss": val_loss,
                "val_smooth_l1": val_metrics["smooth_l1"],
                "val_l1": val_metrics["l1"],
                "val_mse": val_metrics["mse"],
                "val_cos_res": val_metrics["cos"],
                "lr": current_lr,
                **val_stats
            }
            mlflow.log_metrics(log_dict, step=ep)
            
            writer.add_scalar("loss/train", tr_loss, ep)
            writer.add_scalar("loss/val", val_loss, ep)
            writer.add_scalar("lr", current_lr, ep)

            # Save Checkpoint for best result
            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                torch.save({
                    "epoch": ep, "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(), "best_val": best_val,
                    "mlflow_run_id": run_id,
                }, os.path.join(artifact_dir, "checkpoint_best.pth"))
                print(f"   [info] New best val_loss: {best_val:.6f}")
            
            # Save Last Checkpoint
            torch.save({
                "epoch": ep, 
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), 
                "best_val": best_val,
                "mlflow_run_id": run_id,
            }, os.path.join(artifact_dir, "checkpoint_last.pth"))

        # --- FINAL ARTIFACTS ---
        if best_state:
            model.load_state_dict(best_state)

        # Get final data with extra info
        _, pred_all, true_all, extra_info, _ = run_epoch_stream(
            model, optimizer, device, root, tree,
            step_size=chunksize, batch_size=max(batch,256), train=False, amp=False,
            max_chunks=max_chunks, npho_branch=npho_branch, time_branch=time_branch,
            NphoScale=NphoScale, time_shift=time_shift, time_scale=time_scale,
            loss_type=loss_type
        )

        if pred_all is not None:
            csv_path = os.path.join(artifact_dir, f"predictions_{run_name}.csv")
            pd.DataFrame({
                "true_theta": true_all[:,0], "true_phi": true_all[:,1],
                "pred_theta": pred_all[:,0], "pred_phi": pred_all[:,1]
            }).to_csv(csv_path, index=False)
            mlflow.log_artifact(csv_path)

            # --- WORST EVENT PLOTTING ---
            worst_events = extra_info.get("worst_events", [])
            for i, (err, raw_n, raw_t, p, t, vtx, energy) in enumerate(worst_events):

                vtx_str = f"({vtx[0]:.1f}, {vtx[1]:.1f}, {vtx[2]:.1f})"
                base_title = (f"Worst #{i+1} (Loss={err:.4f})\n"
                              f"Truth E={energy:.1f} MeV | VTX={vtx_str}\n"
                              f"Truth: θ={t[0]:.2f}, φ={t[1]:.2f} | Pred: θ={p[0]:.2f}, φ={p[1]:.2f}")

                # 1. Plot Npho Faces
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

            res_pdf = os.path.join(artifact_dir, f"resolution_profile_{run_name}.pdf")
            plot_resolution_profile(pred_all, true_all, outfile=res_pdf)
            mlflow.log_artifact(res_pdf)

            weight_pdf = os.path.join(artifact_dir, f"face_weights_{run_name}.pdf")
            plot_face_weights(model, outfile=weight_pdf)
            mlflow.log_artifact(weight_pdf)

            plot_profile(pred_all[:,0], true_all[:,0], label="Theta", 
                         outfile=os.path.join(artifact_dir, "profile_theta.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "profile_theta.pdf"))
            
            plot_cos_residuals(pred_all, true_all, outfile=os.path.join(artifact_dir, "cos_res.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "cos_res.pdf"))
            
            plot_pred_truth_scatter(pred_all, true_all, outfile=os.path.join(artifact_dir, "scatter.pdf"))
            mlflow.log_artifact(os.path.join(artifact_dir, "scatter.pdf"))

        # --- EXPORT ONNX TO ARTIFACT DIR ---
        if onnx:
            onnx_path = os.path.join(artifact_dir, onnx)
            print(f"[INFO] Exporting ONNX model to {onnx_path}...")
            model.eval()
            dummy_input = torch.randn(1, 4760, 2, device=device)
            try:
                torch.onnx.export(
                    model, dummy_input, onnx_path, export_params=True,
                    opset_version=13, do_constant_folding=True,
                    input_names=['input'], output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
                )
                if os.path.exists(onnx_path):
                    mlflow.log_artifact(onnx_path)
                    print(f"[INFO] ONNX model exported and logged.")
            except Exception as e:
                 print(f"[WARN] Failed to export ONNX: {e}")

        writer.close()