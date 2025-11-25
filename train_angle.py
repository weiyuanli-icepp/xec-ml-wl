#!/usr/bin/env python3

import os, time
import numpy as np
import math
import uproot
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import mlflow
import mlflow.pytorch

from angle_model_geom import AngleRegressorSharedFaces, plot_event_faces  # <-- from above

# ------------------------------------------------------------
# GPU memory monitoring utilities
# ------------------------------------------------------------

def format_mem(bytes_val):
    return f"{bytes_val / (1024**3):.2f} GB"

def get_gpu_memory_stats(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda")
    torch.cuda.synchronize(device)
    alloc   = torch.cuda.memory_allocated(device)
    reserv  = torch.cuda.memory_reserved(device)
    peak    = torch.cuda.max_memory_allocated(device)
    return {
        "allocated": alloc,
        "reserved": reserv,
        "peak": peak,
    }

# ------------------------------------------------------------
#  Utilities: streaming ROOT reading
# ------------------------------------------------------------

def iterate_chunks(path, tree, branches, step_size=4000):
    with uproot.open(path) as f:
        t = f[tree]
        for arrays in t.iterate(branches, step_size=step_size, library="np"):
            yield arrays

# ------------------------------------------------------------
# Angle conversion utilities
# ------------------------------------------------------------
def angles_deg_to_unit_vec(angles: torch.Tensor) -> torch.Tensor:
    """
    angles: (B,2) tensor of [theta_deg, phi_deg].
    Convert to a 3D unit vector consistent with your emiVec convention.

    You said:
      phi = atan2(emiVec[1], -emiVec[0]) * 180 / pi
    We assume theta is the polar angle from +z (0 to 180 deg).

    Then a consistent mapping back is:
      x = -sin(theta) * cos(phi)
      y =  sin(theta) * sin(phi)
      z =  cos(theta)
    """
    # angles in degrees → radians
    theta = torch.deg2rad(angles[:, 0])
    phi   = torch.deg2rad(angles[:, 1])

    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    sin_ph = torch.sin(phi)
    cos_ph = torch.cos(phi)

    x = -sin_th * cos_ph
    y =  sin_th * sin_ph
    z =  cos_th

    v = torch.stack([x, y, z], dim=1)  # (B,3)
    return v

# ------------------------------------------------------------
#  Angle histograms for reweighting
# ------------------------------------------------------------

def scan_angle_hist_1d(root, tree="tree", comp=0, nbins=30, step_size=4000):
    """
    1D histogram over emiAng[:, comp].
    comp = 0 -> theta, comp = 1 -> phi
    Returns:
        edges:   (nbins+1,) array
        weights: (nbins,) inverse-frequency weights (mean ≈ 1)
    """
    root = os.path.expanduser(root)

    # 1st pass: get min/max
    vmin, vmax = +np.inf, -np.inf
    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        vals = ang[:, comp]
        if vals.size == 0:
            continue
        vmin = min(vmin, vals.min())
        vmax = max(vmax, vals.max())

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError("scan_angle_hist_1d: invalid range")

    edges = np.linspace(vmin, vmax, nbins + 1)
    counts = np.zeros(nbins, dtype=np.int64)

    # 2nd pass: count
    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        vals = ang[:, comp]
        if vals.size == 0:
            continue
        h, _ = np.histogram(vals, bins=edges)
        counts += h
        
    weights = np.zeros(nbins, dtype=np.float64)
    valid = counts > 0
    if valid.any():
        total_counts = counts.sum()
        k = total_counts / valid.sum()
        weights[valid] = k / counts[valid]

    return edges, weights


def scan_angle_hist_2d(root, tree="tree", nbins_theta=20, nbins_phi=20, step_size=4000):
    """
    2D histogram over (theta=emiAng[:,0], phi=emiAng[:,1]).
    Returns:
        edges_theta: (nbins_theta+1,)
        edges_phi:   (nbins_phi+1,)
        weights_2d:  (nbins_theta, nbins_phi) inverse-frequency weights
    """
    root = os.path.expanduser(root)

    th_min, th_max = +np.inf, -np.inf
    ph_min, ph_max = +np.inf, -np.inf

    # 1st pass: ranges
    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        theta = ang[:, 0]
        phi   = ang[:, 1]
        if theta.size == 0:
            continue
        th_min = min(th_min, theta.min())
        th_max = max(th_max, theta.max())
        ph_min = min(ph_min, phi.min())
        ph_max = max(ph_max, phi.max())

    if not (np.isfinite(th_min) and np.isfinite(th_max) and
            np.isfinite(ph_min) and np.isfinite(ph_max)):
        raise RuntimeError("scan_angle_hist_2d: invalid range")

    edges_theta = np.linspace(th_min, th_max, nbins_theta + 1)
    edges_phi   = np.linspace(ph_min, ph_max, nbins_phi + 1)
    counts = np.zeros((nbins_theta, nbins_phi), dtype=np.int64)

    # 2nd pass: counts
    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        theta = ang[:, 0]
        phi   = ang[:, 1]
        if theta.size == 0:
            continue
        h, _, _ = np.histogram2d(theta, phi, bins=[edges_theta, edges_phi])
        counts += h.astype(np.int64)
        
    weights_2d = np.zeros_like(counts, dtype=np.float64)
    valid = counts > 0
    if valid.any():
        total_counts = counts.sum()
        k = total_counts / valid.sum()
        weights_2d[valid] = k / counts[valid]

    return edges_theta, edges_phi, weights_2d

# ------------------------------------------------------------
#  Training / Validation loop (streamed)
# ------------------------------------------------------------

def run_epoch_stream(
    model, optimizer, device, root, tree,
    step_size=4000, batch_size=128, train=True, amp=True,
    max_chunks=None,
    npho_branch="relative_npho",         # "relative_npho" or "npho"
    NphoScale=2e5,
    reweight_mode="none",
    edges_theta=None, weights_theta=None,
    edges_phi=None,   weights_phi=None,
    edges2_theta=None, edges2_phi=None, weights_2d=None,
    loss_type="smooth_l1",
):
    model.train(train)
    loss_fn = nn.SmoothL1Loss(reduction="none")    # or MSELoss
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and device_type == "cuda"))

    total_loss, nobs = 0.0, 0
    all_pred, all_true = [], []
    last_npho_chunk = None # for visualization

    chunks_done = 0
    for arr in iterate_chunks(root, tree, [npho_branch, "emiAng", "emiVec"], step_size):
        if max_chunks and chunks_done >= max_chunks:
            break
        chunks_done += 1

        Npho = arr[npho_branch].astype("float32")
        Y = arr["emiAng"].astype("float32")
        V = arr["emiVec"].astype("float32")
        
        last_npho_chunk = Npho.copy()

        # Preprocess photons
        Npho = np.maximum(Npho, 0.0)
        Npho_log = np.log1p(Npho / NphoScale).astype("float32")

        ds = TensorDataset(
            torch.from_numpy(Npho_log),
            torch.from_numpy(Y),
            torch.from_numpy(V),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=False)

        for Npho_b, Y_b, V_b in loader:
            Npho_b = Npho_b.to(device)
            Y_b    = Y_b.to(device)
            V_b    = V_b.to(device)

            # --- compute angle-based weights w (shape: (B,)) ---
            w = None
            if reweight_mode == "theta" and (edges_theta is not None) and (weights_theta is not None):
                th = Y_b[:, 0].detach().cpu().numpy()
                bin_id = np.clip(np.digitize(th, edges_theta) - 1, 0, len(weights_theta) - 1)
                w = torch.from_numpy(weights_theta[bin_id].astype("float32")).to(device)  # (B,)
            elif reweight_mode == "phi" and (edges_phi is not None) and (weights_phi is not None):
                ph = Y_b[:, 1].detach().cpu().numpy()
                bin_id = np.clip(np.digitize(ph, edges_phi) - 1, 0, len(weights_phi) - 1)
                w = torch.from_numpy(weights_phi[bin_id].astype("float32")).to(device)
            elif reweight_mode == "theta_phi" and (edges2_theta is not None) and (edges2_phi is not None) and (weights_2d is not None):
                th = Y_b[:, 0].detach().cpu().numpy()
                ph = Y_b[:, 1].detach().cpu().numpy()
                id_th = np.clip(np.digitize(th, edges2_theta) - 1, 0, len(edges2_theta) - 2)
                id_ph = np.clip(np.digitize(ph, edges2_phi)   - 1, 0, len(edges2_phi)   - 2)
                w_np = weights_2d[id_th, id_ph].astype("float32")
                w = torch.from_numpy(w_np).to(device)

            if train:
                optimizer.zero_grad(set_to_none=True)
                try:
                    with torch.amp.autocast(
                        device_type,
                        enabled=(amp and device_type == "cuda"),
                        dtype=torch.bfloat16
                    ):
                        pred_angles = model(Npho_b)          # (B,2)
                        
                        if loss_type == "smooth_l1":
                            lvec     = loss_fn(pred_angles, Y_b)
                            l_sample = lvec.mean(dim=1)
                        elif loss_type == "cos":
                            v_pred = angles_deg_to_unit_vec(pred_angles)
                            v_true = V_b
                            cos_sim = torch.sum(v_pred * v_true, dim=1)
                            cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                            l_sample = 1.0 - cos_sim
                        else:
                            raise ValueError(f"Unknown loss_type: {loss_type}")
                        
                        if w is not None:
                            loss = (l_sample * w).mean()
                        else:
                            loss = l_sample.mean()

                    if amp and device_type == "cuda":
                        scaler.scale(loss).backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer); scaler.update()
                    else:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        print("[WARN] CUDA out of memory in this batch. Skipping this batch.")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
            else:
                with torch.no_grad():
                    pred_angles = model(Npho_b)
                    
                    if loss_type == "smooth_l1":
                        lvec = loss_fn(pred_angles, Y_b)
                        l_sample = lvec.mean(dim=1)
                    elif loss_type == "cos":
                        v_pred = angles_deg_to_unit_vec(pred_angles)
                        v_true = V_b
                        cos_sim = torch.sum(v_pred * v_true, dim=1)
                        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
                        l_sample = 1.0 - cos_sim
                    else:
                        raise ValueError(f"Unknown loss_type: {loss_type}")
                    
                    if w is not None:
                        loss = (l_sample * w).mean()
                    else:
                        loss = l_sample.mean()
                    
                    all_pred.append(pred_angles.cpu().numpy())
                    all_true.append(Y_b.cpu().numpy())
                    last_arr = arr

            total_loss += loss.item() * Npho_b.size(0)
            nobs       += Npho_b.size(0)

        torch.cuda.empty_cache()

    loss_avg = total_loss / max(1, nobs)
    if not train and all_pred:
        pred_np = np.concatenate(all_pred, axis=0)
        true_np = np.concatenate(all_true, axis=0)
        return loss_avg, pred_np, true_np, last_npho_chunk
    return loss_avg, None, None, None



# ------------------------------------------------------------
#  Plotting helper (optional)
# ------------------------------------------------------------

def eval_plots_angle(pred, true, outfile=None):
    """
    pred, true: (N,2) arrays
    Plot residuals for each component separately.
    """
    res = pred - true  # (N,2)
    labels = ["emiAng[0]", "emiAng[1]"]

    plt.figure(figsize=(10,4))
    for i in range(2):
        plt.subplot(1,2,i+1)
        plt.hist(res[:, i], bins=200)
        plt.title(f"Residuals: {labels[i]} (pred - true)")
        plt.xlabel("Δ")
        plt.ylabel("count")
    plt.tight_layout()
    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile, dpi=100)
        plt.close()

def plot_prediction_examples(pred, true, n_examples=5, outfile=None):
    """
    Show a few examples of predicted vs truth angles.
    """
    idx = np.random.choice(len(pred), size=n_examples, replace=False)
    plt.figure(figsize=(8, 2*n_examples))

    for k, i in enumerate(idx, 1):
        p = pred[i]
        t = true[i]
        plt.subplot(n_examples, 1, k)
        plt.title(f"Example {k}:   Truth={t}   Pred={p}")
        plt.bar(["Δ0","Δ1"], p - t)
        plt.ylim(-1, 1)

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()

def plot_pred_truth_scatter(pred, true, outfile=None):
    plt.figure(figsize=(10,4))

    for comp in range(2):
        plt.subplot(1,2,comp+1)
        plt.scatter(true[:,comp], pred[:,comp], s=3, alpha=0.4)
        plt.xlabel(f"Truth emiAng[{comp}]")
        plt.ylabel(f"Pred emiAng[{comp}]")
        lim = max(true[:,comp].max(), pred[:,comp].max())
        plt.plot([0,lim],[0,lim],'k--')
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()

# ------------------------------------------------------------
#  Main training entry
# ------------------------------------------------------------

def main_angle_with_args(
    root,
    tree="tree",
    epochs=20,
    batch=256,
    chunksize=4000,
    lr=3e-4,
    weight_decay=1e-4,
    amp=True,
    max_chunks=None,
    npho_branch="relative_npho",             # "npho" or "relative_npho"
    NphoScale=2e5,
    onnx="meg2ang.onnx",
    mlflow_experiment="gamma_angle",
    run_name=None,
    outer_mode="finegrid",             # "split" (coarse + central faces) or "finegrid"
    outer_fine_pool=(3,3),           # e.g., (3,3) to downsample 45x72 outer fine grid
    reweight_mode="none",
    nbins_theta=50,
    nbins_phi=50,
    loss_type="smooth_l1",
):
    root = os.path.expanduser(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AngleRegressorSharedFaces(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool,
    ).to(device)

    # optimizer: no decay on bias/BN
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if n.endswith(".bias") or "bn" in n.lower() else decay).append(p)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr
    )

    # MLflow + TB setup
    mlflow.set_experiment(mlflow_experiment)
    if run_name is None:
        run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=os.path.join("runs", run_name))

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            "root": root,
            "tree": tree,
            "epochs": epochs,
            "batch": batch,
            "chunksize": chunksize,
            "lr": lr,
            "weight_decay": weight_decay,
            "amp": amp,
            "max_chunks": max_chunks,
            "npho_branch": npho_branch,
            "NphoScale": NphoScale,
            "backbone_base_channels": 16,
            "outer_mode": outer_mode,
            "outer_fine_pool": str(outer_fine_pool),
            "reweight_mode": reweight_mode,
            "nbins_theta": nbins_theta,
            "nbins_phi": nbins_phi,
            "loss_type": loss_type,
        })
        
        # --- angle histograms for reweighting ---
        edges_theta = weights_theta = None
        edges_phi   = weights_phi   = None
        edges2_theta = edges2_phi = weights_2d = None

        if reweight_mode == "theta":
            print("[info] using theta reweighting")
            edges_theta, weights_theta = scan_angle_hist_1d(
                root, tree=tree, comp=0,
                nbins=nbins_theta, step_size=chunksize
            )

        elif reweight_mode == "phi":
            print("[info] using phi reweighting")
            edges_phi, weights_phi = scan_angle_hist_1d(
                root, tree=tree, comp=1,
                nbins=nbins_phi, step_size=chunksize
            )

        elif reweight_mode == "theta_phi":
            print("[info] using 2D (theta,phi) reweighting")
            edges2_theta, edges2_phi, weights_2d = scan_angle_hist_2d(
                root, tree=tree,
                nbins_theta=nbins_theta,
                nbins_phi=nbins_phi,
                step_size=chunksize
            )
        
        best_val, best_state = float("inf"), None

        for ep in range(1, epochs+1):
            t0 = time.time()
            tr_loss, _, _, _ = run_epoch_stream(
                model, optimizer, device, root, tree,
                step_size=chunksize, batch_size=batch,
                train=True, amp=amp,
                max_chunks=max_chunks,
                npho_branch=npho_branch,
                NphoScale=NphoScale,
                reweight_mode=reweight_mode,
                edges_theta=edges_theta, weights_theta=weights_theta,
                edges_phi=edges_phi,   weights_phi=weights_phi,
                edges2_theta=edges2_theta, edges2_phi=edges2_phi, weights_2d=weights_2d,
                loss_type=loss_type,
            )

            val_loss, pred_val, true_val, _ = run_epoch_stream(
                model, optimizer, device, root, tree,
                step_size=chunksize, batch_size=max(batch,256),
                train=False, amp=False,
                max_chunks=max_chunks,
                npho_branch=npho_branch,
                NphoScale=NphoScale,
                reweight_mode=reweight_mode,
                edges_theta=edges_theta, weights_theta=weights_theta,
                edges_phi=edges_phi,   weights_phi=weights_phi,
                edges2_theta=edges2_theta, edges2_phi=edges2_phi, weights_2d=weights_2d,
                loss_type=loss_type,
            )

            sec = time.time() - t0
            print(f"[{ep:03d}] train {tr_loss:.6f}  val {val_loss:.6f}  time {sec:.1f}s")
            
            if device.type == "cuda":
                stats = get_gpu_memory_stats(device)
                if stats is not None:
                    alloc_gb = stats["allocated"] / (1024**3)
                    peak_gb  = stats["peak"]      / (1024**3)
                    print(f"   [mem] alloc={alloc_gb:.2f} GB, peak={peak_gb:.2f} GB")
                    writer.add_scalar("memory/allocated_GB", alloc_gb, ep)
                    writer.add_scalar("memory/peak_GB",      peak_gb,  ep)
                    mlflow.log_metrics({
                        "memory_allocated_GB": alloc_gb,
                        "memory_peak_GB":      peak_gb,
                    }, step=ep)
                    torch.cuda.reset_peak_memory_stats(device)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "epoch_time_sec": sec,
            }, step=ep)

            writer.add_scalar("loss/train", tr_loss, ep)
            writer.add_scalar("loss/val", val_loss, ep)
            writer.add_scalar("time/epoch_sec", sec, ep)

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"[info] best val_loss = {best_val:.6f}")
        mlflow.log_metric("best_val_loss", best_val)

        # Load best state and do a full evaluation pass
        if best_state:
            model.load_state_dict(best_state)

        _, pred_all, true_all, last_npho_chunk = run_epoch_stream(
            model, optimizer, device, root, tree,
            step_size=chunksize, batch_size=max(batch,256),
            train=False, amp=False,
            max_chunks=max_chunks,
            npho_branch=npho_branch,
            NphoScale=NphoScale,
            reweight_mode=reweight_mode,
            edges_theta=edges_theta, weights_theta=weights_theta,
            edges_phi=edges_phi,   weights_phi=weights_phi,
            edges2_theta=edges2_theta, edges2_phi=edges2_phi, weights_2d=weights_2d,
            loss_type=loss_type,
        )

        # ---------- 1. scatter plot ----------
        scatter_png = f"scatter_pred_truth_{run_name}.png"
        plot_pred_truth_scatter(pred_all, true_all, outfile=scatter_png)
        mlflow.log_artifact(scatter_png)

        # ---------- 2. example predictions ----------
        examples_png = f"examples_pred_truth_{run_name}.png"
        plot_prediction_examples(pred_all, true_all, n_examples=5, outfile=examples_png)
        mlflow.log_artifact(examples_png)

        # ---------- 3. event visualization ----------
        if last_npho_chunk is not None:
            n_in_chunk = last_npho_chunk.shape[0]
            n_vis = min(3, n_in_chunk)
            idx_chunk = np.random.choice(n_in_chunk, size=n_vis, replace=False)
            for k, j in enumerate(idx_chunk):
                event_npho = last_npho_chunk[j]  # shape (4760,)
                face_png = f"event_faces_{k}_{run_name}.png"
                plot_event_faces(
                    event_npho,
                    title=f"Event (last-chunk idx={j})",
                    savepath=face_png,
                    outer_mode=outer_mode,
                    outer_fine_pool=outer_fine_pool,
                )
                mlflow.log_artifact(face_png)

        # Plot residuals
        if pred_all is not None:
            residual_png = f"angle_residuals_{run_name}.png"
            eval_plots_angle(pred_all, true_all, outfile=residual_png)
            mlflow.log_artifact(residual_png)

        # Export ONNX + log model
        model.eval()
        dummy = torch.randn(1, 4760, device=device)
        # torch.onnx.export(
            # model, dummy, onnx,
            # input_names=["Npho4760"],
            # output_names=["emiAng"]
        # )
        print(f"[OK] Exported ONNX to {onnx}")
        if os.path.exists(onnx):
            mlflow.log_artifact(onnx)

        mlflow.pytorch.log_model(
            model,
            "pytorch_angle_model",
            input_example={"Npho4760": dummy.cpu().numpy()},
        )

        writer.close()
