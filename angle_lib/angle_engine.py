import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .angle_utils import iterate_chunks, angles_deg_to_unit_vec
from .angle_metrics import eval_stats, eval_resolution

def run_epoch_stream(
    model, optimizer, device, root, tree,
    step_size=4000, batch_size=128, train=True, amp=True,
    max_chunks=None,
    npho_branch="relative_npho",
    time_branch="relative_time",
    NphoScale=2e5,
    time_shift=0.0,
    time_scale=1e-7,
    reweight_mode="none",
    edges_theta=None, weights_theta=None,
    edges_phi=None,   weights_phi=None,
    edges2_theta=None, edges2_phi=None, weights_2d=None,
    loss_type="smooth_l1",
    scheduler=None
):
    model.train(train)
    # Define all loss functions for tracking
    criterion_smooth = nn.SmoothL1Loss(reduction="none")
    criterion_l1 = nn.L1Loss(reduction="none")
    criterion_mse = nn.MSELoss(reduction="none")
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and device_type == "cuda"))

    # Accumulators for metrics
    loss_sums = {"total_opt": 0.0, "smooth_l1": 0.0, "l1": 0.0, "mse": 0.0, "cos": 0.0}
    nobs = 0
    all_pred, all_true = [], []
    
    val_inputs_npho = [] 
    val_inputs_time = []
    
    # Store tuples: (error_metric, npho_tensor, pred, true) for Worst 5
    worst_events_buffer = [] 

    chunks_done = 0
    branches_to_load = [npho_branch, time_branch, "emiAng", "emiVec"]
    
    for arr in iterate_chunks(root, tree, branches_to_load, step_size):
        if max_chunks and chunks_done >= max_chunks:
            break
        chunks_done += 1

        Npho = arr[npho_branch].astype("float32")
        Time = arr[time_branch].astype("float32")
        Y = arr["emiAng"].astype("float32")
        V = arr["emiVec"].astype("float32")
        
        # --- PREPROCESSING ---
        Npho = np.maximum(Npho, 0.0)
        mask_garbage_time = (np.abs(Time) > 1.0) | np.isnan(Time)
        mask_invalid = (Npho <= 0.0) | mask_garbage_time
        Time[mask_invalid] = 0.0
        
        if not train and len(val_inputs_npho) < 10000:
            n_flat = Npho.flatten()
            t_flat = Time.flatten()
            valid_mask = n_flat > 0
            val_inputs_npho.append(n_flat[valid_mask])
            val_inputs_time.append(t_flat[valid_mask])

        # Customizable scaling
        Time_norm = (Time - time_shift) / time_scale
        Npho_log = np.log1p(Npho / NphoScale).astype("float32")
        
        X_stacked = np.stack([Npho_log, Time_norm], axis=-1)

        ds = TensorDataset(
            torch.from_numpy(X_stacked),
            torch.from_numpy(Y),
            torch.from_numpy(V),
            torch.from_numpy(Npho) # Raw Npho for viz
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=False)

        for i_batch, (Npho_b, Y_b, V_b, Raw_Npho_b) in enumerate(loader):
            Npho_b = Npho_b.to(device)
            Y_b    = Y_b.to(device)
            V_b    = V_b.to(device)

            # Reweighting Logic
            w = None
            if reweight_mode == "theta" and (edges_theta is not None):
                th = Y_b[:, 0].detach().cpu().numpy()
                bin_id = np.clip(np.digitize(th, edges_theta) - 1, 0, len(weights_theta) - 1)
                w = torch.from_numpy(weights_theta[bin_id].astype("float32")).to(device)
            elif reweight_mode == "phi" and (edges_phi is not None):
                ph = Y_b[:, 1].detach().cpu().numpy()
                bin_id = np.clip(np.digitize(ph, edges_phi) - 1, 0, len(weights_phi) - 1)
                w = torch.from_numpy(weights_phi[bin_id].astype("float32")).to(device)
            elif reweight_mode == "theta_phi" and (edges2_theta is not None):
                th = Y_b[:, 0].detach().cpu().numpy()
                ph = Y_b[:, 1].detach().cpu().numpy()
                id_th = np.clip(np.digitize(th, edges2_theta) - 1, 0, len(edges2_theta) - 2)
                id_ph = np.clip(np.digitize(ph, edges2_phi)   - 1, 0, len(edges2_phi)   - 2)
                w_np = weights_2d[id_th, id_ph].astype("float32")
                w = torch.from_numpy(w_np).to(device)

            # --- TRAINING: Compute ONLY specific loss ---
            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type, enabled=(amp and device_type == "cuda"), dtype=torch.bfloat16):
                    pred_angles = model(Npho_b)
                    
                    l_opt = None
                    batch_smooth = 0.0
                    batch_l1 = 0.0
                    batch_mse = 0.0
                    batch_cos = 0.0

                    # Only compute what we need
                    if loss_type == "smooth_l1":
                        l_vec = criterion_smooth(pred_angles, Y_b).mean(dim=1)
                        l_opt = l_vec
                        batch_smooth = l_vec.sum().item()
                    elif loss_type == "l1":
                        l_vec = criterion_l1(pred_angles, Y_b).mean(dim=1)
                        l_opt = l_vec
                        batch_l1 = l_vec.sum().item()
                    elif loss_type == "mse":
                        l_vec = criterion_mse(pred_angles, Y_b).mean(dim=1)
                        l_opt = l_vec
                        batch_mse = l_vec.sum().item()
                    elif loss_type == "cos":
                        v_pred = angles_deg_to_unit_vec(pred_angles)
                        cos_sim = torch.sum(v_pred * V_b, dim=1).clamp(-1.0, 1.0)
                        l_vec = 1.0 - cos_sim
                        l_opt = l_vec
                        batch_cos = l_vec.sum().item()
                    else:
                        # Fallback
                        l_vec = criterion_smooth(pred_angles, Y_b).mean(dim=1)
                        l_opt = l_vec
                        batch_smooth = l_vec.sum().item()

                    if w is not None:
                        loss = (l_opt * w).mean()
                    else:
                        loss = l_opt.mean()

                if amp and device_type == "cuda":
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # Update sums
                loss_sums["total_opt"] += loss.item() * Npho_b.size(0)
                loss_sums["smooth_l1"] += batch_smooth
                loss_sums["l1"]        += batch_l1
                loss_sums["mse"]       += batch_mse
                loss_sums["cos"]       += batch_cos
                nobs += Npho_b.size(0)

            # --- VALIDATION: Compute ALL losses ---
            else:
                with torch.no_grad():
                    pred_angles = model(Npho_b)
                    
                    l_smooth = criterion_smooth(pred_angles, Y_b).mean(dim=1)
                    l_l1 = criterion_l1(pred_angles, Y_b).mean(dim=1)
                    l_mse = criterion_mse(pred_angles, Y_b).mean(dim=1)
                    
                    v_pred = angles_deg_to_unit_vec(pred_angles)
                    cos_sim = torch.sum(v_pred * V_b, dim=1).clamp(-1.0, 1.0)
                    l_cos = 1.0 - cos_sim
                    
                    if loss_type == "smooth_l1": l_opt = l_smooth
                    elif loss_type == "cos": l_opt = l_cos
                    elif loss_type == "mse": l_opt = l_mse
                    elif loss_type == "l1": l_opt = l_l1
                    else: l_opt = l_smooth
                    
                    if w is not None:
                        loss = (l_opt * w).mean()
                    else:
                        loss = l_opt.mean()
                    
                    all_pred.append(pred_angles.cpu().numpy())
                    all_true.append(Y_b.cpu().numpy())
                    
                    # Worst Case Tracking
                    batch_errs_np = l_opt.cpu().numpy()
                    worst_idx = np.argsort(batch_errs_np)[-5:] # Top 5 in batch
                    
                    for idx in worst_idx:
                        err = batch_errs_np[idx]
                        raw_n = Raw_Npho_b[idx].cpu().numpy()
                        p = pred_angles[idx].cpu().numpy()
                        t = Y_b[idx].cpu().numpy()
                        worst_events_buffer.append((err, raw_n, p, t))
                    
                    worst_events_buffer.sort(key=lambda x: x[0], reverse=True)
                    worst_events_buffer = worst_events_buffer[:5]

                    loss_sums["total_opt"] += loss.item() * Npho_b.size(0)
                    loss_sums["smooth_l1"] += l_smooth.sum().item()
                    loss_sums["l1"]        += l_l1.sum().item()
                    loss_sums["mse"]       += l_mse.sum().item()
                    loss_sums["cos"]       += l_cos.sum().item()
                    nobs += Npho_b.size(0)

        torch.cuda.empty_cache()

    # Scheduler Step (if training)
    if train and scheduler is not None:
        scheduler.step()

    # Averages
    metrics = {k: v / max(1, nobs) for k, v in loss_sums.items()}
    
    # Package extra info
    extra_info = {}
    if not train and all_pred:
        pred_np = np.concatenate(all_pred, axis=0)
        true_np = np.concatenate(all_true, axis=0)
        
        val_stats = eval_stats(pred_np, true_np, print_out=False)
        res_68, psi_deg = eval_resolution(pred_np, true_np)
        val_stats["val_resolution_deg"] = res_68
        
        extra_info["worst_events"] = worst_events_buffer
        if val_inputs_npho:
            extra_info["input_dist"] = {
                "npho": np.concatenate(val_inputs_npho),
                "time": np.concatenate(val_inputs_time)
            }
        
        return metrics, pred_np, true_np, extra_info, val_stats
        
    return metrics, None, None, None, {}