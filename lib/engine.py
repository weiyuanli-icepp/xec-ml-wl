import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time

from .utils import iterate_chunks, angles_deg_to_unit_vec
from .metrics import eval_stats, eval_resolution

def run_epoch_stream(
    model, optimizer, device, root, tree,
    scaler=None,
    step_size=4000, batch_size=128, train=True, amp=True,
    max_chunks=None,
    npho_branch="relative_npho",
    time_branch="relative_time",
    NphoScale=1e5,
    NphoScale2=13,
    time_shift=-0.29,
    time_scale=2.32e6,
    sentinel_value=-5.0,
    reweight_mode="none",
    edges_theta=None, weights_theta=None,
    edges_phi=None,   weights_phi=None,
    edges2_theta=None, edges2_phi=None, weights_2d=None,
    loss_type="smooth_l1",
    loss_beta=1.0,
    scheduler=None,
    ema_model=None
):
    model.train(train)
    criterion_smooth = nn.SmoothL1Loss(reduction="none", beta=loss_beta)
    criterion_l1 = nn.L1Loss(reduction="none")
    criterion_mse = nn.MSELoss(reduction="none")
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    # scaler = torch.amp.GradScaler(device_type, enabled=(amp and device_type == "cuda"))
    if scaler is None and train:
        scaler = torch.amp.GradScaler("cuda", enabled=amp)

    loss_sums = {"total_opt": 0.0, "smooth_l1": 0.0, "l1": 0.0, "mse": 0.0, "cos": 0.0}
    nobs = 0
    
    epoch_throughput = []
    epoch_data_time  = []
    epoch_batch_time = []
    t_end = time.time()
    
    # Collections for the Final ROOT file
    val_root_data = {
        "run_id": [],
        "event_id": [],
        "pred_theta": [], "pred_phi": [],
        "true_theta": [], "true_phi": [],
        "opening_angle": [],
        "energy_truth": [],
        "x_truth": [], "y_truth": [], "z_truth": [],
        "x_vtx": [], "y_vtx": [], "z_vtx": []
    }
    
    val_inputs_npho = [] 
    val_inputs_time = []
    worst_events_buffer = [] 

    chunks_done = 0
    branches_to_load = [npho_branch, time_branch, "emiAng", "emiVec", "xyzTruth", "xyzVTX", "energyTruth", "run", "event"]
    
    # ========================== START OF CHUNK LOOP ==========================
    for arr in iterate_chunks(root, tree, branches_to_load, step_size):
        if max_chunks and chunks_done >= max_chunks:
            break
        chunks_done += 1

        Npho     = arr[npho_branch].astype("float32")
        Time     = arr[time_branch].astype("float32")
        Y        = arr["emiAng"].astype("float32")
        V        = arr["emiVec"].astype("float32")
        XYZ_tru  = arr["xyzTruth"].astype("float32") # (N, 3)
        XYZ_vtx  = arr["xyzVTX"].astype("float32") # (N, 3)
        E_truth  = arr["energyTruth"].astype("float32") # (N,)
        RunNum   = arr["run"].astype("int32")  # (N,)
        EventNum = arr["event"].astype("int32")  # (N,)
        
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

        # Time_norm = Time / time_scale - time_shift
        # Npho_log = np.log1p(Npho / NphoScale).astype("float32")
        # Npho_norm = Npho_log / NphoScale2
        # X_stacked = np.stack([Npho_norm, Time_norm], axis=-1)
        X_raw = np.stack([Npho, Time], axis=-1).astype("float32")

        ds = TensorDataset(
            torch.from_numpy(X_raw),  # Raw input features: Npho and Time
            torch.from_numpy(Y),          # True angles in degrees
            torch.from_numpy(V),          # Unit vector of true angles
            torch.from_numpy(RunNum),     # Run Number
            torch.from_numpy(EventNum),   # Event Number
            torch.from_numpy(XYZ_tru),    # XYZ Truth
            torch.from_numpy(XYZ_vtx),    # XYZ VTX
            torch.from_numpy(E_truth)     # Energy Truth (in GeV)
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=False)

        # ========================== START OF BATCH LOOP ==========================
        for i_batch, (X_raw_b, Y_b, V_b, RunNum_b, EventNum_b, XYZ_tru_b, XYZ_vtx_b, E_b) in enumerate(loader):
            t_data  = time.time() - t_end
            
            X_raw_b = X_raw_b.to(device, non_blocking=True)
            Y_b    = Y_b.to(device, non_blocking=True)
            V_b    = V_b.to(device, non_blocking=True)
            
            raw_npho = X_raw_b[:, :, 0]
            raw_time = X_raw_b[:, :, 1]
            Raw_N_b = X_raw_b[:, :, 0]
            Raw_T_b = X_raw_b[:, :, 1]
            # time_norm = raw_time / time_scale - time_shift
            # npho_log = torch.log1p(raw_npho / NphoScale)
            # npho_norm = npho_log / NphoScale2
            # X_batch = torch.stack([npho_norm, time_norm], dim=-1)
            npho_norm = torch.log1p(raw_npho / NphoScale) / NphoScale2
            time_norm = (raw_time / time_scale) - time_shift
            mask_npho_bad = (raw_npho <= 0.0) | (raw_npho > 9e9) | torch.isnan(raw_npho)
            mask_time_bad = mask_npho_bad | (torch.abs(raw_time) > 9e9) | torch.isnan(raw_time)
            npho_norm[mask_npho_bad] = 0.0
            time_norm[mask_time_bad] = sentinel_value
            X_batch = torch.stack([npho_norm, time_norm], dim=-1)
            
            # Reweighting
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

            if train:
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type, enabled=(amp and device_type == "cuda"), dtype=torch.bfloat16):
                    pred_angles = model(X_batch)
                    
                    if loss_type == "smooth_l1": l_opt = criterion_smooth(pred_angles, Y_b).mean(dim=1)
                    elif loss_type == "l1": l_opt = criterion_l1(pred_angles, Y_b).mean(dim=1)
                    elif loss_type == "mse": l_opt = criterion_mse(pred_angles, Y_b).mean(dim=1)
                    elif loss_type == "cos":
                        v_pred = angles_deg_to_unit_vec(pred_angles)
                        l_opt = 1.0 - torch.sum(v_pred * V_b, dim=1).clamp(-1.0, 1.0)
                    else: l_opt = criterion_smooth(pred_angles, Y_b).mean(dim=1)

                    if w is not None: loss = (l_opt * w).mean()
                    else: loss = l_opt.mean()

                if amp and device_type == "cuda":
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    
                if ema_model is not None:
                    ema_model.update_parameters(model)
                
                loss_sums["total_opt"] += loss.item() * X_batch.size(0)
                nobs += X_batch.size(0)

            # --- VALIDATION ---
            else:
                with torch.no_grad():
                    pred_angles = model(X_batch)
                    
                    # Individual losses for tracking
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
                    
                    if w is not None: loss = (l_opt * w).mean()
                    else: loss = l_opt.mean()
                    
                    # --- COLLECT DATA FOR ROOT FILE ---
                    opening_angle = torch.acos(cos_sim) * (180.0 / np.pi)
                    
                    # Append batch data to lists
                    p_np = pred_angles.cpu().numpy()
                    t_np = Y_b.cpu().numpy()
                    xyz_tru_np = XYZ_tru_b.cpu().numpy()
                    xyz_vtx_np = XYZ_vtx_b.cpu().numpy()
                    
                    val_root_data["run_id"].append(RunNum_b.cpu().numpy())
                    val_root_data["event_id"].append(EventNum_b.cpu().numpy())
                    val_root_data["pred_theta"].append(p_np[:, 0])
                    val_root_data["pred_phi"].append(p_np[:, 1])
                    val_root_data["true_theta"].append(t_np[:, 0])
                    val_root_data["true_phi"].append(t_np[:, 1])
                    val_root_data["opening_angle"].append(opening_angle.cpu().numpy())
                    val_root_data["energy_truth"].append(E_b.cpu().numpy())
                    val_root_data["x_truth"].append(xyz_tru_np[:, 0])
                    val_root_data["y_truth"].append(xyz_tru_np[:, 1])
                    val_root_data["z_truth"].append(xyz_tru_np[:, 2])
                    val_root_data["x_vtx"].append(xyz_vtx_np[:, 0])
                    val_root_data["y_vtx"].append(xyz_vtx_np[:, 1])
                    val_root_data["z_vtx"].append(xyz_vtx_np[:, 2])

                    # --- Worst Case Tracking (10 worst cases) ---
                    batch_errs_np = l_opt.cpu().numpy()
                    worst_idx = np.argsort(batch_errs_np)[-5:]
                    for idx in worst_idx:
                        worst_events_buffer.append((
                            batch_errs_np[idx], 
                            Raw_N_b[idx].cpu().numpy(),
                            Raw_T_b[idx].cpu().numpy(),
                            pred_angles[idx].cpu().numpy(), 
                            Y_b[idx].cpu().numpy(),
                            xyz_tru_np[idx],
                            xyz_vtx_np[idx],
                            E_b[idx].cpu().numpy(),
                            RunNum_b[idx].cpu().numpy(),
                            EventNum_b[idx].cpu().numpy()
                        ))
                    worst_events_buffer.sort(key=lambda x: x[0], reverse=True)
                    worst_events_buffer = worst_events_buffer[:10]

                    loss_sums["total_opt"] += loss.item() * X_batch.size(0)
                    loss_sums["smooth_l1"] += l_smooth.sum().item()
                    loss_sums["l1"]        += l_l1.sum().item()
                    loss_sums["mse"]       += l_mse.sum().item()
                    loss_sums["cos"]       += l_cos.sum().item()
                    nobs += X_batch.size(0)
                    
            t_batch = time.time() - t_end
            current_bs = X_batch.size(0)
            if t_batch > 0:
                epoch_throughput.append(current_bs / t_batch)
                epoch_data_time.append(t_data)
                epoch_batch_time.append(t_batch)
            t_end = time.time()
        # ========================== END OF BATCH LOOP ==========================

        torch.cuda.empty_cache()
    # ========================== END OF CHUNK LOOP ==========================

    if train and scheduler is not None:
        scheduler.step()

    metrics = {k: v / max(1, nobs) for k, v in loss_sums.items()}
    
    if len(epoch_throughput) > 0:
        metrics["system/throughput_events_per_sec"] = np.mean(epoch_throughput)
        metrics["system/avg_data_load_sec"]         = np.mean(epoch_data_time)
        metrics["system/avg_batch_process_sec"]     = np.mean(epoch_batch_time)
        metrics["system/compute_efficiency"]        = 1.0 - (metrics["system/avg_data_load_sec"] / metrics["system/avg_batch_process_sec"])
    
    extra_info = {}
    if not train:
        for k, v_list in val_root_data.items():
            if v_list:
                val_root_data[k] = np.concatenate(v_list, axis=0)
            else:
                val_root_data[k] = np.array([])

        pred_np = np.stack([val_root_data["pred_theta"], val_root_data["pred_phi"]], axis=1)
        true_np = np.stack([val_root_data["true_theta"], val_root_data["true_phi"]], axis=1)
        
        val_stats = eval_stats(pred_np, true_np, print_out=False)
        res_68, psi_deg = eval_resolution(pred_np, true_np)
        val_stats["val_resolution_deg"] = res_68
        
        extra_info["worst_events"] = worst_events_buffer
        extra_info["root_data"] = val_root_data
        
        if val_inputs_npho:
            extra_info["input_dist"] = {
                "npho": np.concatenate(val_inputs_npho),
                "time": np.concatenate(val_inputs_time)
            }
        
        return metrics, pred_np, true_np, extra_info, val_stats
        
    return metrics, None, None, None, {}