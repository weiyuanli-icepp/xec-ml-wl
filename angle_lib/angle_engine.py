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
    NphoScale=1e5,
    NphoScale2=13,
    time_shift=-0.29,
    time_scale=2.32e6,
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
    scaler = torch.amp.GradScaler(device_type, enabled=(amp and device_type == "cuda"))

    loss_sums = {"total_opt": 0.0, "smooth_l1": 0.0, "l1": 0.0, "mse": 0.0, "cos": 0.0}
    nobs = 0
    
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

        # Time_norm = (Time - time_shift) / time_scale
        Time_norm = Time / time_scale - time_shift
        Npho_log = np.log1p(Npho / NphoScale).astype("float32")
        Npho_norm = Npho_log / NphoScale2
        X_stacked = np.stack([Npho_norm, Time_norm], axis=-1)

        ds = TensorDataset(
            torch.from_numpy(X_stacked),  # Stacked input features: log-scaled Npho and normalized Time
            torch.from_numpy(Y),          # True angles in degrees
            torch.from_numpy(V),          # Unit vector of true angles
            torch.from_numpy(Npho),       # Raw Npho
            torch.from_numpy(Time),       # Raw Time
            torch.from_numpy(RunNum),     # Run Number
            torch.from_numpy(EventNum),   # Event Number
            torch.from_numpy(XYZ_tru),    # XYZ Truth
            torch.from_numpy(XYZ_vtx),    # XYZ VTX
            torch.from_numpy(E_truth)     # Energy Truth (in GeV)
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=train, drop_last=False)

        for i_batch, (Npho_b, Y_b, V_b, Raw_N_b, Raw_T_b, RunNum_b, EventNum_b, XYZ_tru_b, XYZ_vtx_b, E_b) in enumerate(loader):
            Npho_b = Npho_b.to(device)
            Y_b    = Y_b.to(device)
            V_b    = V_b.to(device)
            
            # Reweighting (Unchanged)
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
                    pred_angles = model(Npho_b)
                    
                    # (Simplified Loss selection for brevity - same as before)
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
                
                loss_sums["total_opt"] += loss.item() * Npho_b.size(0)
                nobs += Npho_b.size(0)

            # --- VALIDATION ---
            else:
                with torch.no_grad():
                    pred_angles = model(Npho_b)
                    
                    # Calculate individual losses for tracking
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
                    # Opening angle in degrees: acos(cos_sim) * 180/pi
                    opening_angle = torch.acos(cos_sim) * (180.0 / np.pi)
                    
                    # Append batch data to lists
                    # Move to CPU numpy
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

                    # --- Worst Case Tracking ---
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
                            E_b[idx].cpu().numpy()
                        ))
                    worst_events_buffer.sort(key=lambda x: x[0], reverse=True)
                    worst_events_buffer = worst_events_buffer[:10]  # Keep only top 10 worst

                    loss_sums["total_opt"] += loss.item() * Npho_b.size(0)
                    loss_sums["smooth_l1"] += l_smooth.sum().item()
                    loss_sums["l1"]        += l_l1.sum().item()
                    loss_sums["mse"]       += l_mse.sum().item()
                    loss_sums["cos"]       += l_cos.sum().item()
                    nobs += Npho_b.size(0)

        torch.cuda.empty_cache()

    if train and scheduler is not None:
        scheduler.step()

    metrics = {k: v / max(1, nobs) for k, v in loss_sums.items()}
    
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