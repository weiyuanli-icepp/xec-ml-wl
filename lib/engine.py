import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings

from .utils import iterate_chunks, angles_deg_to_unit_vec, get_pointwise_loss_fn, SimpleProfiler
from .metrics import eval_stats, eval_resolution


def run_epoch_stream(
    model, optimizer, device, loader,
    scaler=None,
    train=True,
    amp=True,
    task_weights=None,  # dict: {task: {"loss_fn": str, "weight": float}} or {task: float}
    loss_scaler=None,   # for automatic loss scaling
    reweighter=None,    # SampleReweighter instance (new API)
    # Legacy reweighting args (deprecated, use reweighter instead)
    reweight_mode="none",
    edges_theta=None,
    weights_theta=None,
    edges_phi=None,
    weights_phi=None,
    edges2_theta=None,
    edges2_phi=None,
    weights_2d=None,
    channel_dropout_rate=0.1,
    loss_type="smooth_l1",
    loss_beta=1.0,
    scheduler=None,
    ema_model=None,
    grad_clip=1.0,
    grad_accum_steps=1,
    profile=False,
):
    # Warn about deprecated legacy reweighting API
    if reweight_mode != "none" and reweighter is None:
        warnings.warn(
            "Legacy reweighting parameters (reweight_mode, edges_*, weights_*) are deprecated. "
            "Use the 'reweighter' parameter with a SampleReweighter instance instead.",
            DeprecationWarning,
            stacklevel=2
        )

    model.train(train)
    criterion_smooth = nn.SmoothL1Loss(reduction="none", beta=loss_beta)
    criterion_l1 = nn.L1Loss(reduction="none")
    criterion_mse = nn.MSELoss(reduction="none")

    loss_sums = {"total_opt": 0.0, "smooth_l1": 0.0, "l1": 0.0, "mse": 0.0, "cos": 0.0}
    nobs = 0
    
    epoch_throughput = []
    epoch_data_time  = []
    epoch_batch_time = []
    t_end = time.time()
    
    # Collections for the Final ROOT file
    val_root_data = {
        "run_id": [],        "event_id": [],
        # Angle task
        "pred_theta": [],    "pred_phi": [],
        "true_theta": [],    "true_phi": [],
        "opening_angle": [],
        # Energy task
        "pred_energy": [],   "true_energy": [],
        # Timing task
        "pred_timing": [],   "true_timing": [],
        # Position task (uvwFI)
        "pred_u": [],        "pred_v": [],        "pred_w": [],
        "true_u": [],        "true_v": [],        "true_w": [],
        # Legacy position fields
        "x_truth": [], "y_truth": [], "z_truth": [],
        "x_vtx": [],   "y_vtx": [],   "z_vtx": []
    }

    worst_events_buffer = []

    # Initialize profiler
    profiler = SimpleProfiler(enabled=profile, sync_cuda=True)

    # Gradient accumulation
    grad_accum_steps = max(int(grad_accum_steps), 1)
    accum_step = 0
    if train:
        optimizer.zero_grad(set_to_none=True)

    # ========================== START OF BATCH LOOP ==========================
    for i_batch, (X_batch, target_dict) in enumerate(loader):
        t_data      = time.time() - t_end
        profiler.stop()  # data_load

        profiler.start("gpu_transfer")
        X_batch     = X_batch.to(device, non_blocking=True)
        target_dict = {k: v.to(device, non_blocking=True) for k, v in target_dict.items()}
        profiler.stop()

        # Channel Dropout
        if train and channel_dropout_rate > 0.0:
            dropout_mask = (torch.rand(X_batch.shape[0], X_batch.shape[1], 1, device=device) < channel_dropout_rate)
            X_batch = X_batch * (~dropout_mask)
        
        # Reweighting (new API with SampleReweighter)
        w = None
        if reweighter is not None:
            w = reweighter.compute_weights(target_dict, device)
        # Legacy reweighting support
        elif reweight_mode != "none" and "angle" in target_dict:
            angles = target_dict["angle"].cpu().numpy()
            if reweight_mode == "theta" and (edges_theta is not None):
                bin_id = np.clip(np.digitize(angles[:, 0], edges_theta) - 1, 0, len(weights_theta) - 1)
                w = torch.from_numpy(weights_theta[bin_id]).to(device)
            elif reweight_mode == "phi" and (edges_phi is not None):
                bin_id = np.clip(np.digitize(angles[:, 1], edges_phi) - 1, 0, len(weights_phi) - 1)
                w = torch.from_numpy(weights_phi[bin_id]).to(device)
            elif reweight_mode == "theta_phi" and (edges2_theta is not None):
                th = angles[:, 0]
                ph = angles[:, 1]
                id_th = np.clip(np.digitize(th, edges2_theta) - 1, 0, len(edges2_theta) - 2)
                id_ph = np.clip(np.digitize(ph, edges2_phi)   - 1, 0, len(edges2_phi)   - 2)
                w_np = weights_2d[id_th, id_ph].astype("float32")
                w = torch.from_numpy(w_np).to(device)
        
        if train:
            profiler.start("forward")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp):

                # Forward Pass
                preds = model(X_batch)
                if not isinstance(preds, dict):
                    preds = {"angle": preds} # Compatibility for single-head models
                    
                # Multi-task loss computation
                total_loss = 0.0
                for task, pred in preds.items():
                    if task not in target_dict:
                        continue

                    truth = target_dict[task]
                    if truth.ndim == 1 and pred.ndim == 2:
                        truth = truth.unsqueeze(-1)

                    # Get loss function from task_weights config
                    # task_weights can be: {task: {"loss_fn": str, "weight": float}} or {task: float}
                    loss_fn_name = loss_type  # default
                    task_weight = 1.0
                    if task_weights and task in task_weights:
                        cfg = task_weights[task]
                        if isinstance(cfg, dict):
                            loss_fn_name = cfg.get("loss_fn", loss_type)
                            task_weight = cfg.get("weight", 1.0)
                        else:
                            task_weight = cfg
                    valid_names = {"smooth_l1", "huber", "l1", "mse"}
                    if loss_fn_name not in valid_names:
                        warnings.warn(
                            f"Unknown loss function '{loss_fn_name}' for task '{task}'. "
                            f"Falling back to smooth_l1. Supported: smooth_l1, huber, l1, mse.",
                            UserWarning
                        )
                        loss_fn_name = "smooth_l1"

                    loss_fn = get_pointwise_loss_fn(loss_fn_name)
                    l_task = loss_fn(pred, truth).mean(dim=-1)

                    # Apply sample weights if available
                    if w is not None:
                        l_task = l_task * w

                    # Aggregate with loss balancing
                    if loss_scaler is not None:
                        total_loss += loss_scaler(l_task.mean(), task)
                    else:
                        total_loss += l_task.mean() * task_weight

                loss = total_loss

            profiler.stop()  # forward

            # Backward with gradient accumulation
            loss_sums["total_opt"] += loss.item() * X_batch.size(0)
            nobs += X_batch.size(0)

            profiler.start("backward")
            loss = loss / grad_accum_steps
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            profiler.stop()

            accum_step += 1

            # Optimizer step every grad_accum_steps
            if accum_step % grad_accum_steps == 0:
                profiler.start("optimizer")
                if scaler is not None:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

                if ema_model is not None:
                    ema_model.update_parameters(model)

                optimizer.zero_grad(set_to_none=True)
                profiler.stop()

        # --- VALIDATION ---
        else:
            profiler.start("forward")
            with torch.no_grad():
                preds = model(X_batch)
                if not isinstance(preds, dict):
                    preds = {"angle": preds}
            profiler.stop()

            profiler.start("loss_compute")
            with torch.no_grad():
                # Determine active tasks from predictions
                active_tasks = list(preds.keys())

                # Initialize batch losses
                batch_total_loss = 0.0
                batch_size = X_batch.size(0)
                sample_total_loss = torch.zeros(batch_size, device=device)  # Per-sample loss for worst case tracking

                # === ANGLE TASK ===
                if "angle" in preds and "angle" in target_dict:
                    p_angle = preds["angle"]
                    t_angle = target_dict["angle"]
                    t_vec = target_dict.get("emiVec")

                    l_smooth = criterion_smooth(p_angle, t_angle).mean(dim=-1)
                    l_l1 = criterion_l1(p_angle, t_angle).mean(dim=-1)
                    l_mse = criterion_mse(p_angle, t_angle).mean(dim=-1)

                    loss_sums["smooth_l1"] += l_smooth.sum().item()
                    loss_sums["l1"] += l_l1.sum().item()
                    loss_sums["mse"] += l_mse.sum().item()
                    batch_total_loss += l_smooth.mean()
                    sample_total_loss += l_smooth  # Accumulate per-sample loss

                    # Cosine similarity (angle-specific)
                    if t_vec is not None:
                        v_pred = angles_deg_to_unit_vec(p_angle)
                        cos_sim = torch.sum(v_pred * t_vec, dim=1).clamp(-1.0, 1.0)
                        l_cos = 1.0 - cos_sim
                        opening_angle = torch.acos(cos_sim) * (180.0 / np.pi)
                        loss_sums["cos"] += l_cos.sum().item()

                        val_root_data["opening_angle"].append(opening_angle.cpu().numpy())

                    val_root_data["pred_theta"].append(p_angle[:, 0].cpu().numpy())
                    val_root_data["pred_phi"].append(p_angle[:, 1].cpu().numpy())
                    val_root_data["true_theta"].append(t_angle[:, 0].cpu().numpy())
                    val_root_data["true_phi"].append(t_angle[:, 1].cpu().numpy())

                # === POSITION TASK (uvwFI) ===
                if "uvwFI" in preds and "uvwFI" in target_dict:
                    p_uvw = preds["uvwFI"]
                    t_uvw = target_dict["uvwFI"]

                    l_pos_smooth = criterion_smooth(p_uvw, t_uvw).mean(dim=-1)
                    l_pos_l1 = criterion_l1(p_uvw, t_uvw).mean(dim=-1)
                    l_pos_mse = criterion_mse(p_uvw, t_uvw).mean(dim=-1)

                    loss_sums["smooth_l1"] += l_pos_smooth.sum().item()
                    loss_sums["l1"] += l_pos_l1.sum().item()
                    loss_sums["mse"] += l_pos_mse.sum().item()
                    batch_total_loss += l_pos_smooth.mean()
                    sample_total_loss += l_pos_smooth  # Accumulate per-sample loss

                    # Collect predictions for artifacts
                    val_root_data["pred_u"].append(p_uvw[:, 0].cpu().numpy())
                    val_root_data["pred_v"].append(p_uvw[:, 1].cpu().numpy())
                    val_root_data["pred_w"].append(p_uvw[:, 2].cpu().numpy())
                    val_root_data["true_u"].append(t_uvw[:, 0].cpu().numpy())
                    val_root_data["true_v"].append(t_uvw[:, 1].cpu().numpy())
                    val_root_data["true_w"].append(t_uvw[:, 2].cpu().numpy())

                    # Per-axis resolution
                    residual = p_uvw - t_uvw
                    if "uvw_u_res" not in loss_sums:
                        loss_sums["uvw_u_res"] = []
                        loss_sums["uvw_v_res"] = []
                        loss_sums["uvw_w_res"] = []
                        loss_sums["uvw_dist"] = []
                    loss_sums["uvw_u_res"].append(residual[:, 0].cpu().numpy())
                    loss_sums["uvw_v_res"].append(residual[:, 1].cpu().numpy())
                    loss_sums["uvw_w_res"].append(residual[:, 2].cpu().numpy())
                    dist = torch.norm(residual, dim=1)
                    loss_sums["uvw_dist"].append(dist.cpu().numpy())

                # === ENERGY TASK ===
                if "energy" in preds and "energy" in target_dict:
                    p_energy = preds["energy"]
                    t_energy = target_dict["energy"]
                    if t_energy.ndim == 1:
                        t_energy = t_energy.unsqueeze(-1)

                    l_e_smooth = criterion_smooth(p_energy, t_energy).mean(dim=-1)
                    l_e_l1 = criterion_l1(p_energy, t_energy).mean(dim=-1)
                    l_e_mse = criterion_mse(p_energy, t_energy).mean(dim=-1)

                    loss_sums["smooth_l1"] += l_e_smooth.sum().item()
                    loss_sums["l1"] += l_e_l1.sum().item()
                    loss_sums["mse"] += l_e_mse.sum().item()
                    batch_total_loss += l_e_smooth.mean()
                    sample_total_loss += l_e_smooth  # Accumulate per-sample loss

                    # Collect predictions for artifacts
                    val_root_data["pred_energy"].append(p_energy.squeeze(-1).cpu().numpy())
                    val_root_data["true_energy"].append(t_energy.squeeze(-1).cpu().numpy())

                # === TIMING TASK ===
                if "timing" in preds and "timing" in target_dict:
                    p_timing = preds["timing"]
                    t_timing = target_dict["timing"]
                    if t_timing.ndim == 1:
                        t_timing = t_timing.unsqueeze(-1)

                    l_t_smooth = criterion_smooth(p_timing, t_timing).mean(dim=-1)
                    l_t_l1 = criterion_l1(p_timing, t_timing).mean(dim=-1)
                    l_t_mse = criterion_mse(p_timing, t_timing).mean(dim=-1)

                    loss_sums["smooth_l1"] += l_t_smooth.sum().item()
                    loss_sums["l1"] += l_t_l1.sum().item()
                    loss_sums["mse"] += l_t_mse.sum().item()
                    batch_total_loss += l_t_smooth.mean()
                    sample_total_loss += l_t_smooth  # Accumulate per-sample loss

                    # Collect predictions for artifacts
                    val_root_data["pred_timing"].append(p_timing.squeeze(-1).cpu().numpy())
                    val_root_data["true_timing"].append(t_timing.squeeze(-1).cpu().numpy())

                # === Common data collection ===
                # Check for required keys and warn if missing
                required_keys = ["run", "event", "energy", "uvwFI", "xyzVTX"]
                missing_keys = [k for k in required_keys if k not in target_dict]
                if missing_keys and i_batch == 0:
                    warnings.warn(
                        f"Missing keys in target_dict for validation data collection: {missing_keys}. "
                        f"Some validation outputs may be incomplete.",
                        UserWarning
                    )

                if "run" in target_dict:
                    val_root_data["run_id"].append(target_dict["run"].cpu().numpy())
                if "event" in target_dict:
                    val_root_data["event_id"].append(target_dict["event"].cpu().numpy())

                v_uvw = None
                if "uvwFI" in target_dict:
                    v_uvw = target_dict["uvwFI"].cpu().numpy()
                    val_root_data["x_truth"].append(v_uvw[:, 0])
                    val_root_data["y_truth"].append(v_uvw[:, 1])
                    val_root_data["z_truth"].append(v_uvw[:, 2])

                v_vtx = None
                if "xyzVTX" in target_dict:
                    v_vtx = target_dict["xyzVTX"].cpu().numpy()
                    val_root_data["x_vtx"].append(v_vtx[:, 0])
                    val_root_data["y_vtx"].append(v_vtx[:, 1])
                    val_root_data["z_vtx"].append(v_vtx[:, 2])

                # === Worst Case Tracking (based on total loss across all tasks) ===
                if active_tasks:  # Track worst cases if any task is active
                    batch_errs_np = sample_total_loss.cpu().numpy()

                    # Get metadata
                    xyz_truth_np = target_dict["xyzTruth"].cpu().numpy() if "xyzTruth" in target_dict else None
                    energy_np = target_dict["energy"].cpu().numpy() if "energy" in target_dict else None
                    run_np = target_dict["run"].cpu().numpy() if "run" in target_dict else None
                    event_np = target_dict["event"].cpu().numpy() if "event" in target_dict else None

                    # Find worst 5 samples in this batch
                    worst_idx = np.argsort(batch_errs_np)[-5:]
                    for idx in worst_idx:
                        worst_event = {
                            "total_loss": batch_errs_np[idx],
                            "input_npho": X_batch[idx, :, 0].cpu().numpy(),
                            "input_time": X_batch[idx, :, 1].cpu().numpy(),
                            "run": run_np[idx] if run_np is not None else None,
                            "event": event_np[idx] if event_np is not None else None,
                            "energy_truth": energy_np[idx] if energy_np is not None else None,
                            "uvw_truth": v_uvw[idx] if v_uvw is not None else None,
                            "xyz_truth": xyz_truth_np[idx] if xyz_truth_np is not None else None,
                            "vtx": v_vtx[idx] if v_vtx is not None else None,
                        }
                        # Add predictions and truths for active tasks
                        if "angle" in preds:
                            worst_event["pred_angle"] = preds["angle"][idx].cpu().numpy()
                            worst_event["true_angle"] = target_dict["angle"][idx].cpu().numpy()
                        if "energy" in preds:
                            worst_event["pred_energy"] = preds["energy"][idx].cpu().numpy()
                        if "timing" in preds:
                            worst_event["pred_timing"] = preds["timing"][idx].cpu().numpy()
                        if "uvwFI" in preds:
                            worst_event["pred_uvwFI"] = preds["uvwFI"][idx].cpu().numpy()
                            worst_event["true_uvwFI"] = target_dict["uvwFI"][idx].cpu().numpy()

                        worst_events_buffer.append(worst_event)

                    # Keep only top 10 worst events
                    worst_events_buffer.sort(key=lambda x: x["total_loss"], reverse=True)
                    worst_events_buffer = worst_events_buffer[:10]

                loss_sums["total_opt"] += batch_total_loss.item() * X_batch.size(0)
                nobs += X_batch.size(0)

            profiler.stop()  # loss_compute

        t_batch = time.time() - t_end
        current_bs = X_batch.size(0)
        if t_batch > 0:
            epoch_throughput.append(current_bs / t_batch)
            epoch_data_time.append(t_data)
            epoch_batch_time.append(t_batch)
        t_end = time.time()

        profiler.start("data_load")  # For next iteration
    # ========================== END OF BATCH LOOP ==========================

    # Stop any pending timer and print report
    profiler.stop()
    if profile:
        print(profiler.report())

    # Final optimizer step if gradients remain from incomplete accumulation
    if train and accum_step % grad_accum_steps != 0:
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if ema_model is not None:
            ema_model.update_parameters(model)

        optimizer.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    # ========================== END OF CHUNK LOOP ==========================

    if train and scheduler is not None:
        scheduler.step()

    # Compute average metrics (skip list-based metrics)
    metrics = {}
    for k, v in loss_sums.items():
        if isinstance(v, list):
            continue  # Handle separately
        metrics[k] = v / max(1, nobs)

    if len(epoch_throughput) > 0:
        metrics["system/throughput_events_per_sec"] = np.mean(epoch_throughput)
        metrics["system/avg_data_load_sec"] = np.mean(epoch_data_time)
        metrics["system/avg_batch_process_sec"] = np.mean(epoch_batch_time)
        # Compute efficiency: ratio of compute time to total time, clamped to [0, 1]
        avg_batch = metrics["system/avg_batch_process_sec"]
        if avg_batch > 0:
            efficiency = 1.0 - (metrics["system/avg_data_load_sec"] / avg_batch)
            metrics["system/compute_efficiency"] = max(0.0, min(1.0, efficiency))
        else:
            metrics["system/compute_efficiency"] = 0.0

    extra_info = {}
    val_stats = {}

    if not train:
        # Concatenate collected data
        for k, v_list in val_root_data.items():
            if v_list:
                val_root_data[k] = np.concatenate(v_list, axis=0)
            else:
                val_root_data[k] = np.array([])

        # === Position task metrics ===
        if "uvw_u_res" in loss_sums and loss_sums["uvw_u_res"]:
            u_res = np.concatenate(loss_sums["uvw_u_res"])
            v_res = np.concatenate(loss_sums["uvw_v_res"])
            w_res = np.concatenate(loss_sums["uvw_w_res"])
            dist = np.concatenate(loss_sums["uvw_dist"])

            val_stats["uvw_u_res_68pct"] = np.percentile(np.abs(u_res), 68)
            val_stats["uvw_v_res_68pct"] = np.percentile(np.abs(v_res), 68)
            val_stats["uvw_w_res_68pct"] = np.percentile(np.abs(w_res), 68)
            val_stats["uvw_dist_68pct"] = np.percentile(dist, 68)

        # === Angle task metrics ===
        angle_pred_np = None
        angle_true_np = None
        if val_root_data["pred_theta"].size > 0 and val_root_data["pred_phi"].size > 0:
            angle_pred_np = np.stack([val_root_data["pred_theta"], val_root_data["pred_phi"]], axis=1)
            angle_true_np = np.stack([val_root_data["true_theta"], val_root_data["true_phi"]], axis=1)

            angle_stats = eval_stats(angle_pred_np, angle_true_np, print_out=False)
            val_stats.update(angle_stats)

            res_68, psi_deg = eval_resolution(angle_pred_np, angle_true_np)
            val_stats["angle_resolution_68pct"] = res_68

        # === Energy task metrics ===
        if val_root_data["pred_energy"].size > 0 and val_root_data["true_energy"].size > 0:
            energy_res = val_root_data["pred_energy"] - val_root_data["true_energy"]
            val_stats["energy_bias"] = np.mean(energy_res)
            val_stats["energy_res_68pct"] = np.percentile(np.abs(energy_res), 68)

        # === Timing task metrics ===
        if val_root_data["pred_timing"].size > 0 and val_root_data["true_timing"].size > 0:
            timing_res = val_root_data["pred_timing"] - val_root_data["true_timing"]
            val_stats["timing_bias"] = np.mean(timing_res)
            val_stats["timing_res_68pct"] = np.percentile(np.abs(timing_res), 68)

        extra_info["worst_events"] = worst_events_buffer
        extra_info["root_data"] = val_root_data

        return metrics, angle_pred_np, angle_true_np, extra_info, val_stats

    return metrics, None, None, None, {}
