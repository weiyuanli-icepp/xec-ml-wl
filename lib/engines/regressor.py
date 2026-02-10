"""
Training engine for the XEC regressor model.

Provides the main training loop with support for:
- Multi-task learning (angle, energy, timing, position)
- Sample reweighting
- EMA model updates
- Gradient accumulation
- AMP (automatic mixed precision)
"""

import math
import heapq
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from contextlib import nullcontext
from torch.utils.data import TensorDataset, DataLoader
import time
import warnings

from ..utils import iterate_chunks, angles_deg_to_unit_vec, get_pointwise_loss_fn, SimpleProfiler
from ..metrics import eval_stats, eval_resolution
from ..tasks import get_task_handlers
from ..tasks.angle import AngleTaskHandler
from ..tasks.position import PositionTaskHandler


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
    no_sync_ctx=None,
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

    loss_sums = {"total_opt": 0.0, "smooth_l1": 0.0, "l1": 0.0, "mse": 0.0, "cos": 0.0, "cos_pos": 0.0}
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
        "pos_angle": [],     # Angle between pred and true position vectors
        # Legacy position fields
        "x_truth": [], "y_truth": [], "z_truth": [],
        "x_vtx": [],   "y_vtx": [],   "z_vtx": []
    }

    # Use heapq for efficient worst-case tracking (min-heap, so we store negative loss)
    worst_events_heap = []  # (negative_loss, event_dict)
    MAX_WORST_EVENTS = 10

    # Initialize profiler
    profiler = SimpleProfiler(enabled=profile, sync_cuda=True)

    # Gradient accumulation
    grad_accum_steps = max(int(grad_accum_steps), 1)
    accum_step = 0
    max_grad_norm = 0.0  # Track max gradient norm for monitoring
    if train:
        optimizer.zero_grad(set_to_none=True)

    # Get task handlers for validation
    task_handlers = get_task_handlers()  # Get all handlers, filter by active tasks later

    # Pre-cache loss functions and settings for each task (avoid repeated lookups in loop)
    task_loss_cache = {}
    valid_loss_names = {"smooth_l1", "huber", "l1", "mse", "l2",
                        "relative_l1", "relative_smooth_l1", "relative_mse", "relative_l2"}
    if task_weights:
        for task, cfg in task_weights.items():
            if isinstance(cfg, dict):
                loss_fn_name = cfg.get("loss_fn", loss_type)
                task_weight = cfg.get("weight", 1.0)
                use_log_transform = cfg.get("log_transform", False)
            else:
                loss_fn_name = loss_type
                task_weight = cfg
                use_log_transform = False

            if loss_fn_name not in valid_loss_names:
                warnings.warn(
                    f"Unknown loss function '{loss_fn_name}' for task '{task}'. "
                    f"Falling back to smooth_l1. Supported: {valid_loss_names}.",
                    UserWarning
                )
                loss_fn_name = "smooth_l1"

            task_loss_cache[task] = {
                "loss_fn": get_pointwise_loss_fn(loss_fn_name),
                "weight": task_weight,
                "log_transform": use_log_transform,
            }

    # Pre-cache legacy reweighting tensors on GPU (avoid recreation every batch)
    legacy_rw_cache = {}
    if reweight_mode != "none" and reweighter is None:
        if reweight_mode == "theta" and edges_theta is not None:
            legacy_rw_cache["edges"] = torch.as_tensor(edges_theta, device=device)
            legacy_rw_cache["weights"] = torch.as_tensor(weights_theta, device=device, dtype=torch.float32)
        elif reweight_mode == "phi" and edges_phi is not None:
            legacy_rw_cache["edges"] = torch.as_tensor(edges_phi, device=device)
            legacy_rw_cache["weights"] = torch.as_tensor(weights_phi, device=device, dtype=torch.float32)
        elif reweight_mode == "theta_phi" and edges2_theta is not None:
            legacy_rw_cache["edges_th"] = torch.as_tensor(edges2_theta, device=device)
            legacy_rw_cache["edges_ph"] = torch.as_tensor(edges2_phi, device=device)
            legacy_rw_cache["weights"] = torch.as_tensor(weights_2d, device=device, dtype=torch.float32)

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
        # Legacy reweighting support (using pre-cached GPU tensors)
        elif legacy_rw_cache and "angle" in target_dict:
            angles = target_dict["angle"]  # Already on GPU
            if reweight_mode == "theta":
                bin_id = torch.bucketize(angles[:, 0], legacy_rw_cache["edges"]) - 1
                bin_id = bin_id.clamp(0, legacy_rw_cache["weights"].shape[0] - 1)
                w = legacy_rw_cache["weights"][bin_id]
            elif reweight_mode == "phi":
                bin_id = torch.bucketize(angles[:, 1], legacy_rw_cache["edges"]) - 1
                bin_id = bin_id.clamp(0, legacy_rw_cache["weights"].shape[0] - 1)
                w = legacy_rw_cache["weights"][bin_id]
            elif reweight_mode == "theta_phi":
                id_th = (torch.bucketize(angles[:, 0], legacy_rw_cache["edges_th"]) - 1).clamp(0, legacy_rw_cache["weights"].shape[0] - 1)
                id_ph = (torch.bucketize(angles[:, 1], legacy_rw_cache["edges_ph"]) - 1).clamp(0, legacy_rw_cache["weights"].shape[1] - 1)
                w = legacy_rw_cache["weights"][id_th, id_ph]

        if train:
            profiler.start("forward")
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp):

                # Forward Pass
                preds = model(X_batch)
                if not isinstance(preds, dict):
                    preds = {"angle": preds}  # Compatibility for single-head models

                # Multi-task loss computation using cached loss functions
                total_loss = 0.0
                for task, pred in preds.items():
                    if task not in target_dict:
                        continue

                    truth = target_dict[task]
                    if truth.ndim == 1 and pred.ndim == 2:
                        truth = truth.unsqueeze(-1)

                    # Use cached loss function and settings
                    if task in task_loss_cache:
                        cache = task_loss_cache[task]
                        loss_fn = cache["loss_fn"]
                        task_weight = cache["weight"]
                        use_log_transform = cache["log_transform"]
                    else:
                        # Fallback for uncached tasks
                        loss_fn = get_pointwise_loss_fn(loss_type)
                        task_weight = 1.0
                        use_log_transform = False

                    # Apply log transform if configured (for energy/timing)
                    if use_log_transform:
                        truth = torch.log(truth.clamp(min=1e-6))

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

            # Check for NaN/Inf loss - skip batch if detected
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"[WARN] Skipping batch due to non-finite loss: {loss_val}")
                optimizer.zero_grad(set_to_none=True)
                continue

            # Backward with gradient accumulation
            loss_sums["total_opt"] += loss_val * X_batch.size(0)
            nobs += X_batch.size(0)

            profiler.start("backward")
            loss = loss / grad_accum_steps
            # Use no_sync on intermediate accumulation steps to skip AllReduce
            is_sync_step = (accum_step + 1) % grad_accum_steps == 0
            sync_ctx = nullcontext() if (no_sync_ctx is None or is_sync_step) else no_sync_ctx()
            with sync_ctx:
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
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        max_grad_norm = max(max_grad_norm, grad_norm.item())
                    # Sync loss_scaler gradients across DDP ranks (not part of DDP model)
                    if loss_scaler is not None and dist.is_initialized():
                        for p in loss_scaler.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        max_grad_norm = max(max_grad_norm, grad_norm.item())
                    # Sync loss_scaler gradients across DDP ranks (not part of DDP model)
                    if loss_scaler is not None and dist.is_initialized():
                        for p in loss_scaler.parameters():
                            if p.grad is not None:
                                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
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

                # Convert predictions from log space to linear space for tasks with log_transform
                if task_weights:
                    for task, cfg in task_weights.items():
                        if isinstance(cfg, dict) and cfg.get("log_transform", False) and task in preds:
                            preds[task] = torch.exp(preds[task])
            profiler.stop()

            profiler.start("loss_compute")
            with torch.no_grad():
                # Determine active tasks from predictions
                active_tasks = list(preds.keys())

                # Initialize batch losses
                batch_total_loss = 0.0
                batch_size = X_batch.size(0)
                sample_total_loss = torch.zeros(batch_size, device=device)  # Per-sample loss for worst case tracking

                # Process each task using handlers
                for handler in task_handlers:
                    if not handler.is_active(preds, target_dict):
                        continue

                    # Compute losses using handler
                    losses = handler.compute_val_loss(
                        preds, target_dict,
                        criterion_smooth, criterion_l1, criterion_mse
                    )

                    # Accumulate losses
                    loss_sums["smooth_l1"] += losses["smooth_l1"].sum().item()
                    loss_sums["l1"] += losses["l1"].sum().item()
                    loss_sums["mse"] += losses["mse"].sum().item()
                    batch_total_loss += losses["batch_loss"]
                    sample_total_loss += losses["smooth_l1"]

                    # Collect predictions using handler
                    handler.collect_predictions(preds, target_dict, val_root_data)

                    # Task-specific additional processing
                    if isinstance(handler, AngleTaskHandler):
                        # Compute cosine loss for angle task
                        cos_losses = handler.compute_cosine_loss(preds, target_dict)
                        if "cos" in cos_losses:
                            loss_sums["cos"] += cos_losses["cos"].sum().item()
                        if "opening_angle" in cos_losses:
                            val_root_data["opening_angle"].append(
                                cos_losses["opening_angle"].cpu().numpy()
                            )

                    if isinstance(handler, PositionTaskHandler):
                        # Collect residuals for position resolution
                        handler.collect_residuals(preds, target_dict, loss_sums)
                        # Compute cosine loss for position task
                        cos_losses = handler.compute_cosine_loss(preds, target_dict)
                        if "cos_pos" in cos_losses:
                            loss_sums["cos_pos"] += cos_losses["cos_pos"].sum().item()
                        if "pos_angle" in cos_losses:
                            val_root_data["pos_angle"].append(
                                cos_losses["pos_angle"].cpu().numpy()
                            )

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
                    # Legacy position fields
                    val_root_data["x_truth"].append(v_uvw[:, 0])
                    val_root_data["y_truth"].append(v_uvw[:, 1])
                    val_root_data["z_truth"].append(v_uvw[:, 2])
                    # Position task fields (for resolution profiles)
                    val_root_data["true_u"].append(v_uvw[:, 0])
                    val_root_data["true_v"].append(v_uvw[:, 1])
                    val_root_data["true_w"].append(v_uvw[:, 2])

                # Collect predicted uvwFI values
                if "uvwFI" in preds:
                    p_uvw = preds["uvwFI"].cpu().numpy()
                    val_root_data["pred_u"].append(p_uvw[:, 0])
                    val_root_data["pred_v"].append(p_uvw[:, 1])
                    val_root_data["pred_w"].append(p_uvw[:, 2])

                v_vtx = None
                if "xyzVTX" in target_dict:
                    v_vtx = target_dict["xyzVTX"].cpu().numpy()
                    val_root_data["x_vtx"].append(v_vtx[:, 0])
                    val_root_data["y_vtx"].append(v_vtx[:, 1])
                    val_root_data["z_vtx"].append(v_vtx[:, 2])

                # === Worst Case Tracking (based on total loss across all tasks) ===
                # Use GPU topk + heapq for efficient tracking without full CPU transfers
                if active_tasks:
                    # Find top-5 worst in this batch using GPU
                    k = min(5, batch_size)
                    top_losses, top_indices = torch.topk(sample_total_loss, k)

                    # Check if any of these are worse than our current worst
                    min_heap_loss = -worst_events_heap[0][0] if worst_events_heap else float('-inf')

                    for i in range(k):
                        loss_val = top_losses[i].item()
                        # Only process if this could be in our top-N
                        if len(worst_events_heap) < MAX_WORST_EVENTS or loss_val > min_heap_loss:
                            idx = top_indices[i].item()
                            # Now transfer only this sample's data to CPU
                            worst_event = {
                                "total_loss": loss_val,
                                "input_npho": X_batch[idx, :, 0].cpu().numpy(),
                                "input_time": X_batch[idx, :, 1].cpu().numpy(),
                                "run": target_dict["run"][idx].item() if "run" in target_dict else None,
                                "event": target_dict["event"][idx].item() if "event" in target_dict else None,
                                "energy_truth": target_dict["energy"][idx].item() if "energy" in target_dict else None,
                                "uvw_truth": v_uvw[idx] if v_uvw is not None else None,
                                "xyz_truth": target_dict["xyzTruth"][idx].cpu().numpy() if "xyzTruth" in target_dict else None,
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

                            # Use heapq (min-heap with negative loss for max behavior)
                            if len(worst_events_heap) < MAX_WORST_EVENTS:
                                heapq.heappush(worst_events_heap, (-loss_val, worst_event))
                            elif loss_val > -worst_events_heap[0][0]:
                                heapq.heapreplace(worst_events_heap, (-loss_val, worst_event))

                            # Update threshold for early exit
                            min_heap_loss = -worst_events_heap[0][0]

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
        # Print dataset I/O breakdown if available
        # Note: only accurate when num_workers=0 for DataLoader
        if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'get_profile_report'):
            print(loader.dataset.get_profile_report())

    # Final optimizer step if gradients remain from incomplete accumulation
    if train and accum_step % grad_accum_steps != 0:
        if scaler is not None:
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if loss_scaler is not None and dist.is_initialized():
                for p in loss_scaler.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if loss_scaler is not None and dist.is_initialized():
                for p in loss_scaler.parameters():
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
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

    # Log max gradient norm (helps diagnose gradient explosion)
    if train and max_grad_norm > 0:
        metrics["system/grad_norm_max"] = max_grad_norm

    extra_info = {}
    val_stats = {}

    if not train:
        # Concatenate collected data
        for k, v_list in val_root_data.items():
            if v_list:
                val_root_data[k] = np.concatenate(v_list, axis=0)
            else:
                val_root_data[k] = np.array([])

        # Compute metrics using task handlers
        for handler in task_handlers:
            handler_metrics = handler.compute_metrics(val_root_data, loss_sums)
            val_stats.update(handler_metrics)

        # Prepare angle predictions for return (for backward compatibility)
        angle_pred_np = None
        angle_true_np = None
        if val_root_data["pred_theta"].size > 0 and val_root_data["pred_phi"].size > 0:
            angle_pred_np = np.stack([val_root_data["pred_theta"], val_root_data["pred_phi"]], axis=1)
            angle_true_np = np.stack([val_root_data["true_theta"], val_root_data["true_phi"]], axis=1)

        # Convert heap to sorted list (highest loss first)
        worst_events_list = [event for (neg_loss, event) in sorted(worst_events_heap, reverse=True)]
        extra_info["worst_events"] = worst_events_list
        extra_info["root_data"] = val_root_data

        return metrics, angle_pred_np, angle_true_np, extra_info, val_stats

    return metrics, None, None, None, {}
