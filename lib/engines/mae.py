import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from ..dataset import XECStreamingDataset
from ..utils import get_pointwise_loss_fn, SimpleProfiler
from ..geom_utils import build_outer_fine_grid_tensor, gather_face, gather_hex_nodes
from ..geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    OUTER_FINE_COARSE_SCALE, OUTER_FINE_CENTER_SCALE, OUTER_FINE_CENTER_START,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows,
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_VALUE, DEFAULT_NPHO_THRESHOLD
)


def run_epoch_mae(model, optimizer, device, root_files, tree_name,
                  batch_size=8192, step_size=4000,
                  amp=True,
                  npho_branch="npho", time_branch="relative_time",
                  npho_scale=DEFAULT_NPHO_SCALE, npho_scale2=DEFAULT_NPHO_SCALE2,
                  time_scale=DEFAULT_TIME_SCALE, time_shift=DEFAULT_TIME_SHIFT,
                  sentinel_value=DEFAULT_SENTINEL_VALUE,
                  channel_dropout_rate=0.1,
                  loss_fn="mse",
                  npho_weight=1.0,
                  time_weight=1.0,
                  auto_channel_weight=False,
                  grad_clip=1.0,
                  grad_accum_steps=1,
                  scaler=None,
                  dataloader_workers=0,
                  dataset_workers=8,
                  prefetch_factor=2,
                  npho_threshold=None,
                  use_npho_time_weight=True,
                  track_mae_rmse=True,
                  track_train_metrics=True,
                  profile=False,
                  log_invalid_npho=True):
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loss_func = get_pointwise_loss_fn(loss_fn)
    log_vars = getattr(model, "channel_log_vars", None) if auto_channel_weight else None

    # Detect predict_channels from model (default to ["npho", "time"] for legacy)
    predict_channels = getattr(model, 'predict_channels', ['npho', 'time'])
    predict_time = 'time' in predict_channels
    # Map prediction channel indices to input channel indices
    INPUT_CH_MAP = {"npho": 0, "time": 1}
    pred_npho_idx = predict_channels.index("npho") if "npho" in predict_channels else None
    pred_time_idx = predict_channels.index("time") if "time" in predict_channels else None

    # Conditional time loss threshold
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD

    # Convert to normalized space for stratified masking and threshold checks
    npho_threshold_norm = np.log1p(npho_threshold / npho_scale) / npho_scale2

    # Per-face total losses
    face_loss_sums = {
        "inner": 0.0, "us": 0.0, "ds": 0.0,
        "outer": 0.0, "top": 0.0, "bot": 0.0
    }
    # Per-face npho losses
    face_npho_loss_sums = {
        "inner": 0.0, "us": 0.0, "ds": 0.0,
        "outer": 0.0, "top": 0.0, "bot": 0.0
    }
    # Per-face time losses
    face_time_loss_sums = {
        "inner": 0.0, "us": 0.0, "ds": 0.0,
        "outer": 0.0, "top": 0.0, "bot": 0.0
    }
    masked_abs_face_npho = {name: 0.0 for name in face_loss_sums}
    masked_abs_face_time = {name: 0.0 for name in face_loss_sums}
    masked_sq_face_npho = {name: 0.0 for name in face_loss_sums}
    masked_sq_face_time = {name: 0.0 for name in face_loss_sums}
    masked_count_face = {name: 0.0 for name in face_loss_sums}
    masked_abs_sum_npho = 0.0
    masked_abs_sum_time = 0.0
    masked_sq_sum_npho = 0.0
    masked_sq_sum_time = 0.0
    masked_count_npho = 0.0
    masked_count_time = 0.0
    total_loss_sum = 0.0
    n_batches = 0

    # Gradient accumulation
    grad_accum_steps = max(int(grad_accum_steps), 1)
    accum_step = 0
    optimizer.zero_grad(set_to_none=True)

    # Track actual mask ratio (randomly-masked / valid sensors)
    # Use tensors to avoid GPU-CPU sync every batch (only .item() at epoch end)
    total_randomly_masked = torch.tensor(0, dtype=torch.long, device=device)
    total_valid_sensors = torch.tensor(0, dtype=torch.long, device=device)
    # Track time-valid sensors (sensors with npho > threshold)
    total_time_valid_masked = torch.tensor(0.0, dtype=torch.float32, device=device)

    top_indices = torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long()
    bot_indices = torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long()

    # Initialize profiler
    profiler = SimpleProfiler(enabled=profile, sync_cuda=True)

    # Create dataset - data is pre-normalized by XECStreamingDataset
    # Note: dataset I/O profiling requires dataloader_workers=0 for accurate stats
    root_files_list = root_files if isinstance(root_files, list) else [root_files]
    dataset = XECStreamingDataset(
        root_files=root_files_list,
        tree_name=tree_name,
        batch_size=batch_size,
        step_size=step_size,
        npho_branch=npho_branch,
        time_branch=time_branch,
        npho_scale=npho_scale,
        npho_scale2=npho_scale2,
        time_scale=time_scale,
        time_shift=time_shift,
        sentinel_value=sentinel_value,
        npho_threshold=npho_threshold,
        num_workers=dataset_workers,
        log_invalid_npho=log_invalid_npho,
        load_truth_branches=False,  # MAE doesn't need truth branches
        profile=profile,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset yields pre-batched tensors
        num_workers=dataloader_workers,
        pin_memory=True,
        persistent_workers=(dataloader_workers > 0),
        prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
    )

    for x_batch, _ in loader:
        profiler.start("gpu_transfer")
        x_in = x_batch.to(device, non_blocking=True)  # Already normalized: (B, 4760, 2)
        profiler.stop()

        with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            # 1. Forward Pass (pass npho_threshold_norm for stratified masking)
            profiler.start("forward")
            recons, mask = model(x_in, npho_threshold_norm=npho_threshold_norm)
            profiler.stop()

            # Track actual mask ratio (no .item() to avoid GPU-CPU sync)
            # Already-invalid sensors have time == sentinel_value and are NOT in mask
            already_invalid = (x_in[:, :, 1] == sentinel_value)  # (B, N)
            total_valid_sensors += (~already_invalid).sum()
            total_randomly_masked += mask.sum().long()

            profiler.start("loss_compute")
            # 2. Gather Truth Targets
            if hasattr(model, "encoder") and getattr(model.encoder, "outer_fine", False):
                outer_target = build_outer_fine_grid_tensor(
                    x_in,
                    pool_kernel=model.encoder.outer_fine_pool,
                    sentinel_value=getattr(model, "sentinel_value", None)
                )
            else:
                outer_target = gather_face(x_in, OUTER_COARSE_FULL_INDEX_MAP)

            targets = {
                "inner": gather_face(x_in, INNER_INDEX_MAP),
                "us":    gather_face(x_in, US_INDEX_MAP),
                "ds":    gather_face(x_in, DS_INDEX_MAP),
                "outer": outer_target,
                "top":   gather_hex_nodes(x_in, top_indices).permute(0, 2, 1),
                "bot":   gather_hex_nodes(x_in, bot_indices).permute(0, 2, 1)
            }

            # 3. Calculate Loss
            loss = 0.0

            face_to_sensor_indices = {
                "inner": INNER_INDEX_MAP,
                "us":    US_INDEX_MAP,
                "ds":    DS_INDEX_MAP,
                "outer": OUTER_COARSE_FULL_INDEX_MAP,  # Always use coarse for mask lookup
                "top":   top_indices,
                "bot":   bot_indices
            }
            for name, pred in recons.items():
                if name in targets:
                    indices = face_to_sensor_indices.get(name)

                    if indices is not None:
                        m_face = mask[:, indices]  # Shape: (B, num_sensors_in_face)
                        if name in ["top", "bot"]:
                            mask_expanded = m_face.unsqueeze(1)  # (B, 1, num_hex_nodes)
                        elif name == "outer" and getattr(model.encoder, "outer_fine", False):
                            # For outer fine grid: upsample coarse mask to fine grid dimensions
                            H_coarse, W_coarse = OUTER_COARSE_FULL_INDEX_MAP.shape
                            m_coarse = m_face.view(mask.size(0), 1, H_coarse, W_coarse)
                            cr, cc = OUTER_FINE_COARSE_SCALE
                            m_fine = F.interpolate(m_coarse.float(), scale_factor=(float(cr), float(cc)), mode='nearest')
                            # Apply pooling if used
                            pool_kernel = model.encoder.outer_fine_pool
                            if pool_kernel:
                                if isinstance(pool_kernel, int):
                                    ph, pw = pool_kernel, pool_kernel
                                else:
                                    ph, pw = pool_kernel
                                m_fine = F.avg_pool2d(m_fine, kernel_size=(ph, pw), stride=(ph, pw))
                                # Convert avg back to binary (any masked → masked)
                                m_fine = (m_fine > 0).float()
                            mask_expanded = m_fine
                        else:
                            H, W = pred.shape[-2], pred.shape[-1]
                            mask_expanded = m_face.view(mask.size(0), 1, H, W)  # (B, 1, H, W)
                    else:
                        mask_expanded = torch.ones_like(pred[:, 0:1])

                    # Separate npho and time losses (use prediction channel indices)
                    target = targets[name]
                    # npho: prediction channel pred_npho_idx -> input channel 0
                    loss_map_npho = loss_func(pred[:, pred_npho_idx:pred_npho_idx+1], target[:, 0:1])
                    # time: prediction channel pred_time_idx -> input channel 1 (only if predicting time)
                    if predict_time:
                        loss_map_time = loss_func(pred[:, pred_time_idx:pred_time_idx+1], target[:, 1:2])

                    mask_sum = mask_expanded.sum()
                    mask_sum_safe = mask_sum + 1e-8
                    npho_loss = (loss_map_npho * mask_expanded).sum() / mask_sum_safe

                    # Time loss (only if predicting time)
                    time_loss = torch.tensor(0.0, device=device)
                    time_mask_sum = torch.tensor(0.0, device=device)
                    if predict_time:
                        # Conditional time loss: only compute where npho > threshold
                        # Get normalized npho values for this face
                        npho_norm_all = x_in[:, :, 0]  # (B, 4760) - normalized npho
                        if isinstance(indices, torch.Tensor):
                            npho_norm_face = npho_norm_all[:, indices]  # (B, num_sensors)
                        else:
                            npho_norm_face = npho_norm_all[:, indices.flatten()].view(x_in.size(0), *indices.shape)

                        # Create time_valid_mask using normalized threshold
                        if name in ["top", "bot"]:
                            time_valid_base = (npho_norm_face > npho_threshold_norm).unsqueeze(1).float()  # (B, 1, N)
                        elif name == "outer" and getattr(model.encoder, "outer_fine", False):
                            # For outer fine grid: use coarse-level threshold check, then upsample
                            time_valid_coarse = (npho_norm_face > npho_threshold_norm).float()  # (B, H_coarse, W_coarse)
                            time_valid_coarse = time_valid_coarse.view(x_in.size(0), 1, *indices.shape)
                            time_valid_base = F.interpolate(time_valid_coarse, scale_factor=(float(cr), float(cc)), mode='nearest')
                            if pool_kernel:
                                # Pool and convert back to binary (any valid → valid, be permissive)
                                time_valid_base = F.avg_pool2d(time_valid_base, kernel_size=(ph, pw), stride=(ph, pw))
                                time_valid_base = (time_valid_base > 0).float()
                        else:
                            H, W = pred.shape[-2], pred.shape[-1]
                            time_valid_base = (npho_norm_face > npho_threshold_norm).view(x_in.size(0), 1, H, W).float()

                        # Combined mask for time: randomly masked AND time-valid (npho > threshold)
                        time_mask_expanded = mask_expanded * time_valid_base

                        # Npho weighting for time loss (chi-square-like: weight ~ sqrt(npho))
                        if use_npho_time_weight and time_mask_expanded.sum() > 0:
                            # De-normalize to get approximate raw npho for weighting
                            # raw_npho = npho_scale * (exp(npho_norm * npho_scale2) - 1)
                            raw_npho_face = npho_scale * (torch.exp(npho_norm_face * npho_scale2) - 1)
                            if name in ["top", "bot"]:
                                npho_weight_map = torch.sqrt(raw_npho_face.clamp(min=npho_threshold)).unsqueeze(1)
                            elif name == "outer" and getattr(model.encoder, "outer_fine", False):
                                npho_coarse = raw_npho_face.view(x_in.size(0), 1, *indices.shape)
                                npho_fine = F.interpolate(npho_coarse, scale_factor=(float(cr), float(cc)), mode='nearest')
                                if pool_kernel:
                                    npho_fine = F.avg_pool2d(npho_fine, kernel_size=(ph, pw), stride=(ph, pw))
                                npho_weight_map = torch.sqrt(npho_fine.clamp(min=npho_threshold))
                            else:
                                H, W = pred.shape[-2], pred.shape[-1]
                                npho_weight_map = torch.sqrt(raw_npho_face.view(x_in.size(0), 1, H, W).clamp(min=npho_threshold))
                            # Normalize weight so mean is ~1 for stable training
                            npho_weight_map = npho_weight_map / (npho_weight_map[time_mask_expanded.bool()].mean() + 1e-8)
                            weighted_time_loss = (loss_map_time * time_mask_expanded * npho_weight_map).sum()
                        else:
                            weighted_time_loss = (loss_map_time * time_mask_expanded).sum()

                        time_mask_sum = time_mask_expanded.sum()
                        time_mask_sum_safe = time_mask_sum + 1e-8
                        time_loss = weighted_time_loss / time_mask_sum_safe

                        # Track time-valid masked count (no .item() to avoid GPU-CPU sync)
                        total_time_valid_masked += time_mask_sum

                    # MAE/RMSE tracking (expensive - skip if disabled)
                    if track_mae_rmse:
                        diff_npho = pred[:, pred_npho_idx:pred_npho_idx+1] - target[:, 0:1]
                        mask_sum_val = mask_sum.item()
                        if mask_sum_val > 0:
                            masked_abs_sum_npho += (diff_npho.abs() * mask_expanded).sum().item()
                            masked_sq_sum_npho += (diff_npho.pow(2) * mask_expanded).sum().item()
                            masked_count_npho += mask_sum_val
                            masked_abs_face_npho[name] += (diff_npho.abs() * mask_expanded).sum().item()
                            masked_sq_face_npho[name] += (diff_npho.pow(2) * mask_expanded).sum().item()
                        if predict_time:
                            diff_time = pred[:, pred_time_idx:pred_time_idx+1] - target[:, 1:2]
                            time_mask_sum_val = time_mask_sum.item()
                            if time_mask_sum_val > 0:
                                # Time metrics computed only on time-valid sensors
                                masked_abs_sum_time += (diff_time.abs() * time_mask_expanded).sum().item()
                                masked_sq_sum_time += (diff_time.pow(2) * time_mask_expanded).sum().item()
                                masked_count_time += time_mask_sum_val
                                masked_abs_face_time[name] += (diff_time.abs() * time_mask_expanded).sum().item()
                                masked_sq_face_time[name] += (diff_time.pow(2) * time_mask_expanded).sum().item()
                                masked_count_face[name] += time_mask_sum_val

                    if log_vars is not None and log_vars.numel() >= 2 and predict_time:
                        npho_loss = 0.5 * torch.exp(-log_vars[0]) * npho_loss + 0.5 * log_vars[0]
                        time_loss = 0.5 * torch.exp(-log_vars[1]) * time_loss + 0.5 * log_vars[1]
                    elif log_vars is not None and log_vars.numel() >= 1:
                        npho_loss = 0.5 * torch.exp(-log_vars[0]) * npho_loss + 0.5 * log_vars[0]
                    else:
                        npho_loss = npho_loss * npho_weight
                        if predict_time:
                            time_loss = time_loss * time_weight
                    face_loss = npho_loss + time_loss
                    loss += face_loss

                    # Per-face loss tracking (skip if disabled for speed)
                    if track_train_metrics:
                        face_loss_sums[name] += face_loss.item()
                        face_npho_loss_sums[name] += npho_loss.item()
                        if predict_time:
                            face_time_loss_sums[name] += time_loss.item()

            profiler.stop()  # loss_compute

        # Backward with gradient accumulation
        total_loss_sum += loss.item()
        n_batches += 1

        profiler.start("backward")
        loss = loss / grad_accum_steps
        scaler.scale(loss).backward()
        profiler.stop()

        accum_step += 1

        # Optimizer step every grad_accum_steps
        if accum_step % grad_accum_steps == 0:
            profiler.start("optimizer")
            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            profiler.stop()

        profiler.start("data_load_cpu")  # For next iteration

    # Stop any pending timer (last data_load_cpu started but not stopped)
    profiler.stop()

    # Final optimizer step if gradients remain from incomplete accumulation
    if accum_step % grad_accum_steps != 0:
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Print profiler report if enabled
    if profile:
        print(profiler.report())
        # Print dataset I/O breakdown (only accurate when dataloader_workers=0)
        if dataloader_workers == 0:
            print(dataset.get_profile_report())

    # Return averaged losses with detailed breakdown
    metrics = {}

    # Total loss (always computed)
    metrics["total_loss"] = total_loss_sum / max(1, n_batches)

    # Per-face metrics (only if track_train_metrics)
    if track_train_metrics:
        for name, val in face_loss_sums.items():
            metrics[f"loss_{name}"] = val / max(1, n_batches)
        for name, val in face_npho_loss_sums.items():
            metrics[f"loss_{name}_npho"] = val / max(1, n_batches)
        if predict_time:
            for name, val in face_time_loss_sums.items():
                metrics[f"loss_{name}_time"] = val / max(1, n_batches)
        # Aggregate npho/time losses (sum across faces)
        metrics["loss_npho"] = sum(metrics[f"loss_{name}_npho"] for name in face_npho_loss_sums)
        if predict_time:
            metrics["loss_time"] = sum(metrics[f"loss_{name}_time"] for name in face_time_loss_sums)

    # MAE/RMSE metrics (only if track_mae_rmse)
    if track_mae_rmse:
        metrics["mae_npho"] = masked_abs_sum_npho / max(masked_count_npho, 1e-8)
        metrics["rmse_npho"] = (masked_sq_sum_npho / max(masked_count_npho, 1e-8)) ** 0.5
        if predict_time:
            metrics["mae_time"] = masked_abs_sum_time / max(masked_count_time, 1e-8)
            metrics["rmse_time"] = (masked_sq_sum_time / max(masked_count_time, 1e-8)) ** 0.5
        for name in face_loss_sums:
            face_count = masked_count_face[name]
            metrics[f"mae_{name}_npho"] = masked_abs_face_npho[name] / max(face_count, 1e-8)
            metrics[f"rmse_{name}_npho"] = (masked_sq_face_npho[name] / max(face_count, 1e-8)) ** 0.5
            if predict_time:
                metrics[f"mae_{name}_time"] = masked_abs_face_time[name] / max(face_count, 1e-8)
                metrics[f"rmse_{name}_time"] = (masked_sq_face_time[name] / max(face_count, 1e-8)) ** 0.5

    if log_vars is not None and log_vars.numel() >= 1:
        metrics["channel_logvar_npho"] = log_vars[0].item()
        if predict_time and log_vars.numel() >= 2:
            metrics["channel_logvar_time"] = log_vars[1].item()

    # Actual mask ratio (randomly-masked / valid sensors)
    # Convert tensor accumulators to Python scalars
    total_valid_sensors_val = total_valid_sensors.item()
    total_randomly_masked_val = total_randomly_masked.item()
    total_time_valid_masked_val = total_time_valid_masked.item()

    if total_valid_sensors_val > 0:
        metrics["actual_mask_ratio"] = total_randomly_masked_val / total_valid_sensors_val
    else:
        metrics["actual_mask_ratio"] = 0.0

    # Time-valid ratio (masked sensors with npho > threshold / total masked sensors)
    if total_randomly_masked_val > 0:
        metrics["time_valid_ratio"] = total_time_valid_masked_val / total_randomly_masked_val
    else:
        metrics["time_valid_ratio"] = 0.0

    return metrics

def run_eval_mae(model, device, root_files, tree_name,
                 batch_size=8192, step_size=4000,
                 amp=True,
                 npho_branch="npho", time_branch="relative_time",
                 npho_scale=DEFAULT_NPHO_SCALE, npho_scale2=DEFAULT_NPHO_SCALE2,
                 time_scale=DEFAULT_TIME_SCALE, time_shift=DEFAULT_TIME_SHIFT,
                 sentinel_value=DEFAULT_SENTINEL_VALUE,
                 loss_fn="mse",
                 npho_weight=1.0,
                 time_weight=1.0,
                 auto_channel_weight=False,
                 collect_predictions=False, max_events=1000,
                 dataloader_workers=0,
                 dataset_workers=8,
                 prefetch_factor=2,
                 npho_threshold=None,
                 use_npho_time_weight=True,
                 track_mae_rmse=True,
                 profile=False,
                 log_invalid_npho=True):
    """
    Evaluate MAE model on validation data.

    Args:
        collect_predictions: If True, collect sensor-level predictions for ROOT output
        max_events: Max events to collect when collect_predictions=True

    Returns:
        If collect_predictions=False: dict of metrics
        If collect_predictions=True: (dict of metrics, dict of predictions)
    """
    model.eval()
    loss_func = get_pointwise_loss_fn(loss_fn)
    log_vars = getattr(model, "channel_log_vars", None) if auto_channel_weight else None

    # Detect predict_channels from model (default to ["npho", "time"] for legacy)
    predict_channels = getattr(model, 'predict_channels', ['npho', 'time'])
    predict_time = 'time' in predict_channels
    # Map prediction channel indices to input channel indices
    pred_npho_idx = predict_channels.index("npho") if "npho" in predict_channels else None
    pred_time_idx = predict_channels.index("time") if "time" in predict_channels else None

    # Conditional time loss threshold
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD

    # Convert to normalized space for stratified masking and threshold checks
    npho_threshold_norm = np.log1p(npho_threshold / npho_scale) / npho_scale2

    # Loss tracking
    face_loss_sums = {"inner": 0.0, "us": 0.0, "ds": 0.0, "outer": 0.0, "top": 0.0, "bot": 0.0}
    face_npho_loss_sums = {"inner": 0.0, "us": 0.0, "ds": 0.0, "outer": 0.0, "top": 0.0, "bot": 0.0}
    face_time_loss_sums = {"inner": 0.0, "us": 0.0, "ds": 0.0, "outer": 0.0, "top": 0.0, "bot": 0.0}
    masked_abs_face_npho = {name: 0.0 for name in face_loss_sums}
    masked_abs_face_time = {name: 0.0 for name in face_loss_sums}
    masked_sq_face_npho = {name: 0.0 for name in face_loss_sums}
    masked_sq_face_time = {name: 0.0 for name in face_loss_sums}
    masked_count_face = {name: 0.0 for name in face_loss_sums}
    masked_abs_sum_npho = 0.0
    masked_abs_sum_time = 0.0
    masked_sq_sum_npho = 0.0
    masked_sq_sum_time = 0.0
    masked_count_npho = 0.0
    masked_count_time = 0.0
    total_loss = 0.0
    n_batches = 0

    # Prediction collection - only include pred_time if predicting time
    predictions = {
        "truth_npho": [], "truth_time": [],
        "pred_npho": [],
        "mask": [], "x_masked": []
    }
    if predict_time:
        predictions["pred_time"] = []
    n_collected = 0

    # Track actual mask ratio (randomly-masked / valid sensors)
    # Use tensors to avoid GPU-CPU sync every batch (only .item() at epoch end)
    total_randomly_masked = torch.tensor(0, dtype=torch.long, device=device)
    total_valid_sensors = torch.tensor(0, dtype=torch.long, device=device)
    # Track time-valid sensors (sensors with npho > threshold)
    total_time_valid_masked = torch.tensor(0.0, dtype=torch.float32, device=device)

    top_indices = torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long()
    bot_indices = torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long()

    # Number of output channels based on predict_channels
    out_channels = len(predict_channels)

    face_to_sensor_indices = {
        "inner": INNER_INDEX_MAP,
        "us":    US_INDEX_MAP,
        "ds":    DS_INDEX_MAP,
        "outer": OUTER_COARSE_FULL_INDEX_MAP,  # Always use coarse for mask lookup
        "top":   top_indices,
        "bot":   bot_indices
    }

    num_sensors = int(max(
        OUTER_CENTER_INDEX_MAP.max(),
        int(top_indices.max().item()),
        int(bot_indices.max().item()),
    )) + 1

    # Initialize profiler
    profiler = SimpleProfiler(enabled=profile, sync_cuda=True)

    def scatter_rect_face(full, face_pred, index_map):
        face_pred = face_pred.to(full.dtype)
        idx_flat = torch.tensor(index_map.reshape(-1), device=face_pred.device, dtype=torch.long)
        valid = idx_flat >= 0
        idx = idx_flat[valid]
        vals = face_pred.permute(0, 2, 3, 1).reshape(face_pred.size(0), -1, out_channels)[:, valid]
        full[:, idx, :] = vals

    def scatter_hex_face(full, pred_nodes, indices):
        pred_nodes = pred_nodes.to(full.dtype)
        full[:, indices, :] = pred_nodes.permute(0, 2, 1)

    def reconstruct_outer_from_fine(pred_outer, pool_kernel):
        fine_pred = pred_outer
        if pool_kernel:
            if isinstance(pool_kernel, int):
                scale = (pool_kernel, pool_kernel)
            else:
                scale = tuple(pool_kernel)
            fine_pred = F.interpolate(pred_outer, scale_factor=scale, mode="nearest")

        cr, cc = OUTER_FINE_COARSE_SCALE
        Hc, Wc = OUTER_COARSE_FULL_INDEX_MAP.shape

        # Aggregate npho (always present at pred index 0) - sum over sub-pixels
        npho_idx = pred_npho_idx if pred_npho_idx is not None else 0
        npho = fine_pred[:, npho_idx:npho_idx+1].contiguous().view(-1, 1, Hc, cr, Wc, cc).sum(dim=(3, 5))
        coarse_parts = [npho]

        # Aggregate time only if predicting time - mean over sub-pixels
        if predict_time and pred_time_idx is not None:
            time = fine_pred[:, pred_time_idx:pred_time_idx+1].contiguous().view(-1, 1, Hc, cr, Wc, cc).mean(dim=(3, 5))
            coarse_parts.append(time)

        coarse_pred = torch.cat(coarse_parts, dim=1)

        sr, sc = OUTER_FINE_CENTER_SCALE
        Hc_center, Wc_center = OUTER_CENTER_INDEX_MAP.shape
        top = OUTER_FINE_CENTER_START[0] * cr
        left = OUTER_FINE_CENTER_START[1] * cc
        center_fine = fine_pred[:, :, top:top + Hc_center * sr, left:left + Wc_center * sc]
        c_npho = center_fine[:, npho_idx:npho_idx+1].contiguous().view(-1, 1, Hc_center, sr, Wc_center, sc).sum(dim=(3, 5))
        center_parts = [c_npho]

        if predict_time and pred_time_idx is not None:
            c_time = center_fine[:, pred_time_idx:pred_time_idx+1].contiguous().view(-1, 1, Hc_center, sr, Wc_center, sc).mean(dim=(3, 5))
            center_parts.append(c_time)

        center_pred = torch.cat(center_parts, dim=1)

        return coarse_pred, center_pred

    def assemble_full_pred(recons_dict):
        full = torch.zeros(
            (recons_dict["inner"].size(0), num_sensors, out_channels),
            device=recons_dict["inner"].device,
            dtype=recons_dict["inner"].dtype,
        )
        scatter_rect_face(full, recons_dict["inner"], INNER_INDEX_MAP)
        scatter_rect_face(full, recons_dict["us"], US_INDEX_MAP)
        scatter_rect_face(full, recons_dict["ds"], DS_INDEX_MAP)
        if getattr(model.encoder, "outer_fine", False):
            coarse_pred, center_pred = reconstruct_outer_from_fine(
                recons_dict["outer"], model.encoder.outer_fine_pool
            )
            scatter_rect_face(full, coarse_pred, OUTER_COARSE_FULL_INDEX_MAP)
            scatter_rect_face(full, center_pred, OUTER_CENTER_INDEX_MAP)
        else:
            scatter_rect_face(full, recons_dict["outer"], OUTER_COARSE_FULL_INDEX_MAP)
        scatter_hex_face(full, recons_dict["top"], top_indices)
        scatter_hex_face(full, recons_dict["bot"], bot_indices)
        return full

    # Create dataset - data is pre-normalized by XECStreamingDataset
    # Note: dataset I/O profiling requires dataloader_workers=0 for accurate stats
    root_files_list = root_files if isinstance(root_files, list) else [root_files]
    dataset = XECStreamingDataset(
        root_files=root_files_list,
        tree_name=tree_name,
        batch_size=batch_size,
        step_size=step_size,
        npho_branch=npho_branch,
        time_branch=time_branch,
        npho_scale=npho_scale,
        npho_scale2=npho_scale2,
        time_scale=time_scale,
        time_shift=time_shift,
        sentinel_value=sentinel_value,
        npho_threshold=npho_threshold,
        num_workers=dataset_workers,
        log_invalid_npho=log_invalid_npho,
        load_truth_branches=False,  # MAE doesn't need truth branches
        profile=profile,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset yields pre-batched tensors
        num_workers=dataloader_workers,
        pin_memory=True,
        persistent_workers=(dataloader_workers > 0),
        prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
    )

    for x_batch, _ in loader:
        profiler.start("gpu_transfer")
        x_in = x_batch.to(device, non_blocking=True)  # Already normalized: (B, 4760, 2)
        profiler.stop()

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=amp):
                # Get masked input for visualization (pass npho_threshold_norm for stratified masking)
                profiler.start("forward")
                x_masked, mask = model.random_masking(x_in, npho_threshold_norm=npho_threshold_norm)

                # Track actual mask ratio (no .item() to avoid GPU-CPU sync)
                already_invalid = (x_in[:, :, 1] == sentinel_value)  # (B, N)
                total_valid_sensors += (~already_invalid).sum()
                total_randomly_masked += mask.sum().long()

                latent_seq = model.encoder.forward_features(x_masked)

                # Decode each face
                cnn_names = list(model.encoder.cnn_face_names)
                name_to_idx = {name: i for i, name in enumerate(cnn_names)}
                if model.encoder.outer_fine:
                    outer_idx = len(cnn_names)
                    top_idx = outer_idx + 1
                else:
                    outer_idx = name_to_idx.get("outer_coarse", name_to_idx.get("outer_center"))
                    top_idx = len(cnn_names)
                bot_idx = top_idx + 1

                recons = {
                    "inner": model.dec_inner(latent_seq[:, name_to_idx["inner"]]),
                    "us":    model.dec_us(latent_seq[:, name_to_idx["us"]]),
                    "ds":    model.dec_ds(latent_seq[:, name_to_idx["ds"]]),
                    "outer": model.dec_outer(latent_seq[:, outer_idx]),
                    "top":   model.dec_top(latent_seq[:, top_idx]),
                    "bot":   model.dec_bot(latent_seq[:, bot_idx]),
                }
                profiler.stop()  # forward

                profiler.start("loss_compute")
                targets = {
                    "inner": gather_face(x_in, INNER_INDEX_MAP),
                    "us":    gather_face(x_in, US_INDEX_MAP),
                    "ds":    gather_face(x_in, DS_INDEX_MAP),
                    "outer": build_outer_fine_grid_tensor(x_in, model.encoder.outer_fine_pool, sentinel_value=getattr(model, "sentinel_value", None)) if getattr(model.encoder, "outer_fine", False) else gather_face(x_in, OUTER_COARSE_FULL_INDEX_MAP),
                    "top":   gather_hex_nodes(x_in, top_indices).permute(0, 2, 1),
                    "bot":   gather_hex_nodes(x_in, bot_indices).permute(0, 2, 1)
                }

                loss = 0.0
                for name, pred in recons.items():
                    if name in targets:
                        indices = face_to_sensor_indices.get(name)
                        if indices is not None:
                            m_face = mask[:, indices]
                            if name in ["top", "bot"]:
                                mask_expanded = m_face.unsqueeze(1)
                            elif name == "outer" and getattr(model.encoder, "outer_fine", False):
                                # For outer fine grid: upsample coarse mask to fine grid dimensions
                                H_coarse, W_coarse = OUTER_COARSE_FULL_INDEX_MAP.shape
                                m_coarse = m_face.view(mask.size(0), 1, H_coarse, W_coarse)
                                cr, cc = OUTER_FINE_COARSE_SCALE
                                m_fine = F.interpolate(m_coarse.float(), scale_factor=(float(cr), float(cc)), mode='nearest')
                                pool_kernel = model.encoder.outer_fine_pool
                                if pool_kernel:
                                    if isinstance(pool_kernel, int):
                                        ph, pw = pool_kernel, pool_kernel
                                    else:
                                        ph, pw = pool_kernel
                                    m_fine = F.avg_pool2d(m_fine, kernel_size=(ph, pw), stride=(ph, pw))
                                    m_fine = (m_fine > 0).float()
                                mask_expanded = m_fine
                            else:
                                mask_expanded = m_face.view(mask.size(0), 1, *pred.shape[-2:])
                        else:
                            mask_expanded = torch.ones_like(pred[:, 0:1])

                        target = targets[name]
                        # npho: prediction channel pred_npho_idx -> input channel 0
                        loss_map_npho = loss_func(pred[:, pred_npho_idx:pred_npho_idx+1], target[:, 0:1])
                        # time: prediction channel pred_time_idx -> input channel 1 (only if predicting time)
                        if predict_time:
                            loss_map_time = loss_func(pred[:, pred_time_idx:pred_time_idx+1], target[:, 1:2])

                        mask_sum = mask_expanded.sum()
                        mask_sum_safe = mask_sum + 1e-8
                        npho_loss = (loss_map_npho * mask_expanded).sum() / mask_sum_safe

                        # Time loss (only if predicting time)
                        time_loss = torch.tensor(0.0, device=device)
                        time_mask_sum = torch.tensor(0.0, device=device)
                        if predict_time:
                            # Conditional time loss: only compute where npho > threshold
                            # Get normalized npho values for this face
                            npho_norm_all = x_in[:, :, 0]  # (B, 4760) - normalized npho
                            if isinstance(indices, torch.Tensor):
                                npho_norm_face = npho_norm_all[:, indices]
                            else:
                                npho_norm_face = npho_norm_all[:, indices.flatten()].view(x_in.size(0), *indices.shape)

                            # Create time_valid_mask using normalized threshold
                            if name in ["top", "bot"]:
                                time_valid_base = (npho_norm_face > npho_threshold_norm).unsqueeze(1).float()
                            elif name == "outer" and getattr(model.encoder, "outer_fine", False):
                                time_valid_coarse = (npho_norm_face > npho_threshold_norm).float()
                                time_valid_coarse = time_valid_coarse.view(x_in.size(0), 1, *indices.shape)
                                time_valid_base = F.interpolate(time_valid_coarse, scale_factor=(float(cr), float(cc)), mode='nearest')
                                if pool_kernel:
                                    time_valid_base = F.avg_pool2d(time_valid_base, kernel_size=(ph, pw), stride=(ph, pw))
                                    time_valid_base = (time_valid_base > 0).float()
                            else:
                                H, W = pred.shape[-2], pred.shape[-1]
                                time_valid_base = (npho_norm_face > npho_threshold_norm).view(x_in.size(0), 1, H, W).float()

                            time_mask_expanded = mask_expanded * time_valid_base

                            # Npho weighting for time loss
                            if use_npho_time_weight and time_mask_expanded.sum() > 0:
                                # De-normalize to get approximate raw npho for weighting
                                raw_npho_face = npho_scale * (torch.exp(npho_norm_face * npho_scale2) - 1)
                                if name in ["top", "bot"]:
                                    npho_weight_map = torch.sqrt(raw_npho_face.clamp(min=npho_threshold)).unsqueeze(1)
                                elif name == "outer" and getattr(model.encoder, "outer_fine", False):
                                    npho_coarse = raw_npho_face.view(x_in.size(0), 1, *indices.shape)
                                    npho_fine = F.interpolate(npho_coarse, scale_factor=(float(cr), float(cc)), mode='nearest')
                                    if pool_kernel:
                                        npho_fine = F.avg_pool2d(npho_fine, kernel_size=(ph, pw), stride=(ph, pw))
                                    npho_weight_map = torch.sqrt(npho_fine.clamp(min=npho_threshold))
                                else:
                                    H, W = pred.shape[-2], pred.shape[-1]
                                    npho_weight_map = torch.sqrt(raw_npho_face.view(x_in.size(0), 1, H, W).clamp(min=npho_threshold))
                                npho_weight_map = npho_weight_map / (npho_weight_map[time_mask_expanded.bool()].mean() + 1e-8)
                                weighted_time_loss = (loss_map_time * time_mask_expanded * npho_weight_map).sum()
                            else:
                                weighted_time_loss = (loss_map_time * time_mask_expanded).sum()

                            time_mask_sum = time_mask_expanded.sum()
                            time_mask_sum_safe = time_mask_sum + 1e-8
                            time_loss = weighted_time_loss / time_mask_sum_safe

                            # Track time-valid masked count (no .item() to avoid GPU-CPU sync)
                            total_time_valid_masked += time_mask_sum

                        # MAE/RMSE tracking (skip if disabled for speed)
                        if track_mae_rmse:
                            diff_npho = pred[:, pred_npho_idx:pred_npho_idx+1] - target[:, 0:1]
                            mask_sum_val = mask_sum.item()
                            if mask_sum_val > 0:
                                masked_abs_sum_npho += (diff_npho.abs() * mask_expanded).sum().item()
                                masked_sq_sum_npho += (diff_npho.pow(2) * mask_expanded).sum().item()
                                masked_count_npho += mask_sum_val
                                masked_abs_face_npho[name] += (diff_npho.abs() * mask_expanded).sum().item()
                                masked_sq_face_npho[name] += (diff_npho.pow(2) * mask_expanded).sum().item()
                            if predict_time:
                                diff_time = pred[:, pred_time_idx:pred_time_idx+1] - target[:, 1:2]
                                time_mask_sum_val = time_mask_sum.item()
                                if time_mask_sum_val > 0:
                                    masked_abs_sum_time += (diff_time.abs() * time_mask_expanded).sum().item()
                                    masked_sq_sum_time += (diff_time.pow(2) * time_mask_expanded).sum().item()
                                    masked_count_time += time_mask_sum_val
                                    masked_abs_face_time[name] += (diff_time.abs() * time_mask_expanded).sum().item()
                                    masked_sq_face_time[name] += (diff_time.pow(2) * time_mask_expanded).sum().item()
                                    masked_count_face[name] += time_mask_sum_val
                        if log_vars is not None and log_vars.numel() >= 2 and predict_time:
                            npho_loss = 0.5 * torch.exp(-log_vars[0]) * npho_loss + 0.5 * log_vars[0]
                            time_loss = 0.5 * torch.exp(-log_vars[1]) * time_loss + 0.5 * log_vars[1]
                        elif log_vars is not None and log_vars.numel() >= 1:
                            npho_loss = 0.5 * torch.exp(-log_vars[0]) * npho_loss + 0.5 * log_vars[0]
                        else:
                            npho_loss = npho_loss * npho_weight
                            if predict_time:
                                time_loss = time_loss * time_weight
                        face_loss = npho_loss + time_loss

                        face_loss_sums[name] += face_loss.item()
                        face_npho_loss_sums[name] += npho_loss.item()
                        if predict_time:
                            face_time_loss_sums[name] += time_loss.item()
                        loss += face_loss

                # Collect predictions for ROOT output
                if collect_predictions and n_collected < max_events:
                    full_pred = assemble_full_pred(recons)
                    n_to_collect = min(x_in.shape[0], max_events - n_collected)
                    # Truth from input (always both channels)
                    predictions["truth_npho"].append(x_in[:n_to_collect, :, 0].cpu().numpy())
                    predictions["truth_time"].append(x_in[:n_to_collect, :, 1].cpu().numpy())
                    # Predictions - use channel indices from predict_channels
                    predictions["pred_npho"].append(full_pred[:n_to_collect, :, pred_npho_idx].cpu().numpy())
                    if predict_time and pred_time_idx is not None:
                        predictions["pred_time"].append(full_pred[:n_to_collect, :, pred_time_idx].cpu().numpy())
                    predictions["mask"].append(mask[:n_to_collect].cpu().numpy())
                    predictions["x_masked"].append(x_masked[:n_to_collect].cpu().numpy())
                    n_collected += n_to_collect

                profiler.stop()  # loss_compute

        total_loss += loss.item()
        n_batches += 1

        profiler.start("data_load_cpu")  # For next iteration

    # Stop any pending timer
    profiler.stop()

    # Print profiler report if enabled
    if profile:
        print(profiler.report("Validation timing breakdown"))
        # Print dataset I/O breakdown (only accurate when dataloader_workers=0)
        if dataloader_workers == 0:
            print(dataset.get_profile_report())

    # Build metrics dict
    metrics = {}
    metrics["total_loss"] = total_loss / max(1, n_batches)

    # Per-face loss metrics (always included in eval for monitoring)
    for name, val in face_loss_sums.items():
        metrics[f"loss_{name}"] = val / max(1, n_batches)
    for name, val in face_npho_loss_sums.items():
        metrics[f"loss_{name}_npho"] = val / max(1, n_batches)
    if predict_time:
        for name, val in face_time_loss_sums.items():
            metrics[f"loss_{name}_time"] = val / max(1, n_batches)

    # Aggregate npho/time losses (sum across faces, consistent with total_loss)
    metrics["loss_npho"] = sum(metrics[f"loss_{name}_npho"] for name in face_npho_loss_sums)
    if predict_time:
        metrics["loss_time"] = sum(metrics[f"loss_{name}_time"] for name in face_time_loss_sums)

    # MAE/RMSE metrics (only if track_mae_rmse)
    if track_mae_rmse:
        metrics["mae_npho"] = masked_abs_sum_npho / max(masked_count_npho, 1e-8)
        metrics["rmse_npho"] = (masked_sq_sum_npho / max(masked_count_npho, 1e-8)) ** 0.5
        if predict_time:
            metrics["mae_time"] = masked_abs_sum_time / max(masked_count_time, 1e-8)
            metrics["rmse_time"] = (masked_sq_sum_time / max(masked_count_time, 1e-8)) ** 0.5
        for name in face_loss_sums:
            face_count = masked_count_face[name]
            metrics[f"mae_{name}_npho"] = masked_abs_face_npho[name] / max(face_count, 1e-8)
            metrics[f"rmse_{name}_npho"] = (masked_sq_face_npho[name] / max(face_count, 1e-8)) ** 0.5
            if predict_time:
                metrics[f"mae_{name}_time"] = masked_abs_face_time[name] / max(face_count, 1e-8)
                metrics[f"rmse_{name}_time"] = (masked_sq_face_time[name] / max(face_count, 1e-8)) ** 0.5

    if log_vars is not None and log_vars.numel() >= 1:
        metrics["channel_logvar_npho"] = log_vars[0].item()
        if predict_time and log_vars.numel() >= 2:
            metrics["channel_logvar_time"] = log_vars[1].item()

    # Actual mask ratio (randomly-masked / valid sensors)
    # Convert tensor accumulators to Python scalars
    total_valid_sensors_val = total_valid_sensors.item()
    total_randomly_masked_val = total_randomly_masked.item()
    total_time_valid_masked_val = total_time_valid_masked.item()

    if total_valid_sensors_val > 0:
        metrics["actual_mask_ratio"] = total_randomly_masked_val / total_valid_sensors_val
    else:
        metrics["actual_mask_ratio"] = 0.0

    # Time-valid ratio (masked sensors with npho > threshold / total masked sensors)
    if total_randomly_masked_val > 0:
        metrics["time_valid_ratio"] = total_time_valid_masked_val / total_randomly_masked_val
    else:
        metrics["time_valid_ratio"] = 0.0

    if collect_predictions:
        # Concatenate collected predictions
        for key in predictions:
            if predictions[key]:
                predictions[key] = np.concatenate(predictions[key], axis=0)
            else:
                predictions[key] = np.array([])
        return metrics, predictions

    return metrics
