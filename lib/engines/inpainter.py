"""
Training and evaluation engine for dead channel inpainting.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List, Callable
from contextlib import nullcontext
from ..dataset import XECStreamingDataset
from ..geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_TIME, DEFAULT_NPHO_THRESHOLD
)
from ..utils import SimpleProfiler
from ..normalization import NphoTransform


def compute_inpainting_loss_flat(
    pred_all: torch.Tensor,
    original_values: torch.Tensor,
    mask: torch.Tensor,
    loss_fn: str = "smooth_l1",
    loss_beta: float = 1.0,
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
    npho_threshold: float = None,
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    use_npho_time_weight: bool = True,
    track_metrics: bool = True,
    predict_channels: List[str] = None,
    npho_scheme: str = "log1p",
    npho_loss_weight_enabled: bool = False,
    npho_loss_weight_alpha: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute inpainting loss from FLAT predictions (from forward_training).

    This is a simplified, faster loss computation that works directly on flat tensors
    without per-face iteration. All masked positions are treated uniformly.

    Args:
        pred_all: (B, 4760, out_channels) - predictions for all sensors
        original_values: (B, 4760, 2) - ground truth sensor values (normalized)
        mask: (B, 4760) - binary mask of randomly-masked positions (1 = masked)
        loss_fn: "smooth_l1", "mse", or "l1"
        loss_beta: beta parameter for smooth_l1/huber loss (default 1.0)
        npho_weight: weight for npho channel loss
        time_weight: weight for time channel loss
        npho_threshold: raw npho threshold for conditional time loss
        npho_scale, npho_scale2: normalization params for npho
        use_npho_time_weight: whether to weight time loss by sqrt(npho)
        track_metrics: whether to compute MAE/RMSE metrics
        predict_channels: list of channels being predicted (["npho"] or ["npho", "time"])

    Returns:
        total_loss: scalar loss for backprop
        metrics: dict of losses and optional MAE/RMSE
    """
    device = pred_all.device

    # Detect predict_channels (default to ["npho", "time"] for legacy)
    if predict_channels is None:
        predict_channels = ["npho", "time"]
    predict_time = "time" in predict_channels
    pred_npho_idx = predict_channels.index("npho") if "npho" in predict_channels else 0
    pred_time_idx = predict_channels.index("time") if "time" in predict_channels else None

    # Create NphoTransform for denormalization and threshold conversion
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)

    # Threshold for valid time (in normalized space)
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD
    npho_threshold_norm = npho_transform.convert_threshold(npho_threshold)

    # Select loss function
    if loss_fn == "mse":
        loss_func = lambda p, t: F.mse_loss(p, t, reduction="none")
    elif loss_fn == "l1":
        loss_func = lambda p, t: F.l1_loss(p, t, reduction="none")
    else:  # smooth_l1 / huber
        loss_func = lambda p, t: F.smooth_l1_loss(p, t, reduction="none", beta=loss_beta)

    # Get masked positions
    mask_bool = mask.bool()  # (B, N)
    num_masked = mask_bool.sum()

    if num_masked == 0:
        return torch.tensor(0.0, device=device), {"total_loss": 0.0}

    # Gather predictions and targets at masked positions
    pred_masked = pred_all[mask_bool]  # (num_masked, out_channels)
    gt_masked = original_values[mask_bool]  # (num_masked, 2)

    # Compute npho loss (all masked positions) with optional intensity-based weighting
    loss_npho = loss_func(pred_masked[:, pred_npho_idx], gt_masked[:, 0])  # (num_masked,)
    if npho_loss_weight_enabled and num_masked > 0:
        # Denormalize to get raw npho for weighting
        raw_npho_masked = npho_transform.inverse(gt_masked[:, 0])
        weight_npho = (raw_npho_masked.clamp(min=1.0) + 1.0).pow(npho_loss_weight_alpha)
        weight_npho = weight_npho / weight_npho.mean()  # Normalize weights
        avg_npho_loss = (loss_npho * weight_npho).mean()
    else:
        avg_npho_loss = loss_npho.mean()

    # Compute time loss (only if predicting time)
    avg_time_loss = torch.tensor(0.0, device=device)
    time_valid = None
    if predict_time:
        # Compute time loss (only for sensors with sufficient npho)
        npho_gt_norm = gt_masked[:, 0]
        time_valid = npho_gt_norm > npho_threshold_norm  # (num_masked,)

        if time_valid.sum() > 0:
            pred_time_valid = pred_masked[time_valid, pred_time_idx]
            gt_time_valid = gt_masked[time_valid, 1]

            loss_time = loss_func(pred_time_valid, gt_time_valid)  # (num_time_valid,)

            # Optional: weight by sqrt(npho) for chi-square-like weighting
            if use_npho_time_weight:
                # Denormalize npho to get raw values for weighting
                npho_norm_valid = gt_masked[time_valid, 0]
                raw_npho = npho_transform.inverse(npho_norm_valid)
                time_weights = torch.sqrt(raw_npho.clamp(min=1.0))
                time_weights = time_weights / time_weights.mean()  # Normalize weights
                loss_time = loss_time * time_weights

            avg_time_loss = loss_time.mean()

    # Total loss
    total_loss = npho_weight * avg_npho_loss
    if predict_time:
        total_loss = total_loss + time_weight * avg_time_loss

    metrics = {
        "total_loss": total_loss.item(),
        "loss_npho": avg_npho_loss.item(),
    }
    if predict_time:
        metrics["loss_time"] = avg_time_loss.item()

    # Optional MAE/RMSE metrics
    if track_metrics:
        with torch.no_grad():
            # MAE
            mae_npho = torch.abs(pred_masked[:, pred_npho_idx] - gt_masked[:, 0]).mean().item()
            metrics["mae_npho"] = mae_npho

            # RMSE
            rmse_npho = torch.sqrt(((pred_masked[:, pred_npho_idx] - gt_masked[:, 0]) ** 2).mean()).item()
            metrics["rmse_npho"] = rmse_npho

            if predict_time and time_valid is not None and time_valid.sum() > 0:
                mae_time = torch.abs(pred_masked[time_valid, pred_time_idx] - gt_masked[time_valid, 1]).mean().item()
                metrics["mae_time"] = mae_time
                rmse_time = torch.sqrt(((pred_masked[time_valid, pred_time_idx] - gt_masked[time_valid, 1]) ** 2).mean()).item()
                metrics["rmse_time"] = rmse_time

    return total_loss, metrics


def compute_inpainting_loss(
    results: Dict,
    original_values: torch.Tensor,
    mask: torch.Tensor,
    face_index_maps: Dict[str, torch.Tensor],
    loss_fn: str = "smooth_l1",
    loss_beta: float = 1.0,
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
    outer_fine: bool = False,
    outer_fine_pool: Optional[Tuple[int, int]] = None,
    track_mae_rmse: bool = True,
    track_metrics: bool = True,
    npho_threshold: float = None,
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    use_npho_time_weight: bool = True,
    predict_channels: List[str] = None,
    npho_scheme: str = "log1p",
    npho_loss_weight_enabled: bool = False,
    npho_loss_weight_alpha: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute inpainting loss for all faces.

    Args:
        results: dict from XEC_Inpainter.forward()
        original_values: (B, 4760, 2) - ground truth sensor values (normalized)
        mask: (B, 4760) - binary mask (1 = masked)
        face_index_maps: dict mapping face names to their index tensors
        loss_fn: "smooth_l1", "mse", or "l1"
        loss_beta: beta parameter for smooth_l1/huber loss (default 1.0)
        npho_weight: weight for npho channel loss
        time_weight: weight for time channel loss
        outer_fine: whether outer face uses fine grid
        outer_fine_pool: pooling kernel for outer fine grid
        track_mae_rmse: whether to compute MAE/RMSE metrics (adds extra reductions)
        npho_threshold: raw npho threshold for conditional time loss (default: DEFAULT_NPHO_THRESHOLD)
        npho_scale: normalization scale for npho (for de-normalizing)
        npho_scale2: normalization scale2 for npho
        use_npho_time_weight: whether to weight time loss by sqrt(npho)
        predict_channels: list of channels being predicted (["npho"] or ["npho", "time"])

    Returns:
        total_loss: scalar loss for backprop
        metrics: dict of per-face and per-channel losses and MAE/RMSE
    """
    device = original_values.device
    B = original_values.shape[0]

    # Detect predict_channels (default to ["npho", "time"] for legacy)
    if predict_channels is None:
        predict_channels = ["npho", "time"]
    predict_time = "time" in predict_channels
    pred_npho_idx = predict_channels.index("npho") if "npho" in predict_channels else 0
    pred_time_idx = predict_channels.index("time") if "time" in predict_channels else None

    # Conditional time loss threshold (convert raw threshold to normalized space)
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD
    # Create NphoTransform for threshold conversion and denormalization
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
    npho_threshold_norm = npho_transform.convert_threshold(npho_threshold)

    # Select loss function
    if loss_fn == "mse":
        loss_func = F.mse_loss
    elif loss_fn == "l1":
        loss_func = F.l1_loss
    else:  # smooth_l1 / huber
        loss_func = lambda p, t, reduction: F.smooth_l1_loss(p, t, reduction=reduction, beta=loss_beta)

    total_loss = torch.tensor(0.0, device=device)
    metrics = {}

    # Accumulators for aggregate metrics
    total_npho_loss = 0.0
    total_time_loss = 0.0

    # Accumulators for MAE/RMSE (raw errors)
    total_abs_npho = 0.0
    total_abs_time = 0.0
    total_sq_npho = 0.0
    total_sq_time = 0.0
    total_count = 0

    # Per-face MAE/RMSE accumulators
    face_abs_npho = {}
    face_abs_time = {}
    face_sq_npho = {}
    face_sq_time = {}
    face_count = {}

    outer_target = None
    if outer_fine and "outer" in results:
        from ..geom_utils import build_outer_fine_grid_tensor
        outer_target = build_outer_fine_grid_tensor(
            original_values, pool_kernel=outer_fine_pool
        )

    # Rectangular faces (inner, us, ds, outer)
    rect_faces = ["inner", "us", "ds", "outer"]
    for face_name in rect_faces:
        if face_name not in results:
            continue

        face_result = results[face_name]
        pred = face_result["pred"]  # (B, max_masked, 2)
        valid = face_result["valid"]  # (B, max_masked)

        # Check if this is sensor-level prediction (new outer face behavior)
        is_sensor_level = face_result.get("is_sensor_level", False)

        if not valid.any():
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0
            if track_mae_rmse:
                metrics[f"mae_{face_name}_npho"] = 0.0
                metrics[f"mae_{face_name}_time"] = 0.0
                metrics[f"rmse_{face_name}_npho"] = 0.0
                metrics[f"rmse_{face_name}_time"] = 0.0
            continue

        # Gather all valid positions across the batch
        batch_idx, pos_idx = valid.nonzero(as_tuple=True)

        if is_sensor_level:
            # Sensor-level outer face: sensor_ids contains flat sensor indices
            sensor_ids = face_result["sensor_ids"]  # (B, max_masked) - flat sensor IDs
            flat_idx_all = sensor_ids[batch_idx, pos_idx]  # (total,)
            gt_all = original_values[batch_idx, flat_idx_all, :]  # (total, 2)
        else:
            # Grid-level faces: indices contains (h, w) pairs
            indices = face_result["indices"]  # (B, max_masked, 2)
            h_all = indices[batch_idx, pos_idx, 0]
            w_all = indices[batch_idx, pos_idx, 1]

            if face_name == "outer" and outer_fine:
                # Legacy grid-level outer fine (shouldn't happen with new code, but keep for compatibility)
                gt_all = outer_target[batch_idx, :, h_all, w_all]  # (total, 2)
            else:
                idx_map = face_index_maps[face_name]  # (H, W) with flat indices
                flat_idx_all = idx_map[h_all, w_all]
                gt_all = original_values[batch_idx, flat_idx_all, :]  # (total, 2)

        pr_all = pred[batch_idx, pos_idx, :]  # (total, 2)

        # Conditional time loss threshold check (needed for both loss and MAE/RMSE)
        npho_gt_norm = gt_all[:, 0]  # (total,) - normalized npho values
        time_valid = npho_gt_norm > npho_threshold_norm  # (total,) - sensors with valid time

        counts_per_batch = valid.sum(dim=1)  # (B,)
        nonzero_mask = counts_per_batch > 0
        safe_counts = counts_per_batch.clamp_min(1)

        # Count time-valid sensors per batch (needed for time loss averaging)
        time_valid_per_elem = time_valid.float()
        time_counts_per_batch = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, time_valid_per_elem)

        # Compute losses only for channels with non-zero weights
        avg_npho = torch.tensor(0.0, device=device)
        avg_time = torch.tensor(0.0, device=device)

        if npho_weight > 0 and nonzero_mask.any():
            loss_npho_elem = loss_func(pr_all[:, pred_npho_idx], gt_all[:, 0], reduction="none")
            loss_npho_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, loss_npho_elem)
            loss_npho_means = loss_npho_sum / safe_counts
            avg_npho = loss_npho_means[nonzero_mask].mean()

        if predict_time and time_weight > 0 and nonzero_mask.any():
            loss_time_elem = loss_func(pr_all[:, pred_time_idx], gt_all[:, 1], reduction="none")

            # Apply npho weighting to time loss if enabled
            if use_npho_time_weight and time_valid.any():
                # De-normalize to get raw npho for weighting using NphoTransform
                raw_npho_approx = npho_transform.inverse(npho_gt_norm)
                npho_weights = torch.sqrt(raw_npho_approx.clamp(min=npho_threshold))
                npho_weights = npho_weights / (npho_weights[time_valid].mean() + 1e-8)
                loss_time_elem = loss_time_elem * npho_weights

            # Zero out time loss for sensors with npho below threshold
            loss_time_elem_valid = loss_time_elem.clone()
            loss_time_elem_valid[~time_valid] = 0.0

            time_safe_counts = time_counts_per_batch.clamp_min(1)
            loss_time_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, loss_time_elem_valid)
            loss_time_means = loss_time_sum / time_safe_counts

            time_nonzero_mask = time_counts_per_batch > 0
            if time_nonzero_mask.any():
                avg_time = loss_time_means[time_nonzero_mask].mean()

        # Apply npho loss weighting if enabled
        if npho_loss_weight_enabled and nonzero_mask.any() and npho_weight > 0:
            # Recompute npho loss with intensity-based weighting
            raw_npho_for_weight = npho_transform.inverse(gt_all[:, 0])
            weight_npho = (raw_npho_for_weight.clamp(min=1.0) + 1.0).pow(npho_loss_weight_alpha)
            weight_npho = weight_npho / weight_npho.mean()
            loss_npho_elem_weighted = loss_func(pr_all[:, pred_npho_idx], gt_all[:, 0], reduction="none") * weight_npho
            loss_npho_sum_weighted = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, loss_npho_elem_weighted)
            loss_npho_means_weighted = loss_npho_sum_weighted / safe_counts
            avg_npho = loss_npho_means_weighted[nonzero_mask].mean()

        if nonzero_mask.any():
            face_loss = npho_weight * avg_npho + time_weight * avg_time
            total_loss = total_loss + face_loss

            if track_metrics:
                metrics[f"loss_{face_name}"] = face_loss.item()
                metrics[f"loss_{face_name}_npho"] = (npho_weight * avg_npho).item()
                metrics[f"loss_{face_name}_time"] = (time_weight * avg_time).item()

                total_npho_loss += metrics[f"loss_{face_name}_npho"]
                total_time_loss += metrics[f"loss_{face_name}_time"]

        else:
            if track_metrics:
                metrics[f"loss_{face_name}"] = 0.0
                metrics[f"loss_{face_name}_npho"] = 0.0
                metrics[f"loss_{face_name}_time"] = 0.0
                if track_mae_rmse:
                    metrics[f"mae_{face_name}_npho"] = 0.0
                    metrics[f"mae_{face_name}_time"] = 0.0
                    metrics[f"rmse_{face_name}_npho"] = 0.0
                    metrics[f"rmse_{face_name}_time"] = 0.0
            continue

        if track_mae_rmse and track_metrics:
            # Accumulate raw errors for MAE/RMSE
            diff_npho = pr_all[:, pred_npho_idx] - gt_all[:, 0]

            # For npho: use all masked sensors
            abs_npho_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, diff_npho.abs())
            sq_npho_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, diff_npho ** 2)

            # For time: only use time-valid sensors (if predicting time)
            abs_time_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype)
            sq_time_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype)
            if predict_time:
                diff_time = pr_all[:, pred_time_idx] - gt_all[:, 1]
                diff_time_valid = diff_time.clone()
                diff_time_valid[~time_valid] = 0.0
                abs_time_sum = abs_time_sum.scatter_add(0, batch_idx, diff_time_valid.abs())
                sq_time_sum = sq_time_sum.scatter_add(0, batch_idx, diff_time_valid ** 2)

            face_abs_npho[face_name] = abs_npho_sum.sum().item()
            face_abs_time[face_name] = abs_time_sum.sum().item()
            face_sq_npho[face_name] = sq_npho_sum.sum().item()
            face_sq_time[face_name] = sq_time_sum.sum().item()
            face_count[face_name] = counts_per_batch.sum().item()

            # Track time-valid count separately
            time_valid_count = time_counts_per_batch.sum().item()

            # Metrics
            fc = max(face_count[face_name], 1)
            tc = max(time_valid_count, 1)
            metrics[f"mae_{face_name}_npho"] = face_abs_npho[face_name] / fc
            metrics[f"mae_{face_name}_time"] = face_abs_time[face_name] / tc
            metrics[f"rmse_{face_name}_npho"] = (face_sq_npho[face_name] / fc) ** 0.5
            metrics[f"rmse_{face_name}_time"] = (face_sq_time[face_name] / tc) ** 0.5

            # Accumulate for global MAE/RMSE (using time-valid count for time metrics)
            total_abs_npho += face_abs_npho[face_name]
            total_abs_time += face_abs_time[face_name]
            total_sq_npho += face_sq_npho[face_name]
            total_sq_time += face_sq_time[face_name]
            total_count += face_count[face_name]

    # Hex faces
    for face_name, hex_indices in [("top", face_index_maps["top"]), ("bot", face_index_maps["bot"])]:
        if face_name not in results:
            continue

        face_result = results[face_name]
        pred = face_result["pred"]  # (B, max_masked, 2)
        indices = face_result["indices"]  # (B, max_masked) - node indices
        valid = face_result["valid"]  # (B, max_masked)

        if not valid.any():
            if track_metrics:
                metrics[f"loss_{face_name}"] = 0.0
                metrics[f"loss_{face_name}_npho"] = 0.0
                metrics[f"loss_{face_name}_time"] = 0.0
                if track_mae_rmse:
                    metrics[f"mae_{face_name}_npho"] = 0.0
                    metrics[f"mae_{face_name}_time"] = 0.0
                    metrics[f"rmse_{face_name}_npho"] = 0.0
                    metrics[f"rmse_{face_name}_time"] = 0.0
            continue

        # Gather all valid positions across the batch
        batch_idx, pos_idx = valid.nonzero(as_tuple=True)
        node_idx_all = indices[batch_idx, pos_idx]
        flat_idx_all = hex_indices[node_idx_all]
        pr_all = pred[batch_idx, pos_idx, :]
        gt_all = original_values[batch_idx, flat_idx_all, :]

        # Conditional time loss threshold check (needed for both loss and MAE/RMSE)
        npho_gt_norm = gt_all[:, 0]  # (total,) - normalized npho values
        time_valid = npho_gt_norm > npho_threshold_norm  # (total,) - sensors with valid time

        counts_per_batch = valid.sum(dim=1)
        nonzero_mask = counts_per_batch > 0
        safe_counts = counts_per_batch.clamp_min(1)

        # Count time-valid sensors per batch (needed for time loss averaging)
        time_valid_per_elem = time_valid.float()
        time_counts_per_batch = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, time_valid_per_elem)

        # Compute losses only for channels with non-zero weights
        avg_npho = torch.tensor(0.0, device=device)
        avg_time = torch.tensor(0.0, device=device)

        if npho_weight > 0 and nonzero_mask.any():
            loss_npho_elem = loss_func(pr_all[:, pred_npho_idx], gt_all[:, 0], reduction="none")
            loss_npho_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, loss_npho_elem)
            loss_npho_means = loss_npho_sum / safe_counts
            avg_npho = loss_npho_means[nonzero_mask].mean()

        if predict_time and time_weight > 0 and nonzero_mask.any():
            loss_time_elem = loss_func(pr_all[:, pred_time_idx], gt_all[:, 1], reduction="none")

            # Apply npho weighting to time loss if enabled
            if use_npho_time_weight and time_valid.any():
                # De-normalize to get raw npho for weighting using NphoTransform
                raw_npho_approx = npho_transform.inverse(npho_gt_norm)
                npho_weights = torch.sqrt(raw_npho_approx.clamp(min=npho_threshold))
                npho_weights = npho_weights / (npho_weights[time_valid].mean() + 1e-8)
                loss_time_elem = loss_time_elem * npho_weights

            # Zero out time loss for sensors with npho below threshold
            loss_time_elem_valid = loss_time_elem.clone()
            loss_time_elem_valid[~time_valid] = 0.0

            time_safe_counts = time_counts_per_batch.clamp_min(1)
            loss_time_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, loss_time_elem_valid)
            loss_time_means = loss_time_sum / time_safe_counts

            time_nonzero_mask = time_counts_per_batch > 0
            if time_nonzero_mask.any():
                avg_time = loss_time_means[time_nonzero_mask].mean()

        # Apply npho loss weighting if enabled
        if npho_loss_weight_enabled and nonzero_mask.any() and npho_weight > 0:
            # Recompute npho loss with intensity-based weighting
            raw_npho_for_weight = npho_transform.inverse(gt_all[:, 0])
            weight_npho = (raw_npho_for_weight.clamp(min=1.0) + 1.0).pow(npho_loss_weight_alpha)
            weight_npho = weight_npho / weight_npho.mean()
            loss_npho_elem_weighted = loss_func(pr_all[:, pred_npho_idx], gt_all[:, 0], reduction="none") * weight_npho
            loss_npho_sum_weighted = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, loss_npho_elem_weighted)
            loss_npho_means_weighted = loss_npho_sum_weighted / safe_counts
            avg_npho = loss_npho_means_weighted[nonzero_mask].mean()

        if nonzero_mask.any():
            face_loss = npho_weight * avg_npho + time_weight * avg_time
            total_loss = total_loss + face_loss

            if track_metrics:
                metrics[f"loss_{face_name}"] = face_loss.item()
                metrics[f"loss_{face_name}_npho"] = (npho_weight * avg_npho).item()
                metrics[f"loss_{face_name}_time"] = (time_weight * avg_time).item()

                total_npho_loss += metrics[f"loss_{face_name}_npho"]
                total_time_loss += metrics[f"loss_{face_name}_time"]

        if track_mae_rmse and track_metrics:
            # Accumulate raw errors for MAE/RMSE
            diff = pr_all - gt_all
            diff_npho = pr_all[:, pred_npho_idx] - gt_all[:, 0]

            abs_npho_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, diff_npho.abs())
            sq_npho_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype).scatter_add(0, batch_idx, diff_npho ** 2)

            # For time: only use time-valid sensors (if predicting time)
            abs_time_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype)
            sq_time_sum = torch.zeros(B, device=pred.device, dtype=pr_all.dtype)
            if predict_time:
                diff_time = pr_all[:, pred_time_idx] - gt_all[:, 1]
                diff_time_valid = diff_time.clone()
                diff_time_valid[~time_valid] = 0.0
                abs_time_sum = abs_time_sum.scatter_add(0, batch_idx, diff_time_valid.abs())
                sq_time_sum = sq_time_sum.scatter_add(0, batch_idx, diff_time_valid ** 2)

            face_abs_npho[face_name] = abs_npho_sum.sum().item()
            face_abs_time[face_name] = abs_time_sum.sum().item()
            face_sq_npho[face_name] = sq_npho_sum.sum().item()
            face_sq_time[face_name] = sq_time_sum.sum().item()
            face_count[face_name] = counts_per_batch.sum().item()
            time_valid_count = time_counts_per_batch.sum().item()

        # Compute MAE/RMSE for this face
        if track_mae_rmse and track_metrics:
            fc = max(face_count[face_name], 1)
            tc = max(time_valid_count, 1)
            metrics[f"mae_{face_name}_npho"] = face_abs_npho[face_name] / fc
            metrics[f"mae_{face_name}_time"] = face_abs_time[face_name] / tc
            metrics[f"rmse_{face_name}_npho"] = (face_sq_npho[face_name] / fc) ** 0.5
            metrics[f"rmse_{face_name}_time"] = (face_sq_time[face_name] / tc) ** 0.5

            # Accumulate for global MAE/RMSE (using time-valid count for time metrics)
            total_abs_npho += face_abs_npho[face_name]
            total_abs_time += face_abs_time[face_name]
            total_sq_npho += face_sq_npho[face_name]
            total_sq_time += face_sq_time[face_name]
            total_count += face_count[face_name]

    if track_metrics:
        # Aggregate metrics (sum across faces, consistent with total_loss)
        metrics["loss_npho"] = total_npho_loss
        if predict_time:
            metrics["loss_time"] = total_time_loss
        metrics["total_loss"] = total_loss.item()

        # Global MAE/RMSE
        if track_mae_rmse:
            tc = max(total_count, 1)
            metrics["mae_npho"] = total_abs_npho / tc
            metrics["rmse_npho"] = (total_sq_npho / tc) ** 0.5
            if predict_time:
                metrics["mae_time"] = total_abs_time / tc
                metrics["rmse_time"] = (total_sq_time / tc) ** 0.5

        # Mask statistics
        metrics["n_masked_total"] = total_count
    else:
        metrics["total_loss"] = total_loss.item()

    return total_loss, metrics


def run_epoch_inpainter(
    model,
    optimizer,
    device,
    train_files: List[str],
    tree_name: str,
    batch_size: int,
    step_size: int,
    mask_ratio: float = 0.05,
    npho_branch: str = "npho",
    time_branch: str = "relative_time",
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    time_scale: float = DEFAULT_TIME_SCALE,
    time_shift: float = DEFAULT_TIME_SHIFT,
    sentinel_time: float = DEFAULT_SENTINEL_TIME,
    loss_fn: str = "smooth_l1",
    loss_beta: float = 1.0,
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
    track_mae_rmse: bool = True,
    dataloader_workers: int = 0,
    dataset_workers: int = 8,
    prefetch_factor: int = 2,
    grad_accum_steps: int = 1,
    track_metrics: bool = True,
    npho_threshold: float = None,
    use_npho_time_weight: bool = True,
    profile: bool = False,
    log_invalid_npho: bool = True,
    use_fast_forward: Optional[bool] = None,
    npho_scheme: str = "log1p",
    npho_loss_weight_enabled: bool = False,
    npho_loss_weight_alpha: float = 0.5,
    intensity_reweighter: Optional[object] = None,
    no_sync_ctx=None,
    sentinel_npho: float = -1.0,
) -> Dict[str, float]:
    """
    Run one training epoch for inpainter.

    Args:
        ...
        use_fast_forward: If True, use forward_training() which returns fixed-size outputs.
                         If False, use forward() with per-face results (slower due to extraction overhead).
                         If None (default), auto-selects True since vectorized OuterSensorInpaintingHead
                         computes all sensors regardless of path.
        no_sync_ctx: model.no_sync method for DDP gradient accumulation (None for single-GPU).

    Returns:
        metrics: dict of averaged metrics
    """
    model.train()

    # Get predict_channels from model for conditional loss computation
    predict_channels = getattr(model, 'predict_channels', ['npho', 'time'])

    # Auto-select fast forward: always True because the vectorized OuterSensorInpaintingHead
    # computes all sensors regardless of path (forward() calls _compute_all_sensor_preds_vectorized
    # then extracts masked, so it's actually slower than forward_full which skips extraction)
    if use_fast_forward is None:
        use_fast_forward = True

    # Build face index maps
    from ..geom_defs import INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP, OUTER_COARSE_FULL_INDEX_MAP
    from ..geom_defs import TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows

    face_index_maps = {
        "inner": torch.from_numpy(INNER_INDEX_MAP).to(device),
        "us": torch.from_numpy(US_INDEX_MAP).to(device),
        "ds": torch.from_numpy(DS_INDEX_MAP).to(device),
        "outer": torch.from_numpy(OUTER_COARSE_FULL_INDEX_MAP).to(device),
        "top": torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long().to(device),
        "bot": torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long().to(device),
    }

    # Get outer fine config from model
    outer_fine = getattr(model.encoder, "outer_fine", False)
    outer_fine_pool = getattr(model.encoder, "outer_fine_pool", None)

    # Convert npho_threshold to normalized space for stratified masking
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
    npho_threshold_norm = npho_transform.convert_threshold(npho_threshold)

    # Accumulate metrics
    metric_sums = {}
    num_batches = 0

    # Track actual mask ratio (randomly-masked / valid sensors)
    # Use tensors to avoid GPU-CPU sync every batch (only .item() at epoch end)
    total_randomly_masked = torch.tensor(0, dtype=torch.long, device=device)
    total_valid_sensors = torch.tensor(0, dtype=torch.long, device=device)

    accum_step = 0
    optimizer.zero_grad(set_to_none=True)
    grad_accum_steps = max(int(grad_accum_steps), 1)

    total_loss_sum = 0.0
    loss_batches = 0

    # Initialize profiler
    profiler = SimpleProfiler(enabled=profile, sync_cuda=True)

    # Cumulative dataset I/O stats (since we create a new dataset per file)
    cumulative_io_stats = {
        "io_time": 0.0,
        "process_time": 0.0,
        "batch_time": 0.0,
        "chunk_count": 0,
        "event_count": 0,
        "file_count": 0,
    }

    for root_file in train_files:
        profiler.start("data_load")
        # Note: dataset I/O profiling requires dataloader_workers=0 for accurate stats
        dataset = XECStreamingDataset(
            root_files=root_file,
            tree_name=tree_name,
            batch_size=batch_size,  # Pre-batch in dataset
            step_size=step_size,
            npho_branch=npho_branch,
            time_branch=time_branch,
            npho_scale=npho_scale,
            npho_scale2=npho_scale2,
            npho_scheme=npho_scheme,
            time_scale=time_scale,
            time_shift=time_shift,
            sentinel_time=sentinel_time,
            num_workers=dataset_workers,
            log_invalid_npho=log_invalid_npho,
            load_truth_branches=False,  # Inpainter doesn't need truth branches
            profile=profile,
            sentinel_npho=sentinel_npho,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=None,  # Dataset yields pre-batched tensors
            num_workers=dataloader_workers,
            pin_memory=True,
            persistent_workers=(dataloader_workers > 0),
            prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
        )

        profiler.stop()  # data_load

        for batch in loader:
            profiler.start("gpu_transfer")
            if isinstance(batch, dict):
                x_batch = batch["x"]
            else:
                x_batch = batch[0]
            x_batch = x_batch.to(device, non_blocking=True)
            profiler.stop()

            # Forward with AMP (pass npho_threshold_norm for stratified masking)
            profiler.start("forward")
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                if use_fast_forward:
                    # Fast path: fixed-size outputs, simpler loss computation
                    pred_all, original_values, mask = model.forward_training(
                        x_batch, mask_ratio=mask_ratio, npho_threshold_norm=npho_threshold_norm
                    )
                else:
                    # Original path: variable-size outputs, per-face loss
                    results, original_values, mask = model(
                        x_batch, mask_ratio=mask_ratio, npho_threshold_norm=npho_threshold_norm
                    )
            profiler.stop()

            # Track actual mask ratio (no .item() to avoid GPU-CPU sync)
            # Already-invalid sensors are NOT in mask
            if "time" in predict_channels:
                already_invalid = (original_values[:, :, 1] == sentinel_time)  # (B, N)
            else:
                already_invalid = (original_values[:, :, 0] == sentinel_npho)  # (B, N)
            total_valid_sensors += (~already_invalid).sum()
            total_randomly_masked += mask.sum().long()

            profiler.start("loss_compute")
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                if use_fast_forward:
                    # Fast path: simple flat loss computation
                    loss, metrics = compute_inpainting_loss_flat(
                        pred_all, original_values, mask,
                        loss_fn=loss_fn,
                        loss_beta=loss_beta,
                        npho_weight=npho_weight,
                        time_weight=time_weight,
                        npho_threshold=npho_threshold,
                        npho_scale=npho_scale,
                        npho_scale2=npho_scale2,
                        use_npho_time_weight=use_npho_time_weight,
                        track_metrics=track_metrics,
                        predict_channels=predict_channels,
                        npho_scheme=npho_scheme,
                        npho_loss_weight_enabled=npho_loss_weight_enabled,
                        npho_loss_weight_alpha=npho_loss_weight_alpha,
                    )
                else:
                    # Original path: per-face loss computation
                    loss, metrics = compute_inpainting_loss(
                        results, original_values, mask,
                        face_index_maps,
                        loss_fn=loss_fn,
                        loss_beta=loss_beta,
                        npho_weight=npho_weight,
                        time_weight=time_weight,
                        outer_fine=outer_fine,
                        outer_fine_pool=outer_fine_pool,
                        track_mae_rmse=track_mae_rmse,
                        track_metrics=track_metrics,
                        npho_threshold=npho_threshold,
                        npho_scale=npho_scale,
                        npho_scale2=npho_scale2,
                        use_npho_time_weight=use_npho_time_weight,
                        predict_channels=predict_channels,
                        npho_scheme=npho_scheme,
                        npho_loss_weight_enabled=npho_loss_weight_enabled,
                        npho_loss_weight_alpha=npho_loss_weight_alpha,
                    )
            profiler.stop()

            # Backward
            total_loss_sum += loss.item()
            loss_batches += 1

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

            profiler.start("optimizer")
            if scaler is not None:
                if (accum_step + 1) % grad_accum_steps == 0:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                if (accum_step + 1) % grad_accum_steps == 0:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            profiler.stop()

            accum_step += 1

            # Accumulate metrics
            if track_metrics:
                for key, value in metrics.items():
                    if key not in metric_sums:
                        metric_sums[key] = 0.0
                    metric_sums[key] += value
            num_batches += 1

        # Accumulate dataset I/O stats from this file
        if profile and dataloader_workers == 0:
            file_stats = dataset.get_profile_stats()
            for key in cumulative_io_stats:
                cumulative_io_stats[key] += file_stats.get(key, 0)

        profiler.start("data_load")  # For next file/iteration

    # Stop any pending timer
    profiler.stop()

    # Print profiler report if enabled
    if profile:
        print(profiler.report())
        # Print cumulative dataset I/O breakdown (only accurate when dataloader_workers=0)
        if dataloader_workers == 0 and cumulative_io_stats["event_count"] > 0:
            total_time = cumulative_io_stats["io_time"] + cumulative_io_stats["process_time"] + cumulative_io_stats["batch_time"]
            lines = ["[Dataset Profile] Cumulative I/O breakdown:"]
            lines.append(f"  Files: {cumulative_io_stats['file_count']}, Chunks: {cumulative_io_stats['chunk_count']}, Events: {cumulative_io_stats['event_count']:,}")
            timing_items = [
                ("I/O (uproot)", cumulative_io_stats["io_time"]),
                ("CPU (normalize)", cumulative_io_stats["process_time"]),
                ("Batch (numpy→torch)", cumulative_io_stats["batch_time"]),
            ]
            max_name_len = max(len(name) for name, _ in timing_items)
            for name, t in timing_items:
                pct = 100 * t / total_time if total_time > 0 else 0
                lines.append(f"  {name:<{max_name_len}}  {t:>7.2f}s  ({pct:>5.1f}%)")
            lines.append(f"  {'TOTAL':<{max_name_len}}  {total_time:>7.2f}s")
            if total_time > 0:
                throughput = cumulative_io_stats["event_count"] / total_time
                lines.append(f"  Throughput: {throughput:,.0f} events/s")
            print("\n".join(lines))

    # Final optimizer step if grads remain from incomplete accumulation.
    # Previous backward passes used no_sync(), so model gradients are local-only.
    if (accum_step % grad_accum_steps) != 0:
        if dist.is_initialized():
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
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
        optimizer.zero_grad(set_to_none=True)

    # Average metrics
    if track_metrics:
        avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}
    else:
        avg_metrics = {}

    # Always report average total_loss
    avg_metrics["total_loss"] = total_loss_sum / max(1, loss_batches)

    # Actual mask ratio (randomly-masked / valid sensors)
    # Convert tensor accumulators to Python scalars
    total_valid_sensors_val = total_valid_sensors.item()
    total_randomly_masked_val = total_randomly_masked.item()

    if total_valid_sensors_val > 0:
        avg_metrics["actual_mask_ratio"] = total_randomly_masked_val / total_valid_sensors_val
    else:
        avg_metrics["actual_mask_ratio"] = 0.0

    return avg_metrics


def run_eval_inpainter(
    model,
    device,
    val_files: List[str],
    tree_name: str,
    batch_size: int,
    step_size: int,
    mask_ratio: float = 0.05,
    npho_branch: str = "npho",
    time_branch: str = "relative_time",
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    time_scale: float = DEFAULT_TIME_SCALE,
    time_shift: float = DEFAULT_TIME_SHIFT,
    sentinel_time: float = DEFAULT_SENTINEL_TIME,
    loss_fn: str = "smooth_l1",
    loss_beta: float = 1.0,
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
    collect_predictions: bool = False,
    prediction_writer: Optional[Callable[[List[Dict]], None]] = None,
    track_mae_rmse: bool = True,
    dataloader_workers: int = 0,
    dataset_workers: int = 8,
    prefetch_factor: int = 2,
    npho_threshold: float = None,
    use_npho_time_weight: bool = True,
    profile: bool = False,
    log_invalid_npho: bool = True,
    use_fast_forward: Optional[bool] = None,
    npho_scheme: str = "log1p",
    npho_loss_weight_enabled: bool = False,
    npho_loss_weight_alpha: float = 0.5,
    sentinel_npho: float = -1.0,
) -> Dict[str, float]:
    """
    Run evaluation for inpainter.

    Args:
        collect_predictions: If True, collect per-sensor predictions for ROOT output
        prediction_writer: optional callable to stream predictions per batch (avoids keeping everything in memory)
        use_fast_forward: If True, use forward_training() which returns fixed-size outputs.
                         If False, use forward() with per-face results (needed for collect_predictions).
                         If None (default), auto-selects True unless collect_predictions is True.

    Returns:
        metrics: dict of averaged metrics
        predictions: (optional) list of prediction dicts if collect_predictions=True
    """
    model.eval()

    # Get predict_channels from model for conditional loss computation
    predict_channels = getattr(model, 'predict_channels', ['npho', 'time'])

    # Auto-select fast forward: always True (see run_epoch_inpainter comment)
    # Exception: collect_predictions requires the results dict from standard forward
    # (but cross-attention always uses fast path since it has no per-face heads)
    head_type = getattr(model, 'head_type', 'per_face')
    if use_fast_forward is None:
        if head_type == "cross_attention":
            use_fast_forward = True
        else:
            use_fast_forward = not collect_predictions

    from ..geom_defs import INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP, OUTER_COARSE_FULL_INDEX_MAP
    from ..geom_defs import TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows

    face_index_maps = {
        "inner": torch.from_numpy(INNER_INDEX_MAP).to(device),
        "us": torch.from_numpy(US_INDEX_MAP).to(device),
        "ds": torch.from_numpy(DS_INDEX_MAP).to(device),
        "outer": torch.from_numpy(OUTER_COARSE_FULL_INDEX_MAP).to(device),
        "top": torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long().to(device),
        "bot": torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long().to(device),
    }

    # Numpy versions for prediction collection
    face_index_maps_np = {
        "inner": INNER_INDEX_MAP,
        "us": US_INDEX_MAP,
        "ds": DS_INDEX_MAP,
        "outer": OUTER_COARSE_FULL_INDEX_MAP,
        "top": flatten_hex_rows(TOP_HEX_ROWS),
        "bot": flatten_hex_rows(BOTTOM_HEX_ROWS),
    }

    outer_fine = getattr(model.encoder, "outer_fine", False)
    outer_fine_pool = getattr(model.encoder, "outer_fine_pool", None)

    # Convert npho_threshold to normalized space for stratified masking
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
    npho_threshold_norm = npho_transform.convert_threshold(npho_threshold)

    metric_sums = {}
    num_batches = 0

    # Track actual mask ratio (randomly-masked / valid sensors)
    # Use tensors to avoid GPU-CPU sync every batch (only .item() at epoch end)
    total_randomly_masked = torch.tensor(0, dtype=torch.long, device=device)
    total_valid_sensors = torch.tensor(0, dtype=torch.long, device=device)

    # For collecting predictions (per-sensor level)
    all_predictions = [] if (collect_predictions and prediction_writer is None) else None

    # Initialize profiler
    profiler = SimpleProfiler(enabled=profile, sync_cuda=True)

    # Cumulative dataset I/O stats (since we create a new dataset per file)
    cumulative_io_stats = {
        "io_time": 0.0,
        "process_time": 0.0,
        "batch_time": 0.0,
        "chunk_count": 0,
        "event_count": 0,
        "file_count": 0,
    }

    with torch.no_grad():
        for root_file in val_files:
            profiler.start("data_load")
            # Note: dataset I/O profiling requires dataloader_workers=0 for accurate stats
            dataset = XECStreamingDataset(
                root_files=root_file,
                tree_name=tree_name,
                batch_size=batch_size,  # Pre-batch in dataset
                step_size=step_size,
                npho_branch=npho_branch,
                time_branch=time_branch,
                npho_scale=npho_scale,
                npho_scale2=npho_scale2,
                npho_scheme=npho_scheme,
                time_scale=time_scale,
                time_shift=time_shift,
                sentinel_time=sentinel_time,
                num_workers=dataset_workers,
                log_invalid_npho=log_invalid_npho,
                load_truth_branches=False,  # Inpainter doesn't need truth branches
                profile=profile,
                sentinel_npho=sentinel_npho,
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=None,  # Dataset yields pre-batched tensors
                num_workers=dataloader_workers,
                pin_memory=True,
                persistent_workers=(dataloader_workers > 0),
                prefetch_factor=prefetch_factor if dataloader_workers > 0 else None,
            )
            profiler.stop()  # data_load

            for batch in loader:
                batch_predictions = [] if collect_predictions else None
                event_base = num_batches * batch_size

                # Extract input and metadata from batch
                # Dataset returns (x, targets_dict) where targets_dict has "run" and "event"
                profiler.start("gpu_transfer")
                if isinstance(batch, dict):
                    x_batch = batch["x"]
                    run_numbers = batch.get("run", None)
                    event_numbers = batch.get("event", None)
                elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                    x_batch = batch[0]
                    targets = batch[1] if isinstance(batch[1], dict) else {}
                    run_numbers = targets.get("run", None)
                    event_numbers = targets.get("event", None)
                else:
                    x_batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                    run_numbers = None
                    event_numbers = None

                x_batch = x_batch.to(device, non_blocking=True)
                profiler.stop()

                # Convert run/event to numpy for later use
                run_numbers_np = run_numbers.cpu().numpy() if run_numbers is not None else None
                event_numbers_np = event_numbers.cpu().numpy() if event_numbers is not None else None

                profiler.start("forward")
                # Use fast path only if not collecting predictions (which needs the results dict)
                # Exception: cross-attention always uses fast path (no per-face results dict)
                use_fast = use_fast_forward and (not collect_predictions or head_type == "cross_attention")
                with torch.amp.autocast('cuda', enabled=True):
                    if use_fast:
                        pred_all, original_values, mask = model.forward_training(
                            x_batch, mask_ratio=mask_ratio, npho_threshold_norm=npho_threshold_norm
                        )
                    else:
                        results, original_values, mask = model(
                            x_batch, mask_ratio=mask_ratio, npho_threshold_norm=npho_threshold_norm
                        )
                profiler.stop()

                # Track actual mask ratio (no .item() to avoid GPU-CPU sync)
                if "time" in predict_channels:
                    already_invalid = (original_values[:, :, 1] == sentinel_time)  # (B, N)
                else:
                    already_invalid = (original_values[:, :, 0] == sentinel_npho)  # (B, N)
                total_valid_sensors += (~already_invalid).sum()
                total_randomly_masked += mask.sum().long()

                profiler.start("loss_compute")
                with torch.amp.autocast('cuda', enabled=True):
                    if use_fast:
                        _, metrics = compute_inpainting_loss_flat(
                            pred_all, original_values, mask,
                            loss_fn=loss_fn,
                            loss_beta=loss_beta,
                            npho_weight=npho_weight,
                            time_weight=time_weight,
                            npho_threshold=npho_threshold,
                            npho_scale=npho_scale,
                            npho_scale2=npho_scale2,
                            use_npho_time_weight=use_npho_time_weight,
                            track_metrics=True,
                            predict_channels=predict_channels,
                            npho_scheme=npho_scheme,
                            npho_loss_weight_enabled=npho_loss_weight_enabled,
                            npho_loss_weight_alpha=npho_loss_weight_alpha,
                        )
                    else:
                        _, metrics = compute_inpainting_loss(
                            results, original_values, mask,
                            face_index_maps,
                            loss_fn=loss_fn,
                            loss_beta=loss_beta,
                            npho_weight=npho_weight,
                            time_weight=time_weight,
                            outer_fine=outer_fine,
                            outer_fine_pool=outer_fine_pool,
                            track_mae_rmse=track_mae_rmse,
                            track_metrics=True,
                            npho_threshold=npho_threshold,
                            npho_scale=npho_scale,
                            npho_scale2=npho_scale2,
                            use_npho_time_weight=use_npho_time_weight,
                            predict_channels=predict_channels,
                            npho_scheme=npho_scheme,
                            npho_loss_weight_enabled=npho_loss_weight_enabled,
                            npho_loss_weight_alpha=npho_loss_weight_alpha,
                        )
                    if not track_mae_rmse:
                        metrics = {k: v for k, v in metrics.items() if not (k.startswith("mae_") or k.startswith("rmse_"))}
                profiler.stop()  # loss_compute

                for key, value in metrics.items():
                    if key not in metric_sums:
                        metric_sums[key] = 0.0
                    metric_sums[key] += value

                # Collect predictions at sensor level
                if collect_predictions:
                    profiler.start("pred_collect")
                    B = x_batch.shape[0]
                    original_np = original_values.cpu().numpy()

                    if head_type == "cross_attention":
                        # Flat prediction collection from pred_all (B, 4760, C)
                        pred_np = pred_all.cpu().numpy()
                        mask_np = mask.cpu().numpy()
                        # Map sensor_id → face name
                        _face_id_to_name = {0: "inner", 1: "us", 2: "ds", 3: "outer", 4: "top", 5: "bot"}
                        from ..sensor_geometry import build_sensor_face_ids
                        _sensor_face_ids = build_sensor_face_ids()

                        for b in range(B):
                            masked_sensors = np.where(mask_np[b] > 0.5)[0]
                            for sensor_id in masked_sensors:
                                face_name = _face_id_to_name[int(_sensor_face_ids[sensor_id])]
                                pred_dict = {
                                    "event_idx": event_base + b,
                                    "run_number": int(run_numbers_np[b]) if run_numbers_np is not None else -1,
                                    "event_number": int(event_numbers_np[b]) if event_numbers_np is not None else -1,
                                    "sensor_id": int(sensor_id),
                                    "face": face_name,
                                    "truth_npho": float(original_np[b, sensor_id, 0]),
                                    "truth_time": float(original_np[b, sensor_id, 1]),
                                    "pred_npho": float(pred_np[b, sensor_id, 0]),
                                    "pred_time": float(pred_np[b, sensor_id, 1]) if pred_np.shape[2] > 1 else 0.0,
                                }
                                batch_predictions.append(pred_dict)
                    else:
                        # Per-face prediction collection from results dict
                        outer_target_np = None
                        outer_grid_w = None
                        # Only build outer_target for legacy grid-level mode
                        if outer_fine and "outer" in results and not results["outer"].get("is_sensor_level", False):
                            from ..geom_utils import build_outer_fine_grid_tensor
                            outer_target = build_outer_fine_grid_tensor(
                                original_values, pool_kernel=outer_fine_pool
                            )
                            outer_target_np = outer_target.permute(0, 2, 3, 1).cpu().numpy()
                            outer_grid_w = outer_target_np.shape[2]

                        # Process each face and collect predictions
                        for face_name in ["inner", "us", "ds", "outer", "top", "bot"]:
                            if face_name not in results:
                                continue

                            face_result = results[face_name]
                            pred = face_result["pred"].cpu().numpy()  # (B, max_masked, 2)
                            valid = face_result["valid"].cpu().numpy()  # (B, max_masked)

                            # Check if this is sensor-level prediction (new outer face behavior)
                            is_sensor_level = face_result.get("is_sensor_level", False)

                            if face_name in ["top", "bot"]:
                                indices = face_result["indices"].cpu().numpy()  # (B, max_masked)
                                hex_indices = face_index_maps_np[face_name]

                                for b in range(B):
                                    n_valid = int(valid[b].sum())
                                    if n_valid == 0:
                                        continue

                                    node_idx = indices[b, :n_valid]
                                    flat_idx = hex_indices[node_idx]

                                    for i in range(n_valid):
                                        sensor_id = int(flat_idx[i])
                                        pred_dict = {
                                            "event_idx": event_base + b,
                                            "run_number": int(run_numbers_np[b]) if run_numbers_np is not None else -1,
                                            "event_number": int(event_numbers_np[b]) if event_numbers_np is not None else -1,
                                            "sensor_id": sensor_id,
                                            "face": face_name,
                                            "truth_npho": float(original_np[b, sensor_id, 0]),
                                            "truth_time": float(original_np[b, sensor_id, 1]),
                                            "pred_npho": float(pred[b, i, 0]),
                                            "pred_time": float(pred[b, i, 1]),
                                        }
                                        batch_predictions.append(pred_dict)
                            elif is_sensor_level:
                                # Sensor-level outer face: sensor_ids contains flat sensor indices
                                sensor_ids = face_result["sensor_ids"].cpu().numpy()  # (B, max_masked)

                                for b in range(B):
                                    n_valid = int(valid[b].sum())
                                    if n_valid == 0:
                                        continue

                                    for i in range(n_valid):
                                        sensor_id = int(sensor_ids[b, i])
                                        pred_dict = {
                                            "event_idx": event_base + b,
                                            "run_number": int(run_numbers_np[b]) if run_numbers_np is not None else -1,
                                            "event_number": int(event_numbers_np[b]) if event_numbers_np is not None else -1,
                                            "sensor_id": sensor_id,
                                            "face": face_name,
                                            "truth_npho": float(original_np[b, sensor_id, 0]),
                                            "truth_time": float(original_np[b, sensor_id, 1]),
                                            "pred_npho": float(pred[b, i, 0]),
                                            "pred_time": float(pred[b, i, 1]),
                                        }
                                        batch_predictions.append(pred_dict)
                            else:
                                # Grid-level faces: indices contains (h, w) pairs
                                indices = face_result["indices"].cpu().numpy()  # (B, max_masked, 2)
                                if face_name == "outer" and outer_fine:
                                    if outer_target_np is None:
                                        continue

                                for b in range(B):
                                    n_valid = int(valid[b].sum())
                                    if n_valid == 0:
                                        continue

                                    h_idx = indices[b, :n_valid, 0]
                                    w_idx = indices[b, :n_valid, 1]
                                    if face_name == "outer" and outer_fine:
                                        # Legacy grid-level outer fine
                                        truth_vals = outer_target_np[b, h_idx, w_idx]
                                        for i in range(n_valid):
                                            sensor_id = int(h_idx[i] * outer_grid_w + w_idx[i])
                                            pred_dict = {
                                                "event_idx": event_base + b,
                                                "run_number": int(run_numbers_np[b]) if run_numbers_np is not None else -1,
                                                "event_number": int(event_numbers_np[b]) if event_numbers_np is not None else -1,
                                                "sensor_id": sensor_id,
                                                "face": face_name,
                                                "truth_npho": float(truth_vals[i, 0]),
                                                "truth_time": float(truth_vals[i, 1]),
                                                "pred_npho": float(pred[b, i, 0]),
                                                "pred_time": float(pred[b, i, 1]),
                                            }
                                            batch_predictions.append(pred_dict)
                                        continue

                                    idx_map = face_index_maps_np[face_name]
                                    flat_idx = idx_map[h_idx, w_idx]

                                    for i in range(n_valid):
                                        sensor_id = int(flat_idx[i])
                                        pred_dict = {
                                            "event_idx": event_base + b,
                                            "run_number": int(run_numbers_np[b]) if run_numbers_np is not None else -1,
                                            "event_number": int(event_numbers_np[b]) if event_numbers_np is not None else -1,
                                            "sensor_id": sensor_id,
                                            "face": face_name,
                                            "truth_npho": float(original_np[b, sensor_id, 0]),
                                            "truth_time": float(original_np[b, sensor_id, 1]),
                                            "pred_npho": float(pred[b, i, 0]),
                                            "pred_time": float(pred[b, i, 1]),
                                        }
                                        batch_predictions.append(pred_dict)

                if collect_predictions:
                    if prediction_writer is not None:
                        prediction_writer(batch_predictions)
                    else:
                        all_predictions.extend(batch_predictions)
                    profiler.stop()  # pred_collect

                num_batches += 1

            # Accumulate dataset I/O stats from this file
            if profile and dataloader_workers == 0:
                file_stats = dataset.get_profile_stats()
                for key in cumulative_io_stats:
                    cumulative_io_stats[key] += file_stats.get(key, 0)

            profiler.start("data_load")  # For next file/iteration

    # Stop any pending timer
    profiler.stop()

    # Print profiler report if enabled
    if profile:
        print(profiler.report("Validation timing breakdown"))
        # Print cumulative dataset I/O breakdown (only accurate when dataloader_workers=0)
        if dataloader_workers == 0 and cumulative_io_stats["event_count"] > 0:
            total_time = cumulative_io_stats["io_time"] + cumulative_io_stats["process_time"] + cumulative_io_stats["batch_time"]
            lines = ["[Dataset Profile] Cumulative I/O breakdown:"]
            lines.append(f"  Files: {cumulative_io_stats['file_count']}, Chunks: {cumulative_io_stats['chunk_count']}, Events: {cumulative_io_stats['event_count']:,}")
            timing_items = [
                ("I/O (uproot)", cumulative_io_stats["io_time"]),
                ("CPU (normalize)", cumulative_io_stats["process_time"]),
                ("Batch (numpy→torch)", cumulative_io_stats["batch_time"]),
            ]
            max_name_len = max(len(name) for name, _ in timing_items)
            for name, t in timing_items:
                pct = 100 * t / total_time if total_time > 0 else 0
                lines.append(f"  {name:<{max_name_len}}  {t:>7.2f}s  ({pct:>5.1f}%)")
            lines.append(f"  {'TOTAL':<{max_name_len}}  {total_time:>7.2f}s")
            if total_time > 0:
                throughput = cumulative_io_stats["event_count"] / total_time
                lines.append(f"  Throughput: {throughput:,.0f} events/s")
            print("\n".join(lines))

    avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}

    # Actual mask ratio (randomly-masked / valid sensors)
    # Convert tensor accumulators to Python scalars
    total_valid_sensors_val = total_valid_sensors.item()
    total_randomly_masked_val = total_randomly_masked.item()

    if total_valid_sensors_val > 0:
        avg_metrics["actual_mask_ratio"] = total_randomly_masked_val / total_valid_sensors_val
    else:
        avg_metrics["actual_mask_ratio"] = 0.0

    if not track_mae_rmse:
        avg_metrics = {k: v for k, v in avg_metrics.items() if not (k.startswith("mae_") or k.startswith("rmse_"))}

    if collect_predictions:
        return avg_metrics, all_predictions
    return avg_metrics


class RootPredictionWriter:
    """
    Streaming writer that appends prediction batches to a ROOT TTree.

    Stores:
    - Per-sensor predictions (event_idx, run_number, event_number, sensor_id, face, truth/pred values)
    - Normalization metadata in a separate 'metadata' tree
    """

    def __init__(self, save_path: str, epoch: int, run_id: str = None, tree_name: str = "tree",
                 npho_scale: float = None, npho_scale2: float = None,
                 time_scale: float = None, time_shift: float = None,
                 sentinel_time: float = None,
                 predict_channels: List[str] = None,
                 npho_scheme: str = None):
        import os
        import uproot

        os.makedirs(save_path, exist_ok=True)

        self.filename = f"inpainter_predictions_epoch_{epoch}.root"
        self.filepath = os.path.join(save_path, self.filename)
        self.run_id = run_id
        self._file = uproot.recreate(self.filepath)

        # Store normalization factors
        self.npho_scale = npho_scale
        self.npho_scale2 = npho_scale2
        self.time_scale = time_scale
        self.time_shift = time_shift
        self.sentinel_time = sentinel_time

        # Store predict_channels
        self.predict_channels = predict_channels if predict_channels is not None else ['npho', 'time']
        self.predict_time = 'time' in self.predict_channels

        # Store npho_scheme
        self.npho_scheme = npho_scheme if npho_scheme is not None else "log1p"

        # Main predictions tree
        branch_types = {
            "event_idx": np.int32,      # Batch-based index (for backwards compatibility)
            "run_number": np.int64,     # Actual run number from input file
            "event_number": np.int64,   # Actual event number from input file
            "sensor_id": np.int32,
            "face": np.int32,
            "truth_npho": np.float32,
            "truth_time": np.float32,
            "pred_npho": np.float32,
            "error_npho": np.float32,
        }
        # Only include time predictions if time was predicted
        if self.predict_time:
            branch_types["pred_time"] = np.float32
            branch_types["error_time"] = np.float32
        if self.run_id is not None:
            branch_types["run_id"] = str
        self._tree = self._file.mktree(tree_name, branch_types)
        self.count = 0

        # Write normalization metadata tree (single entry)
        self._write_metadata()

    def _write_metadata(self):
        """Write normalization factors and model config to a separate metadata tree."""
        # Use a dict-based approach for metadata that mixes types
        metadata = {
            "predict_channels": np.array([','.join(self.predict_channels)], dtype='U32'),
            "npho_scale": np.array([self.npho_scale if self.npho_scale is not None else np.nan], dtype=np.float64),
            "npho_scale2": np.array([self.npho_scale2 if self.npho_scale2 is not None else np.nan], dtype=np.float64),
            "time_scale": np.array([self.time_scale if self.time_scale is not None else np.nan], dtype=np.float64),
            "time_shift": np.array([self.time_shift if self.time_shift is not None else np.nan], dtype=np.float64),
            "sentinel_time": np.array([self.sentinel_time if self.sentinel_time is not None else np.nan], dtype=np.float64),
            "npho_scheme": np.array([self.npho_scheme], dtype='U32'),
        }
        self._file["metadata"] = metadata

    def write(self, predictions: List[Dict]):
        if not predictions:
            return

        n = len(predictions)
        self.count += n

        event_idx = np.array([p["event_idx"] for p in predictions], dtype=np.int32)
        run_number = np.array([p.get("run_number", -1) for p in predictions], dtype=np.int64)
        event_number = np.array([p.get("event_number", -1) for p in predictions], dtype=np.int64)
        sensor_id = np.array([p["sensor_id"] for p in predictions], dtype=np.int32)

        face_map = {"inner": 0, "us": 1, "ds": 2, "outer": 3, "top": 4, "bot": 5}
        face = np.array([face_map.get(p["face"], -1) for p in predictions], dtype=np.int32)

        truth_npho = np.array([p["truth_npho"] for p in predictions], dtype=np.float32)
        truth_time = np.array([p["truth_time"] for p in predictions], dtype=np.float32)
        pred_npho = np.array([p["pred_npho"] for p in predictions], dtype=np.float32)
        error_npho = pred_npho - truth_npho

        branches = {
            "event_idx": event_idx,
            "run_number": run_number,
            "event_number": event_number,
            "sensor_id": sensor_id,
            "face": face,
            "truth_npho": truth_npho,
            "truth_time": truth_time,
            "pred_npho": pred_npho,
            "error_npho": error_npho,
        }

        # Only include time predictions if time was predicted
        if self.predict_time:
            pred_time = np.array([p["pred_time"] for p in predictions], dtype=np.float32)
            error_time = pred_time - truth_time
            branches["pred_time"] = pred_time
            branches["error_time"] = error_time

        if self.run_id is not None:
            branches["run_id"] = np.full(n, self.run_id, dtype=object)

        self._tree.extend(branches)

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._tree = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def save_predictions_to_root(predictions: List[Dict], save_path: str, epoch: int, run_id: str = None,
                             predict_channels: List[str] = None,
                             npho_scale: float = None, npho_scale2: float = None,
                             time_scale: float = None, time_shift: float = None,
                             sentinel_time: float = None,
                             npho_scheme: str = None):
    """
    Save inpainter predictions to a ROOT file.

    Args:
        predictions: list of dicts with keys: event_idx, sensor_id, face, truth_npho, truth_time, pred_npho, pred_time
        save_path: directory to save the file
        epoch: epoch number
        run_id: optional MLflow run ID
        predict_channels: list of predicted channels (['npho'] or ['npho', 'time'])
        npho_scale, npho_scale2, time_scale, time_shift, sentinel_time: normalization params for metadata
        npho_scheme: Normalization scheme for npho ('log1p', 'anscombe', 'sqrt', 'linear')

    Returns:
        path to saved ROOT file
    """
    if not predictions:
        return None

    with RootPredictionWriter(save_path, epoch, run_id=run_id,
                              predict_channels=predict_channels,
                              npho_scale=npho_scale, npho_scale2=npho_scale2,
                              time_scale=time_scale, time_shift=time_shift,
                              sentinel_time=sentinel_time,
                              npho_scheme=npho_scheme) as writer:
        writer.write(predictions)
        if writer.count == 0:
            return None
        return writer.filepath
