"""
Training and evaluation engine for dead channel inpainting.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from .dataset import XECStreamingDataset


def compute_inpainting_loss(
    results: Dict,
    original_values: torch.Tensor,
    mask: torch.Tensor,
    face_index_maps: Dict[str, torch.Tensor],
    loss_fn: str = "smooth_l1",
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute inpainting loss for all faces.

    Args:
        results: dict from XEC_Inpainter.forward()
        original_values: (B, 4760, 2) - ground truth sensor values
        mask: (B, 4760) - binary mask (1 = masked)
        face_index_maps: dict mapping face names to their index tensors
        loss_fn: "smooth_l1", "mse", or "l1"
        npho_weight: weight for npho channel loss
        time_weight: weight for time channel loss

    Returns:
        total_loss: scalar loss for backprop
        metrics: dict of per-face and per-channel losses
    """
    device = original_values.device
    B = original_values.shape[0]

    # Select loss function
    if loss_fn == "mse":
        loss_func = F.mse_loss
    elif loss_fn == "l1":
        loss_func = F.l1_loss
    else:  # smooth_l1 / huber
        loss_func = F.smooth_l1_loss

    total_loss = torch.tensor(0.0, device=device)
    metrics = {}

    total_npho_loss = 0.0
    total_time_loss = 0.0
    total_count = 0

    # Rectangular faces
    for face_name in ["inner", "us", "ds"]:
        if face_name not in results:
            continue

        face_result = results[face_name]
        pred = face_result["pred"]  # (B, max_masked, 2)
        indices = face_result["indices"]  # (B, max_masked, 2) - (h, w) pairs
        valid = face_result["valid"]  # (B, max_masked)

        if not valid.any():
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0
            continue

        # Get ground truth at masked positions
        idx_map = face_index_maps[face_name]  # (H, W) with flat indices
        H, W = idx_map.shape

        face_losses_npho = []
        face_losses_time = []

        for b in range(B):
            valid_mask = valid[b]  # (max_masked,)
            if not valid_mask.any():
                continue

            n_valid = valid_mask.sum().item()
            h_idx = indices[b, :n_valid, 0]  # (n_valid,)
            w_idx = indices[b, :n_valid, 1]  # (n_valid,)

            # Get flat indices from index map
            flat_idx = idx_map[h_idx, w_idx]  # (n_valid,)

            # Get ground truth
            gt = original_values[b, flat_idx, :]  # (n_valid, 2)
            pr = pred[b, :n_valid, :]  # (n_valid, 2)

            # Compute per-channel loss
            loss_npho = loss_func(pr[:, 0], gt[:, 0], reduction="mean")
            loss_time = loss_func(pr[:, 1], gt[:, 1], reduction="mean")

            face_losses_npho.append(loss_npho)
            face_losses_time.append(loss_time)

        if face_losses_npho:
            avg_npho = torch.stack(face_losses_npho).mean()
            avg_time = torch.stack(face_losses_time).mean()
            face_loss = npho_weight * avg_npho + time_weight * avg_time
            total_loss = total_loss + face_loss

            metrics[f"loss_{face_name}"] = face_loss.item()
            metrics[f"loss_{face_name}_npho"] = avg_npho.item()
            metrics[f"loss_{face_name}_time"] = avg_time.item()

            total_npho_loss += avg_npho.item()
            total_time_loss += avg_time.item()
            total_count += 1
        else:
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0

    # Hex faces
    for face_name, hex_indices in [("top", face_index_maps["top"]), ("bot", face_index_maps["bot"])]:
        if face_name not in results:
            continue

        face_result = results[face_name]
        pred = face_result["pred"]  # (B, max_masked, 2)
        indices = face_result["indices"]  # (B, max_masked) - node indices
        valid = face_result["valid"]  # (B, max_masked)

        if not valid.any():
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0
            continue

        face_losses_npho = []
        face_losses_time = []

        for b in range(B):
            valid_mask = valid[b]
            if not valid_mask.any():
                continue

            n_valid = valid_mask.sum().item()
            node_idx = indices[b, :n_valid]  # (n_valid,)

            # Map to flat indices
            flat_idx = hex_indices[node_idx]  # (n_valid,)

            # Get ground truth
            gt = original_values[b, flat_idx, :]  # (n_valid, 2)
            pr = pred[b, :n_valid, :]  # (n_valid, 2)

            loss_npho = loss_func(pr[:, 0], gt[:, 0], reduction="mean")
            loss_time = loss_func(pr[:, 1], gt[:, 1], reduction="mean")

            face_losses_npho.append(loss_npho)
            face_losses_time.append(loss_time)

        if face_losses_npho:
            avg_npho = torch.stack(face_losses_npho).mean()
            avg_time = torch.stack(face_losses_time).mean()
            face_loss = npho_weight * avg_npho + time_weight * avg_time
            total_loss = total_loss + face_loss

            metrics[f"loss_{face_name}"] = face_loss.item()
            metrics[f"loss_{face_name}_npho"] = avg_npho.item()
            metrics[f"loss_{face_name}_time"] = avg_time.item()

            total_npho_loss += avg_npho.item()
            total_time_loss += avg_time.item()
            total_count += 1
        else:
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0

    # Aggregate metrics
    if total_count > 0:
        metrics["loss_npho"] = total_npho_loss / total_count
        metrics["loss_time"] = total_time_loss / total_count
    else:
        metrics["loss_npho"] = 0.0
        metrics["loss_time"] = 0.0

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
    npho_branch: str = "relative_npho",
    time_branch: str = "relative_time",
    npho_scale: float = 0.58,
    npho_scale2: float = 1.0,
    time_scale: float = 5e-7,
    time_shift: float = 0.1,
    sentinel_value: float = -5.0,
    loss_fn: str = "smooth_l1",
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
    grad_clip: float = 1.0,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Dict[str, float]:
    """
    Run one training epoch for inpainter.

    Returns:
        metrics: dict of averaged metrics
    """
    model.train()

    # Build face index maps
    from .geom_defs import INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP
    from .geom_defs import TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows

    face_index_maps = {
        "inner": torch.from_numpy(INNER_INDEX_MAP).to(device),
        "us": torch.from_numpy(US_INDEX_MAP).to(device),
        "ds": torch.from_numpy(DS_INDEX_MAP).to(device),
        "top": torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long().to(device),
        "bot": torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long().to(device),
    }

    # Accumulate metrics
    metric_sums = {}
    num_batches = 0

    for root_file in train_files:
        dataset = XECStreamingDataset(
            root_path=root_file,
            tree_name=tree_name,
            step_size=step_size,
            npho_branch=npho_branch,
            time_branch=time_branch,
            npho_scale=npho_scale,
            npho_scale2=npho_scale2,
            time_scale=time_scale,
            time_shift=time_shift,
            sentinel_value=sentinel_value,
        )

        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,  # Streaming, shuffle within chunks handled by dataset
            num_workers=0,
            pin_memory=True,
        )

        for batch in loader:
            x_batch = batch["x"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            # Forward with AMP
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                results, original_values, mask = model(x_batch, mask_ratio=mask_ratio)
                loss, metrics = compute_inpainting_loss(
                    results, original_values, mask,
                    face_index_maps,
                    loss_fn=loss_fn,
                    npho_weight=npho_weight,
                    time_weight=time_weight,
                )

            # Backward
            if scaler is not None:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # Accumulate metrics
            for key, value in metrics.items():
                if key not in metric_sums:
                    metric_sums[key] = 0.0
                metric_sums[key] += value
            num_batches += 1

    # Average metrics
    avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}
    return avg_metrics


def run_eval_inpainter(
    model,
    device,
    val_files: List[str],
    tree_name: str,
    batch_size: int,
    step_size: int,
    mask_ratio: float = 0.05,
    npho_branch: str = "relative_npho",
    time_branch: str = "relative_time",
    npho_scale: float = 0.58,
    npho_scale2: float = 1.0,
    time_scale: float = 5e-7,
    time_shift: float = 0.1,
    sentinel_value: float = -5.0,
    loss_fn: str = "smooth_l1",
    npho_weight: float = 1.0,
    time_weight: float = 1.0,
    collect_predictions: bool = False,
) -> Dict[str, float]:
    """
    Run evaluation for inpainter.

    Returns:
        metrics: dict of averaged metrics
        predictions: (optional) dict of predictions if collect_predictions=True
    """
    model.eval()

    from .geom_defs import INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP
    from .geom_defs import TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows

    face_index_maps = {
        "inner": torch.from_numpy(INNER_INDEX_MAP).to(device),
        "us": torch.from_numpy(US_INDEX_MAP).to(device),
        "ds": torch.from_numpy(DS_INDEX_MAP).to(device),
        "top": torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long().to(device),
        "bot": torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long().to(device),
    }

    metric_sums = {}
    num_batches = 0

    # For collecting predictions
    all_predictions = [] if collect_predictions else None

    with torch.no_grad():
        for root_file in val_files:
            dataset = XECStreamingDataset(
                root_path=root_file,
                tree_name=tree_name,
                step_size=step_size,
                npho_branch=npho_branch,
                time_branch=time_branch,
                npho_scale=npho_scale,
                npho_scale2=npho_scale2,
                time_scale=time_scale,
                time_shift=time_shift,
                sentinel_value=sentinel_value,
            )

            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
            )

            for batch in loader:
                x_batch = batch["x"].to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=True):
                    results, original_values, mask = model(x_batch, mask_ratio=mask_ratio)
                    _, metrics = compute_inpainting_loss(
                        results, original_values, mask,
                        face_index_maps,
                        loss_fn=loss_fn,
                        npho_weight=npho_weight,
                        time_weight=time_weight,
                    )

                for key, value in metrics.items():
                    if key not in metric_sums:
                        metric_sums[key] = 0.0
                    metric_sums[key] += value
                num_batches += 1

                if collect_predictions:
                    all_predictions.append({
                        "mask": mask.cpu().numpy(),
                        "original": original_values.cpu().numpy(),
                        "results": {
                            k: {
                                "pred": v["pred"].cpu().numpy(),
                                "valid": v["valid"].cpu().numpy(),
                            }
                            for k, v in results.items()
                        }
                    })

    avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}

    if collect_predictions:
        return avg_metrics, all_predictions
    return avg_metrics
