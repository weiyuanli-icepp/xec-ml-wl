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
    outer_fine: bool = False,
    outer_fine_pool: Optional[Tuple[int, int]] = None,
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
        outer_fine: whether outer face uses fine grid
        outer_fine_pool: pooling kernel for outer fine grid

    Returns:
        total_loss: scalar loss for backprop
        metrics: dict of per-face and per-channel losses and MAE/RMSE
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
        from .geom_utils import build_outer_fine_grid_tensor
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
        indices = face_result["indices"]  # (B, max_masked, 2) - (h, w) pairs
        valid = face_result["valid"]  # (B, max_masked)

        if not valid.any():
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0
            metrics[f"mae_{face_name}_npho"] = 0.0
            metrics[f"mae_{face_name}_time"] = 0.0
            metrics[f"rmse_{face_name}_npho"] = 0.0
            metrics[f"rmse_{face_name}_time"] = 0.0
            continue

        # Get ground truth at masked positions
        idx_map = None
        if not (face_name == "outer" and outer_fine):
            idx_map = face_index_maps[face_name]  # (H, W) with flat indices

        face_losses_npho = []
        face_losses_time = []
        face_abs_npho[face_name] = 0.0
        face_abs_time[face_name] = 0.0
        face_sq_npho[face_name] = 0.0
        face_sq_time[face_name] = 0.0
        face_count[face_name] = 0

        for b in range(B):
            valid_mask = valid[b]  # (max_masked,)
            if not valid_mask.any():
                continue

            n_valid = valid_mask.sum().item()
            h_idx = indices[b, :n_valid, 0]  # (n_valid,)
            w_idx = indices[b, :n_valid, 1]  # (n_valid,)

            if face_name == "outer" and outer_fine:
                # Use fine-grid target for outer face
                gt = outer_target[b, :, h_idx, w_idx].T  # (n_valid, 2)
            else:
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

            # Accumulate raw errors for MAE/RMSE
            diff_npho = pr[:, 0] - gt[:, 0]
            diff_time = pr[:, 1] - gt[:, 1]

            face_abs_npho[face_name] += diff_npho.abs().sum().item()
            face_abs_time[face_name] += diff_time.abs().sum().item()
            face_sq_npho[face_name] += (diff_npho ** 2).sum().item()
            face_sq_time[face_name] += (diff_time ** 2).sum().item()
            face_count[face_name] += n_valid

        if face_losses_npho:
            avg_npho = torch.stack(face_losses_npho).mean()
            avg_time = torch.stack(face_losses_time).mean()
            face_loss = npho_weight * avg_npho + time_weight * avg_time
            total_loss = total_loss + face_loss

            metrics[f"loss_{face_name}"] = face_loss.item()
            metrics[f"loss_{face_name}_npho"] = (npho_weight * avg_npho).item()
            metrics[f"loss_{face_name}_time"] = (time_weight * avg_time).item()

            total_npho_loss += metrics[f"loss_{face_name}_npho"]
            total_time_loss += metrics[f"loss_{face_name}_time"]

            # Compute MAE/RMSE for this face
            fc = max(face_count[face_name], 1)
            metrics[f"mae_{face_name}_npho"] = face_abs_npho[face_name] / fc
            metrics[f"mae_{face_name}_time"] = face_abs_time[face_name] / fc
            metrics[f"rmse_{face_name}_npho"] = (face_sq_npho[face_name] / fc) ** 0.5
            metrics[f"rmse_{face_name}_time"] = (face_sq_time[face_name] / fc) ** 0.5

            # Accumulate for global MAE/RMSE
            total_abs_npho += face_abs_npho[face_name]
            total_abs_time += face_abs_time[face_name]
            total_sq_npho += face_sq_npho[face_name]
            total_sq_time += face_sq_time[face_name]
            total_count += face_count[face_name]
        else:
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0
            metrics[f"mae_{face_name}_npho"] = 0.0
            metrics[f"mae_{face_name}_time"] = 0.0
            metrics[f"rmse_{face_name}_npho"] = 0.0
            metrics[f"rmse_{face_name}_time"] = 0.0

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
            metrics[f"mae_{face_name}_npho"] = 0.0
            metrics[f"mae_{face_name}_time"] = 0.0
            metrics[f"rmse_{face_name}_npho"] = 0.0
            metrics[f"rmse_{face_name}_time"] = 0.0
            continue

        face_losses_npho = []
        face_losses_time = []
        face_abs_npho[face_name] = 0.0
        face_abs_time[face_name] = 0.0
        face_sq_npho[face_name] = 0.0
        face_sq_time[face_name] = 0.0
        face_count[face_name] = 0

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

            # Accumulate raw errors for MAE/RMSE
            diff_npho = pr[:, 0] - gt[:, 0]
            diff_time = pr[:, 1] - gt[:, 1]

            face_abs_npho[face_name] += diff_npho.abs().sum().item()
            face_abs_time[face_name] += diff_time.abs().sum().item()
            face_sq_npho[face_name] += (diff_npho ** 2).sum().item()
            face_sq_time[face_name] += (diff_time ** 2).sum().item()
            face_count[face_name] += n_valid

        if face_losses_npho:
            avg_npho = torch.stack(face_losses_npho).mean()
            avg_time = torch.stack(face_losses_time).mean()
            face_loss = npho_weight * avg_npho + time_weight * avg_time
            total_loss = total_loss + face_loss

            metrics[f"loss_{face_name}"] = face_loss.item()
            metrics[f"loss_{face_name}_npho"] = (npho_weight * avg_npho).item()
            metrics[f"loss_{face_name}_time"] = (time_weight * avg_time).item()

            total_npho_loss += metrics[f"loss_{face_name}_npho"]
            total_time_loss += metrics[f"loss_{face_name}_time"]

            # Compute MAE/RMSE for this face
            fc = max(face_count[face_name], 1)
            metrics[f"mae_{face_name}_npho"] = face_abs_npho[face_name] / fc
            metrics[f"mae_{face_name}_time"] = face_abs_time[face_name] / fc
            metrics[f"rmse_{face_name}_npho"] = (face_sq_npho[face_name] / fc) ** 0.5
            metrics[f"rmse_{face_name}_time"] = (face_sq_time[face_name] / fc) ** 0.5

            # Accumulate for global MAE/RMSE
            total_abs_npho += face_abs_npho[face_name]
            total_abs_time += face_abs_time[face_name]
            total_sq_npho += face_sq_npho[face_name]
            total_sq_time += face_sq_time[face_name]
            total_count += face_count[face_name]
        else:
            metrics[f"loss_{face_name}"] = 0.0
            metrics[f"loss_{face_name}_npho"] = 0.0
            metrics[f"loss_{face_name}_time"] = 0.0
            metrics[f"mae_{face_name}_npho"] = 0.0
            metrics[f"mae_{face_name}_time"] = 0.0
            metrics[f"rmse_{face_name}_npho"] = 0.0
            metrics[f"rmse_{face_name}_time"] = 0.0

    # Aggregate metrics (sum across faces, consistent with total_loss)
    metrics["loss_npho"] = total_npho_loss
    metrics["loss_time"] = total_time_loss
    metrics["total_loss"] = total_loss.item()

    # Global MAE/RMSE
    tc = max(total_count, 1)
    metrics["mae_npho"] = total_abs_npho / tc
    metrics["mae_time"] = total_abs_time / tc
    metrics["rmse_npho"] = (total_sq_npho / tc) ** 0.5
    metrics["rmse_time"] = (total_sq_time / tc) ** 0.5

    # Mask statistics
    metrics["n_masked_total"] = total_count

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
    from .geom_defs import INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP, OUTER_COARSE_FULL_INDEX_MAP
    from .geom_defs import TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows

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

    # Accumulate metrics
    metric_sums = {}
    num_batches = 0

    for root_file in train_files:
        dataset = XECStreamingDataset(
            root_files=root_file,
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
            if isinstance(batch, dict):
                x_batch = batch["x"]
            else:
                x_batch = batch[0]
            x_batch = x_batch.to(device, non_blocking=True)

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
                    outer_fine=outer_fine,
                    outer_fine_pool=outer_fine_pool,
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

    Args:
        collect_predictions: If True, collect per-sensor predictions for ROOT output

    Returns:
        metrics: dict of averaged metrics
        predictions: (optional) list of prediction dicts if collect_predictions=True
    """
    model.eval()

    from .geom_defs import INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP, OUTER_COARSE_FULL_INDEX_MAP
    from .geom_defs import TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows

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

    metric_sums = {}
    num_batches = 0

    # For collecting predictions (per-sensor level)
    all_predictions = [] if collect_predictions else None

    with torch.no_grad():
        for root_file in val_files:
            dataset = XECStreamingDataset(
                root_files=root_file,
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
                if isinstance(batch, dict):
                    x_batch = batch["x"]
                else:
                    x_batch = batch[0]
                x_batch = x_batch.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', enabled=True):
                    results, original_values, mask = model(x_batch, mask_ratio=mask_ratio)
                    _, metrics = compute_inpainting_loss(
                        results, original_values, mask,
                        face_index_maps,
                        loss_fn=loss_fn,
                        npho_weight=npho_weight,
                        time_weight=time_weight,
                        outer_fine=outer_fine,
                        outer_fine_pool=outer_fine_pool,
                    )

                for key, value in metrics.items():
                    if key not in metric_sums:
                        metric_sums[key] = 0.0
                    metric_sums[key] += value
                num_batches += 1

                # Collect predictions at sensor level
                if collect_predictions:
                    B = x_batch.shape[0]
                    original_np = original_values.cpu().numpy()
                    mask_np = mask.cpu().numpy()
                    outer_target_np = None
                    outer_grid_w = None
                    if outer_fine and "outer" in results:
                        from .geom_utils import build_outer_fine_grid_tensor
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
                                    all_predictions.append({
                                        "event_idx": num_batches * batch_size + b,
                                        "sensor_id": sensor_id,
                                        "face": face_name,
                                        "truth_npho": float(original_np[b, sensor_id, 0]),
                                        "truth_time": float(original_np[b, sensor_id, 1]),
                                        "pred_npho": float(pred[b, i, 0]),
                                        "pred_time": float(pred[b, i, 1]),
                                    })
                        else:
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
                                    truth_vals = outer_target_np[b, h_idx, w_idx]
                                    for i in range(n_valid):
                                        sensor_id = int(h_idx[i] * outer_grid_w + w_idx[i])
                                        all_predictions.append({
                                            "event_idx": num_batches * batch_size + b,
                                            "sensor_id": sensor_id,
                                            "face": face_name,
                                            "truth_npho": float(truth_vals[i, 0]),
                                            "truth_time": float(truth_vals[i, 1]),
                                            "pred_npho": float(pred[b, i, 0]),
                                            "pred_time": float(pred[b, i, 1]),
                                        })
                                    continue

                                idx_map = face_index_maps_np[face_name]
                                flat_idx = idx_map[h_idx, w_idx]

                                for i in range(n_valid):
                                    sensor_id = int(flat_idx[i])
                                    all_predictions.append({
                                        "event_idx": num_batches * batch_size + b,
                                        "sensor_id": sensor_id,
                                        "face": face_name,
                                        "truth_npho": float(original_np[b, sensor_id, 0]),
                                        "truth_time": float(original_np[b, sensor_id, 1]),
                                        "pred_npho": float(pred[b, i, 0]),
                                        "pred_time": float(pred[b, i, 1]),
                                    })

    avg_metrics = {k: v / max(1, num_batches) for k, v in metric_sums.items()}

    if collect_predictions:
        return avg_metrics, all_predictions
    return avg_metrics


def save_predictions_to_root(predictions: List[Dict], save_path: str, epoch: int, run_id: str = None):
    """
    Save inpainter predictions to a ROOT file.

    Args:
        predictions: list of dicts with keys: event_idx, sensor_id, face, truth_npho, truth_time, pred_npho, pred_time
        save_path: directory to save the file
        epoch: epoch number
        run_id: optional MLflow run ID

    Returns:
        path to saved ROOT file
    """
    import os
    import uproot

    if not predictions:
        return None

    os.makedirs(save_path, exist_ok=True)

    # Convert to arrays
    n = len(predictions)
    event_idx = np.array([p["event_idx"] for p in predictions], dtype=np.int32)
    sensor_id = np.array([p["sensor_id"] for p in predictions], dtype=np.int32)

    # Map face names to integers
    face_map = {"inner": 0, "us": 1, "ds": 2, "outer": 3, "top": 4, "bot": 5}
    face = np.array([face_map.get(p["face"], -1) for p in predictions], dtype=np.int32)

    truth_npho = np.array([p["truth_npho"] for p in predictions], dtype=np.float32)
    truth_time = np.array([p["truth_time"] for p in predictions], dtype=np.float32)
    pred_npho = np.array([p["pred_npho"] for p in predictions], dtype=np.float32)
    pred_time = np.array([p["pred_time"] for p in predictions], dtype=np.float32)

    # Compute errors
    error_npho = pred_npho - truth_npho
    error_time = pred_time - truth_time

    # Build output dict
    branches = {
        "event_idx": event_idx,
        "sensor_id": sensor_id,
        "face": face,
        "truth_npho": truth_npho,
        "truth_time": truth_time,
        "pred_npho": pred_npho,
        "pred_time": pred_time,
        "error_npho": error_npho,
        "error_time": error_time,
    }

    if run_id:
        branches["run_id"] = np.array([run_id] * n)

    # Write to ROOT file
    filename = f"inpainter_predictions_epoch_{epoch}.root"
    filepath = os.path.join(save_path, filename)

    with uproot.recreate(filepath) as f:
        f["tree"] = branches

    return filepath
