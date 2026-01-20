import torch
import torch.nn.functional as F
import numpy as np
import uproot
from torch.utils.data import TensorDataset, DataLoader
from .utils import iterate_chunks, get_pointwise_loss_fn
from .geom_utils import build_outer_fine_grid_tensor, gather_face, gather_hex_nodes
from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    OUTER_FINE_COARSE_SCALE, OUTER_FINE_CENTER_SCALE, OUTER_FINE_CENTER_START,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows
)

def run_epoch_mae(model, optimizer, device, root, tree,
                  batch_size=8192, step_size=4000,
                  amp=True,
                  npho_branch="relative_npho", time_branch="relative_time",
                  NphoScale=1e5, NphoScale2=13, time_scale=2.32e6, time_shift=-0.29, sentinel_value=-5.0,
                  channel_dropout_rate=0.1,
                  loss_fn="mse",
                  npho_weight=1.0,
                  time_weight=1.0,
                  auto_channel_weight=False,
                  grad_clip=1.0,
                  scaler=None,
                  num_workers=8):
    model.train()
    if scaler is None:
        scaler = torch.amp.GradScaler('cuda', enabled=amp)
    loss_func = get_pointwise_loss_fn(loss_fn)
    log_vars = getattr(model, "channel_log_vars", None) if auto_channel_weight else None

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
   
    branches = [npho_branch, time_branch]
        
    top_indices = torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long()
    bot_indices = torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long()
    
    debug_printed = True
    for arr in iterate_chunks(root, tree, branches, step_size):

        # --- Preprocessing (CPU) ---
        Npho = np.maximum(arr[npho_branch].astype("float32"), 0.0)
        Time = arr[time_branch].astype("float32")

        if not debug_printed:
            print("\n" + "-"*40)
            print("[DEBUG] Sample Data Stats (Pre-Normalization):")
            print(f"Npho: Min={Npho.min():.4f}, Max={Npho.max():.4f}, Mean={Npho.mean():.4f}")
            print(f"Time: Min={Time.min():.4f}, Max={Time.max():.4f}, Mean={Time.mean():.4f}")
            
            median_val = np.median(Time)
            print(f"Time Median: {median_val:.4e}")
            valid_mask = Time < 9e9
            if np.any(valid_mask):
                clean_time = Time[valid_mask]
                print(f"Clean Time Mean: {clean_time.mean():.4e}")
                print(f"Clean Time Std : {clean_time.std():.4e}")
                print(f"Valid Events: {len(clean_time)}/{len(Time)}")
            else:
                print("No valid Time entries found.")
                
        X_raw = np.stack([Npho, Time], axis=-1).astype("float32")
        loader = DataLoader(TensorDataset(torch.from_numpy(X_raw)), 
                            batch_size=batch_size, shuffle=True, drop_last=False,
                            num_workers=num_workers, pin_memory=True, 
                            persistent_workers=True, prefetch_factor=2)
        
        for (X_b,) in loader:
            X_b = X_b.to(device, non_blocking=True)
            
            # --- Normalize (GPU) ---
            raw_n = X_b[:,:,0]
            raw_t = X_b[:,:,1]
            
            time_norm = (raw_t / time_scale) - time_shift
            npho_norm = torch.log1p(raw_n / NphoScale) / NphoScale2
            
            mask_npho_bad = (raw_n <= 0.0) | (raw_n > 9e9) | torch.isnan(raw_n)
            mask_time_bad = mask_npho_bad | (torch.abs(raw_t) > 9e9) | torch.isnan(raw_t)
            
            # Channel Dropout
            # if channel_dropout_rate > 0.0:
            if False: # No Channel Dropout for MAE
                dropout_mask = (torch.rand_like(npho_norm) < channel_dropout_rate)
                mask_npho_bad = mask_npho_bad | dropout_mask
                mask_time_bad = mask_time_bad | dropout_mask
                
            npho_norm[mask_npho_bad] = 0.0
            time_norm[mask_time_bad] = sentinel_value
            x_in = torch.stack([npho_norm, time_norm], dim=-1) # (B, 4760, 2)
                        
            if not debug_printed:
                print("[DEBUG] Sample Data Stats (Post-Normalization):")
                print(f"Input Npho: Min={x_in[...,0].min():.4e}, Max={x_in[...,0].max():.4e}, Mean={x_in[...,0].mean():.4e}")
                print(f"Input Time: Min={x_in[...,1].min():.4e}, Max={x_in[...,1].max():.4e}, Mean={x_in[...,1].mean():.4e}")
                if x_in.abs().max() < 1e-5:
                    print("[WARNING] INPUTS ARE ZERO! Check your scale factors.")
            
            optimizer.zero_grad()
            # with torch.amp.autocast('cuda', enabled=amp):
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                # 1. Forward Pass
                recons, mask = model(x_in)
                
                # 2. Gather Truth Targets
                if hasattr(model, "encoder") and getattr(model.encoder, "outer_fine", False):
                    outer_target = build_outer_fine_grid_tensor(
                        x_in,
                        pool_kernel=model.encoder.outer_fine_pool
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
                
                if not debug_printed:
                    print("\n" + "="*40)
                    print("[DEBUG] DIAGNOSING ZERO LOSS")
                    print("="*40)
                    print(f"Batch Size: {X_b.shape[0]}")
                    print(f"Raw Npho Stats: Min={raw_n.min().item():.4f}, Max={raw_n.max().item():.4f}, Mean={raw_n.mean().item():.4f}")
                    print(f"Model Input Stats: Min={x_in.min().item():.4f}, Max={x_in.max().item():.4f}, Mean={x_in.mean().item():.4f}")
                    
                    # Check Dictionary Keys
                    t_keys = set(targets.keys())
                    r_keys = set(recons.keys())
                    print(f"Target Keys: {t_keys}")
                    print(f"Recons Keys: {r_keys}")
                    print(f"Intersection: {t_keys & r_keys}")
                    print(f"[DEBUG] Top Masked Pixels: {mask[:, top_indices].sum().item()}")
                    print(f"[DEBUG] Bot Masked Pixels: {mask[:, bot_indices].sum().item()}")
                    
                    if len(t_keys & r_keys) == 0:
                        print("[CRITICAL ERROR] No common keys! Loss loop is skipped.")

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
                            m_face = mask[:, indices] # Shape: (B, num_sensors_in_face)
                            if name in ["top", "bot"]:
                                mask_expanded = m_face.unsqueeze(1) # (B, 1, num_hex_nodes)
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
                                mask_expanded = m_face.view(mask.size(0), 1, H, W) # (B, 1, H, W)
                        else:
                            mask_expanded = torch.ones_like(pred[:, 0:1])

                        # Separate npho (channel 0) and time (channel 1) losses
                        target = targets[name]
                        loss_map_npho = loss_func(pred[:, 0:1], target[:, 0:1])
                        loss_map_time = loss_func(pred[:, 1:2], target[:, 1:2])

                        mask_sum = mask_expanded.sum()
                        mask_sum_safe = mask_sum + 1e-8
                        npho_loss = (loss_map_npho * mask_expanded).sum() / mask_sum_safe
                        time_loss = (loss_map_time * mask_expanded).sum() / mask_sum_safe

                        diff_npho = pred[:, 0:1] - target[:, 0:1]
                        diff_time = pred[:, 1:2] - target[:, 1:2]
                        mask_sum_val = mask_sum.item()
                        if mask_sum_val > 0:
                            masked_abs_sum_npho += (diff_npho.abs() * mask_expanded).sum().item()
                            masked_abs_sum_time += (diff_time.abs() * mask_expanded).sum().item()
                            masked_sq_sum_npho += (diff_npho.pow(2) * mask_expanded).sum().item()
                            masked_sq_sum_time += (diff_time.pow(2) * mask_expanded).sum().item()
                            masked_count_npho += mask_sum_val
                            masked_count_time += mask_sum_val
                            masked_abs_face_npho[name] += (diff_npho.abs() * mask_expanded).sum().item()
                            masked_abs_face_time[name] += (diff_time.abs() * mask_expanded).sum().item()
                            masked_sq_face_npho[name] += (diff_npho.pow(2) * mask_expanded).sum().item()
                            masked_sq_face_time[name] += (diff_time.pow(2) * mask_expanded).sum().item()
                            masked_count_face[name] += mask_sum_val

                        if log_vars is not None and log_vars.numel() >= 2:
                            npho_loss = 0.5 * torch.exp(-log_vars[0]) * npho_loss + 0.5 * log_vars[0]
                            time_loss = 0.5 * torch.exp(-log_vars[1]) * time_loss + 0.5 * log_vars[1]
                        else:
                            npho_loss = npho_loss * npho_weight
                            time_loss = time_loss * time_weight
                        face_loss = npho_loss + time_loss

                        face_loss_sums[name] += face_loss.item()
                        face_npho_loss_sums[name] += npho_loss.item()
                        face_time_loss_sums[name] += time_loss.item()
                        loss += face_loss

                        if not debug_printed:
                            print(f"[DEBUG] Loss Component '{name}': {face_loss.item():.6e} (npho: {npho_loss.item():.6e}, time: {time_loss.item():.6e})")
                        
                if not debug_printed:
                    print(f"[DEBUG] Total Loss: {loss.item():.6e}")
                    print("-"*40 + "\n")
                    debug_printed = True

            scaler.scale(loss).backward()

            # Gradient clipping
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update()

            total_loss_sum += loss.item()
            n_batches += 1

    # Return averaged losses with detailed breakdown
    metrics = {}

    # Total loss
    metrics["total_loss"] = total_loss_sum / max(1, n_batches)

    # Per-face total losses
    for name, val in face_loss_sums.items():
        metrics[f"loss_{name}"] = val / max(1, n_batches)

    # Per-face npho losses
    for name, val in face_npho_loss_sums.items():
        metrics[f"loss_{name}_npho"] = val / max(1, n_batches)

    # Per-face time losses
    for name, val in face_time_loss_sums.items():
        metrics[f"loss_{name}_time"] = val / max(1, n_batches)

    # Aggregate npho/time losses (average across faces)
    num_faces = len(face_npho_loss_sums)
    metrics["loss_npho"] = sum(metrics[f"loss_{name}_npho"] for name in face_npho_loss_sums) / max(1, num_faces)
    metrics["loss_time"] = sum(metrics[f"loss_{name}_time"] for name in face_time_loss_sums) / max(1, num_faces)

    metrics["mae_npho"] = masked_abs_sum_npho / max(masked_count_npho, 1e-8)
    metrics["mae_time"] = masked_abs_sum_time / max(masked_count_time, 1e-8)
    metrics["rmse_npho"] = (masked_sq_sum_npho / max(masked_count_npho, 1e-8)) ** 0.5
    metrics["rmse_time"] = (masked_sq_sum_time / max(masked_count_time, 1e-8)) ** 0.5
    for name in face_loss_sums:
        face_count = masked_count_face[name]
        metrics[f"mae_{name}_npho"] = masked_abs_face_npho[name] / max(face_count, 1e-8)
        metrics[f"mae_{name}_time"] = masked_abs_face_time[name] / max(face_count, 1e-8)
        metrics[f"rmse_{name}_npho"] = (masked_sq_face_npho[name] / max(face_count, 1e-8)) ** 0.5
        metrics[f"rmse_{name}_time"] = (masked_sq_face_time[name] / max(face_count, 1e-8)) ** 0.5

    if log_vars is not None and log_vars.numel() >= 2:
        metrics["channel_logvar_npho"] = log_vars[0].item()
        metrics["channel_logvar_time"] = log_vars[1].item()

    return metrics

def run_eval_mae(model, device, root, tree,
                 batch_size=8192, step_size=4000,
                 amp=True,
                 npho_branch="relative_npho", time_branch="relative_time",
                 NphoScale=1e5, NphoScale2=13, time_scale=2.32e6, time_shift=-0.29, sentinel_value=-5.0,
                 loss_fn="mse",
                 npho_weight=1.0,
                 time_weight=1.0,
                 auto_channel_weight=False,
                 collect_predictions=False, max_events=1000,
                 num_workers=8):
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

    # Prediction collection
    predictions = {
        "truth_npho": [], "truth_time": [],
        "pred_npho": [], "pred_time": [],
        "mask": [], "x_masked": []
    }
    n_collected = 0

    branches = [npho_branch, time_branch]

    top_indices = torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long()
    bot_indices = torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long()

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

    def scatter_rect_face(full, face_pred, index_map):
        face_pred = face_pred.to(full.dtype)
        idx_flat = torch.tensor(index_map.reshape(-1), device=face_pred.device, dtype=torch.long)
        valid = idx_flat >= 0
        idx = idx_flat[valid]
        vals = face_pred.permute(0, 2, 3, 1).reshape(face_pred.size(0), -1, 2)[:, valid]
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
        npho = fine_pred[:, 0:1].contiguous().view(-1, 1, Hc, cr, Wc, cc).sum(dim=(3, 5))
        time = fine_pred[:, 1:2].contiguous().view(-1, 1, Hc, cr, Wc, cc).mean(dim=(3, 5))
        coarse_pred = torch.cat([npho, time], dim=1)

        sr, sc = OUTER_FINE_CENTER_SCALE
        Hc_center, Wc_center = OUTER_CENTER_INDEX_MAP.shape
        top = OUTER_FINE_CENTER_START[0] * cr
        left = OUTER_FINE_CENTER_START[1] * cc
        center_fine = fine_pred[:, :, top:top + Hc_center * sr, left:left + Wc_center * sc]
        c_npho = center_fine[:, 0:1].contiguous().view(-1, 1, Hc_center, sr, Wc_center, sc).sum(dim=(3, 5))
        c_time = center_fine[:, 1:2].contiguous().view(-1, 1, Hc_center, sr, Wc_center, sc).mean(dim=(3, 5))
        center_pred = torch.cat([c_npho, c_time], dim=1)

        return coarse_pred, center_pred

    def assemble_full_pred(recons_dict):
        full = torch.zeros(
            (recons_dict["inner"].size(0), num_sensors, 2),
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
    
    for arr in iterate_chunks(root, tree, branches, step_size):
        Npho = np.maximum(arr[npho_branch].astype("float32"), 0.0)
        Time = arr[time_branch].astype("float32")
        
        X_raw = np.stack([Npho, Time], axis=-1).astype("float32")
        loader = DataLoader(TensorDataset(torch.from_numpy(X_raw)),
                            batch_size=batch_size, shuffle=False, drop_last=False,
                            num_workers=num_workers, pin_memory=True)
        
        for (X_b,) in loader:
            X_b = X_b.to(device, non_blocking=True)
            
            raw_n = X_b[:,:,0]
            raw_t = X_b[:,:,1]
            
            npho_norm = torch.log1p(raw_n / NphoScale) / NphoScale2
            time_norm = (raw_t / time_scale) - time_shift
            
            mask_npho_bad = (raw_n <= 0.0) | (raw_n > 9e9) | torch.isnan(raw_n)
            mask_time_bad = mask_npho_bad | (torch.abs(raw_t) > 9e9) | torch.isnan(raw_t)
            
            npho_norm[mask_npho_bad] = 0.0
            time_norm[mask_time_bad] = sentinel_value
            
            x_in = torch.stack([npho_norm, time_norm], dim=-1)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=amp):
                    # Get masked input for visualization
                    x_masked, mask = model.random_masking(x_in)
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
                    
                    # if hasattr(model, "encoder") and getattr(model.encoder, "outer_fine", False):
                    #     outer_target = build_outer_fine_grid_tensor(
                    #         x_in,
                    #         pool_kernel=model.encoder.outer_fine_pool
                    #     )
                    # else:
                    #     outer_target = gather_face(x_in, OUTER_COARSE_FULL_INDEX_MAP)

                    targets = {
                        "inner": gather_face(x_in, INNER_INDEX_MAP),
                        "us":    gather_face(x_in, US_INDEX_MAP),
                        "ds":    gather_face(x_in, DS_INDEX_MAP),
                        "outer": build_outer_fine_grid_tensor(x_in, model.encoder.outer_fine_pool) if getattr(model.encoder, "outer_fine", False) else gather_face(x_in, OUTER_COARSE_FULL_INDEX_MAP),
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
                            loss_map_npho = loss_func(pred[:, 0:1], target[:, 0:1])
                            loss_map_time = loss_func(pred[:, 1:2], target[:, 1:2])

                            mask_sum = mask_expanded.sum()
                            mask_sum_safe = mask_sum + 1e-8
                            npho_loss = (loss_map_npho * mask_expanded).sum() / mask_sum_safe
                            time_loss = (loss_map_time * mask_expanded).sum() / mask_sum_safe

                            diff_npho = pred[:, 0:1] - target[:, 0:1]
                            diff_time = pred[:, 1:2] - target[:, 1:2]
                            mask_sum_val = mask_sum.item()
                            if mask_sum_val > 0:
                                masked_abs_sum_npho += (diff_npho.abs() * mask_expanded).sum().item()
                                masked_abs_sum_time += (diff_time.abs() * mask_expanded).sum().item()
                                masked_sq_sum_npho += (diff_npho.pow(2) * mask_expanded).sum().item()
                                masked_sq_sum_time += (diff_time.pow(2) * mask_expanded).sum().item()
                                masked_count_npho += mask_sum_val
                                masked_count_time += mask_sum_val
                                masked_abs_face_npho[name] += (diff_npho.abs() * mask_expanded).sum().item()
                                masked_abs_face_time[name] += (diff_time.abs() * mask_expanded).sum().item()
                                masked_sq_face_npho[name] += (diff_npho.pow(2) * mask_expanded).sum().item()
                                masked_sq_face_time[name] += (diff_time.pow(2) * mask_expanded).sum().item()
                                masked_count_face[name] += mask_sum_val
                            if log_vars is not None and log_vars.numel() >= 2:
                                npho_loss = 0.5 * torch.exp(-log_vars[0]) * npho_loss + 0.5 * log_vars[0]
                                time_loss = 0.5 * torch.exp(-log_vars[1]) * time_loss + 0.5 * log_vars[1]
                            else:
                                npho_loss = npho_loss * npho_weight
                                time_loss = time_loss * time_weight
                            face_loss = npho_loss + time_loss

                            face_loss_sums[name] += face_loss.item()
                            face_npho_loss_sums[name] += npho_loss.item()
                            face_time_loss_sums[name] += time_loss.item()
                            loss += face_loss

                    # Collect predictions for ROOT output
                    if collect_predictions and n_collected < max_events:
                        full_pred = assemble_full_pred(recons)
                        n_to_collect = min(X_b.shape[0], max_events - n_collected)
                        predictions["truth_npho"].append(x_in[:n_to_collect, :, 0].cpu().numpy())
                        predictions["truth_time"].append(x_in[:n_to_collect, :, 1].cpu().numpy())
                        predictions["pred_npho"].append(full_pred[:n_to_collect, :, 0].cpu().numpy())
                        predictions["pred_time"].append(full_pred[:n_to_collect, :, 1].cpu().numpy())
                        predictions["mask"].append(mask[:n_to_collect].cpu().numpy())
                        predictions["x_masked"].append(x_masked[:n_to_collect].cpu().numpy())
                        # Note: pred reconstruction is per-face, reassembled to full sensor array
                        n_collected += n_to_collect

            total_loss += loss.item()
            n_batches += 1

    # Build metrics dict
    metrics = {}
    metrics["total_loss"] = total_loss / max(1, n_batches)

    for name, val in face_loss_sums.items():
        metrics[f"loss_{name}"] = val / max(1, n_batches)
    for name, val in face_npho_loss_sums.items():
        metrics[f"loss_{name}_npho"] = val / max(1, n_batches)
    for name, val in face_time_loss_sums.items():
        metrics[f"loss_{name}_time"] = val / max(1, n_batches)

    # Aggregate npho/time losses (average across faces)
    num_faces = len(face_npho_loss_sums)
    metrics["loss_npho"] = sum(metrics[f"loss_{name}_npho"] for name in face_npho_loss_sums) / max(1, num_faces)
    metrics["loss_time"] = sum(metrics[f"loss_{name}_time"] for name in face_time_loss_sums) / max(1, num_faces)

    metrics["mae_npho"] = masked_abs_sum_npho / max(masked_count_npho, 1e-8)
    metrics["mae_time"] = masked_abs_sum_time / max(masked_count_time, 1e-8)
    metrics["rmse_npho"] = (masked_sq_sum_npho / max(masked_count_npho, 1e-8)) ** 0.5
    metrics["rmse_time"] = (masked_sq_sum_time / max(masked_count_time, 1e-8)) ** 0.5
    for name in face_loss_sums:
        face_count = masked_count_face[name]
        metrics[f"mae_{name}_npho"] = masked_abs_face_npho[name] / max(face_count, 1e-8)
        metrics[f"mae_{name}_time"] = masked_abs_face_time[name] / max(face_count, 1e-8)
        metrics[f"rmse_{name}_npho"] = (masked_sq_face_npho[name] / max(face_count, 1e-8)) ** 0.5
        metrics[f"rmse_{name}_time"] = (masked_sq_face_time[name] / max(face_count, 1e-8)) ** 0.5

    if log_vars is not None and log_vars.numel() >= 2:
        metrics["channel_logvar_npho"] = log_vars[0].item()
        metrics["channel_logvar_time"] = log_vars[1].item()

    if collect_predictions:
        # Concatenate collected predictions
        for key in predictions:
            if predictions[key]:
                predictions[key] = np.concatenate(predictions[key], axis=0)
            else:
                predictions[key] = np.array([])
        return metrics, predictions

    return metrics
