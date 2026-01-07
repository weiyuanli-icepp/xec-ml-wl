import torch
import torch.nn.functional as F
import numpy as np
import uproot
from torch.utils.data import TensorDataset, DataLoader
from .utils import iterate_chunks
from .geom_utils import build_outer_fine_grid_tensor, gather_face, gather_hex_nodes
from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows
)

def run_epoch_mae(model, optimizer, device, root, tree, 
                  batch_size=8192, step_size=4000, 
                  amp=True,
                  npho_branch="relative_npho", time_branch="relative_time",
                  NphoScale=1e5, NphoScale2=13, time_scale=2.32e6, time_shift=-0.29, sentinel_value=-5.0):
    model.train()
    scaler = torch.amp.GradScaler('cuda', enabled=amp)

    total_loss = 0.0
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
                            batch_size=batch_size, shuffle=True, drop_last=False)
        
        for (X_b,) in loader:
            X_b = X_b.to(device, non_blocking=True)
            
            # --- Normalize (GPU) ---
            raw_n = X_b[:,:,0]
            raw_t = X_b[:,:,1]
            
            time_norm = (raw_t / time_scale) - time_shift
            npho_norm = torch.log1p(raw_n / NphoScale) / NphoScale2
            
            mask_npho_bad = (raw_n <= 0.0) | (raw_n > 9e9) | torch.isnan(raw_n)
            mask_time_bad = mask_npho_bad | (torch.abs(raw_t) > 9e9) | torch.isnan(raw_t)
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
            with torch.amp.autocast('cuda', enabled=amp):
                # 1. Forward Pass
                recons, _ = model(x_in)
                
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
                    
                    if len(t_keys & r_keys) == 0:
                        print("[CRITICAL ERROR] No common keys! Loss loop is skipped.")
                
                for name, pred in recons.items():
                    if name in targets:
                        # Simple MSE between Reconstruction and Original
                        # (Ideally we mask this, but global MSE is stable)
                        l = F.mse_loss(pred, targets[name])
                        loss += l
                        if not debug_printed:
                            print(f"[DEBUG] Loss Component '{name}': {l.item():.6e}")
                        
                if not debug_printed:
                    print(f"[DEBUG] Total Loss: {loss.item():.6e}")
                    print("-"*40 + "\n")
                    debug_printed = True

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            n_batches += 1
                        
    return total_loss / max(1, n_batches)

def run_eval_mae(model, device, root, tree,
                 batch_size=8192, step_size=4000,
                 amp=True,
                 npho_branch="relative_npho", time_branch="relative_time",
                 NphoScale=1e5, NphoScale2=13, time_scale=2.32e6, time_shift=-0.29, sentinel_value=-5.0):
    model.eval()
    
    total_loss = 0.0
    n_batches = 0
    branches = [npho_branch, time_branch]
    
    top_indices = torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long()
    bot_indices = torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long()
    
    for arr in iterate_chunks(root, tree, branches, step_size):
        Npho = np.maximum(arr[npho_branch].astype("float32"), 0.0)
        Time = arr[time_branch].astype("float32")
        
        X_raw = np.stack([Npho, Time], axis=-1).astype("float32")
        loader = DataLoader(TensorDataset(torch.from_numpy(X_raw)),
                            batch_size=batch_size, shuffle=False, drop_last=False)
        
        for (X_b,) in loader:
            X_b = X_b.to(device, non_blocking=True)
            
            raw_n = X_b[:,:,0]
            raw_t = X_b[:,:,1]
            
            time_norm = (raw_t / time_scale) - time_shift
            npho_norm = torch.log1p(raw_n / NphoScale) / NphoScale2
            
            mask_npho_bad = (raw_n <= 0.0) | (raw_n > 9e9) | torch.isnan(raw_n)
            mask_time_bad = mask_npho_bad | (torch.abs(raw_t) > 9e9) | torch.isnan(raw_t)
            
            npho_norm[mask_npho_bad] = 0.0
            time_norm[mask_time_bad] = sentinel_value
            
            x_in = torch.stack([npho_norm, time_norm], dim=-1)
            
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=amp):
                    recons, _ = model(x_in)
                    
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
                    
                    loss = 0.0
                    for name, pred in recons.items():
                        if name in targets:
                            l = F.mse_loss(pred, targets[name])
                            loss += l
                            
            total_loss += loss.item()
            n_batches += 1
            
    return total_loss / max(1, n_batches)