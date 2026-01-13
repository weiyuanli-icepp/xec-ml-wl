import torch
import torch.nn.functional as F
import numpy as np
from .geom_defs import (
    OUTER_COARSE_FULL_INDEX_MAP,
    OUTER_CENTER_INDEX_MAP,
    OUTER_FINE_COARSE_SCALE,
    OUTER_FINE_CENTER_SCALE,
    OUTER_FINE_CENTER_START
)

# Cache for index tensors to avoid CPU->GPU transfers during cudagraph capture
_INDEX_CACHE = {}

def gather_face(x_batch: torch.Tensor, index_map: np.ndarray) -> torch.Tensor:
    device = x_batch.device
    B, N, C = x_batch.shape
    H, W = index_map.shape

    # Cache key based on index_map id and device
    cache_key = (id(index_map), device)

    if cache_key not in _INDEX_CACHE:
        # Create and cache index tensor on device
        idx_flat = torch.tensor(index_map.reshape(-1), device=device, dtype=torch.long)
        mask = (idx_flat >= 0)
        safe_idx = idx_flat.clone()
        safe_idx[~mask] = 0
        _INDEX_CACHE[cache_key] = (idx_flat, mask, safe_idx, H, W)

    idx_flat, mask, safe_idx, H, W = _INDEX_CACHE[cache_key]

    # Gather
    vals = torch.index_select(x_batch, 1, safe_idx)

    # Mask
    mask_expanded = mask.view(1, -1, 1).expand(B, -1, C)
    vals = vals * mask_expanded.float()
    vals = vals.view(B, H, W, C).permute(0, 3, 1, 2)
    return vals

def gather_hex_nodes(x_batch: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    safe_indices = flat_indices.to(x_batch.device).long()
    vals = torch.index_select(x_batch, 1, safe_indices)
    return vals

def build_outer_fine_grid_tensor(x_batch: torch.Tensor, pool_kernel=None) -> torch.Tensor:
    device = x_batch.device
    B, N, C = x_batch.shape
    
    coarse = gather_face(x_batch, OUTER_COARSE_FULL_INDEX_MAP)
    center = gather_face(x_batch, OUTER_CENTER_INDEX_MAP)
    cr, cc = OUTER_FINE_COARSE_SCALE 
    sr, sc = OUTER_FINE_CENTER_SCALE 
    
    fine_from_coarse = F.interpolate(coarse, scale_factor=(float(cr), float(cc)), mode='nearest')
    fine_from_center = F.interpolate(center, scale_factor=(float(sr), float(sc)), mode='nearest')
    
    # Extensive vs Intensive scaling
    # ori
    # fine_from_coarse[:, 0] = fine_from_coarse[:, 0] / float(cr * cc)
    # fine_from_center[:, 0] = fine_from_center[:, 0] / float(sr * sc)
    # -- v1 --
    # fine_from_coarse[:, 0:1, :, :] = torch.div(fine_from_coarse[:, 0:1, :, :], float(cr * cc))
    # fine_from_center[:, 0:1, :, :] = torch.div(fine_from_center[:, 0:1, :, :], float(sr * sc))
    # --------
    # -- v2 --
    scale_c = torch.tensor([float(cr * cc)], device=device, dtype=x_batch.dtype)
    scale_s = torch.tensor([float(sr * sc)], device=device, dtype=x_batch.dtype)
    
    c_npho = fine_from_coarse[:, 0:1, :, :] / scale_c
    c_time = fine_from_coarse[:, 1:,  :, :]
    fine_coarse_scaled = torch.cat([c_npho, c_time], dim=1)
    
    s_npho = fine_from_center[:, 0:1, :, :] / scale_s
    s_time = fine_from_center[:, 1:, :, :]
    fine_center_scaled = torch.cat([s_npho, s_time], dim=1)
    # --------
    
    c_start_r, c_start_c = OUTER_FINE_CENTER_START 
    # -- ori --
    # top_fine = c_start_r * cr
    # left_fine = c_start_c * cc
    # h_fine_c = fine_from_center.shape[2]
    # w_fine_c = fine_from_center.shape[3]
    # ---------
    # -- v2 --
    top = c_start_r * cr
    left = c_start_c * cc
    h_fine_c = fine_center_scaled.shape[2]
    w_fine_c = fine_center_scaled.shape[3]
    H_total = fine_coarse_scaled.shape[2]
    W_total = fine_coarse_scaled.shape[3]
    
    pad_left = left
    pad_right = W_total - (left + w_fine_c)
    pad_top = top
    pad_bottom = H_total - (top + h_fine_c)
    
    center_padded = F.pad(
        fine_center_scaled, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='constant', value=0.0
    )
    
    ones_patch = torch.ones((1, 1, h_fine_c, w_fine_c), device=device, dtype=x_batch.dtype)
    mask_padded = F.pad(
        ones_patch, 
        (pad_left, pad_right, pad_top, pad_bottom), 
        mode='constant', value=0.0
    )
    # --------
    
    # fine_grid = fine_from_coarse.clone()
    # fine_grid = fine_from_coarse
    # fine_grid[:, :, top_fine : top_fine + h_fine_c, left_fine : left_fine + w_fine_c] = fine_from_center
    fine_grid = fine_coarse_scaled * (1.0 - mask_padded) + center_padded
    
    if pool_kernel:
        fine_grid = F.avg_pool2d(fine_grid, kernel_size=pool_kernel, stride=pool_kernel)
    return fine_grid