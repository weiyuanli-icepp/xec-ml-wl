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

def gather_face(x_batch: torch.Tensor, index_map: np.ndarray) -> torch.Tensor:
    device = x_batch.device
    B, N, C = x_batch.shape
    H, W = index_map.shape
    idx_flat = torch.from_numpy(index_map.reshape(-1)).to(device)
    mask = (idx_flat >= 0)
    safe_idx = idx_flat.clone()
    safe_idx[~mask] = 0
    vals = torch.index_select(x_batch, 1, safe_idx)
    vals[:, ~mask] = 0.0 
    vals = vals.view(B, H, W, C).permute(0, 3, 1, 2)
    return vals

def gather_hex_nodes(x_batch: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    vals = torch.index_select(x_batch, 1, flat_indices.to(x_batch.device))
    return vals

def build_outer_fine_grid_tensor(x_batch: torch.Tensor, pool_kernel=None) -> torch.Tensor:
    device = x_batch.device
    B, N, C = x_batch.shape
    coarse = gather_face(x_batch, OUTER_COARSE_FULL_INDEX_MAP)
    center = gather_face(x_batch, OUTER_CENTER_INDEX_MAP)
    cr, cc = OUTER_FINE_COARSE_SCALE 
    sr, sc = OUTER_FINE_CENTER_SCALE 
    
    fine_from_coarse = F.interpolate(coarse, scale_factor=(cr, cc), mode='nearest')
    fine_from_center = F.interpolate(center, scale_factor=(sr, sc), mode='nearest')
    
    # Extensive vs Intensive scaling
    fine_from_coarse[:, 0] = fine_from_coarse[:, 0] / float(cr * cc)
    fine_from_center[:, 0] = fine_from_center[:, 0] / float(sr * sc)
    
    c_start_r, c_start_c = OUTER_FINE_CENTER_START 
    top_fine = c_start_r * cr
    left_fine = c_start_c * cc
    h_fine_c = fine_from_center.shape[2]
    w_fine_c = fine_from_center.shape[3]
    
    fine_grid = fine_from_coarse.clone()
    fine_grid[:, :, top_fine : top_fine + h_fine_c, left_fine : left_fine + w_fine_c] = fine_from_center
    
    if pool_kernel:
        fine_grid = F.avg_pool2d(fine_grid, kernel_size=pool_kernel, stride=pool_kernel)
    return fine_grid