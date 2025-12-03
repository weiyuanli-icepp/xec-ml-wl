# angle_model_geom.py

"""
======================================================================
 Geometry-Aware CNN for LXe Photon-Angle Regression
======================================================================

INPUT
------
- npho_batch : Tensor (B, 4760)
    Log-scaled photon counts from:
      • Inner SiPM face (93x44)
      • Outer PMT face:
            - Coarse region  (9x24)
            - Fine central patch (5x6)  [or fused fine-grid outer face 45x72]
      • Upstream  (6x24)
      • Downstream (6x24)
      • Top    (staggered hex rows 11,12,11,12,13,14)
      • Bottom (staggered hex rows 11,12,11,12,13,14)

GEOMETRY HANDLING
-----------------
Each face is reconstructed into a small 2D image by mapping npho indices
into (HxW) grids. Missing channels are padded with -1 → replaced by 0.
Top/Bottom use a staggered hex layout (rows: 11,12,11,12,13,14) with 6-neighbor
connectivity and are encoded by a small graph conv. Outer face can be processed
either as coarse+central faces ("split") or as a single fine 45x72 grid
("finegrid") that embeds both regions.

CNN BACKBONE (Shared for all faces)
-----------------------------------
FaceBackbone:
    Conv2d(1 → C, 3x3)
    BatchNorm2d
    LeakyReLU(0.1)

    Conv2d(C → 2C, 3x3)
    BatchNorm2d
    LeakyReLU(0.1)

    AdaptiveAvgPool2d(4x4)
    → flatten → embedding_dim = 2C * 16

HEAD (Fully-Connected Regression)
---------------------------------
Concatenate embeddings from CNN faces (either 5 split faces or 3 + outer fine grid)
plus 2 hex graph faces → Linear(total_embed → hidden=256)
    LeakyReLU(0.1)
    Dropout(0.2)
    Linear(256 → 2)     # predicts (emiAng[0], emiAng[1])

LOSS FUNCTION
--------------
SmoothL1Loss (Huber loss)

ACTIVATIONS
------------
LeakyReLU(0.1)

OPTIMIZER
----------
AdamW with decoupled weight decay

======================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================================================
# 1. GEOMETRY DEFINITIONS
# =========================================================

# --- Inner SiPM face (93x44) ---
# Row 0 is Top (Index 0..43). 
INNER_INDEX_MAP = np.arange(0, 4092, dtype=np.int32).reshape(93, 44)

# --- US and DS faces (24x6) ---
# Reshaped to be tall (24 rows, 6 cols) to match physical aspect ratio
# US: 4308 is Top-Left (adjacent to Inner Right edge)
US_INDEX_MAP = np.arange(4308, 4308 + 6*24, dtype=np.int32).reshape(24, 6)

# DS: 4452 is Top-Right (adjacent to Inner Left edge)
DS_INDEX_MAP = np.arange(4452, 4452 + 6*24, dtype=np.int32).reshape(24, 6)

# --- Outer Face (Coarse + Center) ---
CENTRAL_COARSE_IDS = [
    4185, 4186, 4187, 4194, 4195, 4196,
    4203, 4204, 4205, 4212, 4213, 4214,
]
OUTER_COARSE_FULL_INDEX_MAP = np.arange(4092, 4308, dtype=np.int32).reshape(9, 24)
def make_outer_coarse_index_map():
    base = np.arange(4092, 4308, dtype=np.int32).reshape(9, 24)
    mask_central = np.isin(base, CENTRAL_COARSE_IDS)
    base[mask_central] = -1
    return base
OUTER_COARSE_INDEX_MAP = make_outer_coarse_index_map()
OUTER_CENTER_INDEX_MAP = np.array([
    [4185, 4742, 4186, 4743, 4187],
    [4744, 4745, 4746, 4747, 4748],
    [4194, 4749, 4195, 4750, 4196],
    [4203, 4751, 4204, 4752, 4205],
    [4753, 4754, 4755, 4756, 4757],
    [4212, 4758, 4213, 4759, 4214],
], dtype=np.int32).T

# --- Outer Fine Grid Scaling ---
OUTER_FINE_COARSE_SCALE = (5, 3)
OUTER_FINE_CENTER_SCALE = (3, 2)
OUTER_FINE_CENTER_START = (3, 10)
OUTER_FINE_H = OUTER_COARSE_FULL_INDEX_MAP.shape[0] * OUTER_FINE_COARSE_SCALE[0]
OUTER_FINE_W = OUTER_COARSE_FULL_INDEX_MAP.shape[1] * OUTER_FINE_COARSE_SCALE[1]

# --- TOP & BOTTOM HEX DEFINITIONS (Rows from PDF) ---
# Top starts at 4596. Rows: 11, 12, 11, 12, 13, 14 items.
TOP_ROWS_LIST = [
    np.arange(4596, 4607), # 11
    np.arange(4607, 4619), # 12
    np.arange(4619, 4630), # 11
    np.arange(4630, 4642), # 12
    np.arange(4642, 4655), # 13
    np.arange(4655, 4669), # 14
]

# Bottom starts at 4669. Rows: 11, 12, 11, 12, 13, 14 items.
BOTTOM_ROWS_LIST = [
    np.arange(4669, 4680), # 11
    np.arange(4680, 4692), # 12
    np.arange(4692, 4703), # 11
    np.arange(4703, 4715), # 12
    np.arange(4715, 4728), # 13
    np.arange(4728, 4742), # 14
]

TOP_HEX_ROWS = TOP_ROWS_LIST
BOTTOM_HEX_ROWS = BOTTOM_ROWS_LIST

def flatten_hex_rows(rows) -> np.ndarray:
    return np.concatenate([np.asarray(r, dtype=np.int32) for r in rows])

def build_hex_edge_index(row_lengths):
    id_map = {}
    node = 0
    for r, L in enumerate(row_lengths):
        for c in range(L):
            id_map[(r, c)] = node
            node += 1
    edges = set()
    for (r, c), u in id_map.items():
        if r % 2 == 0:
            neigh = [(r, c-1), (r, c+1), (r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
        else:
            neigh = [(r, c-1), (r, c+1), (r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]
        for rr, cc in neigh:
            if (rr, cc) in id_map:
                edges.add((u, id_map[(rr, cc)]))
                edges.add((id_map[(rr, cc)], u))
    if edges:
        edge_index = np.array(list(edges), dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    dst = edge_index[1] if edge_index.size else np.array([], dtype=np.int64)
    deg = np.bincount(dst, minlength=node) if dst.size else np.zeros(node, dtype=np.int64)
    return edge_index, deg

HEX_EDGE_INDEX_NP, HEX_DEG_NP = build_hex_edge_index([len(r) for r in TOP_ROWS_LIST])

def hex_rows_to_padded_map(rows, pad_value=-1):
    max_len = max(len(r) for r in rows) + 1
    H = len(rows)
    grid = pad_value * np.ones((H, max_len), dtype=np.int32)
    for ridx, row in enumerate(rows):
        shift = ridx % 2 
        grid[ridx, shift:shift+len(row)] = row
    return grid

TOP_INDEX_MAP = hex_rows_to_padded_map(TOP_ROWS_LIST)
BOTTOM_INDEX_MAP = hex_rows_to_padded_map(BOTTOM_ROWS_LIST)


# =========================================================
# 2. DATA GATHERING HELPERS
# =========================================================

def gather_face(x_batch: torch.Tensor, index_map: np.ndarray) -> torch.Tensor:
    """
    x_batch: (B, 4760, C)
    Returns: (B, C, H, W)
    """
    device = x_batch.device
    B, N, C = x_batch.shape
    H, W = index_map.shape
    
    idx_flat = torch.from_numpy(index_map.reshape(-1)).to(device)
    mask = (idx_flat >= 0)
    
    safe_idx = idx_flat.clone()
    safe_idx[~mask] = 0
    
    # Gather: (B, H*W, C)
    vals = torch.index_select(x_batch, 1, safe_idx)
    
    # Zero out padding pixels
    # (B, H*W, C)
    vals[:, ~mask] = 0.0 # mask is (H*W,)
    
    # Reshape to image: (B, H*W, C) -> (B, C, H, W)
    vals = vals.view(B, H, W, C).permute(0, 3, 1, 2)
    return vals

def gather_hex_nodes(x_batch: torch.Tensor, flat_indices: torch.Tensor) -> torch.Tensor:
    """
    x_batch: (B, 4760, C)
    Returns: (B, N_hex, C)
    """
    vals = torch.index_select(x_batch, 1, flat_indices.to(x_batch.device))
    return vals

def build_outer_fine_grid_tensor(x_batch: torch.Tensor, pool_kernel=None) -> torch.Tensor:
    """
    Constructs the high-res Outer Face image (Finegrid mode).
    Handles N-channel inputs (e.g., Npho + Time).
    
    Logic:
    - Channel 0 (Npho): Extensive property. Value is divided by pixel area (density).
    - Channel 1+ (Time): Intensive property. Value is broadcasted (copied).
    """
    device = x_batch.device
    B, N, C = x_batch.shape
    
    # 1. Gather Coarse (B,C,9,24) and Center (B,C,5,6) faces
    coarse = gather_face(x_batch, OUTER_COARSE_FULL_INDEX_MAP)
    center = gather_face(x_batch, OUTER_CENTER_INDEX_MAP)
    
    # 2. Define Scales
    # Coarse PMT -> 5x3 pixels (Area=15)
    cr, cc = OUTER_FINE_COARSE_SCALE 
    # Center PMT -> 3x2 pixels (Area=6)
    sr, sc = OUTER_FINE_CENTER_SCALE 
    
    # 3. Upsample to Fine Grid using Nearest Neighbor (Broadcasting)
    # Coarse: (9, 24) -> (45, 72)
    fine_from_coarse = F.interpolate(coarse, scale_factor=(cr, cc), mode='nearest')
    # Center: (5, 6) -> (15, 12)
    fine_from_center = F.interpolate(center, scale_factor=(sr, sc), mode='nearest')
    
    # 4. Apply Physical Scaling Rules
    # Channel 0 (Npho): Divide by area to conserve total charge sum
    # We clone the slice to avoid in-place modification errors if needed
    
    # Scale Coarse Channel 0 by 15.0
    fine_from_coarse[:, 0] = fine_from_coarse[:, 0] / float(cr * cc)
    
    # Scale Center Channel 0 by 6.0
    fine_from_center[:, 0] = fine_from_center[:, 0] / float(sr * sc)
    
    # (Channel 1 is Time: We leave it as is. T=10ns on 1 PMT becomes T=10ns on 15 pixels.)

    # 5. Stitching
    # Place the Center patch into the hole in the Coarse grid
    
    # Calculate offset in Fine coordinates
    c_start_r, c_start_c = OUTER_FINE_CENTER_START # (3, 10) in coarse indices
    top_fine = c_start_r * cr
    left_fine = c_start_c * cc
    
    # Get dimensions of the center patch
    h_fine_c = fine_from_center.shape[2]
    w_fine_c = fine_from_center.shape[3]
    
    # Final assembly
    fine_grid = fine_from_coarse.clone()
    fine_grid[:, :, top_fine : top_fine + h_fine_c, left_fine : left_fine + w_fine_c] = fine_from_center
    
    if pool_kernel:
        fine_grid = F.avg_pool2d(fine_grid, kernel_size=pool_kernel, stride=pool_kernel)
        
    return fine_grid

def build_face_tensors(npho_batch: torch.Tensor):
    faces = {}
    faces["inner"] = gather_face(npho_batch, INNER_INDEX_MAP)
    faces["us"]    = gather_face(npho_batch, US_INDEX_MAP)
    faces["ds"]    = gather_face(npho_batch, DS_INDEX_MAP)
    return faces


# =========================================================
# 3. VISUALIZATION
# =========================================================

def plot_event_faces(npho_event, title="Event Faces", savepath=None, outer_mode="split", outer_fine_pool=None):
    """
    Geometry-aware visualization with correct physical arrangement.
    Layout (Left to Right): [Downstream] [Inner] [Upstream] [Outer]
    Top/Bottom are aligned with Inner.
    Orientation corrected as per user instructions.
    """
    # 1. Prepare Data
    npho_np = npho_event.reshape(-1)
    
    # Reshape to (1, 4760, 1) for gather functions
    x = torch.from_numpy(npho_np.reshape(1, -1, 1).astype("float32"))
    
    faces = build_face_tensors(x)
    outer_fine_fused = build_outer_fine_grid_tensor(x, pool_kernel=None)

    def to_np(t): return t.squeeze(0).squeeze(0).cpu().numpy()

    # Determine global min/max for color consistency
    valid_all = npho_np[npho_np > 0]
    if valid_all.size > 0:
        is_relative = (valid_all.max() <= 1.0001)
        floor = 0.001 if is_relative else 0.1
        vmin = max(valid_all.min(), floor)
        vmax = valid_all.max()
        if vmax <= vmin:
            vmax = vmin + (0.1 if is_relative else 1.0)
            vmin = vmin * 0.99
    else:
        vmin, vmax = 0.001, 1.0

    norm = LogNorm(vmin=vmin, vmax=vmax)
    # norm = Normalize(vmin=vmin, vmax=vmax)

    # 2. Setup Figure
    # New Layout: DS | Inner | US | Outer
    # Widths: 45 | 100 | 45 | 100
    # Heights: Top=45, Mid=160, Bot=45
    fig = plt.figure(figsize=(20, 14))
    width_ratios = [45, 88, 45, 100]
    height_ratios = [100, 183, 100]
    gs = gridspec.GridSpec(3, 4, 
                           width_ratios=width_ratios, 
                           height_ratios=height_ratios,
                           wspace=0.1, hspace=0.1)

    # --- PLOT HELPER (RECTANGULAR) ---
    def plot_rect_face(ax, data, title, flip_lr=False):
        if flip_lr:
            data = np.fliplr(data)
        im = ax.imshow(data, aspect='auto', origin='upper', cmap="viridis", norm=norm)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return im

    # --- PLOT HELPER (HEXAGONAL) ---
    def plot_hex_face(ax, row_list, full_npho, title, mode):
        """
        mode='top': Row 0 is at Bottom. X inverted (Right->Left).
        mode='bottom': Row 0 is at Top. X inverted (Right->Left).
        """
        xs, ys, vals = [], [], []
        
        pitch_y = 7.5
        pitch_x = 7.1
        
        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            # Standard center: -(N-1)*P/2 .. +(N-1)*P/2
            x_start = -(n_items - 1) * pitch_x / 2.0
            
            # Determine X and Y based on mode
            if mode == 'top':
                # Row 0 (r_idx=0) is "Inner Edge" -> Bottom of Plot (Y=0)
                # Row 5 is Top (Y=max)
                y_pos = r_idx * pitch_y
            elif mode == 'bottom':
                # Row 0 (r_idx=0) is "Inner Edge" -> Top of Plot (Y=max)
                # Row 5 is Bottom (Y=0)
                # We start high and subtract
                y_pos = (5 - r_idx) * pitch_y
            
            for c_idx, pmt_id in enumerate(ids):
                # Standard order is Left->Right.
                # User wants 0th element on Right (Upstream side), Last on Left.
                # So we invert X.
                x_raw = x_start + c_idx * pitch_x
                x = -x_raw 
                y = y_pos

                val = full_npho[pmt_id]
                xs.append(x)
                ys.append(y)
                vals.append(val)
        
        xs = np.array(xs)
        ys = np.array(ys)
        vals = np.array(vals)
        
        marker_size = 280 
        
        # Background
        ax.scatter(xs, ys, s=marker_size, c='lightgray', marker='h', alpha=0.3)
        
        # Active
        mask = vals > 0
        if np.any(mask):
            ax.scatter(xs[mask], ys[mask], c=vals[mask], s=marker_size, 
                       cmap="viridis", norm=norm, marker='h', edgecolors='none')
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        
        # Limits
        limit_x = 55
        # Y is approx 0 to 5*7.5 = 37.5. Add padding.
        ax.set_xlim(-limit_x, limit_x)
        ax.set_ylim(-5, 45) 
        ax.axis('off')

    # --- TOP ROW ---
    # Top Face: Row 0 is Bottom, X inverted.
    ax_top = plt.subplot(gs[0, 1])
    plot_hex_face(ax_top, TOP_ROWS_LIST, npho_np, "Top", mode='top')

    # --- MIDDLE ROW ---
    # 1. Downstream (Left)
    ax_ds = plt.subplot(gs[1, 0])
    # DS: 4452 (0,0) is Inner/Top -> Top-Right.
    # Standard imshow puts (0,0) at Top-Left. So we Flip LR.
    plot_rect_face(ax_ds, to_np(faces["ds"]), "Downstream", flip_lr=True)

    # 2. Inner (Center)
    ax_inner = plt.subplot(gs[1, 1])
    # Inner: 0 (0,0) is Inner/Top/US -> Top-Right.
    # Standard imshow puts (0,0) at Top-Left. So we Flip LR.
    im_main = plot_rect_face(ax_inner, to_np(faces["inner"]), "Inner", flip_lr=True)

    # 3. Upstream (Right)
    ax_us = plt.subplot(gs[1, 2])
    # US: 4308 (0,0) is Inner/Top -> Top-Left.
    # Standard imshow puts (0,0) at Top-Left. No Flip.
    plot_rect_face(ax_us, to_np(faces["us"]), "Upstream", flip_lr=False)

    # 4. Outer (Far Right)
    ax_outer = plt.subplot(gs[1, 3])
    plot_rect_face(ax_outer, to_np(outer_fine_fused), "Outer", flip_lr=True)
    rect = patches.Rectangle((29.5, 14.5), 12, 15, linewidth=1.5, edgecolor='white', facecolor='none', linestyle=':')
    ax_outer.add_patch(rect)

    # --- BOTTOM ROW ---
    # Bottom Face: Row 0 is Top, X inverted.
    ax_bot = plt.subplot(gs[2, 1])
    plot_hex_face(ax_bot, BOTTOM_ROWS_LIST, npho_np, "Bottom", mode='bottom')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    fig.colorbar(im_main, cax=cbar_ax, label="NPho")
    fig.suptitle(title, fontsize=16)
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=120)
        plt.close()
    else:
        plt.show()

def plot_event_time(npho_data, time_data, title="Event Time", savepath=None):
    """
    Visualizes the time distribution.
    - Uses Robust Scaling (5th-95th percentile) to ignore outliers and enhance contrast.
    - Uses 'coolwarm' colormap.
    """
    
    # 1. Data Cleaning
    t_clean = time_data.copy()
    
    # Garbage Filter (safeguard against huge default values like 1e10)
    mask_garbage = np.abs(t_clean) > 1.0 
    
    # Validity Mask: Positive photons AND valid time
    mask_valid = (npho_data > 0) & (~mask_garbage)
    
    # Zero out invalid data for safety
    t_clean[~mask_valid] = 0.0
    
    # 2. Determine Dynamic Range (Robust)
    valid_vals = t_clean[mask_valid]
    
    if valid_vals.size > 0:
        # --- CHANGE: ROBUST SCALING ---
        # Instead of min/max, we take the 5th and 95th percentiles.
        # This ignores the top/bottom 5% of extreme outliers.
        vmin, vmax = np.percentile(valid_vals, [0, 100])
        
        # Safety check: if vmin == vmax (e.g. all values are identical), expand slightly
        if vmin == vmax:
            vmin -= 1e-9
            vmax += 1e-9
            
        # Optional: If you want to ensure the range is symmetric around 0 
        # (so 0 is always white), uncomment these lines:
        # limit = max(abs(vmin), abs(vmax))
        # vmin, vmax = -limit, limit
    else:
        vmin, vmax = -1.0, 1.0

    # Linear scale with robust limits
    # Values outside [vmin, vmax] will appear as the darkest blue/red (saturated)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "coolwarm" 

    # 3. Build Tensors (Same as before)
    x_npho = torch.from_numpy(npho_data.reshape(1, -1, 1).astype("float32"))
    x_time = torch.from_numpy(t_clean.reshape(1, -1, 1).astype("float32"))
    
    faces_npho = build_face_tensors(x_npho)
    faces_time = build_face_tensors(x_time)
    
    outer_time_fused = build_outer_fine_grid_tensor(x_time, pool_kernel=None)
    outer_npho_fused = build_outer_fine_grid_tensor(x_npho, pool_kernel=None)

    def to_np(t): return t.squeeze(0).squeeze(0).cpu().numpy()

    # 4. Setup Figure Layout
    fig = plt.figure(figsize=(22, 14))
    width_ratios = [45, 100, 45, 100]
    height_ratios = [45, 160, 45]
    gs = gridspec.GridSpec(3, 4, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.1, hspace=0.1)

    # --- HELPER: Rectangular Faces ---
    def plot_face(ax, t_tensor, n_tensor, title, flip_lr=False):
        t_img = to_np(t_tensor)
        n_img = to_np(n_tensor)
        
        if flip_lr:
            t_img = np.fliplr(t_img)
            n_img = np.fliplr(n_img)
        
        masked_t = np.ma.masked_where(n_img <= 0, t_img)
        
        im = ax.imshow(masked_t, aspect='auto', origin='upper', cmap=cmap, norm=norm)
        ax.set_facecolor('lightgray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return im

    # --- HELPER: Hex Faces ---
    def plot_hex_time(ax, row_list, full_t, full_n, title, mode):
        pitch_y, pitch_x = 7.5, 7.1
        xs, ys, vals = [], [], []
        
        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            x_start = -(n_items - 1) * pitch_x / 2.0
            
            if mode == 'top':
                y_pos = r_idx * pitch_y
            elif mode == 'bottom':
                y_pos = (5 - r_idx) * pitch_y
            
            for c_idx, pmt_id in enumerate(ids):
                if not mask_valid[pmt_id]:
                    continue
                
                x = -(x_start + c_idx * pitch_x)
                y = y_pos
                
                xs.append(x)
                ys.append(y)
                vals.append(full_t[pmt_id])

        # Background dots
        all_xs, all_ys = [], []
        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            x_start = -(n_items - 1) * pitch_x / 2.0
            y_pos = r_idx * pitch_y if mode == 'top' else (5 - r_idx) * pitch_y
            for c_idx, _ in enumerate(ids):
                all_xs.append(-(x_start + c_idx * pitch_x))
                all_ys.append(y_pos)
        
        ax.scatter(all_xs, all_ys, s=280, c='lightgray', marker='h', alpha=0.3)

        # Active dots
        if vals:
            ax.scatter(xs, ys, c=vals, s=280, cmap=cmap, norm=norm, marker='h', edgecolors='none')

        ax.set_xlim(-55, 55)
        ax.set_ylim(-5, 45)
        ax.axis('off')
        ax.set_title(title, fontsize=10, fontweight='bold')

    # --- EXECUTE PLOTS ---
    ax_top = plt.subplot(gs[0, 1])
    plot_hex_time(ax_top, TOP_ROWS_LIST, t_clean, npho_data, "Top (Time)", mode='top')

    ax_ds = plt.subplot(gs[1, 0])
    plot_face(ax_ds, faces_time["ds"], faces_npho["ds"], "Downstream", flip_lr=True)
    
    ax_inner = plt.subplot(gs[1, 1])
    im_main = plot_face(ax_inner, faces_time["inner"], faces_npho["inner"], "Inner", flip_lr=True)
    
    ax_us = plt.subplot(gs[1, 2])
    plot_face(ax_us, faces_time["us"], faces_npho["us"], "Upstream", flip_lr=False)
    
    ax_outer = plt.subplot(gs[1, 3])
    plot_face(ax_outer, outer_time_fused, outer_npho_fused, "Outer", flip_lr=True)
    rect = patches.Rectangle((29.5, 14.5), 12, 15, linewidth=1.5, edgecolor='black', facecolor='none', linestyle=':')
    ax_outer.add_patch(rect)

    ax_bot = plt.subplot(gs[2, 1])
    plot_hex_time(ax_bot, BOTTOM_ROWS_LIST, t_clean, npho_data, "Bottom (Time)", mode='bottom')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    fmt = "%.1e" # Force scientific notation for small numbers like 1e-7
    fig.colorbar(im_main, cax=cbar_ax, label=f"Time [{title}]", format=fmt)
    fig.suptitle(title, fontsize=16)
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=120)
        plt.close()
    else:
        plt.show()
        
# =========================================================
# 4. MODEL CLASSES
# =========================================================

class FaceBackbone(nn.Module):
    def __init__(self, in_channels=1, base_channels=16, pooled_hw=(4, 4)):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(pooled_hw)
        self.out_dim = base_channels*2 * pooled_hw[0] * pooled_hw[1]
    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        # return x.view(x.size(0), -1)
        return x.flatten(1)

class HexGraphConv(nn.Module):
    """
    Simple message-passing layer for hex grids.
    out = W_self * x + mean_neigh( W_neigh * x_neigh )
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.self_lin = nn.Linear(in_dim, out_dim)
        self.neigh_lin = nn.Linear(in_dim, out_dim)
        self.act = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x, edge_index, deg):
        """
        x: (B, N, Cin)
        edge_index: (2, E) [src, dst]
        deg: (N,) in-degree per node
        """
        src, dst = edge_index  # (E,)
        B, N, _ = x.shape

        # 1. Force calculation in float32 for numerical stability in aggregation
        x_f = x.float() 

        # 2. Compute messages
        # Note: Even if input is float32, under AMP this linear layer 
        # might output bfloat16/float16.
        msgs = self.neigh_lin(x_f[:, src, :])  # (B, E, Cout)
        
        # 3. Prepare accumulator with specific dtype (float32)
        agg = torch.zeros(B, N, msgs.size(-1), device=x.device, dtype=x_f.dtype)
        idx = dst.view(1, -1, 1).expand(B, -1, msgs.size(-1))

        msgs = msgs.to(agg.dtype) 

        agg.scatter_add_(1, idx, msgs)
        agg = agg / deg.to(agg.dtype).clamp(min=1).view(1, -1, 1)

        self_out = self.self_lin(x_f)
        out = self_out + agg
        out = self.act(out)
        
        # Cast back to original input dtype (e.g. if input was half precision)
        return out.to(x.dtype)

class HexGraphEncoder(nn.Module):
    def __init__(self, in_dim=1, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.conv1 = HexGraphConv(in_dim, hidden_dim)
        self.conv2 = HexGraphConv(hidden_dim, hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.LeakyReLU(0.1), 
            nn.Linear(hidden_dim, embed_dim)
        )
    def forward(self, node_feats, edge_index, deg):
        x = self.conv1(node_feats, edge_index, deg)
        x = self.conv2(x, edge_index, deg)
        return self.proj(x.mean(dim=1))

class AngleRegressorSharedFaces(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=2, outer_mode="split", outer_fine_pool=None):
        super().__init__()
        self.outer_mode = outer_mode
        self.outer_fine_pool = outer_fine_pool
        
        input_channels = 2 # Npho, Time
        
        self.backbone = FaceBackbone(in_channels=input_channels, base_channels=16, pooled_hw=(4, 4))
        self.hex_embed_dim = self.backbone.out_dim
        self.hex_encoder = HexGraphEncoder(in_dim=input_channels, embed_dim=self.hex_embed_dim, hidden_dim=64)
        
        if outer_mode == "split":
            self.cnn_face_names = ["inner", "outer_coarse", "outer_center", "us", "ds"]
            self.outer_fine = False
            extra_cnn = 0
        elif outer_mode == "finegrid":
            self.cnn_face_names = ["inner", "us", "ds"]
            self.outer_fine = True
            extra_cnn = 1
        
        self.register_buffer("top_hex_indices", torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long())
        self.register_buffer("bottom_hex_indices", torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long())
        self.register_buffer("hex_edge_index", torch.from_numpy(HEX_EDGE_INDEX_NP).long())
        self.register_buffer("hex_deg", torch.from_numpy(HEX_DEG_NP.astype(np.float32)))

        in_fc = self.backbone.out_dim * (len(self.cnn_face_names) + extra_cnn) + self.hex_embed_dim * 2
        self.head = nn.Sequential(
            nn.Linear(in_fc, hidden_dim), nn.LeakyReLU(0.1), nn.Dropout(0.2), nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x_batch):
        """
        x_batch: (B, 4760, 2)
        """
        # faces = build_face_tensors(npho_batch)
        faces = {}
        faces["inner"] = gather_face(x_batch, INNER_INDEX_MAP)
        faces["us"]    = gather_face(x_batch, US_INDEX_MAP)
        faces["ds"]    = gather_face(x_batch, DS_INDEX_MAP)
        if self.outer_mode == "split":
            faces["outer_coarse"] = gather_face(x_batch, OUTER_COARSE_FULL_INDEX_MAP)
            faces["outer_center"] = gather_face(x_batch, OUTER_CENTER_INDEX_MAP)

        embeddings = []
        for name in self.cnn_face_names:
            embeddings.append(self.backbone(faces[name]))
        
        if self.outer_fine:
            outer_fine = build_outer_fine_grid_tensor(x_batch, pool_kernel=self.outer_fine_pool)
            embeddings.append(self.backbone(outer_fine))
        
        edge_index, deg = self.hex_edge_index, self.hex_deg
        top_nodes = gather_hex_nodes(x_batch, self.top_hex_indices)
        bot_nodes = gather_hex_nodes(x_batch, self.bottom_hex_indices)
        
        embeddings.append(self.hex_encoder(top_nodes, edge_index, deg))
        embeddings.append(self.hex_encoder(bot_nodes, edge_index, deg))
        
        return self.head(torch.cat(embeddings, dim=1))