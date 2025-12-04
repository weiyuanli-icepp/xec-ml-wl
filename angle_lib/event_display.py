import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, Normalize
import torch
import torch.nn.functional as F

# --- IMPORTS FROM GEOMETRY MODULES ---
from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS
)
from .geom_utils import (
    gather_face, 
    build_outer_fine_grid_tensor
)

# =========================================================
# HELPER: Build Face Dictionary
# =========================================================

def build_face_tensors(npho_batch: torch.Tensor):
    """
    Local helper to organize faces into a dictionary for plotting.
    Uses the imported gather_face function.
    """
    faces = {}
    faces["inner"] = gather_face(npho_batch, INNER_INDEX_MAP)
    faces["us"]    = gather_face(npho_batch, US_INDEX_MAP)
    faces["ds"]    = gather_face(npho_batch, DS_INDEX_MAP)
    return faces

# =========================================================
# 3. VISUALIZATION FUNCTIONS
# =========================================================

def plot_event_faces(npho_event, title="Event Faces", savepath=None, outer_mode="split", outer_fine_pool=None):
    """
    Visualizes Photon Counts (Npho) on the geometry.
    Layout: [Downstream] [Inner] [Upstream] [Outer]
    """
    if isinstance(npho_event, torch.Tensor):
        npho_event = npho_event.cpu().numpy()
        
    npho_np = npho_event.reshape(-1)
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

    fig = plt.figure(figsize=(20, 14))
    width_ratios = [45, 88, 45, 100]
    height_ratios = [100, 183, 100]
    gs = gridspec.GridSpec(3, 4, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.1, hspace=0.1)

    # --- Plot Helpers ---
    def plot_rect_face(ax, data, title, flip_lr=False):
        if flip_lr:
            data = np.fliplr(data)
        im = ax.imshow(data, aspect='auto', origin='upper', cmap="viridis", norm=norm)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return im

    def plot_hex_face(ax, row_list, full_npho, title, mode):
        xs, ys, vals = [], [], []
        pitch_y, pitch_x = 7.5, 7.1
        
        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            x_start = -(n_items - 1) * pitch_x / 2.0
            y_pos = r_idx * pitch_y if mode == 'top' else (5 - r_idx) * pitch_y
            
            for c_idx, pmt_id in enumerate(ids):
                x = -(x_start + c_idx * pitch_x)
                y = y_pos
                val = full_npho[pmt_id]
                xs.append(x); ys.append(y); vals.append(val)
        
        xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)
        ax.scatter(xs, ys, s=280, c='lightgray', marker='h', alpha=0.3)
        
        mask = vals > 0
        if np.any(mask):
            ax.scatter(xs[mask], ys[mask], c=vals[mask], s=280, cmap="viridis", norm=norm, marker='h', edgecolors='none')
        
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.set_xlim(-55, 55)
        ax.set_ylim(-5, 45)
        ax.axis('off')

    # --- Execute Plots ---
    ax_top = plt.subplot(gs[0, 1])
    plot_hex_face(ax_top, TOP_HEX_ROWS, npho_np, "Top", mode='top')

    ax_ds = plt.subplot(gs[1, 0])
    plot_rect_face(ax_ds, to_np(faces["ds"]), "Downstream", flip_lr=True)

    ax_inner = plt.subplot(gs[1, 1])
    im_main = plot_rect_face(ax_inner, to_np(faces["inner"]), "Inner", flip_lr=True)

    ax_us = plt.subplot(gs[1, 2])
    plot_rect_face(ax_us, to_np(faces["us"]), "Upstream", flip_lr=False)

    ax_outer = plt.subplot(gs[1, 3])
    plot_rect_face(ax_outer, to_np(outer_fine_fused), "Outer", flip_lr=True)
    rect = patches.Rectangle((29.5, 14.5), 12, 15, linewidth=1.5, edgecolor='white', facecolor='none', linestyle=':')
    ax_outer.add_patch(rect)

    ax_bot = plt.subplot(gs[2, 1])
    plot_hex_face(ax_bot, BOTTOM_HEX_ROWS, npho_np, "Bottom", mode='bottom')

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
    Visualizes the time distribution on the geometry.
    Uses Robust Scaling (0-100th percentile of valid hits) and 'coolwarm' map.
    """
    if isinstance(npho_data, torch.Tensor): npho_data = npho_data.cpu().numpy()
    if isinstance(time_data, torch.Tensor): time_data = time_data.cpu().numpy()

    t_clean = time_data.copy()
    mask_garbage = np.abs(t_clean) > 1.0 
    mask_valid = (npho_data > 0) & (~mask_garbage)
    t_clean[~mask_valid] = 0.0
    
    valid_vals = t_clean[mask_valid]
    if valid_vals.size > 0:
        # Use full range of valid hits (0-100%) for contrast
        vmin, vmax = np.percentile(valid_vals, [0, 100])
        if vmin == vmax:
            vmin -= 1e-9; vmax += 1e-9
    else:
        vmin, vmax = -1.0, 1.0

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = "coolwarm" 

    # Build Tensors
    x_npho = torch.from_numpy(npho_data.reshape(1, -1, 1).astype("float32"))
    x_time = torch.from_numpy(t_clean.reshape(1, -1, 1).astype("float32"))
    
    faces_npho = build_face_tensors(x_npho)
    faces_time = build_face_tensors(x_time)
    
    outer_time_fused = build_outer_fine_grid_tensor(x_time, pool_kernel=None)
    outer_npho_fused = build_outer_fine_grid_tensor(x_npho, pool_kernel=None)

    def to_np(t): return t.squeeze(0).squeeze(0).cpu().numpy()

    fig = plt.figure(figsize=(22, 14))
    width_ratios = [45, 100, 45, 100]
    height_ratios = [45, 160, 45]
    gs = gridspec.GridSpec(3, 4, width_ratios=width_ratios, height_ratios=height_ratios, wspace=0.1, hspace=0.1)

    # --- Plot Helpers ---
    def plot_face(ax, t_tensor, n_tensor, title, flip_lr=False):
        t_img = to_np(t_tensor)
        n_img = to_np(n_tensor)
        if flip_lr:
            t_img = np.fliplr(t_img); n_img = np.fliplr(n_img)
        
        masked_t = np.ma.masked_where(n_img <= 0, t_img)
        im = ax.imshow(masked_t, aspect='auto', origin='upper', cmap=cmap, norm=norm)
        ax.set_facecolor('lightgray')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return im

    def plot_hex_time(ax, row_list, full_t, full_n, title, mode):
        pitch_y, pitch_x = 7.5, 7.1
        xs, ys, vals = [], [], []
        
        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            x_start = -(n_items - 1) * pitch_x / 2.0
            y_pos = r_idx * pitch_y if mode == 'top' else (5 - r_idx) * pitch_y
            
            for c_idx, pmt_id in enumerate(ids):
                if not mask_valid[pmt_id]: continue
                x = -(x_start + c_idx * pitch_x)
                y = y_pos
                xs.append(x); ys.append(y); vals.append(full_t[pmt_id])

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
        if vals:
            ax.scatter(xs, ys, c=vals, s=280, cmap=cmap, norm=norm, marker='h', edgecolors='none')

        ax.set_xlim(-55, 55); ax.set_ylim(-5, 45)
        ax.axis('off'); ax.set_title(title, fontsize=10, fontweight='bold')

    # --- Execute Plots ---
    ax_top = plt.subplot(gs[0, 1])
    plot_hex_time(ax_top, TOP_HEX_ROWS, t_clean, npho_data, "Top (Time)", mode='top')

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
    plot_hex_time(ax_bot, BOTTOM_HEX_ROWS, t_clean, npho_data, "Bottom (Time)", mode='bottom')

    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    fmt = "%.1e" # Force scientific notation for small numbers like 1e-7
    fig.colorbar(im_main, cax=cbar_ax, label=f"Time [{title}]", format=fmt)
    fig.suptitle(title, fontsize=16)
    
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=120)
        plt.close()
    else:
        plt.show()