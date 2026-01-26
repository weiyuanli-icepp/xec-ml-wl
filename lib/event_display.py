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
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    DEFAULT_NPHO_THRESHOLD
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


# =========================================================
# MAE Event Display: Truth vs Masked vs Prediction
# =========================================================

def plot_mae_comparison(x_truth, x_masked, mask, x_pred=None, event_idx=0,
                        channel="npho", title="MAE Reconstruction", savepath=None,
                        include_top_bottom=False,
                        time_invalid_mask=None, npho_threshold=None):
    """
    Side-by-side comparison of truth, masked input, and MAE prediction.

    Args:
        x_truth: (B, 4760, 2) or (4760, 2) - ground truth [npho, time]
        x_masked: (B, 4760, 2) or (4760, 2) - masked input
        mask: (B, 4760) or (4760,) - 1 where masked, 0 where visible
        x_pred: (B, 4760, 2) or (4760, 2) - MAE prediction (optional)
        event_idx: which event to display if batched
        channel: "npho" (channel 0) or "time" (channel 1)
        title: plot title
        savepath: path to save figure (None to display)
        include_top_bottom: include top/bottom hex faces in the comparison grid
        time_invalid_mask: (B, 4760) or (4760,) - 1 where time is invalid (npho <= threshold)
                          If None and channel=="time" and npho_threshold is set, computed automatically
        npho_threshold: threshold for time validity (in normalized npho space)
                       If None, uses DEFAULT_NPHO_THRESHOLD converted to normalized space
    """
    # Handle batched input
    if x_truth.ndim == 3:
        x_truth = x_truth[event_idx]
        x_masked = x_masked[event_idx]
        mask = mask[event_idx]
        if x_pred is not None:
            x_pred = x_pred[event_idx]
        if time_invalid_mask is not None and time_invalid_mask.ndim == 2:
            time_invalid_mask = time_invalid_mask[event_idx]

    if isinstance(x_truth, torch.Tensor):
        x_truth = x_truth.cpu().numpy()
    if isinstance(x_masked, torch.Tensor):
        x_masked = x_masked.cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    if isinstance(x_pred, torch.Tensor):
        x_pred = x_pred.cpu().numpy()
    if isinstance(time_invalid_mask, torch.Tensor):
        time_invalid_mask = time_invalid_mask.cpu().numpy()

    ch_idx = 0 if channel == "npho" else 1

    # Auto-compute time_invalid_mask for time channel if not provided
    if channel == "time" and time_invalid_mask is None:
        # Use npho channel (index 0) from truth to determine time validity
        npho_truth = x_truth[:, 0]
        if npho_threshold is None:
            # Convert default raw threshold to normalized space
            # Assuming normalization: npho_norm = log1p(raw / 0.58) / 1.0
            from .geom_defs import DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2
            npho_threshold = np.log1p(DEFAULT_NPHO_THRESHOLD / DEFAULT_NPHO_SCALE) / DEFAULT_NPHO_SCALE2
        time_invalid_mask = (npho_truth <= npho_threshold).astype(np.float32)
    ch_label = "Npho" if channel == "npho" else "Time"

    truth_ch = x_truth[:, ch_idx]
    masked_ch = x_masked[:, ch_idx]
    pred_ch = x_pred[:, ch_idx] if x_pred is not None else masked_ch

    # Build tensor for gathering faces
    def to_tensor(arr):
        return torch.from_numpy(arr.reshape(1, -1, 1).astype("float32"))

    truth_t = to_tensor(truth_ch)
    masked_t = to_tensor(masked_ch)
    pred_t = to_tensor(pred_ch)
    mask_t = to_tensor(mask.astype("float32"))

    # Gather faces
    faces_truth = build_face_tensors(truth_t)
    faces_masked = build_face_tensors(masked_t)
    faces_pred = build_face_tensors(pred_t)
    faces_mask = build_face_tensors(mask_t)

    outer_truth = build_outer_fine_grid_tensor(truth_t, pool_kernel=None)
    outer_masked = build_outer_fine_grid_tensor(masked_t, pool_kernel=None)
    outer_pred = build_outer_fine_grid_tensor(pred_t, pool_kernel=None)
    outer_mask = build_outer_fine_grid_tensor(mask_t, pool_kernel=None)

    # Time-invalid mask for visualization (only used for time channel)
    if time_invalid_mask is not None:
        time_invalid_t = to_tensor(time_invalid_mask.astype("float32"))
        faces_time_invalid = build_face_tensors(time_invalid_t)
        outer_time_invalid = build_outer_fine_grid_tensor(time_invalid_t, pool_kernel=None)
    else:
        faces_time_invalid = None
        outer_time_invalid = None

    def to_np(t):
        return t.squeeze(0).squeeze(0).cpu().numpy()

    # Compute percentage residual where masked: (pred - truth) / |truth| * 100
    # Use small epsilon to avoid division by zero
    residual_ch = np.where(
        mask > 0.5,
        (pred_ch - truth_ch) / (np.abs(truth_ch) + 1e-6) * 100.0,
        0.0
    )
    residual_t = to_tensor(residual_ch)
    faces_residual = build_face_tensors(residual_t)
    outer_residual = build_outer_fine_grid_tensor(residual_t, pool_kernel=None)

    # Determine color scale
    valid_truth = truth_ch[mask < 0.5]  # visible pixels
    if len(valid_truth) > 0:
        vmin, vmax = np.percentile(valid_truth, [2, 98])
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1
    else:
        vmin, vmax = -1, 1

    norm_main = Normalize(vmin=vmin, vmax=vmax)
    cmap_main_name = "viridis" if channel == "npho" else "coolwarm"
    cmap_main = plt.get_cmap(cmap_main_name)
    cmap_main_masked = cmap_main.copy()
    cmap_main_masked.set_bad("black")

    # Residual scale (symmetric around 0)
    res_abs = np.abs(residual_ch[mask > 0.5])
    if len(res_abs) > 0:
        res_max = np.percentile(res_abs, 98)
    else:
        res_max = 1.0
    norm_res = Normalize(vmin=-res_max, vmax=res_max)
    cmap_res = "RdBu_r"

    def plot_hex_grid(ax, row_list, full_vals, title, mode, cmap, norm, mask_vals=None, time_invalid_vals=None):
        pitch_y, pitch_x = 7.5, 7.1
        xs, ys, vals = [], [], []
        xs_masked_valid, ys_masked_valid = [], []  # Masked with valid time
        xs_masked_invalid, ys_masked_invalid = [], []  # Masked with invalid time (npho <= threshold)
        all_xs, all_ys = [], []

        for r_idx, ids in enumerate(row_list):
            n_items = len(ids)
            x_start = -(n_items - 1) * pitch_x / 2.0
            y_pos = r_idx * pitch_y if mode == 'top' else (5 - r_idx) * pitch_y

            for c_idx, pmt_id in enumerate(ids):
                x = -(x_start + c_idx * pitch_x)
                y = y_pos
                all_xs.append(x)
                all_ys.append(y)
                if mask_vals is not None and mask_vals[pmt_id] > 0.5:
                    # Check if this masked sensor has invalid time
                    if time_invalid_vals is not None and time_invalid_vals[pmt_id] > 0.5:
                        xs_masked_invalid.append(x)
                        ys_masked_invalid.append(y)
                    else:
                        xs_masked_valid.append(x)
                        ys_masked_valid.append(y)
                else:
                    xs.append(x)
                    ys.append(y)
                    vals.append(full_vals[pmt_id])

        ax.scatter(all_xs, all_ys, s=280, c='lightgray', marker='h', alpha=0.3)
        if xs:
            ax.scatter(xs, ys, c=vals, s=280, cmap=cmap, norm=norm, marker='h', edgecolors='none')
        if xs_masked_valid:
            ax.scatter(xs_masked_valid, ys_masked_valid, s=280, c='black', marker='h', edgecolors='none')
        if xs_masked_invalid:
            # Time-invalid masked sensors: gray with white edge (clearly "no data")
            ax.scatter(xs_masked_invalid, ys_masked_invalid, s=280, c='#606060', marker='h',
                      edgecolors='white', linewidths=2.0)
        ax.set_xlim(-55, 55)
        ax.set_ylim(-5, 45)
        ax.axis('off')
        ax.set_title(title)

    ncols = 6 if include_top_bottom else 4
    fig_width = 22 if include_top_bottom else 16
    fig, axes = plt.subplots(5, ncols, figsize=(fig_width, 15))
    row_labels = ["Truth", "Masked", "Pred", "Residual (%)", "Mask"]
    col_labels = ["Inner", "Upstream", "Downstream", "Outer"]

    face_keys = ["inner", "us", "ds"]

    for col_idx, face_key in enumerate(face_keys):
        # Truth
        im = axes[0, col_idx].imshow(to_np(faces_truth[face_key]), aspect='auto',
                                      origin='upper', cmap=cmap_main, norm=norm_main)
        axes[0, col_idx].set_title(f"{col_labels[col_idx]} - Truth")
        axes[0, col_idx].axis('off')

        # Masked - show time-invalid sensors differently
        mask_face = to_np(faces_mask[face_key]) > 0.5
        masked_img = np.ma.array(to_np(faces_masked[face_key]), mask=mask_face)
        axes[1, col_idx].imshow(masked_img, aspect='auto',
                                 origin='upper', cmap=cmap_main_masked, norm=norm_main)
        # Overlay time-invalid masked sensors with hatching pattern (gray with crosshatch)
        if faces_time_invalid is not None:
            time_inv_face = to_np(faces_time_invalid[face_key]) > 0.5
            combined_mask = mask_face & time_inv_face  # masked AND time-invalid
            if np.any(combined_mask):
                # Create gray overlay with distinct pattern for time-invalid masked pixels
                overlay = np.zeros((*mask_face.shape, 4))
                overlay[combined_mask] = [0.4, 0.4, 0.4, 0.8]  # semi-transparent gray
                axes[1, col_idx].imshow(overlay, aspect='auto', origin='upper')
                # Add hatching using contourf for visual distinction
                axes[1, col_idx].contourf(combined_mask.astype(float), levels=[0.5, 1.5],
                                          colors='none', hatches=['//'], alpha=0)
        axes[1, col_idx].set_title(f"{col_labels[col_idx]} - Masked")
        axes[1, col_idx].axis('off')

        # Pred
        axes[2, col_idx].imshow(to_np(faces_pred[face_key]), aspect='auto',
                                 origin='upper', cmap=cmap_main, norm=norm_main)
        axes[2, col_idx].set_title(f"{col_labels[col_idx]} - Pred")
        axes[2, col_idx].axis('off')

        # Residual
        axes[3, col_idx].imshow(to_np(faces_residual[face_key]), aspect='auto',
                                 origin='upper', cmap=cmap_res, norm=norm_res)
        axes[3, col_idx].set_title(f"{col_labels[col_idx]} - Residual (%)")
        axes[3, col_idx].axis('off')

        # Mask - show time-invalid as different shade
        mask_display = to_np(faces_mask[face_key]).copy()
        if faces_time_invalid is not None:
            time_inv_face = to_np(faces_time_invalid[face_key]) > 0.5
            # Mark time-invalid masked sensors as 0.5 (gray) instead of 1 (white)
            mask_display[mask_face & time_inv_face] = 0.5
        axes[4, col_idx].imshow(mask_display, aspect='auto',
                                 origin='upper', cmap='gray', vmin=0, vmax=1)
        axes[4, col_idx].set_title(f"{col_labels[col_idx]} - Mask")
        axes[4, col_idx].axis('off')

    # Outer column
    axes[0, 3].imshow(to_np(outer_truth), aspect='auto', origin='upper', cmap=cmap_main, norm=norm_main)
    axes[0, 3].set_title("Outer - Truth")
    axes[0, 3].axis('off')

    outer_mask_face = to_np(outer_mask) > 0.5
    outer_mask_img = np.ma.array(to_np(outer_masked), mask=outer_mask_face)
    axes[1, 3].imshow(outer_mask_img, aspect='auto', origin='upper', cmap=cmap_main_masked, norm=norm_main)
    # Overlay time-invalid masked sensors with gray + hatching
    if outer_time_invalid is not None:
        outer_time_inv = to_np(outer_time_invalid) > 0.5
        combined_mask = outer_mask_face & outer_time_inv
        if np.any(combined_mask):
            overlay = np.zeros((*outer_mask_face.shape, 4))
            overlay[combined_mask] = [0.4, 0.4, 0.4, 0.8]  # gray instead of red
            axes[1, 3].imshow(overlay, aspect='auto', origin='upper')
            axes[1, 3].contourf(combined_mask.astype(float), levels=[0.5, 1.5],
                                colors='none', hatches=['//'], alpha=0)
    axes[1, 3].set_title("Outer - Masked")
    axes[1, 3].axis('off')

    axes[2, 3].imshow(to_np(outer_pred), aspect='auto', origin='upper', cmap=cmap_main, norm=norm_main)
    axes[2, 3].set_title("Outer - Pred")
    axes[2, 3].axis('off')

    axes[3, 3].imshow(to_np(outer_residual), aspect='auto', origin='upper', cmap=cmap_res, norm=norm_res)
    axes[3, 3].set_title("Outer - Residual (%)")
    axes[3, 3].axis('off')

    # Mask - show time-invalid as different shade
    outer_mask_display = to_np(outer_mask).copy()
    if outer_time_invalid is not None:
        outer_time_inv = to_np(outer_time_invalid) > 0.5
        outer_mask_display[outer_mask_face & outer_time_inv] = 0.5
    axes[4, 3].imshow(outer_mask_display, aspect='auto', origin='upper', cmap='gray', vmin=0, vmax=1)
    axes[4, 3].set_title("Outer - Mask")
    axes[4, 3].axis('off')

    if include_top_bottom:
        mask_norm = Normalize(vmin=0, vmax=1)
        row_vals = [truth_ch, masked_ch, pred_ch, residual_ch, mask.astype("float32")]
        row_cmaps = [cmap_main, cmap_main_masked, cmap_main, cmap_res, "gray"]
        row_norms = [norm_main, norm_main, norm_main, norm_res, mask_norm]
        hex_specs = [("Top", TOP_HEX_ROWS, "top"), ("Bottom", BOTTOM_HEX_ROWS, "bottom")]

        for row_idx, label in enumerate(row_labels):
            mask_vals = mask if row_idx == 1 else None
            # Only pass time_invalid for "Masked" row (row_idx=1)
            time_invalid_vals = time_invalid_mask if (row_idx == 1 and time_invalid_mask is not None) else None
            for col_offset, (hex_title, row_list, mode) in enumerate(hex_specs):
                ax = axes[row_idx, 4 + col_offset]
                subplot_title = f"{hex_title} - {label}"
                plot_hex_grid(
                    ax,
                    row_list,
                    row_vals[row_idx],
                    subplot_title,
                    mode=mode,
                    cmap=row_cmaps[row_idx],
                    norm=row_norms[row_idx],
                    mask_vals=mask_vals,
                    time_invalid_vals=time_invalid_vals,
                )

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.55, 0.015, 0.35])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_main, cmap=cmap_main), cax=cbar_ax1, label=ch_label)

    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.015, 0.35])
    fig.colorbar(plt.cm.ScalarMappable(norm=norm_res, cmap=cmap_res), cax=cbar_ax2, label="Residual (%)")

    # Summary stats
    mask_ratio = mask.mean() * 100
    masked_residuals = residual_ch[mask > 0.5]
    if len(masked_residuals) > 0:
        mape = np.mean(np.abs(masked_residuals))  # Already percentage
        rmspe = np.sqrt(np.mean(masked_residuals**2))  # Root mean squared percentage error
        stats_text = f"Mask: {mask_ratio:.1f}% | MAPE: {mape:.1f}% | RMSPE: {rmspe:.1f}%"
    else:
        stats_text = f"Mask: {mask_ratio:.1f}%"

    # Add time-invalid stats for time channel
    if time_invalid_mask is not None and channel == "time":
        n_masked = (mask > 0.5).sum()
        n_time_invalid = ((mask > 0.5) & (time_invalid_mask > 0.5)).sum()
        time_valid_pct = 100 * (1 - n_time_invalid / max(n_masked, 1))
        stats_text += f" | Time-valid: {time_valid_pct:.1f}% (gray+hatch=invalid)"

    fig.suptitle(f"{title}\n{stats_text}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])

    if savepath:
        plt.savefig(savepath, bbox_inches='tight', dpi=120)
        plt.close()
    else:
        plt.show()

    return fig
