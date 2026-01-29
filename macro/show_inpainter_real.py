#!/usr/bin/env python3
"""
Event Display for Real Data Inpainter Validation

Displays inpainter predictions on real data with dead channels.
Distinguishes between:
- Artificially masked sensors (mask_type=0): Has ground truth, shows residual
- Originally dead sensors (mask_type=1): No ground truth, shows prediction only

Usage:
    python macro/show_inpainter_real.py 0 \
        --predictions validation_real/real_data_predictions.root \
        --original DataGammaAngle_430000-431000.root \
        --channel npho --save event_0.pdf

    # Show specific run/event
    python macro/show_inpainter_real.py \
        --predictions validation_real/real_data_predictions.root \
        --original DataGammaAngle_430000-431000.root \
        --run 430123 --event 456 \
        --channel npho --save event.pdf
"""

import sys
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize, TwoSlopeNorm
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    DEFAULT_NPHO_SCALE, DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE
)
from lib.geom_utils import build_outer_fine_grid_tensor
import torch

# Constants
N_CHANNELS = 4760
REAL_DATA_SENTINEL = 1e10  # Sentinel value in PrepareRealData.C output

# Face configurations for rectangular faces
FACE_CONFIGS = {
    'inner': {'index_map': INNER_INDEX_MAP, 'title': 'Inner', 'type': 'rect'},
    'us': {'index_map': US_INDEX_MAP, 'title': 'Upstream', 'type': 'rect'},
    'ds': {'index_map': DS_INDEX_MAP, 'title': 'Downstream', 'type': 'rect'},
    # Note: outer face uses build_outer_fine_grid_tensor, not direct index map
}

# Hex face configurations (top/bottom PMT arrays)
HEX_FACE_CONFIGS = {
    'top': {'rows': TOP_HEX_ROWS, 'title': 'Top', 'mode': 'top'},
    'bot': {'rows': BOTTOM_HEX_ROWS, 'title': 'Bottom', 'mode': 'bottom'},
}

FACE_ID_TO_NAME = {0: 'inner', 1: 'us', 2: 'ds', 3: 'outer', 4: 'top', 5: 'bot'}


def load_real_data_event(original_file: str, event_idx: int = None,
                         run: int = None, event: int = None,
                         tree_name: str = "tree") -> dict:
    """
    Load a single event from real data ROOT file.

    Args:
        original_file: Path to original ROOT file
        event_idx: Event index (0-based)
        run: Run number (alternative to event_idx)
        event: Event number (alternative to event_idx)
        tree_name: Tree name

    Returns:
        Dictionary with event data
    """
    with uproot.open(original_file) as f:
        tree = f[tree_name]
        n_entries = tree.num_entries

        if run is not None and event is not None:
            # Find event by run/event number
            runs = tree['run'].array(library='np')
            events = tree['event'].array(library='np')

            match_idx = np.where((runs == run) & (events == event))[0]
            if len(match_idx) == 0:
                raise ValueError(f"Event run={run}, event={event} not found in file")
            event_idx = int(match_idx[0])
            print(f"[INFO] Found run={run}, event={event} at index {event_idx}")

        if event_idx is None:
            raise ValueError("Must specify either event_idx or (run, event)")

        if event_idx < 0 or event_idx >= n_entries:
            raise ValueError(f"Event index {event_idx} out of range (0-{n_entries-1})")

        # Load required branches
        branches = ['run', 'event', 'relative_npho', 'relative_time']

        # Optional branches
        optional = ['energyReco', 'timeReco', 'xyzRecoFI', 'uvwRecoFI', 'emiAng']
        available = tree.keys()
        branches += [b for b in optional if b in available]

        arrays = tree.arrays(branches, library='np',
                             entry_start=event_idx, entry_stop=event_idx+1)

        data = {
            'event_idx': event_idx,
            'run': int(arrays['run'][0]),
            'event': int(arrays['event'][0]),
            'relative_npho': arrays['relative_npho'][0].astype(np.float32),
            'relative_time': arrays['relative_time'][0].astype(np.float32),
        }

        # Add optional fields
        if 'energyReco' in arrays:
            data['energyReco'] = float(arrays['energyReco'][0])
        if 'xyzRecoFI' in arrays:
            data['xyzRecoFI'] = arrays['xyzRecoFI'][0]
        if 'uvwRecoFI' in arrays:
            data['uvwRecoFI'] = arrays['uvwRecoFI'][0]
        if 'emiAng' in arrays:
            data['emiAng'] = arrays['emiAng'][0]

    return data


def load_predictions_for_event(pred_file: str, event_idx: int) -> dict:
    """
    Load predictions for a specific event from validation output.

    Args:
        pred_file: Path to predictions ROOT file
        event_idx: Event index

    Returns:
        Dictionary with predictions organized by mask_type
    """
    with uproot.open(pred_file) as f:
        tree = f['predictions']
        arrays = tree.arrays(library='np')

    # Filter for this event
    mask = arrays['event_idx'] == event_idx

    if mask.sum() == 0:
        raise ValueError(f"No predictions found for event_idx={event_idx}")

    result = {
        'sensor_id': arrays['sensor_id'][mask],
        'face': arrays['face'][mask],
        'mask_type': arrays['mask_type'][mask],
        'pred_npho': arrays['pred_npho'][mask],
        'pred_time': arrays['pred_time'][mask],
        'truth_npho': arrays['truth_npho'][mask],
        'truth_time': arrays['truth_time'][mask],
        'error_npho': arrays['error_npho'][mask],
        'error_time': arrays['error_time'][mask],
        'run': int(arrays['run'][mask][0]),
        'event': int(arrays['event'][mask][0]),
    }

    # Count by mask type
    n_artificial = (result['mask_type'] == 0).sum()
    n_dead = (result['mask_type'] == 1).sum()
    print(f"[INFO] Loaded {len(result['sensor_id'])} predictions: "
          f"{n_artificial} artificial, {n_dead} dead")

    return result


def convert_sentinel(data: np.ndarray, from_val: float = REAL_DATA_SENTINEL,
                     to_val: float = np.nan, threshold: float = 1e9) -> np.ndarray:
    """Convert sentinel values."""
    result = data.copy()
    result[np.abs(data) > threshold] = to_val
    return result


def create_face_grid(sensor_values: np.ndarray, index_map: np.ndarray,
                     fill_value: float = np.nan) -> np.ndarray:
    """
    Create 2D grid for a face from flat sensor values.

    Args:
        sensor_values: Flat array of sensor values (4760,)
        index_map: 2D index map (H, W) with sensor indices
        fill_value: Value for invalid indices

    Returns:
        2D array (H, W)
    """
    H, W = index_map.shape
    grid = np.full((H, W), fill_value, dtype=np.float32)

    valid_mask = index_map >= 0
    valid_indices = index_map[valid_mask]
    grid[valid_mask] = sensor_values[valid_indices]

    return grid


def create_mask_grid(sensor_mask: np.ndarray, index_map: np.ndarray) -> np.ndarray:
    """
    Create 2D boolean mask grid for a face.

    Args:
        sensor_mask: Flat boolean array (4760,)
        index_map: 2D index map (H, W)

    Returns:
        2D boolean array (H, W)
    """
    H, W = index_map.shape
    grid = np.zeros((H, W), dtype=bool)

    valid_mask = index_map >= 0
    valid_indices = index_map[valid_mask]
    grid[valid_mask] = sensor_mask[valid_indices]

    return grid


def plot_face_panel(ax, grid: np.ndarray, title: str,
                    cmap: str = 'viridis', vmin: float = None, vmax: float = None,
                    norm=None, dead_mask: np.ndarray = None,
                    artificial_mask: np.ndarray = None,
                    show_colorbar: bool = True):
    """
    Plot a single face panel with optional mask overlays.

    Args:
        ax: Matplotlib axes
        grid: 2D array to plot
        title: Panel title
        cmap: Colormap name
        vmin, vmax: Color limits
        norm: Matplotlib normalization (overrides vmin/vmax)
        dead_mask: 2D boolean mask of dead channels (show with hatching)
        artificial_mask: 2D boolean mask of artificial masks (show with edge)
        show_colorbar: Whether to add colorbar
    """
    if norm is not None:
        im = ax.imshow(grid, cmap=cmap, norm=norm, aspect='auto', origin='lower')
    else:
        im = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto', origin='lower')

    # Overlay dead channels with hatching
    if dead_mask is not None and dead_mask.any():
        # Create hatched overlay for dead channels
        dead_overlay = np.ma.masked_where(~dead_mask, np.ones_like(grid))
        ax.imshow(dead_overlay, cmap='gray', alpha=0.7, aspect='auto', origin='lower')

        # Add hatching pattern
        H, W = grid.shape
        for i in range(H):
            for j in range(W):
                if dead_mask[i, j]:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                          fill=False, hatch='///',
                                          edgecolor='red', linewidth=0.5)
                    ax.add_patch(rect)

    # Overlay artificial masks with edge highlight
    if artificial_mask is not None and artificial_mask.any():
        H, W = grid.shape
        for i in range(H):
            for j in range(W):
                if artificial_mask[i, j]:
                    rect = plt.Rectangle((j-0.5, i-0.5), 1, 1,
                                          fill=False, edgecolor='blue',
                                          linewidth=1.5)
                    ax.add_patch(rect)

    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

    if show_colorbar:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return im


def build_outer_grid_from_sensor_values(sensor_values: np.ndarray) -> np.ndarray:
    """
    Build the outer face fine grid using build_outer_fine_grid_tensor.

    Args:
        sensor_values: Flat array of sensor values (4760,)

    Returns:
        2D array with the interpolated outer face grid
    """
    # Convert to tensor format expected by build_outer_fine_grid_tensor
    x = torch.from_numpy(sensor_values.reshape(1, -1, 1).astype(np.float32))
    # Get the fine grid (B, C, H, W) -> squeeze to (H, W)
    fine_grid = build_outer_fine_grid_tensor(x, pool_kernel=None)
    return fine_grid.squeeze(0).squeeze(0).cpu().numpy()


def plot_hex_face(ax, row_list: list, sensor_values: np.ndarray,
                  title: str, mode: str,
                  cmap: str = 'viridis', vmin: float = None, vmax: float = None,
                  norm=None, dead_mask: np.ndarray = None,
                  artificial_mask: np.ndarray = None):
    """
    Plot a hexagonal face (top/bottom PMT arrays).

    Args:
        ax: Matplotlib axes
        row_list: List of rows, each containing sensor IDs
        sensor_values: Flat array of sensor values (4760,)
        title: Panel title
        mode: 'top' or 'bottom' for layout direction
        cmap: Colormap name
        vmin, vmax: Color limits
        norm: Matplotlib normalization (overrides vmin/vmax)
        dead_mask: Boolean mask (4760,) of dead channels
        artificial_mask: Boolean mask (4760,) of artificial masks
    """
    pitch_y, pitch_x = 7.5, 7.1

    xs, ys, vals = [], [], []
    dead_xs, dead_ys = [], []
    art_xs, art_ys = [], []

    for r_idx, ids in enumerate(row_list):
        n_items = len(ids)
        x_start = -(n_items - 1) * pitch_x / 2.0
        y_pos = r_idx * pitch_y if mode == 'top' else (5 - r_idx) * pitch_y

        for c_idx, pmt_id in enumerate(ids):
            x = -(x_start + c_idx * pitch_x)
            y = y_pos
            val = sensor_values[pmt_id]

            # Check mask status
            is_dead = dead_mask is not None and dead_mask[pmt_id]
            is_art = artificial_mask is not None and artificial_mask[pmt_id]

            if is_dead:
                dead_xs.append(x)
                dead_ys.append(y)
            elif is_art:
                art_xs.append(x)
                art_ys.append(y)

            if not np.isnan(val):
                xs.append(x)
                ys.append(y)
                vals.append(val)

    xs, ys, vals = np.array(xs), np.array(ys), np.array(vals)

    # Background (all positions)
    all_xs, all_ys = [], []
    for r_idx, ids in enumerate(row_list):
        n_items = len(ids)
        x_start = -(n_items - 1) * pitch_x / 2.0
        y_pos = r_idx * pitch_y if mode == 'top' else (5 - r_idx) * pitch_y
        for c_idx, _ in enumerate(ids):
            all_xs.append(-(x_start + c_idx * pitch_x))
            all_ys.append(y_pos)

    ax.scatter(all_xs, all_ys, s=200, c='lightgray', marker='h', alpha=0.3)

    # Plot values
    if len(xs) > 0:
        if norm is not None:
            ax.scatter(xs, ys, c=vals, s=200, cmap=cmap, norm=norm,
                      marker='h', edgecolors='none')
        else:
            ax.scatter(xs, ys, c=vals, s=200, cmap=cmap, vmin=vmin, vmax=vmax,
                      marker='h', edgecolors='none')

    # Overlay dead channels
    if len(dead_xs) > 0:
        ax.scatter(dead_xs, dead_ys, s=200, c='gray', marker='h',
                  edgecolors='red', linewidths=1.5, hatch='///')

    # Overlay artificial masks
    if len(art_xs) > 0:
        ax.scatter(art_xs, art_ys, s=200, facecolors='none', marker='h',
                  edgecolors='blue', linewidths=2)

    ax.set_xlim(-55, 55)
    ax.set_ylim(-5, 45)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def plot_real_data_event(event_data: dict, predictions: dict,
                         channel: str = 'npho',
                         title: str = None,
                         save_path: str = None,
                         npho_scale: float = DEFAULT_NPHO_SCALE,
                         time_scale: float = DEFAULT_TIME_SCALE,
                         time_shift: float = DEFAULT_TIME_SHIFT):
    """
    Plot real data event with inpainter predictions.

    Layout (6 faces per row):
    - Row 1: Original data (with dead channels marked)
    - Row 2: Prediction (all masked sensors filled)
    - Row 3: Residual (only for artificial masks, dead channels marked as N/A)

    Faces: Top, Downstream, Inner, Upstream, Outer, Bottom

    Args:
        event_data: Dictionary from load_real_data_event
        predictions: Dictionary from load_predictions_for_event
        channel: 'npho' or 'time'
        title: Custom title
        save_path: Path to save figure
        npho_scale, time_scale, time_shift: Normalization parameters
    """
    # Get sensor values
    if channel == 'npho':
        original_raw = event_data['relative_npho'].copy()
        ch_idx = 0
    else:
        original_raw = event_data['relative_time'].copy()
        ch_idx = 1

    # Convert sentinel values to NaN for visualization
    original = convert_sentinel(original_raw, REAL_DATA_SENTINEL, np.nan)

    # Create masks
    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    artificial_mask = np.zeros(N_CHANNELS, dtype=bool)
    pred_values = np.full(N_CHANNELS, np.nan, dtype=np.float32)
    residual_values = np.full(N_CHANNELS, np.nan, dtype=np.float32)

    for i in range(len(predictions['sensor_id'])):
        sid = predictions['sensor_id'][i]
        mtype = predictions['mask_type'][i]

        if mtype == 0:
            artificial_mask[sid] = True
        else:
            dead_mask[sid] = True

        # Store prediction
        if channel == 'npho':
            pred_values[sid] = predictions['pred_npho'][i]
            if mtype == 0 and predictions['error_npho'][i] > -900:
                residual_values[sid] = predictions['error_npho'][i]
        else:
            pred_values[sid] = predictions['pred_time'][i]
            if mtype == 0 and predictions['error_time'][i] > -900:
                residual_values[sid] = predictions['error_time'][i]

    # Create filled data (original + predictions for masked)
    filled = original.copy()
    all_masked = dead_mask | artificial_mask
    filled[all_masked] = pred_values[all_masked]

    # Build title
    if title is None:
        run = predictions['run']
        event = predictions['event']
        title_parts = [f"Run {run} Event {event}"]

        if 'energyReco' in event_data:
            # energyReco is in GeV, convert to MeV for display
            energy_MeV = event_data['energyReco'] * 1000
            title_parts.append(f"E={energy_MeV:.1f} MeV")
        if 'uvwRecoFI' in event_data:
            uvw = event_data['uvwRecoFI']
            title_parts.append(f"uvw=({uvw[0]:.1f}, {uvw[1]:.1f}, {uvw[2]:.1f})")

        n_art = artificial_mask.sum()
        n_dead = dead_mask.sum()
        title_parts.append(f"Masked: {n_art} artificial + {n_dead} dead")

        title = " | ".join(title_parts)

    # Determine color limits
    valid_original = original[~np.isnan(original)]
    valid_residual = residual_values[~np.isnan(residual_values)]

    if len(valid_original) > 0:
        vmin_data = np.percentile(valid_original, 1)
        vmax_data = np.percentile(valid_original, 99)
    else:
        vmin_data, vmax_data = 0, 1

    if len(valid_residual) > 0:
        res_abs_max = np.percentile(np.abs(valid_residual), 95)
        vmin_res, vmax_res = -res_abs_max, res_abs_max
    else:
        vmin_res, vmax_res = -0.1, 0.1

    # Create diverging norm for residual
    res_norm = TwoSlopeNorm(vmin=vmin_res, vcenter=0, vmax=vmax_res)
    data_norm = Normalize(vmin=vmin_data, vmax=vmax_data)

    # Setup figure with 6 columns: Top, DS, Inner, US, Outer, Bottom
    # Using gridspec for better control over hex face aspect ratios
    fig = plt.figure(figsize=(24, 12))

    # Width ratios: hex faces need different aspect ratio
    # Top(hex), DS(rect), Inner(rect), US(rect), Outer(rect), Bottom(hex)
    width_ratios = [1, 0.5, 2, 0.5, 2, 1]
    gs = gridspec.GridSpec(3, 6, width_ratios=width_ratios, wspace=0.15, hspace=0.2)

    # Build grids for rectangular faces
    rect_faces = ['ds', 'inner', 'us']
    rect_cols = [1, 2, 3]  # Column positions for DS, Inner, US

    # Build outer face grids using proper interpolation
    outer_orig = build_outer_grid_from_sensor_values(np.nan_to_num(original, nan=0))
    outer_filled = build_outer_grid_from_sensor_values(np.nan_to_num(filled, nan=0))
    outer_residual = build_outer_grid_from_sensor_values(np.nan_to_num(residual_values, nan=0))
    outer_dead = build_outer_grid_from_sensor_values(dead_mask.astype(np.float32)) > 0.01
    outer_art = build_outer_grid_from_sensor_values(artificial_mask.astype(np.float32)) > 0.01

    # Data arrays for each row
    data_arrays = [
        {'values': original, 'label': 'Original'},
        {'values': filled, 'label': 'Filled'},
        {'values': residual_values, 'label': 'Residual'},
    ]

    outer_arrays = [
        {'grid': outer_orig, 'dead': outer_dead, 'art': outer_art},
        {'grid': outer_filled, 'dead': outer_dead, 'art': outer_art},
        {'grid': outer_residual, 'dead': outer_dead, 'art': outer_art},
    ]

    for row_idx in range(3):
        row_data = data_arrays[row_idx]
        values = row_data['values']
        row_label = row_data['label']

        # Determine colormap and norm for this row
        if row_idx == 2:  # Residual row
            cmap = 'RdBu_r'
            norm = res_norm
        else:
            cmap = 'viridis'
            norm = data_norm

        # Column 0: Top hex face
        ax_top = fig.add_subplot(gs[row_idx, 0])
        plot_hex_face(ax_top, TOP_HEX_ROWS, values,
                     f"Top - {row_label}", mode='top',
                     cmap=cmap, norm=norm,
                     dead_mask=dead_mask if row_idx > 0 else None,
                     artificial_mask=artificial_mask if row_idx > 0 else None)

        # Columns 1-3: DS, Inner, US rectangular faces
        for face_name, col in zip(rect_faces, rect_cols):
            ax = fig.add_subplot(gs[row_idx, col])
            config = FACE_CONFIGS[face_name]
            idx_map = config['index_map']
            face_title = config['title']

            grid = create_face_grid(values, idx_map)
            dead_grid = create_mask_grid(dead_mask, idx_map) if row_idx > 0 else None
            art_grid = create_mask_grid(artificial_mask, idx_map) if row_idx > 0 else None

            plot_face_panel(ax, grid, f"{face_title} - {row_label}",
                           cmap=cmap, norm=norm,
                           dead_mask=dead_grid, artificial_mask=art_grid,
                           show_colorbar=False)

        # Column 4: Outer face (using fine grid)
        ax_outer = fig.add_subplot(gs[row_idx, 4])
        outer_data = outer_arrays[row_idx]
        plot_face_panel(ax_outer, outer_data['grid'], f"Outer - {row_label}",
                       cmap=cmap, norm=norm,
                       dead_mask=outer_data['dead'] if row_idx > 0 else None,
                       artificial_mask=outer_data['art'] if row_idx > 0 else None,
                       show_colorbar=False)

        # Column 5: Bottom hex face
        ax_bot = fig.add_subplot(gs[row_idx, 5])
        plot_hex_face(ax_bot, BOTTOM_HEX_ROWS, values,
                     f"Bottom - {row_label}", mode='bottom',
                     cmap=cmap, norm=norm,
                     dead_mask=dead_mask if row_idx > 0 else None,
                     artificial_mask=artificial_mask if row_idx > 0 else None)

    # Add colorbars
    cbar_ax1 = fig.add_axes([0.92, 0.4, 0.012, 0.5])
    fig.colorbar(plt.cm.ScalarMappable(norm=data_norm, cmap='viridis'),
                 cax=cbar_ax1, label='Value')

    cbar_ax2 = fig.add_axes([0.92, 0.1, 0.012, 0.25])
    fig.colorbar(plt.cm.ScalarMappable(norm=res_norm, cmap='RdBu_r'),
                 cax=cbar_ax2, label='Residual')

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor='gray', edgecolor='red', hatch='///',
                       label='Dead channel (no truth)'),
        mpatches.Patch(facecolor='none', edgecolor='blue', linewidth=2,
                       label='Artificial mask (has truth)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2,
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 0.01))

    # Main title
    channel_label = 'Npho' if channel == 'npho' else 'Time'
    fig.suptitle(f"{channel_label}: {title}", fontsize=12, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0.02, 0.05, 0.9, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def print_event_summary(event_data: dict, predictions: dict):
    """Print summary of event predictions."""
    print("\n" + "=" * 60)
    print(f"Event Summary: Run {predictions['run']}, Event {predictions['event']}")
    print("=" * 60)

    # Count by face and mask type
    face_counts = {}
    for face_id in range(6):
        face_name = FACE_ID_TO_NAME.get(face_id, f'face{face_id}')
        face_mask = predictions['face'] == face_id
        n_art = ((predictions['mask_type'] == 0) & face_mask).sum()
        n_dead = ((predictions['mask_type'] == 1) & face_mask).sum()
        if n_art > 0 or n_dead > 0:
            face_counts[face_name] = {'artificial': n_art, 'dead': n_dead}

    print("\nPredictions by face:")
    print(f"{'Face':<10} {'Artificial':>12} {'Dead':>10} {'Total':>10}")
    print("-" * 44)
    total_art, total_dead = 0, 0
    for face, counts in face_counts.items():
        total = counts['artificial'] + counts['dead']
        print(f"{face:<10} {counts['artificial']:>12} {counts['dead']:>10} {total:>10}")
        total_art += counts['artificial']
        total_dead += counts['dead']
    print("-" * 44)
    print(f"{'Total':<10} {total_art:>12} {total_dead:>10} {total_art+total_dead:>10}")

    # Metrics for artificial masks only
    art_mask = predictions['mask_type'] == 0
    if art_mask.sum() > 0:
        print("\nMetrics (artificial masks only):")

        for var in ['npho', 'time']:
            error = predictions[f'error_{var}'][art_mask]
            valid = error > -900  # Filter out invalid values
            if valid.sum() > 0:
                error_valid = error[valid]
                mae = np.mean(np.abs(error_valid))
                rmse = np.sqrt(np.mean(error_valid ** 2))
                bias = np.mean(error_valid)
                print(f"  {var}: MAE={mae:.4f}, RMSE={rmse:.4f}, Bias={bias:.4f}")

    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Display inpainter predictions on real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Event selection (one of: event_idx, or run+event)
    parser.add_argument("event_idx", type=int, nargs='?', default=None,
                        help="Event index (0-based)")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number (use with --event)")
    parser.add_argument("--event", type=int, default=None,
                        help="Event number (use with --run)")

    # Required files
    parser.add_argument("--predictions", "-p", required=True,
                        help="Path to predictions ROOT file from validate_inpainter_real.py")
    parser.add_argument("--original", "-i", required=True,
                        help="Path to original real data ROOT file")

    # Options
    parser.add_argument("--channel", type=str, choices=['npho', 'time', 'both'],
                        default='npho', help="Channel to display")
    parser.add_argument("--tree", type=str, default="tree",
                        help="Tree name in original file")
    parser.add_argument("--save", "-o", type=str, default=None,
                        help="Save path for figure")
    parser.add_argument("--no-summary", action="store_true",
                        help="Don't print event summary")

    # Normalization (for proper denormalization)
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE)
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT)

    args = parser.parse_args()

    # Validate event selection
    if args.event_idx is None and (args.run is None or args.event is None):
        parser.error("Must specify either event_idx or both --run and --event")

    # Validate files exist
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    if not os.path.exists(args.original):
        print(f"Error: Original file not found: {args.original}")
        sys.exit(1)

    # Load original event data
    print(f"[INFO] Loading original data from: {args.original}")
    event_data = load_real_data_event(
        args.original,
        event_idx=args.event_idx,
        run=args.run,
        event=args.event,
        tree_name=args.tree
    )

    # Load predictions
    print(f"[INFO] Loading predictions from: {args.predictions}")
    predictions = load_predictions_for_event(args.predictions, event_data['event_idx'])

    # Print summary
    if not args.no_summary:
        print_event_summary(event_data, predictions)

    # Determine channels to plot
    channels = ['npho', 'time'] if args.channel == 'both' else [args.channel]

    for ch in channels:
        # Handle save path for multiple channels
        if args.save and args.channel == 'both':
            base, ext = os.path.splitext(args.save)
            if not ext:
                ext = '.pdf'
            save_path = f"{base}_{ch}{ext}"
        elif args.save:
            save_path = args.save
            if '.' not in os.path.basename(save_path):
                save_path += '.pdf'
        else:
            save_path = None

        # Plot
        plot_real_data_event(
            event_data, predictions,
            channel=ch,
            save_path=save_path,
            npho_scale=args.npho_scale,
            time_scale=args.time_scale,
            time_shift=args.time_shift
        )


if __name__ == "__main__":
    main()
