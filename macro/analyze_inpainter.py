#!/usr/bin/env python3
"""
Analyze Inpainter Predictions

Generates detailed metrics and plots from inpainter validation output.
Works with both MC and real data validation results.

Mode is auto-detected based on whether 'mask_type' column exists:
- If mask_type exists: Real data mode (separates artificial vs dead channel metrics)
- If mask_type missing: MC mode (all predictions have ground truth)

Usage:
    python macro/analyze_inpainter.py predictions.root --output analysis/
    python macro/analyze_inpainter.py predictions.root --output analysis/ --no-plots

    # With custom normalization parameters for denormalized plots
    python macro/analyze_inpainter.py predictions.root --output analysis/ \\
        --npho-scale 1000 --npho-scale2 4.08 --time-scale 1.14e-7 --time-shift -0.46
"""

import os
import sys
import argparse
import numpy as np
import uproot
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib not available, skipping plots")

from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_VALUE
)

FACE_NAMES = ['inner', 'us', 'ds', 'outer', 'top', 'bot']
FACE_INT_TO_NAME = {0: 'inner', 1: 'us', 2: 'ds', 3: 'outer', 4: 'top', 5: 'bot'}


def load_predictions(input_path: str) -> Dict[str, np.ndarray]:
    """Load predictions from ROOT file."""
    print(f"[INFO] Loading predictions from {input_path}")

    with uproot.open(input_path) as f:
        # Try different tree names
        tree_name = None
        for name in ['predictions', 'tree', 'Tree']:
            if name in f:
                tree_name = name
                break
        if tree_name is None:
            # Use first tree
            tree_name = [k for k in f.keys() if not k.startswith('_')][0].split(';')[0]

        tree = f[tree_name]
        data = {key: tree[key].array(library='np') for key in tree.keys()}

    n_preds = len(data['event_idx'])
    print(f"[INFO] Loaded {n_preds:,} predictions from tree '{tree_name}'")

    # Check if mask_type exists (real data validation)
    has_mask_type = 'mask_type' in data
    if has_mask_type:
        n_artificial = (data['mask_type'] == 0).sum()
        n_dead = (data['mask_type'] == 1).sum()
        print(f"[INFO] Artificial masks: {n_artificial:,}, Dead channels: {n_dead:,}")

    return data


def denormalize_npho(npho_norm: np.ndarray, npho_scale: float, npho_scale2: float) -> np.ndarray:
    """Convert normalized npho back to raw scale."""
    return npho_scale * (np.exp(npho_norm * npho_scale2) - 1.0)


def denormalize_time(time_norm: np.ndarray, time_scale: float, time_shift: float) -> np.ndarray:
    """Convert normalized time back to raw scale (seconds)."""
    return (time_norm + time_shift) * time_scale


def compute_detailed_metrics(data: Dict[str, np.ndarray], sentinel_value: float = DEFAULT_SENTINEL_VALUE) -> Dict:
    """Compute detailed metrics from predictions."""
    metrics = {}

    # Filter valid predictions (has ground truth)
    valid = data['error_npho'] > -999
    has_mask_type = 'mask_type' in data

    if has_mask_type:
        # Real data mode: separate artificial vs dead
        artificial = (data['mask_type'] == 0) & valid
        dead = data['mask_type'] == 1

        metrics['mode'] = 'real_data'
        metrics['n_total'] = len(data['event_idx'])
        metrics['n_artificial'] = artificial.sum()
        metrics['n_dead'] = dead.sum()
        metrics['n_with_truth'] = valid.sum()

        eval_mask = artificial
    else:
        # MC mode: all predictions have truth
        metrics['mode'] = 'mc'
        metrics['n_total'] = len(data['event_idx'])
        metrics['n_with_truth'] = valid.sum()

        eval_mask = valid

    if eval_mask.sum() == 0:
        print("[WARNING] No predictions with ground truth to evaluate")
        return metrics

    # Global metrics
    err_npho = data['error_npho'][eval_mask]
    err_time = data['error_time'][eval_mask]

    # Time metrics excluding invalid sensors (where truth_time == sentinel)
    # Valid time = sensors where time was actually measured (not set to sentinel due to low npho)
    valid_time_mask = eval_mask & (data['truth_time'] != sentinel_value)
    err_time_valid = data['error_time'][valid_time_mask]

    metrics['global'] = {
        'n': len(err_npho),
        'n_valid_time': len(err_time_valid),
        'npho_mae': np.mean(np.abs(err_npho)),
        'npho_rmse': np.sqrt(np.mean(err_npho ** 2)),
        'npho_bias': np.mean(err_npho),
        'npho_std': np.std(err_npho),
        'npho_68pct': np.percentile(np.abs(err_npho), 68),
        'npho_95pct': np.percentile(np.abs(err_npho), 95),
        # Time metrics including all sensors
        'time_mae': np.mean(np.abs(err_time)),
        'time_rmse': np.sqrt(np.mean(err_time ** 2)),
        'time_bias': np.mean(err_time),
        'time_std': np.std(err_time),
        'time_68pct': np.percentile(np.abs(err_time), 68),
        'time_95pct': np.percentile(np.abs(err_time), 95),
        # Time metrics excluding invalid sensors
        'time_mae_valid': np.mean(np.abs(err_time_valid)) if len(err_time_valid) > 0 else np.nan,
        'time_rmse_valid': np.sqrt(np.mean(err_time_valid ** 2)) if len(err_time_valid) > 0 else np.nan,
        'time_bias_valid': np.mean(err_time_valid) if len(err_time_valid) > 0 else np.nan,
        'time_std_valid': np.std(err_time_valid) if len(err_time_valid) > 0 else np.nan,
        'time_68pct_valid': np.percentile(np.abs(err_time_valid), 68) if len(err_time_valid) > 0 else np.nan,
        'time_95pct_valid': np.percentile(np.abs(err_time_valid), 95) if len(err_time_valid) > 0 else np.nan,
    }

    # Per-face metrics
    metrics['per_face'] = {}
    for face_int, face_name in FACE_INT_TO_NAME.items():
        face_mask = eval_mask & (data['face'] == face_int)
        if face_mask.sum() == 0:
            continue

        face_err_npho = data['error_npho'][face_mask]
        face_err_time = data['error_time'][face_mask]

        # Valid time for this face
        face_valid_time_mask = face_mask & (data['truth_time'] != sentinel_value)
        face_err_time_valid = data['error_time'][face_valid_time_mask]

        metrics['per_face'][face_name] = {
            'n': len(face_err_npho),
            'n_valid_time': len(face_err_time_valid),
            'npho_mae': np.mean(np.abs(face_err_npho)),
            'npho_rmse': np.sqrt(np.mean(face_err_npho ** 2)),
            'npho_bias': np.mean(face_err_npho),
            'npho_68pct': np.percentile(np.abs(face_err_npho), 68),
            'time_mae': np.mean(np.abs(face_err_time)),
            'time_rmse': np.sqrt(np.mean(face_err_time ** 2)),
            'time_bias': np.mean(face_err_time),
            'time_68pct': np.percentile(np.abs(face_err_time), 68),
            'time_mae_valid': np.mean(np.abs(face_err_time_valid)) if len(face_err_time_valid) > 0 else np.nan,
            'time_bias_valid': np.mean(face_err_time_valid) if len(face_err_time_valid) > 0 else np.nan,
        }

    # Dead channel statistics (real data mode)
    if has_mask_type and metrics['n_dead'] > 0:
        dead_mask = data['mask_type'] == 1
        metrics['dead_stats'] = {
            'n': dead_mask.sum(),
            'pred_npho_mean': np.mean(data['pred_npho'][dead_mask]),
            'pred_npho_std': np.std(data['pred_npho'][dead_mask]),
            'pred_npho_min': np.min(data['pred_npho'][dead_mask]),
            'pred_npho_max': np.max(data['pred_npho'][dead_mask]),
            'pred_time_mean': np.mean(data['pred_time'][dead_mask]),
            'pred_time_std': np.std(data['pred_time'][dead_mask]),
            'pred_time_min': np.min(data['pred_time'][dead_mask]),
            'pred_time_max': np.max(data['pred_time'][dead_mask]),
        }

    return metrics


def plot_residual_distributions(data: Dict[str, np.ndarray], output_dir: str,
                                 has_mask_type: bool,
                                 npho_scale: float, npho_scale2: float,
                                 time_scale: float, time_shift: float,
                                 sentinel_value: float):
    """Plot residual distributions (normalized and denormalized)."""
    if not HAS_MATPLOTLIB:
        return

    # Filter valid
    if has_mask_type:
        valid = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        valid = data['error_npho'] > -999

    if valid.sum() == 0:
        return

    err_npho = data['error_npho'][valid]
    err_time = data['error_time'][valid]

    # === Normalized residuals ===
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Npho residuals
    ax = axes[0]
    ax.hist(err_npho, bins=100, range=(-0.3, 0.4), density=True, alpha=0.7, color='blue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.3, 0.4)
    ax.set_xlabel('Npho Error (pred - truth) [normalized]')
    ax.set_ylabel('Density')
    ax.set_title(f'Npho Residuals (n={len(err_npho):,})\n'
                 f'MAE={np.mean(np.abs(err_npho)):.4f}, Bias={np.mean(err_npho):.4f}')

    # Time residuals
    ax = axes[1]
    ax.hist(err_time, bins=100, range=(-0.4, 0.4), density=True, alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.4, 0.4)
    ax.set_xlabel('Time Error (pred - truth) [normalized]')
    ax.set_ylabel('Density')
    ax.set_title(f'Time Residuals (n={len(err_time):,})\n'
                 f'MAE={np.mean(np.abs(err_time)):.4f}, Bias={np.mean(err_time):.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distributions.pdf'))
    plt.close()
    print("[INFO] Saved residual_distributions.pdf")

    # === Denormalized residuals ===
    # Get truth values and compute denormalized errors
    truth_npho_norm = data['truth_npho'][valid]
    pred_npho_norm = data['pred_npho'][valid]
    truth_time_norm = data['truth_time'][valid]
    pred_time_norm = data['pred_time'][valid]

    # Denormalize
    truth_npho_raw = denormalize_npho(truth_npho_norm, npho_scale, npho_scale2)
    pred_npho_raw = denormalize_npho(pred_npho_norm, npho_scale, npho_scale2)
    truth_time_raw = denormalize_time(truth_time_norm, time_scale, time_shift)
    pred_time_raw = denormalize_time(pred_time_norm, time_scale, time_shift)

    err_npho_raw = pred_npho_raw - truth_npho_raw
    err_time_raw = pred_time_raw - truth_time_raw

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Npho residuals (raw)
    ax = axes[0]
    # Auto-range based on percentiles
    npho_low, npho_high = np.percentile(err_npho_raw, [1, 99])
    npho_range = (max(npho_low, -1000), min(npho_high, 1000))
    ax.hist(err_npho_raw, bins=100, range=npho_range, density=True, alpha=0.7, color='blue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Npho Error (pred - truth) [photons]')
    ax.set_ylabel('Density')
    ax.set_title(f'Npho Residuals - Denormalized\n'
                 f'MAE={np.mean(np.abs(err_npho_raw)):.1f}, Bias={np.mean(err_npho_raw):.1f} photons')

    # Time residuals (raw)
    ax = axes[1]
    err_time_ns = err_time_raw * 1e9  # Convert to nanoseconds
    time_low, time_high = np.percentile(err_time_ns, [1, 99])
    time_range = (max(time_low, -10), min(time_high, 10))
    ax.hist(err_time_ns, bins=100, range=time_range, density=True, alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Error (pred - truth) [ns]')
    ax.set_ylabel('Density')
    ax.set_title(f'Time Residuals - Denormalized\n'
                 f'MAE={np.mean(np.abs(err_time_ns)):.2f}, Bias={np.mean(err_time_ns):.2f} ns')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distributions_denorm.pdf'))
    plt.close()
    print("[INFO] Saved residual_distributions_denorm.pdf")


def plot_per_face_residuals(data: Dict[str, np.ndarray], output_dir: str,
                             has_mask_type: bool,
                             npho_scale: float, npho_scale2: float,
                             time_scale: float, time_shift: float):
    """Plot per-face residual distributions (normalized and denormalized)."""
    if not HAS_MATPLOTLIB:
        return

    if has_mask_type:
        valid = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        valid = data['error_npho'] > -999

    # === Normalized Npho residuals ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
        ax = axes[idx]
        face_mask = valid & (data['face'] == face_int)

        if face_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{face_name}\nNo data', ha='center', va='center')
            ax.set_xlim(-0.4, 0.2)
            continue

        err_npho = data['error_npho'][face_mask]
        ax.hist(err_npho, bins=50, range=(-0.4, 0.2), density=True, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.4, 0.2)
        ax.set_xlabel('Npho Error [normalized]')
        ax.set_title(f'{face_name} (n={len(err_npho):,})\n'
                     f'MAE={np.mean(np.abs(err_npho)):.4f}, Bias={np.mean(err_npho):.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_per_face_npho.pdf'))
    plt.close()

    # === Normalized Time residuals ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
        ax = axes[idx]
        face_mask = valid & (data['face'] == face_int)

        if face_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{face_name}\nNo data', ha='center', va='center')
            continue

        err_time = data['error_time'][face_mask]
        ax.hist(err_time, bins=50, range=(-0.3, 0.3), density=True, alpha=0.7, color='green')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlim(-0.3, 0.3)
        ax.set_xlabel('Time Error [normalized]')
        ax.set_title(f'{face_name} (n={len(err_time):,})\n'
                     f'MAE={np.mean(np.abs(err_time)):.4f}, Bias={np.mean(err_time):.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_per_face_time.pdf'))
    plt.close()
    print("[INFO] Saved residual_per_face_*.pdf")

    # === Denormalized Npho residuals ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
        ax = axes[idx]
        face_mask = valid & (data['face'] == face_int)

        if face_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{face_name}\nNo data', ha='center', va='center')
            continue

        truth_npho_raw = denormalize_npho(data['truth_npho'][face_mask], npho_scale, npho_scale2)
        pred_npho_raw = denormalize_npho(data['pred_npho'][face_mask], npho_scale, npho_scale2)
        err_npho_raw = pred_npho_raw - truth_npho_raw

        npho_low, npho_high = np.percentile(err_npho_raw, [2, 98])
        npho_range = (max(npho_low, -500), min(npho_high, 500))
        ax.hist(err_npho_raw, bins=50, range=npho_range, density=True, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Npho Error [photons]')
        ax.set_title(f'{face_name} (n={len(err_npho_raw):,})\n'
                     f'MAE={np.mean(np.abs(err_npho_raw)):.1f}, Bias={np.mean(err_npho_raw):.1f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_per_face_npho_denorm.pdf'))
    plt.close()

    # === Denormalized Time residuals ===
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
        ax = axes[idx]
        face_mask = valid & (data['face'] == face_int)

        if face_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{face_name}\nNo data', ha='center', va='center')
            continue

        truth_time_raw = denormalize_time(data['truth_time'][face_mask], time_scale, time_shift)
        pred_time_raw = denormalize_time(data['pred_time'][face_mask], time_scale, time_shift)
        err_time_ns = (pred_time_raw - truth_time_raw) * 1e9  # ns

        time_low, time_high = np.percentile(err_time_ns, [2, 98])
        time_range = (max(time_low, -5), min(time_high, 5))
        ax.hist(err_time_ns, bins=50, range=time_range, density=True, alpha=0.7, color='green')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Error [ns]')
        ax.set_title(f'{face_name} (n={len(err_time_ns):,})\n'
                     f'MAE={np.mean(np.abs(err_time_ns)):.2f}, Bias={np.mean(err_time_ns):.2f} ns')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_per_face_time_denorm.pdf'))
    plt.close()
    print("[INFO] Saved residual_per_face_*_denorm.pdf")


def plot_scatter_truth_vs_pred(data: Dict[str, np.ndarray], output_dir: str,
                                has_mask_type: bool,
                                npho_scale: float, npho_scale2: float,
                                time_scale: float, time_shift: float):
    """Plot scatter of truth vs prediction with log-scale density."""
    if not HAS_MATPLOTLIB:
        return

    from matplotlib.colors import LogNorm

    if has_mask_type:
        valid = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        valid = data['error_npho'] > -999

    if valid.sum() == 0:
        return

    truth_npho = data['truth_npho'][valid]
    pred_npho = data['pred_npho'][valid]
    truth_time = data['truth_time'][valid]
    pred_time = data['pred_time'][valid]

    # === Normalized scatter ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Npho - log scale density
    ax = axes[0]
    h = ax.hist2d(truth_npho, pred_npho, bins=100, cmap='Blues', norm=LogNorm(), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count (log)')
    lims = [min(truth_npho.min(), pred_npho.min()), max(truth_npho.max(), pred_npho.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y=x')
    ax.set_xlabel('Truth Npho (normalized)')
    ax.set_ylabel('Pred Npho (normalized)')
    ax.set_title('Npho: Truth vs Prediction')
    ax.legend()

    # Time - log scale density with fixed range
    ax = axes[1]
    # Filter to valid time range [0, 1]
    time_range_mask = (truth_time >= 0) & (truth_time <= 1)
    h = ax.hist2d(truth_time[time_range_mask], pred_time[time_range_mask],
                  bins=100, range=[[0, 1], [pred_time.min(), pred_time.max()]],
                  cmap='Greens', norm=LogNorm(), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count (log)')
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, linewidth=2, label='y=x')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Truth Time (normalized)')
    ax.set_ylabel('Pred Time (normalized)')
    ax.set_title('Time: Truth vs Prediction')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_truth_vs_pred.pdf'))
    plt.close()

    # === Denormalized scatter ===
    truth_npho_raw = denormalize_npho(truth_npho, npho_scale, npho_scale2)
    pred_npho_raw = denormalize_npho(pred_npho, npho_scale, npho_scale2)
    truth_time_raw = denormalize_time(truth_time, time_scale, time_shift) * 1e9  # ns
    pred_time_raw = denormalize_time(pred_time, time_scale, time_shift) * 1e9  # ns

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Npho denormalized
    ax = axes[0]
    # Clip to reasonable range for visualization
    npho_max = np.percentile(truth_npho_raw, 99)
    mask_npho = (truth_npho_raw < npho_max) & (pred_npho_raw < npho_max * 1.5)
    h = ax.hist2d(truth_npho_raw[mask_npho], pred_npho_raw[mask_npho],
                  bins=100, cmap='Blues', norm=LogNorm(), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count (log)')
    lims = [0, npho_max]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y=x')
    ax.set_xlabel('Truth Npho [photons]')
    ax.set_ylabel('Pred Npho [photons]')
    ax.set_title('Npho: Truth vs Prediction (denormalized)')
    ax.legend()

    # Time denormalized
    ax = axes[1]
    time_low, time_high = np.percentile(truth_time_raw, [1, 99])
    mask_time = (truth_time_raw >= time_low) & (truth_time_raw <= time_high)
    h = ax.hist2d(truth_time_raw[mask_time], pred_time_raw[mask_time],
                  bins=100, cmap='Greens', norm=LogNorm(), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count (log)')
    ax.plot([time_low, time_high], [time_low, time_high], 'r--', alpha=0.7, linewidth=2, label='y=x')
    ax.set_xlabel('Truth Time [ns]')
    ax.set_ylabel('Pred Time [ns]')
    ax.set_title('Time: Truth vs Prediction (denormalized)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_truth_vs_pred_denorm.pdf'))
    plt.close()

    # --- Zoomed scatter for high-signal region (npho > 0.1) ---
    high_signal = truth_npho > 0.1
    if high_signal.sum() > 100:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        h = ax.hist2d(truth_npho[high_signal], pred_npho[high_signal],
                      bins=50, cmap='Blues', norm=LogNorm(), cmin=1)
        plt.colorbar(h[3], ax=ax, label='Count (log)')
        lims = [truth_npho[high_signal].min(), truth_npho[high_signal].max()]
        ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y=x')
        ax.set_xlabel('Truth Npho (normalized)')
        ax.set_ylabel('Pred Npho (normalized)')
        ax.set_title(f'Npho: High Signal (npho > 0.1, n={high_signal.sum():,})')
        ax.legend()

        ax = axes[1]
        h = ax.hist2d(truth_time[high_signal], pred_time[high_signal],
                      bins=50, cmap='Greens', norm=LogNorm(), cmin=1)
        plt.colorbar(h[3], ax=ax, label='Count (log)')
        lims = [truth_time[high_signal].min(), truth_time[high_signal].max()]
        ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y=x')
        ax.set_xlabel('Truth Time (normalized)')
        ax.set_ylabel('Pred Time (normalized)')
        ax.set_title(f'Time: High Signal (npho > 0.1)')
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'scatter_truth_vs_pred_highsignal.pdf'))
        plt.close()
        print("[INFO] Saved scatter_truth_vs_pred*.pdf")
    else:
        print("[INFO] Saved scatter_truth_vs_pred.pdf and scatter_truth_vs_pred_denorm.pdf")


def plot_metrics_summary(metrics: Dict, output_dir: str, sentinel_value: float):
    """Plot bar chart of per-face metrics."""
    if not HAS_MATPLOTLIB:
        return

    if 'per_face' not in metrics or not metrics['per_face']:
        return

    faces = list(metrics['per_face'].keys())
    npho_mae = [metrics['per_face'][f]['npho_mae'] for f in faces]
    time_mae = [metrics['per_face'][f]['time_mae'] for f in faces]
    time_mae_valid = [metrics['per_face'][f].get('time_mae_valid', np.nan) for f in faces]

    x = np.arange(len(faces))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width, npho_mae, width, label='Npho MAE', color='blue', alpha=0.7)
    bars2 = ax.bar(x, time_mae, width, label='Time MAE (all)', color='green', alpha=0.7)
    bars3 = ax.bar(x + width, time_mae_valid, width, label='Time MAE (valid only)', color='orange', alpha=0.7)

    ax.set_xlabel('Face')
    ax.set_ylabel('MAE (normalized)')
    ax.set_title('Per-Face MAE Comparison\n(Time valid = excluding sensors with truth_time == sentinel)')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.legend()

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(f'{height:.4f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=7, rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_summary.pdf'))
    plt.close()
    print("[INFO] Saved metrics_summary.pdf")


def save_metrics_csv(metrics: Dict, output_dir: str):
    """Save metrics to CSV files."""
    import csv

    # Global metrics
    if 'global' in metrics:
        with open(os.path.join(output_dir, 'global_metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            writer.writerow(['mode', metrics['mode']])
            writer.writerow(['n_total', metrics['n_total']])
            writer.writerow(['n_with_truth', metrics['n_with_truth']])
            if 'n_artificial' in metrics:
                writer.writerow(['n_artificial', metrics['n_artificial']])
                writer.writerow(['n_dead', metrics['n_dead']])
            for k, v in metrics['global'].items():
                writer.writerow([k, v])
        print("[INFO] Saved global_metrics.csv")

    # Per-face metrics
    if 'per_face' in metrics and metrics['per_face']:
        with open(os.path.join(output_dir, 'face_metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['face', 'n', 'n_valid_time',
                      'npho_mae', 'npho_rmse', 'npho_bias', 'npho_68pct',
                      'time_mae', 'time_rmse', 'time_bias', 'time_68pct',
                      'time_mae_valid', 'time_bias_valid']
            writer.writerow(header)
            for face_name in FACE_NAMES:
                if face_name in metrics['per_face']:
                    m = metrics['per_face'][face_name]
                    writer.writerow([face_name, m['n'], m.get('n_valid_time', ''),
                                     m['npho_mae'], m['npho_rmse'], m['npho_bias'], m['npho_68pct'],
                                     m['time_mae'], m['time_rmse'], m['time_bias'], m['time_68pct'],
                                     m.get('time_mae_valid', ''), m.get('time_bias_valid', '')])
        print("[INFO] Saved face_metrics.csv")

    # Dead channel stats
    if 'dead_stats' in metrics:
        with open(os.path.join(output_dir, 'dead_channel_stats.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in metrics['dead_stats'].items():
                writer.writerow([k, v])
        print("[INFO] Saved dead_channel_stats.csv")


def print_report(metrics: Dict):
    """Print metrics report to console."""
    print("\n" + "=" * 70)
    print("INPAINTER ANALYSIS REPORT")
    print("=" * 70)

    mode_str = "Real Data" if metrics['mode'] == 'real_data' else "MC"
    print(f"Mode: {mode_str}")
    print(f"Total predictions: {metrics['n_total']:,}")
    print(f"Predictions with truth: {metrics['n_with_truth']:,}")
    if 'n_artificial' in metrics:
        print(f"  - Artificial masks: {metrics['n_artificial']:,}")
        print(f"  - Dead channels: {metrics['n_dead']:,}")

    if 'global' in metrics:
        g = metrics['global']
        print(f"\n--- Global Metrics ---")
        print(f"{'Metric':<20} {'Npho':>12} {'Time (all)':>12} {'Time (valid)':>12}")
        print("-" * 60)
        print(f"{'N':<20} {g['n']:>12,} {g['n']:>12,} {g['n_valid_time']:>12,}")
        print(f"{'MAE':<20} {g['npho_mae']:>12.4f} {g['time_mae']:>12.4f} {g['time_mae_valid']:>12.4f}")
        print(f"{'RMSE':<20} {g['npho_rmse']:>12.4f} {g['time_rmse']:>12.4f} {g['time_rmse_valid']:>12.4f}")
        print(f"{'Bias':<20} {g['npho_bias']:>12.4f} {g['time_bias']:>12.4f} {g['time_bias_valid']:>12.4f}")
        print(f"{'Std':<20} {g['npho_std']:>12.4f} {g['time_std']:>12.4f} {g['time_std_valid']:>12.4f}")
        print(f"{'68th pct':<20} {g['npho_68pct']:>12.4f} {g['time_68pct']:>12.4f} {g['time_68pct_valid']:>12.4f}")
        print(f"{'95th pct':<20} {g['npho_95pct']:>12.4f} {g['time_95pct']:>12.4f} {g['time_95pct_valid']:>12.4f}")

    if 'per_face' in metrics and metrics['per_face']:
        print(f"\n--- Per-Face Metrics ---")
        print(f"{'Face':<8} {'N':>8} {'Npho MAE':>10} {'Npho Bias':>10} {'Time MAE':>10} {'Time MAE*':>10}")
        print("-" * 60)
        print("(* = valid sensors only, excluding sentinel)")
        for face_name in FACE_NAMES:
            if face_name in metrics['per_face']:
                m = metrics['per_face'][face_name]
                time_mae_valid = m.get('time_mae_valid', np.nan)
                time_mae_valid_str = f"{time_mae_valid:>10.4f}" if not np.isnan(time_mae_valid) else "       N/A"
                print(f"{face_name:<8} {m['n']:>8,} {m['npho_mae']:>10.4f} {m['npho_bias']:>10.4f} "
                      f"{m['time_mae']:>10.4f} {time_mae_valid_str}")

    if 'dead_stats' in metrics:
        d = metrics['dead_stats']
        print(f"\n--- Dead Channel Predictions (No Ground Truth) ---")
        print(f"Total: {d['n']:,}")
        print(f"Npho: mean={d['pred_npho_mean']:.4f}, std={d['pred_npho_std']:.4f}, "
              f"range=[{d['pred_npho_min']:.4f}, {d['pred_npho_max']:.4f}]")
        print(f"Time: mean={d['pred_time_mean']:.4f}, std={d['pred_time_std']:.4f}, "
              f"range=[{d['pred_time_min']:.4f}, {d['pred_time_max']:.4f}]")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inpainter predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument("input", type=str,
                        help="Path to predictions ROOT file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory for analysis results")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")

    # Normalization parameters for denormalized plots
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE,
                        help=f"Npho scale for denormalization (default: {DEFAULT_NPHO_SCALE})")
    parser.add_argument("--npho-scale2", type=float, default=DEFAULT_NPHO_SCALE2,
                        help=f"Npho scale2 for denormalization (default: {DEFAULT_NPHO_SCALE2})")
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE,
                        help=f"Time scale for denormalization (default: {DEFAULT_TIME_SCALE})")
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT,
                        help=f"Time shift for denormalization (default: {DEFAULT_TIME_SHIFT})")
    parser.add_argument("--sentinel", type=float, default=DEFAULT_SENTINEL_VALUE,
                        help=f"Sentinel value for invalid sensors (default: {DEFAULT_SENTINEL_VALUE})")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load predictions
    data = load_predictions(args.input)
    has_mask_type = 'mask_type' in data

    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = compute_detailed_metrics(data, sentinel_value=args.sentinel)

    # Print report
    print_report(metrics)

    # Save metrics
    save_metrics_csv(metrics, args.output)

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("[INFO] Generating plots...")
        plot_residual_distributions(data, args.output, has_mask_type,
                                     args.npho_scale, args.npho_scale2,
                                     args.time_scale, args.time_shift,
                                     args.sentinel)
        plot_per_face_residuals(data, args.output, has_mask_type,
                                 args.npho_scale, args.npho_scale2,
                                 args.time_scale, args.time_shift)
        plot_scatter_truth_vs_pred(data, args.output, has_mask_type,
                                    args.npho_scale, args.npho_scale2,
                                    args.time_scale, args.time_shift)
        plot_metrics_summary(metrics, args.output, args.sentinel)

    print(f"\n[INFO] Analysis complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
