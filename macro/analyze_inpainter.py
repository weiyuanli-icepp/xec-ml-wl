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
    DEFAULT_SENTINEL_TIME
)

FACE_NAMES = ['inner', 'us', 'ds', 'outer', 'top', 'bot']
FACE_INT_TO_NAME = {0: 'inner', 1: 'us', 2: 'ds', 3: 'outer', 4: 'top', 5: 'bot'}


def load_metadata(input_path: str) -> Dict:
    """
    Load metadata from ROOT file.

    Returns dict with:
    - predict_channels: list like ['npho'] or ['npho', 'time']
    - npho_scale, npho_scale2, time_scale, time_shift: normalization params
    """
    metadata = {
        'predict_channels': ['npho', 'time'],  # Default for legacy files
        'npho_scale': DEFAULT_NPHO_SCALE,
        'npho_scale2': DEFAULT_NPHO_SCALE2,
        'time_scale': DEFAULT_TIME_SCALE,
        'time_shift': DEFAULT_TIME_SHIFT,
    }

    with uproot.open(input_path) as f:
        if 'metadata' in f:
            meta_tree = f['metadata']
            meta_keys = meta_tree.keys()

            # Read predict_channels
            if 'predict_channels' in meta_keys:
                pc_val = meta_tree['predict_channels'].array(library='np')[0]
                if isinstance(pc_val, bytes):
                    pc_val = pc_val.decode()
                metadata['predict_channels'] = pc_val.split(',')

            # Read normalization params
            for key in ['npho_scale', 'npho_scale2', 'time_scale', 'time_shift']:
                if key in meta_keys:
                    val = meta_tree[key].array(library='np')[0]
                    if not np.isnan(val):
                        metadata[key] = float(val)

    return metadata


def load_predictions(input_path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Load predictions and metadata from ROOT file.

    Returns:
        Tuple of (data dict, metadata dict)
    """
    print(f"[INFO] Loading predictions from {input_path}")

    # Load metadata first
    metadata = load_metadata(input_path)
    predict_channels = metadata['predict_channels']
    predict_time = 'time' in predict_channels
    print(f"[INFO] Predict channels: {predict_channels}")

    with uproot.open(input_path) as f:
        # Try different tree names
        tree_name = None
        for name in ['predictions', 'tree', 'Tree']:
            if name in f:
                tree_name = name
                break
        if tree_name is None:
            # Use first tree
            tree_name = [k for k in f.keys() if not k.startswith('_') and k != 'metadata'][0].split(';')[0]

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

    # Detect time prediction from data if not in metadata
    if 'pred_time' not in data and 'time' in predict_channels:
        print(f"[INFO] pred_time branch missing - switching to npho-only mode")
        metadata['predict_channels'] = ['npho']

    # Detect baseline branches
    has_avg = 'baseline_avg_npho' in data
    has_sa = 'baseline_sa_npho' in data
    if has_avg or has_sa:
        baselines = []
        if has_avg:
            baselines.append('neighbor_avg')
        if has_sa:
            baselines.append('solid_angle')
        print(f"[INFO] Baseline data found: {', '.join(baselines)}")

    return data, metadata


def denormalize_npho(npho_norm: np.ndarray, npho_scale: float, npho_scale2: float) -> np.ndarray:
    """Convert normalized npho back to raw scale."""
    return npho_scale * (np.exp(npho_norm * npho_scale2) - 1.0)


def denormalize_time(time_norm: np.ndarray, time_scale: float, time_shift: float) -> np.ndarray:
    """Convert normalized time back to raw scale (seconds)."""
    return (time_norm + time_shift) * time_scale


def compute_detailed_metrics(data: Dict[str, np.ndarray], sentinel_time: float = DEFAULT_SENTINEL_TIME,
                             predict_time: bool = True) -> Dict:
    """
    Compute detailed metrics from predictions.

    Args:
        data: Prediction data dict
        sentinel_time: Sentinel value for invalid time
        predict_time: Whether time was predicted (if False, skip time metrics)
    """
    metrics = {}
    metrics['predict_time'] = predict_time

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

    metrics['global'] = {
        'n': len(err_npho),
        'npho_mae': np.mean(np.abs(err_npho)),
        'npho_rmse': np.sqrt(np.mean(err_npho ** 2)),
        'npho_bias': np.mean(err_npho),
        'npho_std': np.std(err_npho),
        'npho_68pct': np.percentile(np.abs(err_npho), 68),
        'npho_95pct': np.percentile(np.abs(err_npho), 95),
    }

    # Time metrics only if time was predicted
    if predict_time and 'error_time' in data:
        err_time = data['error_time'][eval_mask]

        # Time metrics excluding invalid sensors (where truth_time == sentinel)
        valid_time_mask = eval_mask & (data['truth_time'] != sentinel_time)
        err_time_valid = data['error_time'][valid_time_mask]

        metrics['global'].update({
            'n_valid_time': len(err_time_valid),
            'time_mae': np.mean(np.abs(err_time)),
            'time_rmse': np.sqrt(np.mean(err_time ** 2)),
            'time_bias': np.mean(err_time),
            'time_std': np.std(err_time),
            'time_68pct': np.percentile(np.abs(err_time), 68),
            'time_95pct': np.percentile(np.abs(err_time), 95),
            'time_mae_valid': np.mean(np.abs(err_time_valid)) if len(err_time_valid) > 0 else np.nan,
            'time_rmse_valid': np.sqrt(np.mean(err_time_valid ** 2)) if len(err_time_valid) > 0 else np.nan,
            'time_bias_valid': np.mean(err_time_valid) if len(err_time_valid) > 0 else np.nan,
            'time_std_valid': np.std(err_time_valid) if len(err_time_valid) > 0 else np.nan,
            'time_68pct_valid': np.percentile(np.abs(err_time_valid), 68) if len(err_time_valid) > 0 else np.nan,
            'time_95pct_valid': np.percentile(np.abs(err_time_valid), 95) if len(err_time_valid) > 0 else np.nan,
        })

    # Per-face metrics
    metrics['per_face'] = {}
    for face_int, face_name in FACE_INT_TO_NAME.items():
        face_mask = eval_mask & (data['face'] == face_int)
        if face_mask.sum() == 0:
            continue

        face_err_npho = data['error_npho'][face_mask]

        face_metrics = {
            'n': len(face_err_npho),
            'npho_mae': np.mean(np.abs(face_err_npho)),
            'npho_rmse': np.sqrt(np.mean(face_err_npho ** 2)),
            'npho_bias': np.mean(face_err_npho),
            'npho_68pct': np.percentile(np.abs(face_err_npho), 68),
        }

        if predict_time and 'error_time' in data:
            face_err_time = data['error_time'][face_mask]
            face_valid_time_mask = face_mask & (data['truth_time'] != sentinel_time)
            face_err_time_valid = data['error_time'][face_valid_time_mask]

            face_metrics.update({
                'n_valid_time': len(face_err_time_valid),
                'time_mae': np.mean(np.abs(face_err_time)),
                'time_rmse': np.sqrt(np.mean(face_err_time ** 2)),
                'time_bias': np.mean(face_err_time),
                'time_68pct': np.percentile(np.abs(face_err_time), 68),
                'time_mae_valid': np.mean(np.abs(face_err_time_valid)) if len(face_err_time_valid) > 0 else np.nan,
                'time_bias_valid': np.mean(face_err_time_valid) if len(face_err_time_valid) > 0 else np.nan,
            })

        metrics['per_face'][face_name] = face_metrics

    # Dead channel statistics (real data mode)
    if has_mask_type and metrics['n_dead'] > 0:
        dead_mask = data['mask_type'] == 1
        dead_stats = {
            'n': dead_mask.sum(),
            'pred_npho_mean': np.mean(data['pred_npho'][dead_mask]),
            'pred_npho_std': np.std(data['pred_npho'][dead_mask]),
            'pred_npho_min': np.min(data['pred_npho'][dead_mask]),
            'pred_npho_max': np.max(data['pred_npho'][dead_mask]),
        }
        if predict_time and 'pred_time' in data:
            dead_stats.update({
                'pred_time_mean': np.mean(data['pred_time'][dead_mask]),
                'pred_time_std': np.std(data['pred_time'][dead_mask]),
                'pred_time_min': np.min(data['pred_time'][dead_mask]),
                'pred_time_max': np.max(data['pred_time'][dead_mask]),
            })
        metrics['dead_stats'] = dead_stats

    return metrics


def compute_baseline_metrics(data: Dict[str, np.ndarray],
                              sentinel_time: float = DEFAULT_SENTINEL_TIME) -> Dict:
    """Compute metrics for baseline predictions found in data.

    Returns dict mapping baseline name ('avg', 'sa') to their metrics.
    """
    has_mask_type = 'mask_type' in data
    if has_mask_type:
        eval_mask = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        eval_mask = data['error_npho'] > -999

    baseline_metrics = {}

    for prefix, label in [('baseline_avg', 'avg'), ('baseline_sa', 'sa')]:
        error_key = f'{prefix}_error_npho'
        pred_key = f'{prefix}_npho'
        if error_key not in data:
            continue

        err = data[error_key][eval_mask]
        if len(err) == 0:
            continue

        global_m = {
            'n': len(err),
            'npho_mae': np.mean(np.abs(err)),
            'npho_rmse': np.sqrt(np.mean(err ** 2)),
            'npho_bias': np.mean(err),
            'npho_std': np.std(err),
            'npho_68pct': np.percentile(np.abs(err), 68),
            'npho_95pct': np.percentile(np.abs(err), 95),
        }

        # Per-face
        per_face = {}
        for face_int, face_name in FACE_INT_TO_NAME.items():
            face_mask = eval_mask & (data['face'] == face_int)
            if face_mask.sum() == 0:
                continue
            face_err = data[error_key][face_mask]
            per_face[face_name] = {
                'n': len(face_err),
                'npho_mae': np.mean(np.abs(face_err)),
                'npho_rmse': np.sqrt(np.mean(face_err ** 2)),
                'npho_bias': np.mean(face_err),
                'npho_68pct': np.percentile(np.abs(face_err), 68),
            }

        baseline_metrics[label] = {'global': global_m, 'per_face': per_face}

    return baseline_metrics


def plot_baseline_comparison(data: Dict[str, np.ndarray], output_dir: str,
                              has_mask_type: bool, baseline_metrics: Dict,
                              npho_scale: float, npho_scale2: float):
    """Plot ML vs baseline comparison: overlay residuals and per-face bar chart."""
    if not HAS_MATPLOTLIB or not baseline_metrics:
        return

    if has_mask_type:
        valid = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        valid = data['error_npho'] > -999

    if valid.sum() == 0:
        return

    # --- Overlay residual distributions ---
    n_baselines = len(baseline_metrics)
    fig, ax = plt.subplots(figsize=(8, 5))

    # ML residuals
    err_ml = data['error_npho'][valid]
    ax.hist(err_ml, bins=100, range=(-0.3, 0.4), density=True, alpha=0.5,
            color='blue', label=f'ML (MAE={np.mean(np.abs(err_ml)):.4f})')

    colors = {'avg': 'orange', 'sa': 'green'}
    labels = {'avg': 'Neighbor Avg', 'sa': 'Solid Angle'}
    for bname in ['avg', 'sa']:
        error_key = f'baseline_{bname}_error_npho'
        if error_key not in data:
            continue
        err_b = data[error_key][valid]
        ax.hist(err_b, bins=100, range=(-0.3, 0.4), density=True, alpha=0.4,
                color=colors.get(bname, 'gray'),
                label=f'{labels.get(bname, bname)} (MAE={np.mean(np.abs(err_b)):.4f})')

    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.3, 0.4)
    ax.set_xlabel('Npho Error (pred - truth) [normalized]')
    ax.set_ylabel('Density')
    ax.set_title('Residual Comparison: ML vs Baselines')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_residual_overlay.pdf'))
    plt.close()

    # --- Per-face MAE comparison bar chart ---
    faces = list(FACE_INT_TO_NAME.values())
    methods = ['ML'] + [labels.get(b, b) for b in baseline_metrics.keys()]
    method_colors = ['blue'] + [colors.get(b, 'gray') for b in baseline_metrics.keys()]

    # Gather ML per-face MAEs
    ml_maes = []
    for face_int, face_name in FACE_INT_TO_NAME.items():
        face_mask = valid & (data['face'] == face_int)
        if face_mask.sum() > 0:
            ml_maes.append(np.mean(np.abs(data['error_npho'][face_mask])))
        else:
            ml_maes.append(0)

    all_maes = [ml_maes]
    for bname, bm in baseline_metrics.items():
        bmaes = []
        for face_name in faces:
            if face_name in bm['per_face']:
                bmaes.append(bm['per_face'][face_name]['npho_mae'])
            else:
                bmaes.append(0)
        all_maes.append(bmaes)

    x = np.arange(len(faces))
    n_methods = len(methods)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (method_name, maes, color) in enumerate(zip(methods, all_maes, method_colors)):
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, maes, width, label=method_name, color=color, alpha=0.7)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.annotate(f'{h:.4f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 2), textcoords="offset points",
                            ha='center', va='bottom', fontsize=6, rotation=45)

    ax.set_xlabel('Face')
    ax.set_ylabel('Npho MAE (normalized)')
    ax.set_title('Per-Face Npho MAE: ML vs Baselines')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'baseline_per_face_comparison.pdf'))
    plt.close()

    print("[INFO] Saved baseline_residual_overlay.pdf + baseline_per_face_comparison.pdf")


def plot_residual_distributions(data: Dict[str, np.ndarray], output_dir: str,
                                 has_mask_type: bool,
                                 npho_scale: float, npho_scale2: float,
                                 time_scale: float, time_shift: float,
                                 sentinel_time: float, predict_time: bool = True):
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

    # === Normalized residuals ===
    n_cols = 2 if predict_time else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

    # Npho residuals
    ax = axes[0]
    ax.hist(err_npho, bins=100, range=(-0.3, 0.4), density=True, alpha=0.7, color='blue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(-0.3, 0.4)
    ax.set_xlabel('Npho Error (pred - truth) [normalized]')
    ax.set_ylabel('Density')
    ax.set_title(f'Npho Residuals (n={len(err_npho):,})\n'
                 f'MAE={np.mean(np.abs(err_npho)):.4f}, Bias={np.mean(err_npho):.4f}')

    # Time residuals (only if predicted)
    if predict_time and 'error_time' in data:
        err_time = data['error_time'][valid]
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

    # Denormalize npho
    truth_npho_raw = denormalize_npho(truth_npho_norm, npho_scale, npho_scale2)
    pred_npho_raw = denormalize_npho(pred_npho_norm, npho_scale, npho_scale2)
    err_npho_raw = pred_npho_raw - truth_npho_raw

    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

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

    # Time residuals (raw) - only if predicted
    if predict_time and 'pred_time' in data:
        truth_time_norm = data['truth_time'][valid]
        pred_time_norm = data['pred_time'][valid]
        truth_time_raw = denormalize_time(truth_time_norm, time_scale, time_shift)
        pred_time_raw = denormalize_time(pred_time_norm, time_scale, time_shift)
        err_time_raw = pred_time_raw - truth_time_raw

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
                             time_scale: float, time_shift: float,
                             predict_time: bool = True):
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

    # === Normalized Time residuals (only if predicted) ===
    if predict_time and 'error_time' in data:
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

    print("[INFO] Saved residual_per_face_npho.pdf" + (" + time.pdf" if predict_time else ""))

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

    # === Denormalized Time residuals (only if predicted) ===
    if predict_time and 'pred_time' in data:
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

    print("[INFO] Saved residual_per_face_npho_denorm.pdf" + (" + time_denorm.pdf" if predict_time else ""))


def plot_scatter_truth_vs_pred(data: Dict[str, np.ndarray], output_dir: str,
                                has_mask_type: bool,
                                npho_scale: float, npho_scale2: float,
                                time_scale: float, time_shift: float,
                                predict_time: bool = True):
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

    # Determine layout based on whether time is predicted
    has_time = predict_time and 'pred_time' in data
    n_cols = 2 if has_time else 1

    # === Normalized scatter ===
    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

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

    # Time - log scale density with fixed range (only if predicted)
    if has_time:
        truth_time = data['truth_time'][valid]
        pred_time = data['pred_time'][valid]
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

    fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
    if n_cols == 1:
        axes = [axes]

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

    # Time denormalized (only if predicted)
    if has_time:
        truth_time_raw = denormalize_time(truth_time, time_scale, time_shift) * 1e9  # ns
        pred_time_raw = denormalize_time(pred_time, time_scale, time_shift) * 1e9  # ns
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
        fig, axes = plt.subplots(1, n_cols, figsize=(7 * n_cols, 5))
        if n_cols == 1:
            axes = [axes]

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

        if has_time:
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


def _compute_slice_metrics(truth: np.ndarray, error: np.ndarray,
                            bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute MAE, bias, 68th percentile in each bin.

    Returns bin_centers, mae, bias, pct68 arrays.
    """
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices = np.digitize(truth, bin_edges) - 1
    indices = np.clip(indices, 0, len(bin_centers) - 1)

    mae = np.full(len(bin_centers), np.nan)
    bias = np.full(len(bin_centers), np.nan)
    pct68 = np.full(len(bin_centers), np.nan)

    for i in range(len(bin_centers)):
        m = indices == i
        if m.sum() < 10:
            continue
        err = error[m]
        mae[i] = np.mean(np.abs(err))
        bias[i] = np.mean(err)
        pct68[i] = np.percentile(np.abs(err), 68)

    return bin_centers, mae, bias, pct68


def plot_resolution_vs_signal(data: Dict[str, np.ndarray], output_dir: str,
                               has_mask_type: bool,
                               npho_scale: float, npho_scale2: float,
                               n_bins: int = 30, predict_time: bool = True):
    """Plot error statistics (MAE, bias, 68%) as a function of truth npho (slice plot).

    Binning strategy:
    - Normalized: equal-width bins in [0, 95th percentile]
    - Denormalized: log-spaced bins with log x-axis
    - Per-face: same equal-width bins as normalized
    """
    if not HAS_MATPLOTLIB:
        return

    if has_mask_type:
        valid = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        valid = data['error_npho'] > -999

    if valid.sum() == 0:
        return

    truth_npho = data['truth_npho'][valid]
    error_npho = data['error_npho'][valid]

    # --- Normalized space: equal-width bins ---
    x_max = np.percentile(truth_npho, 95)
    x_min = max(0.0, truth_npho.min())
    if x_max <= x_min:
        print("[WARNING] Not enough range for slice plot")
        return
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)

    centers, mae_vals, bias_vals, pct68_vals = _compute_slice_metrics(
        truth_npho, error_npho, bin_edges)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(centers, mae_vals, 'o', color='blue', markersize=4)
    axes[0].set_xlabel('Truth Npho (normalized)')
    axes[0].set_ylabel('MAE (normalized)')
    axes[0].set_title('MAE vs Truth Npho')

    axes[1].plot(centers, bias_vals, 'o', color='red', markersize=4)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Truth Npho (normalized)')
    axes[1].set_ylabel('Bias (normalized)')
    axes[1].set_title('Bias vs Truth Npho')

    axes[2].plot(centers, pct68_vals, 'o', color='green', markersize=4)
    axes[2].set_xlabel('Truth Npho (normalized)')
    axes[2].set_ylabel('68th Percentile |Error|')
    axes[2].set_title('Resolution (68%) vs Truth Npho')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resolution_vs_signal.pdf'))
    plt.close()

    # --- Denormalized space: relative resolution, truth >= 100 photons ---
    truth_raw = denormalize_npho(truth_npho, npho_scale, npho_scale2)
    pred_raw = denormalize_npho(data['pred_npho'][valid], npho_scale, npho_scale2)
    error_raw = pred_raw - truth_raw

    # Cut: only sensors with truth >= 100 photons
    raw_cut = truth_raw >= 100.0
    if raw_cut.sum() < 100:
        print("[INFO] Saved resolution_vs_signal.pdf (too few sensors above 100 photons for denorm)")
        return

    truth_raw_cut = truth_raw[raw_cut]
    error_raw_cut = error_raw[raw_cut]
    rel_error = error_raw_cut / truth_raw_cut

    raw_max = np.percentile(truth_raw_cut, 99)
    bin_edges_raw = np.logspace(np.log10(100.0), np.log10(raw_max), n_bins + 1)

    centers_raw, rel_mae, rel_bias, rel_pct68 = _compute_slice_metrics(
        truth_raw_cut, rel_error, bin_edges_raw)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(centers_raw, rel_mae, 'o', color='blue', markersize=4)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Truth Npho [photons]')
    axes[0].set_ylabel('Relative MAE (|error|/truth)')
    axes[0].set_title('Relative MAE vs Truth Npho')

    axes[1].plot(centers_raw, rel_bias, 'o', color='red', markersize=4)
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Truth Npho [photons]')
    axes[1].set_ylabel('Relative Bias (error/truth)')
    axes[1].set_title('Relative Bias vs Truth Npho')

    axes[2].plot(centers_raw, rel_pct68, 'o', color='green', markersize=4)
    axes[2].set_xscale('log')
    axes[2].set_xlabel('Truth Npho [photons]')
    axes[2].set_ylabel('Relative 68th Pct (|error|/truth)')
    axes[2].set_title('Relative Resolution (68%) vs Truth Npho')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resolution_vs_signal_denorm.pdf'))
    plt.close()

    # --- Per-face overlay (normalized, same equal-width bins) ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    face_colors = {'inner': 'blue', 'us': 'orange', 'ds': 'green',
                   'outer': 'red', 'top': 'purple', 'bot': 'brown'}

    for face_int, face_name in FACE_INT_TO_NAME.items():
        face_valid = valid & (data['face'] == face_int)
        if face_valid.sum() < 100:
            continue

        ft = data['truth_npho'][face_valid]
        fe = data['error_npho'][face_valid]

        _, f_mae, f_bias, f_pct68 = _compute_slice_metrics(ft, fe, bin_edges)

        color = face_colors.get(face_name, 'gray')
        axes[0].plot(centers, f_mae, 'o', color=color, markersize=3, alpha=0.7, label=face_name)
        axes[1].plot(centers, f_bias, 'o', color=color, markersize=3, alpha=0.7, label=face_name)
        axes[2].plot(centers, f_pct68, 'o', color=color, markersize=3, alpha=0.7, label=face_name)

    axes[0].set_xlabel('Truth Npho (normalized)')
    axes[0].set_ylabel('MAE (normalized)')
    axes[0].set_title('MAE vs Truth Npho (per face)')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel('Truth Npho (normalized)')
    axes[1].set_ylabel('Bias (normalized)')
    axes[1].set_title('Bias vs Truth Npho (per face)')
    axes[1].axhline(0, color='gray', linestyle='--', alpha=0.5)
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel('Truth Npho (normalized)')
    axes[2].set_ylabel('68th Percentile |Error|')
    axes[2].set_title('Resolution (68%) vs Truth Npho (per face)')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'resolution_vs_signal_per_face.pdf'))
    plt.close()
    print("[INFO] Saved resolution_vs_signal.pdf, resolution_vs_signal_denorm.pdf, resolution_vs_signal_per_face.pdf")


def plot_metrics_summary(metrics: Dict, output_dir: str, sentinel_time: float, predict_time: bool = True):
    """Plot bar chart of per-face metrics."""
    if not HAS_MATPLOTLIB:
        return

    if 'per_face' not in metrics or not metrics['per_face']:
        return

    faces = list(metrics['per_face'].keys())
    npho_mae = [metrics['per_face'][f]['npho_mae'] for f in faces]

    x = np.arange(len(faces))

    if predict_time:
        time_mae = [metrics['per_face'][f].get('time_mae', np.nan) for f in faces]
        time_mae_valid = [metrics['per_face'][f].get('time_mae_valid', np.nan) for f in faces]
        # Scale down time_mae (all) by 10 for better visualization
        time_mae_scaled = [t / 10 if not np.isnan(t) else np.nan for t in time_mae]
        width = 0.25
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width, npho_mae, width, label='Npho MAE', color='blue', alpha=0.7)
        bars2 = ax.bar(x, time_mae_scaled, width, label='Time MAE (all) ÷10', color='green', alpha=0.7)
        bars3 = ax.bar(x + width, time_mae_valid, width, label='Time MAE (valid only)', color='orange', alpha=0.7)
        ax.set_title('Per-Face MAE Comparison\n(Time valid = excluding sensors with truth_time == sentinel)')
        all_bars = [bars1, bars2, bars3]
    else:
        width = 0.4
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x, npho_mae, width, label='Npho MAE', color='blue', alpha=0.7)
        ax.set_title('Per-Face Npho MAE')
        all_bars = [bars1]

    ax.set_xlabel('Face')
    ax.set_ylabel('MAE (normalized)')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.legend()

    # Add value labels
    for bars in all_bars:
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
        predict_time = metrics.get('predict_time', True)
        with open(os.path.join(output_dir, 'face_metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            if predict_time:
                header = ['face', 'n', 'n_valid_time',
                          'npho_mae', 'npho_rmse', 'npho_bias', 'npho_68pct',
                          'time_mae', 'time_rmse', 'time_bias', 'time_68pct',
                          'time_mae_valid', 'time_bias_valid']
            else:
                header = ['face', 'n',
                          'npho_mae', 'npho_rmse', 'npho_bias', 'npho_68pct']
            writer.writerow(header)
            for face_name in FACE_NAMES:
                if face_name in metrics['per_face']:
                    m = metrics['per_face'][face_name]
                    if predict_time:
                        writer.writerow([face_name, m['n'], m.get('n_valid_time', ''),
                                         m['npho_mae'], m['npho_rmse'], m['npho_bias'], m['npho_68pct'],
                                         m.get('time_mae', ''), m.get('time_rmse', ''),
                                         m.get('time_bias', ''), m.get('time_68pct', ''),
                                         m.get('time_mae_valid', ''), m.get('time_bias_valid', '')])
                    else:
                        writer.writerow([face_name, m['n'],
                                         m['npho_mae'], m['npho_rmse'], m['npho_bias'], m['npho_68pct']])
        print("[INFO] Saved face_metrics.csv")

    # Dead channel stats
    if 'dead_stats' in metrics:
        with open(os.path.join(output_dir, 'dead_channel_stats.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['metric', 'value'])
            for k, v in metrics['dead_stats'].items():
                writer.writerow([k, v])
        print("[INFO] Saved dead_channel_stats.csv")


def print_report(metrics: Dict, predict_time: bool = True,
                 baseline_metrics: Optional[Dict] = None):
    """Print metrics report to console."""
    print("\n" + "=" * 70)
    print("INPAINTER ANALYSIS REPORT")
    print("=" * 70)

    mode_str = "Real Data" if metrics['mode'] == 'real_data' else "MC"
    print(f"Mode: {mode_str}")
    print(f"Predict channels: {'npho, time' if predict_time else 'npho only'}")
    print(f"Total predictions: {metrics['n_total']:,}")
    print(f"Predictions with truth: {metrics['n_with_truth']:,}")
    if 'n_artificial' in metrics:
        print(f"  - Artificial masks: {metrics['n_artificial']:,}")
        print(f"  - Dead channels: {metrics['n_dead']:,}")

    if 'global' in metrics:
        g = metrics['global']
        print(f"\n--- Global Metrics ---")
        if predict_time:
            print(f"{'Metric':<20} {'Npho':>12} {'Time (all)':>12} {'Time (valid)':>12}")
            print("-" * 60)
            print(f"{'N':<20} {g['n']:>12,} {g['n']:>12,} {g.get('n_valid_time', 0):>12,}")
            print(f"{'MAE':<20} {g['npho_mae']:>12.4f} {g['time_mae']:>12.4f} {g['time_mae_valid']:>12.4f}")
            print(f"{'RMSE':<20} {g['npho_rmse']:>12.4f} {g['time_rmse']:>12.4f} {g['time_rmse_valid']:>12.4f}")
            print(f"{'Bias':<20} {g['npho_bias']:>12.4f} {g['time_bias']:>12.4f} {g['time_bias_valid']:>12.4f}")
            print(f"{'Std':<20} {g['npho_std']:>12.4f} {g['time_std']:>12.4f} {g['time_std_valid']:>12.4f}")
            print(f"{'68th pct':<20} {g['npho_68pct']:>12.4f} {g['time_68pct']:>12.4f} {g['time_68pct_valid']:>12.4f}")
            print(f"{'95th pct':<20} {g['npho_95pct']:>12.4f} {g['time_95pct']:>12.4f} {g['time_95pct_valid']:>12.4f}")
        else:
            print(f"{'Metric':<20} {'Npho':>12}")
            print("-" * 35)
            print(f"{'N':<20} {g['n']:>12,}")
            print(f"{'MAE':<20} {g['npho_mae']:>12.4f}")
            print(f"{'RMSE':<20} {g['npho_rmse']:>12.4f}")
            print(f"{'Bias':<20} {g['npho_bias']:>12.4f}")
            print(f"{'Std':<20} {g['npho_std']:>12.4f}")
            print(f"{'68th pct':<20} {g['npho_68pct']:>12.4f}")
            print(f"{'95th pct':<20} {g['npho_95pct']:>12.4f}")

    if 'per_face' in metrics and metrics['per_face']:
        print(f"\n--- Per-Face Metrics ---")
        if predict_time:
            print(f"{'Face':<8} {'N':>8} {'Npho MAE':>10} {'Npho Bias':>10} {'Time MAE':>10} {'Time MAE*':>10}")
            print("-" * 60)
            print("(* = valid sensors only, excluding sentinel)")
            for face_name in FACE_NAMES:
                if face_name in metrics['per_face']:
                    m = metrics['per_face'][face_name]
                    time_mae = m.get('time_mae', np.nan)
                    time_mae_valid = m.get('time_mae_valid', np.nan)
                    time_mae_str = f"{time_mae:>10.4f}" if not np.isnan(time_mae) else "       N/A"
                    time_mae_valid_str = f"{time_mae_valid:>10.4f}" if not np.isnan(time_mae_valid) else "       N/A"
                    print(f"{face_name:<8} {m['n']:>8,} {m['npho_mae']:>10.4f} {m['npho_bias']:>10.4f} "
                          f"{time_mae_str} {time_mae_valid_str}")
        else:
            print(f"{'Face':<8} {'N':>8} {'Npho MAE':>10} {'Npho Bias':>10}")
            print("-" * 40)
            for face_name in FACE_NAMES:
                if face_name in metrics['per_face']:
                    m = metrics['per_face'][face_name]
                    print(f"{face_name:<8} {m['n']:>8,} {m['npho_mae']:>10.4f} {m['npho_bias']:>10.4f}")

    if 'dead_stats' in metrics:
        d = metrics['dead_stats']
        print(f"\n--- Dead Channel Predictions (No Ground Truth) ---")
        print(f"Total: {d['n']:,}")
        print(f"Npho: mean={d['pred_npho_mean']:.4f}, std={d['pred_npho_std']:.4f}, "
              f"range=[{d['pred_npho_min']:.4f}, {d['pred_npho_max']:.4f}]")
        if predict_time and 'pred_time_mean' in d:
            print(f"Time: mean={d['pred_time_mean']:.4f}, std={d['pred_time_std']:.4f}, "
                  f"range=[{d['pred_time_min']:.4f}, {d['pred_time_max']:.4f}]")

    # Baseline comparison
    if baseline_metrics and 'global' in metrics:
        baseline_name_map = {'avg': 'Neighbor Avg', 'sa': 'Solid Angle'}
        print(f"\n--- ML vs Baselines (npho, normalized) ---")
        print(f"{'Method':<20} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'68%':>8} {'95%':>8}")
        print("-" * 58)
        g = metrics['global']
        print(f"{'ML Model':<20} {g['npho_mae']:>8.4f} {g['npho_rmse']:>8.4f} "
              f"{g['npho_bias']:>8.4f} {g['npho_68pct']:>8.4f} {g['npho_95pct']:>8.4f}")
        for bname, bm in baseline_metrics.items():
            bg = bm['global']
            label = baseline_name_map.get(bname, bname)
            print(f"{label:<20} {bg['npho_mae']:>8.4f} {bg['npho_rmse']:>8.4f} "
                  f"{bg['npho_bias']:>8.4f} {bg['npho_68pct']:>8.4f} {bg['npho_95pct']:>8.4f}")

        # Per-face comparison
        print(f"\n--- Per-Face MAE: ML vs Baselines ---")
        header = f"{'Face':<8} {'ML':>10}"
        for bname in baseline_metrics:
            header += f" {baseline_name_map.get(bname, bname):>14}"
        print(header)
        print("-" * len(header))
        for face_name in FACE_NAMES:
            ml_mae = metrics['per_face'].get(face_name, {}).get('npho_mae', np.nan)
            line = f"{face_name:<8} {ml_mae:>10.4f}" if not np.isnan(ml_mae) else f"{face_name:<8} {'N/A':>10}"
            for bname, bm in baseline_metrics.items():
                b_mae = bm['per_face'].get(face_name, {}).get('npho_mae', np.nan)
                if not np.isnan(b_mae):
                    line += f" {b_mae:>14.4f}"
                else:
                    line += f" {'N/A':>14}"
            print(line)

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
    parser.add_argument("--sentinel", type=float, default=DEFAULT_SENTINEL_TIME,
                        help=f"Sentinel value for invalid sensors (default: {DEFAULT_SENTINEL_TIME})")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load predictions and metadata
    data, metadata = load_predictions(args.input)
    has_mask_type = 'mask_type' in data

    # Get predict_time from metadata
    predict_channels = metadata.get('predict_channels', ['npho', 'time'])
    predict_time = 'time' in predict_channels

    # Use metadata normalization params if not overridden by CLI
    # (CLI args use defaults, so check if they match defaults to detect override)
    npho_scale = metadata.get('npho_scale', args.npho_scale)
    npho_scale2 = metadata.get('npho_scale2', args.npho_scale2)
    time_scale = metadata.get('time_scale', args.time_scale)
    time_shift = metadata.get('time_shift', args.time_shift)

    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = compute_detailed_metrics(data, sentinel_time=args.sentinel, predict_time=predict_time)

    # Compute baseline metrics (auto-detected from data branches)
    baseline_metrics = compute_baseline_metrics(data, sentinel_time=args.sentinel)
    if baseline_metrics:
        print(f"[INFO] Baseline metrics computed for: {', '.join(baseline_metrics.keys())}")

    # Print report
    print_report(metrics, predict_time=predict_time, baseline_metrics=baseline_metrics or None)

    # Save metrics
    save_metrics_csv(metrics, args.output)

    # Save baseline metrics CSV if available
    if baseline_metrics:
        import csv
        baseline_name_map = {'avg': 'neighbor_avg', 'sa': 'solid_angle'}
        with open(os.path.join(args.output, 'baseline_metrics.csv'), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['method', 'scope', 'metric', 'value'])
            for bname, bm in baseline_metrics.items():
                method = baseline_name_map.get(bname, bname)
                for k, v in bm['global'].items():
                    writer.writerow([method, 'global', k, v])
                for face_name, fm in bm['per_face'].items():
                    for k, v in fm.items():
                        writer.writerow([method, face_name, k, v])
        print("[INFO] Saved baseline_metrics.csv")

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("[INFO] Generating plots...")
        plot_residual_distributions(data, args.output, has_mask_type,
                                     npho_scale, npho_scale2,
                                     time_scale, time_shift,
                                     args.sentinel, predict_time=predict_time)
        plot_per_face_residuals(data, args.output, has_mask_type,
                                 npho_scale, npho_scale2,
                                 time_scale, time_shift, predict_time=predict_time)
        plot_scatter_truth_vs_pred(data, args.output, has_mask_type,
                                    npho_scale, npho_scale2,
                                    time_scale, time_shift, predict_time=predict_time)
        plot_resolution_vs_signal(data, args.output, has_mask_type,
                                   npho_scale, npho_scale2, predict_time=predict_time)
        plot_metrics_summary(metrics, args.output, args.sentinel, predict_time=predict_time)
        if baseline_metrics:
            plot_baseline_comparison(data, args.output, has_mask_type,
                                      baseline_metrics, npho_scale, npho_scale2)

    print(f"\n[INFO] Analysis complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
