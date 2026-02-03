#!/usr/bin/env python3
"""
Analyze Inpainter Predictions

Generates detailed metrics and plots from inpainter validation output.
Works with both MC and real data validation results.

Usage:
    python macro/analyze_inpainter.py predictions.root --output analysis/

    # Specify mode explicitly
    python macro/analyze_inpainter.py predictions.root --mode real --output analysis/
"""

import os
import sys
import argparse
import numpy as np
import uproot
from pathlib import Path
from typing import Dict, List, Optional

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

FACE_NAMES = ['inner', 'us', 'ds', 'outer', 'top', 'bot']
FACE_INT_TO_NAME = {0: 'inner', 1: 'us', 2: 'ds', 3: 'outer', 4: 'top', 5: 'bot'}


def load_predictions(input_path: str) -> Dict[str, np.ndarray]:
    """Load predictions from ROOT file."""
    print(f"[INFO] Loading predictions from {input_path}")

    with uproot.open(input_path) as f:
        tree = f['predictions']
        data = {key: tree[key].array(library='np') for key in tree.keys()}

    n_preds = len(data['event_idx'])
    print(f"[INFO] Loaded {n_preds:,} predictions")

    # Check if mask_type exists (real data validation)
    has_mask_type = 'mask_type' in data
    if has_mask_type:
        n_artificial = (data['mask_type'] == 0).sum()
        n_dead = (data['mask_type'] == 1).sum()
        print(f"[INFO] Artificial masks: {n_artificial:,}, Dead channels: {n_dead:,}")

    return data


def compute_detailed_metrics(data: Dict[str, np.ndarray]) -> Dict:
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

    metrics['global'] = {
        'n': len(err_npho),
        'npho_mae': np.mean(np.abs(err_npho)),
        'npho_rmse': np.sqrt(np.mean(err_npho ** 2)),
        'npho_bias': np.mean(err_npho),
        'npho_std': np.std(err_npho),
        'npho_68pct': np.percentile(np.abs(err_npho), 68),
        'npho_95pct': np.percentile(np.abs(err_npho), 95),
        'time_mae': np.mean(np.abs(err_time)),
        'time_rmse': np.sqrt(np.mean(err_time ** 2)),
        'time_bias': np.mean(err_time),
        'time_std': np.std(err_time),
        'time_68pct': np.percentile(np.abs(err_time), 68),
        'time_95pct': np.percentile(np.abs(err_time), 95),
    }

    # Per-face metrics
    metrics['per_face'] = {}
    for face_int, face_name in FACE_INT_TO_NAME.items():
        face_mask = eval_mask & (data['face'] == face_int)
        if face_mask.sum() == 0:
            continue

        face_err_npho = data['error_npho'][face_mask]
        face_err_time = data['error_time'][face_mask]

        metrics['per_face'][face_name] = {
            'n': len(face_err_npho),
            'npho_mae': np.mean(np.abs(face_err_npho)),
            'npho_rmse': np.sqrt(np.mean(face_err_npho ** 2)),
            'npho_bias': np.mean(face_err_npho),
            'npho_68pct': np.percentile(np.abs(face_err_npho), 68),
            'time_mae': np.mean(np.abs(face_err_time)),
            'time_rmse': np.sqrt(np.mean(face_err_time ** 2)),
            'time_bias': np.mean(face_err_time),
            'time_68pct': np.percentile(np.abs(face_err_time), 68),
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
                                 has_mask_type: bool):
    """Plot residual distributions."""
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

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Npho residuals
    ax = axes[0]
    ax.hist(err_npho, bins=100, density=True, alpha=0.7, color='blue')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Npho Error (pred - truth)')
    ax.set_ylabel('Density')
    ax.set_title(f'Npho Residuals (n={len(err_npho):,})\n'
                 f'MAE={np.mean(np.abs(err_npho)):.4f}, Bias={np.mean(err_npho):.4f}')

    # Time residuals
    ax = axes[1]
    ax.hist(err_time, bins=100, density=True, alpha=0.7, color='green')
    ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time Error (pred - truth)')
    ax.set_ylabel('Density')
    ax.set_title(f'Time Residuals (n={len(err_time):,})\n'
                 f'MAE={np.mean(np.abs(err_time)):.4f}, Bias={np.mean(err_time):.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_distributions.pdf'))
    plt.close()
    print("[INFO] Saved residual_distributions.pdf")


def plot_per_face_residuals(data: Dict[str, np.ndarray], output_dir: str,
                             has_mask_type: bool):
    """Plot per-face residual distributions."""
    if not HAS_MATPLOTLIB:
        return

    if has_mask_type:
        valid = (data['mask_type'] == 0) & (data['error_npho'] > -999)
    else:
        valid = data['error_npho'] > -999

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
        ax = axes[idx]
        face_mask = valid & (data['face'] == face_int)

        if face_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{face_name}\nNo data', ha='center', va='center')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        err_npho = data['error_npho'][face_mask]
        ax.hist(err_npho, bins=50, density=True, alpha=0.7)
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Npho Error')
        ax.set_title(f'{face_name} (n={len(err_npho):,})\n'
                     f'MAE={np.mean(np.abs(err_npho)):.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_per_face_npho.pdf'))
    plt.close()

    # Time residuals
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
        ax = axes[idx]
        face_mask = valid & (data['face'] == face_int)

        if face_mask.sum() == 0:
            ax.text(0.5, 0.5, f'{face_name}\nNo data', ha='center', va='center')
            continue

        err_time = data['error_time'][face_mask]
        ax.hist(err_time, bins=50, density=True, alpha=0.7, color='green')
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('Time Error')
        ax.set_title(f'{face_name} (n={len(err_time):,})\n'
                     f'MAE={np.mean(np.abs(err_time)):.4f}')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'residual_per_face_time.pdf'))
    plt.close()
    print("[INFO] Saved residual_per_face_*.pdf")


def plot_scatter_truth_vs_pred(data: Dict[str, np.ndarray], output_dir: str,
                                has_mask_type: bool):
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

    # --- Full range scatter (log scale density) ---
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

    # Time - log scale density
    ax = axes[1]
    h = ax.hist2d(truth_time, pred_time, bins=100, cmap='Greens', norm=LogNorm(), cmin=1)
    plt.colorbar(h[3], ax=ax, label='Count (log)')
    lims = [min(truth_time.min(), pred_time.min()), max(truth_time.max(), pred_time.max())]
    ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y=x')
    ax.set_xlabel('Truth Time (normalized)')
    ax.set_ylabel('Pred Time (normalized)')
    ax.set_title('Time: Truth vs Prediction')
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scatter_truth_vs_pred.pdf'))
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
        print("[INFO] Saved scatter_truth_vs_pred.pdf and scatter_truth_vs_pred_highsignal.pdf")
    else:
        print("[INFO] Saved scatter_truth_vs_pred.pdf")


def plot_metrics_summary(metrics: Dict, output_dir: str):
    """Plot bar chart of per-face metrics."""
    if not HAS_MATPLOTLIB:
        return

    if 'per_face' not in metrics or not metrics['per_face']:
        return

    faces = list(metrics['per_face'].keys())
    npho_mae = [metrics['per_face'][f]['npho_mae'] for f in faces]
    time_mae = [metrics['per_face'][f]['time_mae'] for f in faces]

    x = np.arange(len(faces))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, npho_mae, width, label='Npho MAE', color='blue', alpha=0.7)
    bars2 = ax.bar(x + width/2, time_mae, width, label='Time MAE', color='green', alpha=0.7)

    ax.set_xlabel('Face')
    ax.set_ylabel('MAE')
    ax.set_title('Per-Face MAE Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)

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
            header = ['face', 'n', 'npho_mae', 'npho_rmse', 'npho_bias', 'npho_68pct',
                      'time_mae', 'time_rmse', 'time_bias', 'time_68pct']
            writer.writerow(header)
            for face_name in FACE_NAMES:
                if face_name in metrics['per_face']:
                    m = metrics['per_face'][face_name]
                    writer.writerow([face_name, m['n'],
                                     m['npho_mae'], m['npho_rmse'], m['npho_bias'], m['npho_68pct'],
                                     m['time_mae'], m['time_rmse'], m['time_bias'], m['time_68pct']])
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
        print(f"{'Metric':<15} {'Npho':>12} {'Time':>12}")
        print("-" * 40)
        print(f"{'MAE':<15} {g['npho_mae']:>12.4f} {g['time_mae']:>12.4f}")
        print(f"{'RMSE':<15} {g['npho_rmse']:>12.4f} {g['time_rmse']:>12.4f}")
        print(f"{'Bias':<15} {g['npho_bias']:>12.4f} {g['time_bias']:>12.4f}")
        print(f"{'Std':<15} {g['npho_std']:>12.4f} {g['time_std']:>12.4f}")
        print(f"{'68th pct':<15} {g['npho_68pct']:>12.4f} {g['time_68pct']:>12.4f}")
        print(f"{'95th pct':<15} {g['npho_95pct']:>12.4f} {g['time_95pct']:>12.4f}")

    if 'per_face' in metrics and metrics['per_face']:
        print(f"\n--- Per-Face Metrics ---")
        print(f"{'Face':<8} {'N':>8} {'Npho MAE':>10} {'Time MAE':>10}")
        print("-" * 40)
        for face_name in FACE_NAMES:
            if face_name in metrics['per_face']:
                m = metrics['per_face'][face_name]
                print(f"{face_name:<8} {m['n']:>8,} {m['npho_mae']:>10.4f} {m['time_mae']:>10.4f}")

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

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load predictions
    data = load_predictions(args.input)
    has_mask_type = 'mask_type' in data

    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = compute_detailed_metrics(data)

    # Print report
    print_report(metrics)

    # Save metrics
    save_metrics_csv(metrics, args.output)

    # Generate plots
    if not args.no_plots and HAS_MATPLOTLIB:
        print("[INFO] Generating plots...")
        plot_residual_distributions(data, args.output, has_mask_type)
        plot_per_face_residuals(data, args.output, has_mask_type)
        plot_scatter_truth_vs_pred(data, args.output, has_mask_type)
        plot_metrics_summary(metrics, args.output)

    print(f"\n[INFO] Analysis complete! Results saved to {args.output}/")


if __name__ == "__main__":
    main()
