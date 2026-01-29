#!/usr/bin/env python3
"""
Inpainter Prediction Analysis Macro

Analyzes inpainter predictions from ROOT files and generates evaluation plots.
Supports both training predictions and real data validation predictions.

For real data validation (with mask_type column):
- mask_type=0 (artificial): Has ground truth, used for quantitative metrics
- mask_type=1 (dead): No ground truth, plausibility checks only

Usage:
    # Training predictions
    python macro/analyze_inpainter.py predictions.root --output analysis_output/

    # Real data validation predictions
    python macro/analyze_inpainter.py validation_real/real_data_predictions.root --output analysis_output/
"""

import argparse
import os
import numpy as np
import pandas as pd
import uproot
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from pathlib import Path

# Face mapping
FACE_ID_TO_NAME = {0: "inner", 1: "us", 2: "ds", 3: "outer", 4: "top", 5: "bot"}
FACE_NAME_TO_ID = {v: k for k, v in FACE_ID_TO_NAME.items()}

# Mask type mapping (for real data validation)
MASK_TYPE_NAMES = {0: "artificial", 1: "dead"}

# Face grid dimensions (H, W) for rectangular faces
FACE_DIMENSIONS = {
    "inner": (93, 44),
    "us": (24, 6),
    "ds": (24, 6),
    "outer": (9, 24),  # coarse grid
}

# Default normalization parameters (must match training config)
DEFAULT_NPHO_SCALE = 1000.0
DEFAULT_NPHO_SCALE2 = 4.08
DEFAULT_TIME_SCALE = 1.14e-7
DEFAULT_TIME_SHIFT = -0.46

# Invalid value marker (used for dead channels in real data validation)
INVALID_VALUE = -999.0


def load_predictions(root_file: str) -> pd.DataFrame:
    """
    Load inpainter predictions from ROOT file into pandas DataFrame.

    Args:
        root_file: Path to the prediction ROOT file

    Returns:
        DataFrame with prediction data
    """
    with uproot.open(root_file) as f:
        tree = f["predictions"]
        df = tree.arrays(library="pd")

    # Add face name column
    df["face_name"] = df["face"].map(FACE_ID_TO_NAME)

    # Check for mask_type column (real data validation)
    has_mask_type = "mask_type" in df.columns

    print(f"[INFO] Loaded {len(df):,} predictions from {root_file}")
    print(f"[INFO] Events: {df['event_idx'].nunique():,} unique")
    print(f"[INFO] Faces: {df['face_name'].value_counts().to_dict()}")

    if has_mask_type:
        n_artificial = (df['mask_type'] == 0).sum()
        n_dead = (df['mask_type'] == 1).sum()
        print(f"[INFO] Real data mode detected:")
        print(f"       - Artificial masks (mask_type=0): {n_artificial:,}")
        print(f"       - Dead channels (mask_type=1): {n_dead:,}")

    return df


def filter_valid_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter predictions to only those with valid ground truth.

    For real data validation, this filters to mask_type=0 (artificial).
    Also filters out rows with invalid error values (-999).

    Args:
        df: Full prediction DataFrame

    Returns:
        Filtered DataFrame with only valid predictions
    """
    if "mask_type" in df.columns:
        # Real data validation: only use artificial masks
        valid_df = df[df["mask_type"] == 0].copy()
    else:
        # Training predictions: use all
        valid_df = df.copy()

    # Filter out invalid error values
    valid_df = valid_df[
        (valid_df["error_npho"] > INVALID_VALUE + 1) &
        (valid_df["error_time"] > INVALID_VALUE + 1)
    ]

    return valid_df


def compute_global_metrics(df: pd.DataFrame) -> dict:
    """
    Compute global metrics for npho and time predictions.

    Returns:
        Dictionary with MAE, RMSE, bias, and 68th percentile for npho and time
    """
    # Filter to valid predictions only
    valid_df = filter_valid_predictions(df)

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions for metrics computation")
        return {f"{metric}_{var}": np.nan
                for metric in ["mae", "rmse", "bias", "res68", "res95", "std", "count"]
                for var in ["npho", "time"]}

    metrics = {}

    for var in ["npho", "time"]:
        error = valid_df[f"error_{var}"].values
        abs_error = np.abs(error)

        metrics[f"mae_{var}"] = np.mean(abs_error)
        metrics[f"rmse_{var}"] = np.sqrt(np.mean(error ** 2))
        metrics[f"bias_{var}"] = np.mean(error)
        metrics[f"res68_{var}"] = np.percentile(abs_error, 68)
        metrics[f"res95_{var}"] = np.percentile(abs_error, 95)
        metrics[f"std_{var}"] = np.std(error)
        metrics[f"count_{var}"] = len(error)

    return metrics


def compute_per_face_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics grouped by face.

    Returns:
        DataFrame with metrics per face
    """
    # Filter to valid predictions only
    valid_df = filter_valid_predictions(df)

    results = []

    for face_id, face_name in FACE_ID_TO_NAME.items():
        face_df = valid_df[valid_df["face"] == face_id]
        if len(face_df) == 0:
            continue

        row = {"face": face_name, "count": len(face_df)}

        for var in ["npho", "time"]:
            error = face_df[f"error_{var}"].values
            abs_error = np.abs(error)

            row[f"mae_{var}"] = np.mean(abs_error)
            row[f"rmse_{var}"] = np.sqrt(np.mean(error ** 2))
            row[f"bias_{var}"] = np.mean(error)
            row[f"res68_{var}"] = np.percentile(abs_error, 68)

        results.append(row)

    return pd.DataFrame(results)


def compute_dead_channel_stats(df: pd.DataFrame) -> dict:
    """
    Compute statistics for dead channel predictions (plausibility checks).

    These predictions have no ground truth, so we check for:
    - Physical plausibility (npho >= 0, reasonable time range)
    - Distribution statistics

    Args:
        df: Full prediction DataFrame

    Returns:
        Dictionary with dead channel statistics
    """
    if "mask_type" not in df.columns:
        return {}

    dead_df = df[df["mask_type"] == 1]
    if len(dead_df) == 0:
        return {}

    stats = {
        "n_dead": len(dead_df),
        "n_events": dead_df["event_idx"].nunique(),
    }

    # Npho statistics
    pred_npho = dead_df["pred_npho"].values
    stats["npho_mean"] = np.mean(pred_npho)
    stats["npho_std"] = np.std(pred_npho)
    stats["npho_min"] = np.min(pred_npho)
    stats["npho_max"] = np.max(pred_npho)
    stats["npho_negative_frac"] = (pred_npho < 0).sum() / len(pred_npho)

    # Time statistics
    pred_time = dead_df["pred_time"].values
    stats["time_mean"] = np.mean(pred_time)
    stats["time_std"] = np.std(pred_time)
    stats["time_min"] = np.min(pred_time)
    stats["time_max"] = np.max(pred_time)

    # Per-face breakdown
    stats["by_face"] = {}
    for face_id, face_name in FACE_ID_TO_NAME.items():
        face_dead = dead_df[dead_df["face"] == face_id]
        if len(face_dead) > 0:
            stats["by_face"][face_name] = {
                "count": len(face_dead),
                "npho_mean": face_dead["pred_npho"].mean(),
                "time_mean": face_dead["pred_time"].mean(),
            }

    return stats


def plot_residual_distributions(df: pd.DataFrame, save_dir: str, suffix: str = ""):
    """
    Plot residual (error) distributions for npho and time.

    Creates 5 plots in a 3x2 grid:
    - Row 1: All data (npho, time)
    - Row 2: truth_npho > 100 filter (npho, time)
    - Row 3: Normalized error (pred-truth)/truth for npho only (left), empty (right)
    """
    # Filter to valid predictions
    valid_df = filter_valid_predictions(df)

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions for residual plot")
        return

    # Create 3x2 figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # Fixed ranges
    npho_range = (-1000, 2000)
    time_range = (-1e-7, 0.5e-7)
    nbins = 100

    # Row 1: All data
    for col, var in enumerate(["npho", "time"]):
        ax = axes[0, col]
        error = valid_df[f"error_{var}"].values

        mean = np.mean(error)
        std = np.std(error)

        if var == "npho":
            bins = np.linspace(npho_range[0], npho_range[1], nbins)
        else:
            bins = np.linspace(time_range[0], time_range[1], nbins)

        ax.hist(error, bins=bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)

        ax.set_xlabel(f"error_{var} (pred - truth)")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual: {var} (all data, n={len(error):,})\nμ={mean:.4g}, σ={std:.4g}")
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(bins[0], bins[-1])

    # Row 2: truth_npho > 100
    high_npho_df = valid_df[valid_df["truth_npho"] > 100]
    for col, var in enumerate(["npho", "time"]):
        ax = axes[1, col]
        error = high_npho_df[f"error_{var}"].values

        if len(error) == 0:
            ax.text(0.5, 0.5, "No data with truth_npho > 100", ha='center', va='center', transform=ax.transAxes)
            continue

        mean = np.mean(error)
        std = np.std(error)

        if var == "npho":
            bins = np.linspace(npho_range[0], npho_range[1], nbins)
        else:
            bins = np.linspace(time_range[0], time_range[1], nbins)

        ax.hist(error, bins=bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)

        ax.set_xlabel(f"error_{var} (pred - truth)")
        ax.set_ylabel("Density")
        ax.set_title(f"Residual: {var} (truth_npho>100, n={len(error):,})\nμ={mean:.4g}, σ={std:.4g}")
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlim(bins[0], bins[-1])

    # Row 3: Normalized error for npho only (left), hide right panel
    # Normalized npho error
    ax = axes[2, 0]
    mask = high_npho_df["truth_npho"] > 100
    truth = high_npho_df.loc[mask, "truth_npho"].values
    pred = high_npho_df.loc[mask, "pred_npho"].values

    if len(truth) == 0:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center', transform=ax.transAxes)
    else:
        norm_error = (pred - truth) / truth

        # Remove extreme outliers for plotting
        pct_low, pct_high = np.percentile(norm_error, [1, 99])
        plot_mask = (norm_error >= pct_low) & (norm_error <= pct_high)
        norm_error_plot = norm_error[plot_mask]

        mean = np.mean(norm_error_plot)
        std = np.std(norm_error_plot)

        bins = np.linspace(pct_low, pct_high, nbins)
        ax.hist(norm_error_plot, bins=bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.3)

        ax.set_xlabel("(pred - truth) / truth [npho]")
        ax.set_ylabel("Density")
        ax.set_title(f"Normalized Residual: npho (truth_npho>100, n={len(norm_error_plot):,})\nμ={mean:.4g}, σ={std:.4g}")
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    # Hide right panel in row 3
    axes[2, 1].set_visible(False)

    # Add count info
    n_total = len(df)
    n_valid = len(valid_df)
    if "mask_type" in df.columns:
        fig.suptitle(f"Residual Distributions - Artificial Masks Only (n={n_valid:,} / {n_total:,} total)", fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"residual_distributions{suffix}.pdf"), dpi=150)
    plt.close()
    print(f"[INFO] Saved residual_distributions{suffix}.pdf")


def plot_residual_per_face(df: pd.DataFrame, save_dir: str, suffix: str = ""):
    """
    Plot residual distributions separated by face.
    """
    # Filter to valid predictions
    valid_df = filter_valid_predictions(df)

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions for per-face residual plot")
        return

    faces = sorted(valid_df["face"].unique())

    for var in ["npho", "time"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for i, face_id in enumerate(faces):
            if i >= 6:
                break
            ax = axes[i]
            face_name = FACE_ID_TO_NAME.get(face_id, f"face_{face_id}")
            face_df = valid_df[valid_df["face"] == face_id]

            if len(face_df) == 0:
                ax.set_visible(False)
                continue

            error = face_df[f"error_{var}"].values
            mean = np.mean(error)
            std = np.std(error)

            bins = np.linspace(mean - 5*std, mean + 5*std, 50)
            ax.hist(error, bins=bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)

            ax.set_xlabel(f"error_{var}")
            ax.set_title(f"{face_name} (n={len(face_df):,})\nμ={mean:.4f}, σ={std:.4f}")
            ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

        # Hide unused axes
        for i in range(len(faces), 6):
            axes[i].set_visible(False)

        plt.suptitle(f"Residual Distribution by Face: {var}", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"residual_per_face_{var}{suffix}.pdf"), dpi=150)
        plt.close()

    print(f"[INFO] Saved residual_per_face_npho/time{suffix}.pdf")


def plot_scatter_truth_vs_pred(df: pd.DataFrame, save_dir: str, max_points: int = 50000, suffix: str = ""):
    """
    Plot 2D histogram of truth vs prediction with truth_npho > 100 cut.
    """
    # Filter to valid predictions
    valid_df = filter_valid_predictions(df)

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions for scatter plot")
        return

    # Apply truth_npho > 100 cut
    valid_df = valid_df[valid_df["truth_npho"] > 100]

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions with truth_npho > 100 for scatter plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fixed ranges
    npho_range = (-500, 4000)
    time_range = (-0.2e-7, 1e-7)

    for ax, var in zip(axes, ["npho", "time"]):
        truth = valid_df[f"truth_{var}"].values
        pred = valid_df[f"pred_{var}"].values

        # Set range based on variable
        if var == "npho":
            plot_range = npho_range
        else:
            plot_range = time_range

        # 2D histogram with square bins
        h = ax.hist2d(truth, pred, bins=50, cmap='viridis', norm=LogNorm(),
                      range=[plot_range, plot_range])
        plt.colorbar(h[3], ax=ax, label='Count')

        # Add diagonal line
        ax.plot(plot_range, plot_range, 'r--', lw=2, alpha=0.7, label='y=x')

        ax.set_xlabel(f"truth_{var}")
        ax.set_ylabel(f"pred_{var}")
        ax.set_title(f"Prediction vs Truth: {var} (truth_npho>100, n={len(truth):,})")
        ax.legend()
        ax.set_xlim(plot_range)
        ax.set_ylim(plot_range)
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"scatter_truth_vs_pred{suffix}.pdf"), dpi=150)
    plt.close()
    print(f"[INFO] Saved scatter_truth_vs_pred{suffix}.pdf")


def plot_resolution_vs_signal(df: pd.DataFrame, save_dir: str, n_bins: int = 20, suffix: str = ""):
    """
    Plot resolution and bias as function of truth_npho.

    Creates two files:
    1. resolution_npho_vs_truthnpho.pdf: npho resolution/bias vs truth_npho, log(truth_npho), sqrt(truth_npho)
    2. resolution_time_vs_truthnpho.pdf: time resolution/bias vs truth_npho, log(truth_npho), sqrt(truth_npho)

    Note: Gray bars in bias plots show the count (number of samples) per bin.
    """
    # Filter to valid predictions
    valid_df = filter_valid_predictions(df)

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions for resolution plot")
        return

    # Filter to positive npho for log scale
    valid_df = valid_df[valid_df["truth_npho"] > 0].copy()

    if len(valid_df) == 0:
        print("[WARNING] No valid predictions with truth_npho > 0")
        return

    truth_npho = valid_df["truth_npho"].values
    log_truth_npho = np.log10(truth_npho)
    sqrt_truth_npho = np.sqrt(truth_npho)

    # ============================================================
    # File 1: Npho resolution/bias profiled by truth_npho transforms
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    error_npho = valid_df["error_npho"].values
    abs_error_npho = np.abs(error_npho)

    x_transforms = [
        (truth_npho, "truth_npho"),
        (log_truth_npho, "log10(truth_npho)"),
        (sqrt_truth_npho, "sqrt(truth_npho)"),
    ]

    for col, (x_vals, x_label) in enumerate(x_transforms):
        bins = np.percentile(x_vals, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_indices = np.digitize(x_vals, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        mae_per_bin, bias_per_bin, res68_per_bin, count_per_bin = [], [], [], []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                mae_per_bin.append(np.mean(abs_error_npho[mask]))
                bias_per_bin.append(np.mean(error_npho[mask]))
                res68_per_bin.append(np.percentile(abs_error_npho[mask], 68))
                count_per_bin.append(mask.sum())
            else:
                mae_per_bin.append(np.nan)
                bias_per_bin.append(np.nan)
                res68_per_bin.append(np.nan)
                count_per_bin.append(0)

        ax = axes[0, col]
        ax.plot(bin_centers, mae_per_bin, 'o-', color='blue', label='MAE')
        ax.plot(bin_centers, res68_per_bin, 's-', color='green', label='68th pct')
        ax.set_xlabel(x_label)
        ax.set_ylabel("Resolution (|error_npho|)")
        ax.set_title(f"Npho Resolution vs {x_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(bin_centers, bias_per_bin, 'o-', color='red')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Bias (mean error_npho)")
        ax.set_title(f"Npho Bias vs {x_label}")
        ax.grid(True, alpha=0.3)
        ax_twin = ax.twinx()
        ax_twin.bar(bin_centers, count_per_bin, width=np.diff(bins).mean() * 0.8, alpha=0.2, color='gray')
        ax_twin.set_ylabel("Count per bin (gray bars)", color='gray')

    plt.suptitle("Npho Resolution and Bias profiled by truth_npho", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"resolution_npho_vs_truthnpho{suffix}.pdf"), dpi=150)
    plt.close()
    print(f"[INFO] Saved resolution_npho_vs_truthnpho{suffix}.pdf")

    # ============================================================
    # File 2: Time resolution/bias profiled by truth_npho transforms
    # ============================================================
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    error_time = valid_df["error_time"].values
    abs_error_time = np.abs(error_time)

    x_transforms = [
        (truth_npho, "truth_npho"),
        (log_truth_npho, "log10(truth_npho)"),
        (sqrt_truth_npho, "sqrt(truth_npho)"),
    ]

    for col, (x_vals, x_label) in enumerate(x_transforms):
        bins = np.percentile(x_vals, np.linspace(0, 100, n_bins + 1))
        bins = np.unique(bins)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_indices = np.digitize(x_vals, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(bins) - 2)

        mae_per_bin, bias_per_bin, res68_per_bin, count_per_bin = [], [], [], []
        for i in range(len(bins) - 1):
            mask = bin_indices == i
            if mask.sum() > 0:
                mae_per_bin.append(np.mean(abs_error_time[mask]))
                bias_per_bin.append(np.mean(error_time[mask]))
                res68_per_bin.append(np.percentile(abs_error_time[mask], 68))
                count_per_bin.append(mask.sum())
            else:
                mae_per_bin.append(np.nan)
                bias_per_bin.append(np.nan)
                res68_per_bin.append(np.nan)
                count_per_bin.append(0)

        ax = axes[0, col]
        ax.plot(bin_centers, mae_per_bin, 'o-', color='blue', label='MAE')
        ax.plot(bin_centers, res68_per_bin, 's-', color='green', label='68th pct')
        ax.set_xlabel(x_label)
        ax.set_ylabel("Resolution (|error_time|)")
        ax.set_title(f"Time Resolution vs {x_label}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        ax.plot(bin_centers, bias_per_bin, 'o-', color='red')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Bias (mean error_time)")
        ax.set_title(f"Time Bias vs {x_label}")
        ax.grid(True, alpha=0.3)
        ax_twin = ax.twinx()
        ax_twin.bar(bin_centers, count_per_bin, width=np.diff(bins).mean() * 0.8, alpha=0.2, color='gray')
        ax_twin.set_ylabel("Count per bin (gray bars)", color='gray')

    plt.suptitle("Time Resolution and Bias profiled by truth_npho", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"resolution_time_vs_truthnpho{suffix}.pdf"), dpi=150)
    plt.close()
    print(f"[INFO] Saved resolution_time_vs_truthnpho{suffix}.pdf")


def plot_dead_channel_distributions(df: pd.DataFrame, save_dir: str):
    """
    Plot prediction distributions for dead channels (no ground truth).
    """
    if "mask_type" not in df.columns:
        return

    dead_df = df[df["mask_type"] == 1]
    if len(dead_df) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Npho distribution - fixed range [-500, 3000]
    ax = axes[0, 0]
    pred_npho = dead_df["pred_npho"].values
    npho_bins = np.linspace(-500, 3000, 100)
    ax.hist(pred_npho, bins=npho_bins, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7, label='Zero')
    ax.set_xlabel("pred_npho")
    ax.set_ylabel("Density")
    ax.set_title(f"Dead Channel Npho Predictions (n={len(dead_df):,})")
    ax.set_xlim(-500, 3000)
    ax.legend()

    # Time distribution - auto range
    ax = axes[0, 1]
    pred_time = dead_df["pred_time"].values
    ax.hist(pred_time, bins=100, density=True, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel("pred_time")
    ax.set_ylabel("Density")
    ax.set_title("Dead Channel Time Predictions")

    # Npho by face
    ax = axes[1, 0]
    face_data = []
    face_labels = []
    for face_id, face_name in FACE_ID_TO_NAME.items():
        face_dead = dead_df[dead_df["face"] == face_id]
        if len(face_dead) > 0:
            face_data.append(face_dead["pred_npho"].values)
            face_labels.append(f"{face_name}\n(n={len(face_dead)})")

    if face_data:
        ax.boxplot(face_data, tick_labels=face_labels)
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.set_ylabel("pred_npho")
        ax.set_title("Dead Channel Npho by Face")

    # Time by face
    ax = axes[1, 1]
    face_data = []
    face_labels = []
    for face_id, face_name in FACE_ID_TO_NAME.items():
        face_dead = dead_df[dead_df["face"] == face_id]
        if len(face_dead) > 0:
            face_data.append(face_dead["pred_time"].values)
            face_labels.append(f"{face_name}\n(n={len(face_dead)})")

    if face_data:
        ax.boxplot(face_data, tick_labels=face_labels)
        ax.set_ylabel("pred_time")
        ax.set_title("Dead Channel Time by Face")

    plt.suptitle("Dead Channel Predictions (No Ground Truth)", fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_dir, "dead_channel_distributions.pdf"), dpi=150)
    plt.close()
    print(f"[INFO] Saved dead_channel_distributions.pdf")


def plot_metrics_summary(global_metrics: dict, face_metrics: pd.DataFrame, save_dir: str,
                         dead_stats: dict = None, suffix: str = ""):
    """
    Create a summary bar chart of metrics per face.

    Uses separate y-axes for npho (left) and time (right) since they have
    very different scales (10 orders of magnitude difference).
    """
    if len(face_metrics) == 0:
        print("[WARNING] No face metrics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    faces = face_metrics["face"].values
    x = np.arange(len(faces))
    width = 0.35

    # MAE comparison - dual y-axis
    ax = axes[0, 0]
    ax.bar(x - width/2, face_metrics["mae_npho"], width, label='npho', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.set_ylabel("MAE (npho)", color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_title("Mean Absolute Error by Face")
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = ax.twinx()
    ax2.bar(x + width/2, face_metrics["mae_time"], width, label='time', color='coral')
    ax2.set_ylabel("MAE (time)", color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')

    # Combined legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='steelblue', label='npho'),
                       Patch(facecolor='coral', label='time')]
    ax.legend(handles=legend_elements, loc='upper right')

    # RMSE comparison - dual y-axis
    ax = axes[0, 1]
    ax.bar(x - width/2, face_metrics["rmse_npho"], width, label='npho', color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.set_ylabel("RMSE (npho)", color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_title("Root Mean Square Error by Face")
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = ax.twinx()
    ax2.bar(x + width/2, face_metrics["rmse_time"], width, label='time', color='coral')
    ax2.set_ylabel("RMSE (time)", color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.legend(handles=legend_elements, loc='upper right')

    # Bias comparison - dual y-axis
    ax = axes[1, 0]
    ax.bar(x - width/2, face_metrics["bias_npho"], width, label='npho', color='steelblue')
    ax.axhline(0, color='steelblue', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.set_ylabel("Bias (npho)", color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax.set_title("Bias by Face")
    ax.grid(True, alpha=0.3, axis='y')

    ax2 = ax.twinx()
    ax2.bar(x + width/2, face_metrics["bias_time"], width, label='time', color='coral')
    ax2.axhline(0, color='coral', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_ylabel("Bias (time)", color='coral')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax.legend(handles=legend_elements, loc='upper right')

    # Count per face
    ax = axes[1, 1]
    ax.bar(x, face_metrics["count"], color='gray', edgecolor='black')
    ax.set_xticks(x)
    ax.set_xticklabels(faces)
    ax.set_ylabel("Count")
    ax.set_title("Number of Predictions by Face (Artificial Masks)")
    ax.grid(True, alpha=0.3, axis='y')

    # Add global metrics as text
    text = f"Global Metrics (Artificial Masks Only):\n"
    text += f"  npho: MAE={global_metrics['mae_npho']:.4g}, RMSE={global_metrics['rmse_npho']:.4g}, bias={global_metrics['bias_npho']:.4g}\n"
    text += f"  time: MAE={global_metrics['mae_time']:.4g}, RMSE={global_metrics['rmse_time']:.4g}, bias={global_metrics['bias_time']:.4g}"

    if dead_stats and dead_stats.get("n_dead", 0) > 0:
        text += f"\n\nDead Channels: {dead_stats['n_dead']:,} predictions"

    fig.text(0.5, 0.02, text, ha='center', fontsize=10, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.1, 1, 1])
    plt.savefig(os.path.join(save_dir, f"metrics_summary{suffix}.pdf"), dpi=150)
    plt.close()
    print(f"[INFO] Saved metrics_summary{suffix}.pdf")


def identify_outliers(df: pd.DataFrame, sigma_threshold: float = 5.0) -> pd.DataFrame:
    """
    Identify predictions with large errors (outliers).

    Args:
        df: DataFrame with predictions
        sigma_threshold: Number of standard deviations to consider as outlier

    Returns:
        DataFrame with outlier predictions
    """
    # Filter to valid predictions
    valid_df = filter_valid_predictions(df)

    if len(valid_df) == 0:
        return pd.DataFrame()

    outliers = []

    for var in ["npho", "time"]:
        error = valid_df[f"error_{var}"].values
        std = np.std(error)
        mean = np.mean(error)

        if std == 0:
            continue

        mask = np.abs(error - mean) > sigma_threshold * std
        outlier_df = valid_df[mask].copy()
        outlier_df["outlier_var"] = var
        outlier_df["outlier_zscore"] = (error[mask] - mean) / std
        outliers.append(outlier_df)

    if outliers:
        result = pd.concat(outliers, ignore_index=True)
        result = result.sort_values("outlier_zscore", key=abs, ascending=False)
        return result
    return pd.DataFrame()


def print_summary(global_metrics: dict, face_metrics: pd.DataFrame, outliers: pd.DataFrame,
                  dead_stats: dict = None, has_mask_type: bool = False):
    """
    Print a text summary of the analysis.
    """
    print("\n" + "=" * 70)
    print("INPAINTER EVALUATION SUMMARY")
    if has_mask_type:
        print("(Real Data Validation Mode)")
    print("=" * 70)

    print("\n--- Global Metrics (Artificial Masks Only) ---")
    print(f"{'Metric':<15} {'npho':>12} {'time':>12}")
    print("-" * 40)
    print(f"{'MAE':<15} {global_metrics['mae_npho']:>12.6f} {global_metrics['mae_time']:>12.6f}")
    print(f"{'RMSE':<15} {global_metrics['rmse_npho']:>12.6f} {global_metrics['rmse_time']:>12.6f}")
    print(f"{'Bias':<15} {global_metrics['bias_npho']:>12.6f} {global_metrics['bias_time']:>12.6f}")
    print(f"{'Res (68%)':<15} {global_metrics['res68_npho']:>12.6f} {global_metrics['res68_time']:>12.6f}")
    print(f"{'Res (95%)':<15} {global_metrics['res95_npho']:>12.6f} {global_metrics['res95_time']:>12.6f}")
    print(f"{'Std':<15} {global_metrics['std_npho']:>12.6f} {global_metrics['std_time']:>12.6f}")
    print(f"{'Count':<15} {global_metrics['count_npho']:>12,}")

    if len(face_metrics) > 0:
        print("\n--- Per-Face Metrics ---")
        print(face_metrics.to_string(index=False, float_format=lambda x: f"{x:.6f}" if isinstance(x, float) else str(x)))

    # Dead channel statistics (if available)
    if dead_stats and dead_stats.get("n_dead", 0) > 0:
        print("\n--- Dead Channel Predictions (No Ground Truth) ---")
        print(f"Total dead channel predictions: {dead_stats['n_dead']:,}")
        print(f"Events with dead channels: {dead_stats['n_events']:,}")
        print(f"\nNpho predictions:")
        print(f"  Mean: {dead_stats['npho_mean']:.4f}, Std: {dead_stats['npho_std']:.4f}")
        print(f"  Range: [{dead_stats['npho_min']:.4f}, {dead_stats['npho_max']:.4f}]")
        print(f"  Negative fraction: {dead_stats['npho_negative_frac']*100:.2f}%")
        print(f"\nTime predictions:")
        print(f"  Mean: {dead_stats['time_mean']:.4f}, Std: {dead_stats['time_std']:.4f}")
        print(f"  Range: [{dead_stats['time_min']:.4f}, {dead_stats['time_max']:.4f}]")

        if dead_stats.get("by_face"):
            print(f"\nDead by face:")
            for face, stats in dead_stats["by_face"].items():
                print(f"  {face}: {stats['count']:,} (npho_mean={stats['npho_mean']:.4f})")

    print(f"\n--- Outliers (>{5}σ) ---")
    if len(outliers) > 0:
        print(f"Total outliers: {len(outliers)}")
        print("Top 10 outliers:")
        cols = ["event_idx", "sensor_id", "face_name", "outlier_var", "outlier_zscore"]
        if "run" in outliers.columns:
            cols = ["run", "event"] + cols
        available_cols = [c for c in cols if c in outliers.columns]
        print(outliers[available_cols].head(10).to_string(index=False))
    else:
        print("No outliers found.")

    print("\n" + "=" * 70)


def save_metrics_csv(global_metrics: dict, face_metrics: pd.DataFrame, save_dir: str,
                     dead_stats: dict = None):
    """
    Save metrics to CSV files.
    """
    # Global metrics
    global_df = pd.DataFrame([global_metrics])
    global_df.to_csv(os.path.join(save_dir, "global_metrics.csv"), index=False)

    # Face metrics
    if len(face_metrics) > 0:
        face_metrics.to_csv(os.path.join(save_dir, "face_metrics.csv"), index=False)

    # Dead channel stats
    if dead_stats and dead_stats.get("n_dead", 0) > 0:
        dead_df = pd.DataFrame([{
            k: v for k, v in dead_stats.items() if k != "by_face"
        }])
        dead_df.to_csv(os.path.join(save_dir, "dead_channel_stats.csv"), index=False)

    print(f"[INFO] Saved global_metrics.csv and face_metrics.csv")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze inpainter predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("input", type=str, help="Input ROOT file with predictions")
    parser.add_argument("--output", "-o", type=str, default="inpainter_analysis",
                        help="Output directory for plots and metrics")
    parser.add_argument("--denorm", action="store_true",
                        help="Denormalize values to physical units (not implemented yet)")
    parser.add_argument("--max-points", type=int, default=50000,
                        help="Maximum points for scatter plots")
    parser.add_argument("--outlier-sigma", type=float, default=5.0,
                        help="Sigma threshold for outlier detection")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    print(f"[INFO] Output directory: {args.output}")

    # Load data
    df = load_predictions(args.input)

    # Check for real data validation mode
    has_mask_type = "mask_type" in df.columns

    # Compute metrics (uses only valid/artificial predictions)
    print("\n[INFO] Computing metrics...")
    global_metrics = compute_global_metrics(df)
    face_metrics = compute_per_face_metrics(df)
    outliers = identify_outliers(df, sigma_threshold=args.outlier_sigma)

    # Compute dead channel statistics (if applicable)
    dead_stats = compute_dead_channel_stats(df) if has_mask_type else None

    # Print summary
    print_summary(global_metrics, face_metrics, outliers, dead_stats, has_mask_type)

    # Save metrics
    save_metrics_csv(global_metrics, face_metrics, args.output, dead_stats)

    # Save outliers
    if len(outliers) > 0:
        outliers.to_csv(os.path.join(args.output, "outliers.csv"), index=False)
        print(f"[INFO] Saved outliers.csv ({len(outliers)} rows)")

    # Generate plots
    print("\n[INFO] Generating plots...")
    plot_residual_distributions(df, args.output)
    plot_residual_per_face(df, args.output)
    plot_scatter_truth_vs_pred(df, args.output, max_points=args.max_points)
    plot_resolution_vs_signal(df, args.output)
    plot_metrics_summary(global_metrics, face_metrics, args.output, dead_stats)

    # Dead channel plots (if applicable)
    if has_mask_type:
        plot_dead_channel_distributions(df, args.output)

    print(f"\n[INFO] Analysis complete. Results saved to {args.output}/")


if __name__ == "__main__":
    main()
