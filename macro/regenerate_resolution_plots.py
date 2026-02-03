#!/usr/bin/env python3
"""
Regenerate resolution plots from saved predictions CSV files.

This macro reads predictions_*.csv files from an artifact directory and
regenerates the resolution_*.pdf plots with the updated plotting functions.

Usage:
    python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/
    python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --output_dir plots/
    python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --tasks energy angle
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.plotting import (
    plot_resolution_profile,
    plot_energy_resolution_profile,
    plot_timing_resolution_profile,
    plot_position_resolution_profile,
)


def load_predictions(artifact_dir, task):
    """
    Load predictions CSV file for a given task.

    Args:
        artifact_dir: Path to artifact directory
        task: Task name ('angle', 'energy', 'timing', 'uvwFI')

    Returns:
        DataFrame with predictions, or None if not found
    """
    # Look for predictions_<task>_*.csv files (without epoch suffix first, then with)
    patterns = [
        f"predictions_{task}_*.csv",
    ]

    import glob
    for pattern in patterns:
        matches = glob.glob(os.path.join(artifact_dir, pattern))
        if matches:
            # Sort and get the one without epoch suffix if available
            matches.sort()
            # Prefer file without _ep suffix
            for f in matches:
                if "_ep" not in os.path.basename(f):
                    print(f"[INFO] Loading {f}")
                    return pd.read_csv(f)
            # Fall back to first match
            print(f"[INFO] Loading {matches[0]}")
            return pd.read_csv(matches[0])

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate resolution plots from predictions CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Regenerate all available plots
    python macro/regenerate_resolution_plots.py artifacts/my_run/

    # Regenerate only energy plots
    python macro/regenerate_resolution_plots.py artifacts/my_run/ --tasks energy

    # Save to a different directory
    python macro/regenerate_resolution_plots.py artifacts/my_run/ --output_dir plots/
"""
    )

    parser.add_argument("artifact_dir", type=str,
                        help="Path to artifact directory containing predictions_*.csv files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots (default: same as artifact_dir)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Tasks to regenerate plots for (default: all available)")
    parser.add_argument("--suffix", type=str, default="_regenerated",
                        help="Suffix to add to output filenames (default: _regenerated)")

    args = parser.parse_args()

    if not os.path.isdir(args.artifact_dir):
        print(f"[ERROR] Artifact directory not found: {args.artifact_dir}")
        sys.exit(1)

    output_dir = args.output_dir or args.artifact_dir
    os.makedirs(output_dir, exist_ok=True)

    tasks = args.tasks or ["angle", "energy", "timing", "uvwFI"]
    suffix = args.suffix

    # Build root_data dict for energy plots (needs uvwFI data)
    root_data = {}

    # Load uvwFI predictions first (needed for energy resolution vs position plots)
    uvwFI_df = load_predictions(args.artifact_dir, "uvwFI")
    if uvwFI_df is not None:
        root_data['true_u'] = uvwFI_df['true_u'].values
        root_data['true_v'] = uvwFI_df['true_v'].values
        root_data['true_w'] = uvwFI_df['true_w'].values
        root_data['pred_u'] = uvwFI_df['pred_u'].values
        root_data['pred_v'] = uvwFI_df['pred_v'].values
        root_data['pred_w'] = uvwFI_df['pred_w'].values

    # Process each task
    for task in tasks:
        df = load_predictions(args.artifact_dir, task)
        if df is None:
            print(f"[WARN] No predictions found for task: {task}")
            continue

        outfile = os.path.join(output_dir, f"resolution_{task}{suffix}.pdf")

        try:
            if task == "angle":
                pred = np.stack([df['pred_theta'].values, df['pred_phi'].values], axis=1)
                true = np.stack([df['true_theta'].values, df['true_phi'].values], axis=1)
                plot_resolution_profile(pred, true, outfile=outfile)
                print(f"[OK] Generated: {outfile}")

            elif task == "energy":
                pred = df['pred_energy'].values
                true = df['true_energy'].values
                # Build root_data for position-profiled plots
                # First try from uvwFI predictions, then from energy CSV itself
                energy_root_data = dict(root_data)  # Copy from uvwFI if available
                # Check if energy CSV has uvw columns (newer format)
                for key in ['true_u', 'true_v', 'true_w']:
                    if key in df.columns and key not in energy_root_data:
                        energy_root_data[key] = df[key].values
                plot_energy_resolution_profile(pred, true, root_data=energy_root_data, outfile=outfile)
                print(f"[OK] Generated: {outfile}")

            elif task == "timing":
                pred = df['pred_timing'].values
                true = df['true_timing'].values
                plot_timing_resolution_profile(pred, true, outfile=outfile)
                print(f"[OK] Generated: {outfile}")

            elif task == "uvwFI":
                pred_uvw = np.stack([df['pred_u'].values, df['pred_v'].values, df['pred_w'].values], axis=1)
                true_uvw = np.stack([df['true_u'].values, df['true_v'].values, df['true_w'].values], axis=1)
                plot_position_resolution_profile(pred_uvw, true_uvw, outfile=outfile)
                print(f"[OK] Generated: {outfile}")

        except Exception as e:
            print(f"[ERROR] Failed to generate {task} plot: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n[INFO] Done. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
