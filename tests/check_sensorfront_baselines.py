#!/usr/bin/env python3
"""
Diagnostic script to check sensorfront baseline predictions.

Shows per-event truth, ML prediction, neighbor avg, and solid-angle
weighted predictions to verify whether large SA errors are from the
omega_m/sum(omega) amplification or a bug.

Usage:
    python tests/check_sensorfront_baselines.py
    python tests/check_sensorfront_baselines.py --step 3 --n 50
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(
        description="Diagnose sensorfront baseline predictions")
    parser.add_argument("--step", type=int, default=1,
                        help="Scan step number (default: 1)")
    parser.add_argument("--n", type=int, default=20,
                        help="Number of events to display (default: 20)")
    parser.add_argument("--shared-dir", type=str,
                        default="artifacts/sensorfront_shared",
                        help="Shared sensorfront data directory")
    args = parser.parse_args()

    labels = {
        1: "s1_baseline", 2: "s2_flatmask", 3: "s3_nphowt",
        4: "s4_flat_nphowt", 5: "s5_sqrt", 6: "s6_mask015",
        7: "s7_sqrt_nphowt_mask015", 8: "s8_mask020",
    }
    label = labels.get(args.step)
    if label is None:
        print(f"Unknown step: {args.step}")
        sys.exit(1)

    # Load prediction file
    import uproot
    pred_path = f"artifacts/inp_scan_{label}/validation_sensorfront/predictions_sensorfront.root"
    if not os.path.exists(pred_path):
        print(f"Not found: {pred_path}")
        sys.exit(1)

    print(f"Loading: {pred_path}")
    with uproot.open(pred_path) as f:
        t = f['predictions']
        data = {k: t[k].array(library='np') for k in t.keys()}

    truth = data['truth_npho']
    ml_pred = data['pred_npho']
    sensor_id = data['sensor_id']
    has_avg = 'baseline_avg_npho' in data
    has_sa = 'baseline_sa_npho' in data
    has_lf = 'baseline_lf_npho' in data

    n = min(args.n, len(truth))

    # Header
    cols = f"{'idx':>4} {'sensor':>6} {'truth':>10} {'ML':>10}"
    if has_avg:
        cols += f" {'Avg':>10}"
    if has_sa:
        cols += f" {'SA':>12} {'omega_ratio':>11}"
    if has_lf:
        cols += f" {'LocalFit':>10}"
    print(cols)
    print("-" * len(cols))

    # Load raw baselines for omega inspection
    omega_ratio_vals = None
    if has_sa and os.path.exists(args.shared_dir):
        baselines_path = os.path.join(args.shared_dir, "_baselines_raw.npz")
        data_path = os.path.join(args.shared_dir, "_sensorfront_data.npz")
        if os.path.exists(baselines_path) and os.path.exists(data_path):
            bl_data = np.load(baselines_path)
            sf_data = np.load(data_path)
            if "solid_angles" in sf_data and "sa_pred_raw" in bl_data:
                solid_angles = sf_data["solid_angles"]
                matched_sid = sf_data["matched_sid"] if "matched_sid" in sf_data else None
                if matched_sid is not None:
                    omega_m = np.array([
                        solid_angles[i, int(matched_sid[i])]
                        for i in range(len(matched_sid))
                    ])
                    # Estimate omega_sum from neighbors
                    # We can't easily recompute here, so show omega_m
                    omega_ratio_vals = omega_m

    for i in range(n):
        row = f"{i:4d} {sensor_id[i]:6d} {truth[i]:10.1f} {ml_pred[i]:10.1f}"
        if has_avg:
            row += f" {data['baseline_avg_npho'][i]:10.1f}"
        if has_sa:
            sa_val = data['baseline_sa_npho'][i]
            if omega_ratio_vals is not None and i < len(omega_ratio_vals):
                row += f" {sa_val:12.1f} {omega_ratio_vals[i]:11.6f}"
            else:
                row += f" {sa_val:12.1f} {'---':>11}"
        if has_lf:
            row += f" {data['baseline_lf_npho'][i]:10.1f}"
        print(row)

    # Summary statistics
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    valid = np.isfinite(truth) & (truth > 100)
    print(f"Events with truth > 100: {valid.sum()}")

    def print_stats(name, pred):
        v = valid & np.isfinite(pred)
        if v.sum() == 0:
            print(f"  {name:20s}: no valid entries")
            return
        err = pred[v] - truth[v]
        rel_err = err / truth[v]
        print(f"  {name:20s}: MAE={np.mean(np.abs(err)):10.1f}  "
              f"Bias={np.mean(err):+10.1f}  "
              f"RelMAE={np.mean(np.abs(rel_err)):.3f}  "
              f"N={v.sum()}")

    print_stats("ML", ml_pred)
    if has_avg:
        print_stats("Neighbor Avg", data['baseline_avg_npho'])
    if has_sa:
        print_stats("Solid-Angle Wt", data['baseline_sa_npho'])
    if has_lf:
        print_stats("LocalFit", data['baseline_lf_npho'])

    # Show worst SA predictions
    if has_sa:
        sa_pred = data['baseline_sa_npho']
        sa_err = sa_pred - truth
        worst = np.argsort(np.abs(sa_err))[-10:][::-1]
        print()
        print("Top 10 worst SA predictions:")
        print(f"{'idx':>6} {'truth':>10} {'SA pred':>12} {'error':>12} {'ratio':>8}")
        for idx in worst:
            ratio = sa_pred[idx] / truth[idx] if truth[idx] > 0 else float('inf')
            print(f"{idx:6d} {truth[idx]:10.1f} {sa_pred[idx]:12.1f} "
                  f"{sa_err[idx]:+12.1f} {ratio:8.1f}x")


if __name__ == "__main__":
    main()
