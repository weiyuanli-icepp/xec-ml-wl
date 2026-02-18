#!/usr/bin/env python3
"""
Debug script to verify inpainter predictions data.
Run this to diagnose normalization and data issues.

Usage:
    python macro/debug_inpainter_predictions.py test_inpainter_predictions.root data/large_val.root
"""

import sys
import os
import argparse
import numpy as np
import uproot

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME
)


def normalize_input(raw_npho, raw_time,
                    npho_scale=DEFAULT_NPHO_SCALE,
                    npho_scale2=DEFAULT_NPHO_SCALE2,
                    time_scale=DEFAULT_TIME_SCALE,
                    time_shift=DEFAULT_TIME_SHIFT,
                    sentinel_time=DEFAULT_SENTINEL_TIME):
    """Apply normalization to raw input data."""
    mask_npho_bad = (raw_npho <= 0.0) | (raw_npho > 9e9) | np.isnan(raw_npho)
    mask_time_bad = mask_npho_bad | (np.abs(raw_time) > 9e9) | np.isnan(raw_time)

    raw_npho_safe = np.where(mask_npho_bad, 0.0, raw_npho)
    npho_norm = np.log1p(raw_npho_safe / npho_scale) / npho_scale2
    time_norm = (raw_time / time_scale) - time_shift

    npho_norm[mask_npho_bad] = 0.0
    time_norm[mask_time_bad] = sentinel_time

    return npho_norm, time_norm


def main():
    parser = argparse.ArgumentParser(description="Debug inpainter predictions data")
    parser.add_argument("predictions", help="Path to inpainter predictions ROOT file")
    parser.add_argument("original", help="Path to original validation ROOT file")
    parser.add_argument("--event", type=int, default=0, help="Event index to debug")
    parser.add_argument("--npho_branch", default="relative_npho")
    parser.add_argument("--time_branch", default="relative_time")
    args = parser.parse_args()

    print("="*60)
    print("INPAINTER PREDICTIONS DEBUG")
    print("="*60)

    # --- Check predictions file ---
    print(f"\n[1] Checking predictions file: {args.predictions}")
    with uproot.open(args.predictions) as f:
        print(f"   Keys: {list(f.keys())}")

        # Check metadata
        if "metadata" in f:
            meta = f["metadata"]
            print("\n   Metadata from predictions file:")
            metadata = {}
            for key in ["npho_scale", "npho_scale2", "time_scale", "time_shift", "sentinel_value"]:
                if key in meta:
                    val = meta[key].array(library="np")[0]
                    metadata[key] = float(val)
                    print(f"      {key}: {val}")
        else:
            print("   WARNING: No metadata tree found!")
            metadata = {}

        # Get normalization params
        npho_scale = metadata.get("npho_scale", DEFAULT_NPHO_SCALE)
        npho_scale2 = metadata.get("npho_scale2", DEFAULT_NPHO_SCALE2)
        time_scale = metadata.get("time_scale", DEFAULT_TIME_SCALE)
        time_shift = metadata.get("time_shift", DEFAULT_TIME_SHIFT)
        sentinel_time = metadata.get("sentinel_value", DEFAULT_SENTINEL_TIME)

        # Load predictions
        tree = f["tree"]
        print(f"\n   Predictions tree: {tree.num_entries} entries")

        arrays = tree.arrays(library="np")

        event_mask = arrays["event_idx"] == args.event
        n_masked = event_mask.sum()
        print(f"\n   Event {args.event}: {n_masked} masked sensors")

        if n_masked == 0:
            print("   ERROR: No predictions for this event!")
            return

        # Face distribution
        print("\n   Face distribution:")
        faces = arrays["face"][event_mask]
        for fid, fname in enumerate(["inner", "us", "ds", "outer", "top", "bot"]):
            count = (faces == fid).sum()
            print(f"      {fname} (face={fid}): {count} sensors")

        # Filter out outer face
        valid_mask = faces != 3
        n_valid = valid_mask.sum()
        print(f"\n   After excluding outer face: {n_valid} sensors")

        sensor_ids = arrays["sensor_id"][event_mask][valid_mask]
        pred_npho = arrays["pred_npho"][event_mask][valid_mask]
        pred_time = arrays["pred_time"][event_mask][valid_mask]
        truth_npho_file = arrays["truth_npho"][event_mask][valid_mask]
        truth_time_file = arrays["truth_time"][event_mask][valid_mask]

    # --- Check original file ---
    print(f"\n[2] Checking original file: {args.original}")
    with uproot.open(args.original) as f:
        tree = f["tree"]
        print(f"   Tree entries: {tree.num_entries}")

        # Load one event
        arrs = tree.arrays([args.npho_branch, args.time_branch], library="np",
                           entry_start=args.event, entry_stop=args.event+1)
        raw_npho = arrs[args.npho_branch][0].astype("float32")
        raw_time = arrs[args.time_branch][0].astype("float32")

    # Normalize
    npho_norm, time_norm = normalize_input(
        raw_npho, raw_time,
        npho_scale=npho_scale, npho_scale2=npho_scale2,
        time_scale=time_scale, time_shift=time_shift,
        sentinel_time=sentinel_time
    )

    # --- Compare truth values ---
    print(f"\n[3] Comparing truth values (predictions file vs normalized original)")
    truth_from_original = np.stack([npho_norm[sensor_ids], time_norm[sensor_ids]], axis=-1)

    npho_diff = np.abs(truth_from_original[:, 0] - truth_npho_file)
    time_diff = np.abs(truth_from_original[:, 1] - truth_time_file)

    # Compute percentage difference
    npho_pct_diff = 100 * npho_diff / (np.abs(truth_npho_file) + 1e-8)
    time_pct_diff = 100 * time_diff / (np.abs(truth_time_file) + 1e-8)

    print(f"   Npho diff: mean={npho_diff.mean():.6f}, max={npho_diff.max():.6f} ({npho_pct_diff.mean():.1f}%)")
    print(f"   Time diff: mean={time_diff.mean():.6f}, max={time_diff.max():.6f} ({time_pct_diff.mean():.1f}%)")

    if npho_diff.mean() > 0.01 or time_diff.mean() > 0.01:
        print("\n   *** NORMALIZATION MISMATCH DETECTED! ***")
        print("   The truth values in predictions file don't match normalized original data.")
        print("   This will cause incorrect visualization.")
    else:
        print("   OK: Truth values match (normalization is consistent)")

    # --- Check prediction quality ---
    print(f"\n[4] Checking prediction vs truth (from predictions file)")
    pred_error_npho = pred_npho - truth_npho_file
    pred_error_time = pred_time - truth_time_file

    # Compute percentage errors (avoid division by zero)
    pct_error_npho = 100 * pred_error_npho / (np.abs(truth_npho_file) + 1e-8)
    pct_error_time = 100 * pred_error_time / (np.abs(truth_time_file) + 1e-8)

    print(f"   Pred - Truth (npho): mean={pred_error_npho.mean():.6f}, std={pred_error_npho.std():.6f}")
    print(f"   Pred - Truth (time): mean={pred_error_time.mean():.6f}, std={pred_error_time.std():.6f}")
    print(f"   Percentage error (npho): mean={pct_error_npho.mean():.2f}%, std={pct_error_npho.std():.2f}%")
    print(f"   Percentage error (time): mean={pct_error_time.mean():.2f}%, std={pct_error_time.std():.2f}%")

    # Show sample sensors
    print(f"\n[5] Sample sensors (first 10):")
    print("   sensor_id | truth_npho | pred_npho |   diff   |  pct%  | face")
    print("   " + "-"*65)
    for i in range(min(10, len(sensor_ids))):
        sid = sensor_ids[i]
        t_n = truth_npho_file[i]
        p_n = pred_npho[i]
        diff = p_n - t_n
        pct = 100 * diff / (abs(t_n) + 1e-8)
        face = faces[valid_mask][i]
        fname = ["inner", "us", "ds", "outer", "top", "bot"][face]
        print(f"   {sid:8d} | {t_n:10.4f} | {p_n:9.4f} | {diff:+8.4f} | {pct:+6.1f}% | {fname}")

    # --- Verify assignment would work ---
    print(f"\n[6] Verifying data shapes for visualization:")
    print(f"   sensor_ids shape: {sensor_ids.shape}")
    print(f"   pred_npho shape: {pred_npho.shape}")
    print(f"   sensor_ids range: [{sensor_ids.min()}, {sensor_ids.max()}]")
    print(f"   Expected range: [0, 4759]")

    invalid_ids = (sensor_ids < 0) | (sensor_ids >= 4760)
    if invalid_ids.any():
        print(f"   WARNING: {invalid_ids.sum()} invalid sensor_ids!")
    else:
        print("   OK: All sensor_ids are valid")

    print("\n" + "="*60)
    print("DEBUG COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
