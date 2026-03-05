#!/usr/bin/env python3
"""
Combine per-patch CEX validation results into a single ROOT file.

Reads prediction CSVs from val_data/cex/patch{1..24}/ and writes a
combined ROOT file with all events and a patch ID column.

Usage:
    python macro/combine_cex_results.py
    python macro/combine_cex_results.py --input-base val_data/cex --output val_data/cex/CEX23_combined.root
"""

import argparse
import os
import sys
import glob

import numpy as np

# Optional: try uproot for ROOT output, fall back to CSV
try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


ALL_PATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


def find_csv(patch_dir):
    """Find the predictions CSV in a patch directory."""
    candidates = sorted(glob.glob(os.path.join(patch_dir, "predictions_energy_*.csv")))
    if candidates:
        return candidates[-1]  # latest
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-patch CEX validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-base", default="val_data/cex",
                        help="Base directory with patch*/ subdirectories")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <input-base>/CEX23_combined.root or .csv)")
    parser.add_argument("--patches", type=int, nargs="*", default=None,
                        help="Specific patches to combine (default: all 1-24)")
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("[ERROR] pandas is required. Install with: pip install pandas")
        sys.exit(1)

    patches = args.patches or ALL_PATCHES
    input_base = args.input_base

    print(f"Input base: {input_base}")
    print(f"Patches:    {patches}")
    print()

    all_dfs = []
    found = 0
    missing = []

    for patch in patches:
        patch_dir = os.path.join(input_base, f"patch{patch}")
        csv_path = find_csv(patch_dir)

        if csv_path is None:
            missing.append(patch)
            continue

        df = pd.read_csv(csv_path)
        df["patch"] = patch
        n = len(df)
        all_dfs.append(df)
        found += 1

        # Per-patch stats
        if "pred_energy" in df.columns and "true_energy" in df.columns:
            valid = df["true_energy"] < 1e9
            if valid.sum() > 0:
                err = (df.loc[valid, "pred_energy"] - df.loc[valid, "true_energy"]) * 1e3
                print(f"  Patch {patch:>2d}: {n:>6d} events | "
                      f"MAE={np.mean(np.abs(err)):.2f} MeV | "
                      f"bias={np.mean(err):+.2f} MeV | "
                      f"res68={np.percentile(np.abs(err), 68):.2f} MeV")
            else:
                pred_mev = df["pred_energy"] * 1e3
                print(f"  Patch {patch:>2d}: {n:>6d} events | "
                      f"pred mean={np.mean(pred_mev):.2f} MeV | "
                      f"pred std={np.std(pred_mev):.2f} MeV (no truth)")
        else:
            print(f"  Patch {patch:>2d}: {n:>6d} events")

    if not all_dfs:
        print("\n[ERROR] No patch results found. Check that validation jobs have completed.")
        sys.exit(1)

    if missing:
        print(f"\n[WARN] Missing patches: {missing}")

    combined = pd.concat(all_dfs, ignore_index=True)
    n_total = len(combined)
    n_patches = found

    # Overall stats
    print(f"\n{'='*60}")
    print(f"Combined: {n_total} events from {n_patches} patches")

    if "pred_energy" in combined.columns and "true_energy" in combined.columns:
        valid = combined["true_energy"] < 1e9
        if valid.sum() > 0:
            err = (combined.loc[valid, "pred_energy"] - combined.loc[valid, "true_energy"]) * 1e3
            print(f"  MAE:    {np.mean(np.abs(err)):.2f} MeV")
            print(f"  RMSE:   {np.sqrt(np.mean(err**2)):.2f} MeV")
            print(f"  Bias:   {np.mean(err):+.2f} MeV")
            print(f"  Res68:  {np.percentile(np.abs(err), 68):.2f} MeV")
        else:
            pred_mev = combined["pred_energy"] * 1e3
            print(f"  Pred mean: {np.mean(pred_mev):.2f} MeV")
            print(f"  Pred std:  {np.std(pred_mev):.2f} MeV")
            print(f"  (No truth available for real data)")

    # Determine output path and format
    if args.output:
        outpath = args.output
    elif HAS_UPROOT:
        outpath = os.path.join(input_base, "CEX23_combined.root")
    else:
        outpath = os.path.join(input_base, "CEX23_combined.csv")

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    if outpath.endswith(".root"):
        if not HAS_UPROOT:
            print("[WARN] uproot not available, falling back to CSV")
            outpath = outpath.replace(".root", ".csv")

    if outpath.endswith(".root"):
        # Write as ROOT file
        branches = {}
        for col in combined.columns:
            arr = combined[col].values
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            branches[col] = arr

        with uproot.recreate(outpath) as f:
            f["tree"] = branches

        print(f"\nOutput: {outpath} ({n_total} events)")
    else:
        # Write as CSV
        combined.to_csv(outpath, index=False)
        print(f"\nOutput: {outpath} ({n_total} events)")

    print("Done!")


if __name__ == "__main__":
    main()
