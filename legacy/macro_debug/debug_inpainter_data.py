#!/usr/bin/env python
"""Debug script to check inpainter data files."""
import sys
import argparse
import numpy as np
import uproot

def main():
    parser = argparse.ArgumentParser(description="Debug inpainter data")
    parser.add_argument("--predictions", type=str, help="Inpainter predictions file")
    parser.add_argument("--original", type=str, help="Original validation file")
    parser.add_argument("--event_idx", type=int, default=0, help="Event to inspect")
    args = parser.parse_args()

    if args.original:
        print(f"\n=== Original file: {args.original} ===")
        with uproot.open(args.original) as f:
            tree = f["tree"]
            print(f"Available branches: {list(tree.keys())}")

            # Check energy-related branches
            for branch in tree.keys():
                if "energy" in branch.lower() or "Energy" in branch:
                    arr = tree[branch].array(library="np", entry_stop=5)
                    print(f"\n{branch}: dtype={arr.dtype}, shape={arr.shape}")
                    print(f"  First 5 values: {arr[:5]}")

            # Load specific event
            if args.event_idx is not None:
                print(f"\n--- Event {args.event_idx} ---")
                branches_to_check = ["energyTruth", "xyzVTX", "emiAng"]
                for branch in branches_to_check:
                    if branch in tree.keys():
                        arr = tree[branch].array(library="np",
                                                  entry_start=args.event_idx,
                                                  entry_stop=args.event_idx+1)
                        print(f"{branch}: {arr[0]}")

    if args.predictions:
        print(f"\n=== Predictions file: {args.predictions} ===")
        with uproot.open(args.predictions) as f:
            tree = f["tree"]
            print(f"Available branches: {list(tree.keys())}")
            print(f"Total entries: {tree.num_entries}")

            # Load sample data
            arrays = tree.arrays(library="np", entry_stop=100)

            print(f"\nSample statistics (first 100 entries):")
            for key in arrays.keys():
                arr = arrays[key]
                if np.issubdtype(arr.dtype, np.floating):
                    print(f"  {key}: min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}")
                else:
                    print(f"  {key}: unique values = {np.unique(arr)[:10]}...")

            # Check specific event
            if args.event_idx is not None:
                event_mask = arrays["event_idx"] == args.event_idx
                n_masked = event_mask.sum()
                print(f"\n--- Event {args.event_idx}: {n_masked} masked sensors ---")
                if n_masked > 0:
                    print(f"  pred_npho: min={arrays['pred_npho'][event_mask].min():.4f}, max={arrays['pred_npho'][event_mask].max():.4f}")
                    print(f"  pred_time: min={arrays['pred_time'][event_mask].min():.4f}, max={arrays['pred_time'][event_mask].max():.4f}")
                    print(f"  truth_npho: min={arrays['truth_npho'][event_mask].min():.4f}, max={arrays['truth_npho'][event_mask].max():.4f}")
                    print(f"  truth_time: min={arrays['truth_time'][event_mask].min():.4f}, max={arrays['truth_time'][event_mask].max():.4f}")

if __name__ == "__main__":
    main()
