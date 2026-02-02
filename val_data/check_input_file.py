#!/usr/bin/env python3
import sys
import os
import glob
import numpy as np
import uproot
import argparse

# Usage:
# $ python check_input_files.py val_data/DataGammaAngle_430026-432559.root

def check_file(filepath, tree_name="tree"):
    print(f"[CHECK] Processing: {filepath}")
    try:
        with uproot.open(filepath) as f:
            if tree_name not in f:
                print(f"  [ERROR] Tree '{tree_name}' not found!")
                return False
            
            tree = f[tree_name]
            num_entries = tree.num_entries
            print(f"  [INFO] Entries: {num_entries}")
            
            if num_entries == 0:
                print("  [WARN] Tree is empty.")
                return True

            # --- Check Event IDs (First/Last) ---
            # Load just the run/event branches for the first and last entry
            ids = tree.arrays(["run", "event"], library="np", entry_start=0, entry_stop=num_entries)
            first_run = ids["run"][0]
            first_evt = ids["event"][0]
            last_run  = ids["run"][-1]
            last_evt  = ids["event"][-1]
            
            print(f"  [INFO] Range: Run {first_run} Evt {first_evt}  ->  Run {last_run} Evt {last_evt}")

            # --- Check Data Quality ---
            # Try 'npho' first (new format), fall back to 'relative_npho' (deprecated)
            available = set(tree.keys())
            if "npho" in available:
                npho_branch = "npho"
            elif "relative_npho" in available:
                npho_branch = "relative_npho"
                print("  [WARN] Using deprecated 'relative_npho' branch. Consider updating data format.")
            else:
                print("  [ERROR] Neither 'npho' nor 'relative_npho' branch found!")
                return False
            branches = [npho_branch, "relative_time"]
            chunk_idx = 0
            has_error = False
            
            # Track global min/max for sanity check
            global_min_npho, global_max_npho = float('inf'), float('-inf')
            global_min_time, global_max_time = float('inf'), float('-inf')

            for arrays in tree.iterate(branches, step_size=10000, library="np"):
                chunk_idx += 1
                npho = arrays[npho_branch]
                time = arrays["relative_time"]
                
                # Check NaNs
                if np.isnan(npho).any() or np.isnan(time).any():
                    print(f"  [FAIL] Chunk {chunk_idx}: NaNs found!")
                    has_error = True
                    break
                
                # Check Infs
                if np.isinf(npho).any() or np.isinf(time).any():
                    print(f"  [FAIL] Chunk {chunk_idx}: Infs found!")
                    has_error = True
                    break

                # Update stats (ignoring purely empty events if any)
                if npho.size > 0:
                    global_min_npho = min(global_min_npho, np.min(npho))
                    global_max_npho = max(global_max_npho, np.max(npho))
                    global_min_time = min(global_min_time, np.min(time))
                    global_max_time = max(global_max_time, np.max(time))

                # Check for "Garbage" (Uninitialized values like 1e10)
                # Typically valid relative times are small (< 1000 ns). 1e9 is definitely garbage.
                GARBAGE_THRESHOLD = 1e9 
                if (np.abs(time) > GARBAGE_THRESHOLD).any():
                    print(f"  [WARN] Chunk {chunk_idx}: Found TIME values > {GARBAGE_THRESHOLD:.0e} (Likely uninitialized/dead channels)")
                    # We don't fail here because inference script should mask these, but good to know.
                
                if (npho > GARBAGE_THRESHOLD).any():
                    print(f"  [WARN] Chunk {chunk_idx}: Found NPHO values > {GARBAGE_THRESHOLD:.0e}")

            print(f"  [STATS] Npho Range: {global_min_npho:.2e} to {global_max_npho:.2e}")
            print(f"  [STATS] Time Range: {global_min_time:.2e} to {global_max_time:.2e}")

            if not has_error:
                print("  [OK] File numerically valid.")
                return True
            else:
                return False

    except Exception as e:
        print(f"  [CRITICAL] Failed to open/read file: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Check ROOT files for validity (NaNs/Infs/Garbage)")
    parser.add_argument("files", nargs='+', help="ROOT files to check")
    parser.add_argument("--tree", default="tree", help="Tree name")
    args = parser.parse_args()
    
    file_list = []
    for f in args.files:
        if "*" in f:
            file_list.extend(glob.glob(f))
        else:
            file_list.append(f)
            
    print(f"Found {len(file_list)} files to check.")
    
    bad_files = []
    for fpath in sorted(file_list):
        if not check_file(fpath, args.tree):
            bad_files.append(fpath)
            
    if bad_files:
        print(f"\nFound {len(bad_files)} BAD files:")
        for bf in bad_files:
            print(f"  - {bf}")
        sys.exit(1)
    else:
        print("\nAll files passed checks.")

if __name__ == "__main__":
    main()