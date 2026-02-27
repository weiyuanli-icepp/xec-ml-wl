#!/usr/bin/env python3
"""
Run LocalFitBaseline on a single file for sensor-front validation.

This script is designed to be run as parallel batch jobs, one per input file.
It reads a manifest produced by validate_inpainter_sensorfront.py, processes
one file's worth of matched events through the ROOT C++ macro, and saves
results as a .npz file.

Usage:
    # Run for a single file index:
    python macro/run_localfit_sensorfront.py \\
        --manifest artifacts/sensorfront_validation/_sensorfront_manifest.npz \\
        --file-index 0

    # Run all files sequentially:
    python macro/run_localfit_sensorfront.py \\
        --manifest artifacts/sensorfront_validation/_sensorfront_manifest.npz \\
        --all

    # Batch submission (e.g. SLURM):
    for i in $(seq 0 9); do
        sbatch --wrap="python macro/run_localfit_sensorfront.py \\
            --manifest path/to/_sensorfront_manifest.npz --file-index $i"
    done
"""

from __future__ import annotations

import os
import sys
import argparse
import subprocess
import tempfile
import numpy as np
import uproot
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.dataset import expand_path


# Branches the C++ macro reads from the input ROOT file
_MACRO_BRANCHES = [
    "run", "event", "npho", "uvwRecoFI", "uvwTruth",
    "xyzTruth", "energyTruth",
]


def run_localfit_one_file(
    root_file: str,
    file_index: int,
    matched_orig_idx: np.ndarray,
    matched_sid: np.ndarray,
    global_offset: int,
    n_in_file: int,
    output_dir: str,
) -> Optional[str]:
    """Run LocalFitBaseline macro on one file's matched events.

    Args:
        root_file: Path to input ROOT file.
        file_index: Index of this file in the manifest's file list.
        matched_orig_idx: Global event indices of ALL matched events.
        matched_sid: Sensor IDs of ALL matched events.
        global_offset: Starting global event index for this file.
        n_in_file: Number of events in this file.
        output_dir: Directory to save results.

    Returns:
        Path to saved .npz file, or None if no matches / failure.
    """
    macro_path = os.path.join(
        os.path.dirname(__file__), '..', 'others', 'LocalFitBaseline.C')
    macro_path = os.path.abspath(macro_path)
    if not os.path.isfile(macro_path):
        print(f"[ERROR] LocalFitBaseline.C not found at {macro_path}")
        return None

    # Find matched events in this file's range
    file_start = global_offset
    file_end = global_offset + n_in_file
    in_file = ((matched_orig_idx >= file_start) &
               (matched_orig_idx < file_end))

    if not in_file.any():
        print(f"[INFO] File {file_index}: no matched events, skipping")
        return None

    file_sids = np.unique(matched_sid[in_file])
    n_file_matched = int(in_file.sum())

    # File-local event indices of matched events
    local_indices = matched_orig_idx[in_file] - global_offset

    print(f"[INFO] File {file_index}: {os.path.basename(root_file)}")
    print(f"       {n_file_matched} matched events, "
          f"{len(file_sids)} unique sensors")

    # Write a filtered ROOT file containing only matched events
    filtered_tmp = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
    filtered_tmp.close()
    with uproot.open(root_file) as rf:
        tree = rf["tree"]
        avail = [b for b in _MACRO_BRANCHES if b in tree.keys()]
        branches = {b: tree[b].array(library="np")[local_indices]
                    for b in avail}
    with uproot.recreate(filtered_tmp.name) as wf:
        wf.mktree("tree", branches)

    # Map from filtered event index back to original file-local index
    filtered_to_local = {i: int(local_indices[i])
                         for i in range(n_file_matched)}

    dead_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    out_tmp = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
    try:
        for ch in file_sids:
            dead_tmp.write(f"{ch}\n")
        dead_tmp.close()
        out_tmp.close()

        cmd = (f'root -l -b -q \'{macro_path}("{filtered_tmp.name}", '
               f'"{dead_tmp.name}", "{out_tmp.name}")\'')
        print(f"[INFO] Running: {cmd}")
        sys.stdout.flush()
        result = subprocess.run(cmd, shell=True)

        ok = True
        if result.returncode != 0:
            try:
                _t = uproot.open(out_tmp.name)
                _t['predictions']
                _t.close()
            except Exception:
                print(f"[ERROR] Macro failed on {root_file}")
                ok = False

        if not ok:
            return None

        # Read macro output
        with uproot.open(out_tmp.name) as f:
            pt = f['predictions']
            lf_ev = pt['event_idx'].array(library='np')
            lf_sid = pt['sensor_id'].array(library='np')
            lf_pred_raw = pt['pred_npho'].array(library='np')

        # Map filtered event indices back to global indices
        global_ev_list = []
        sid_list = []
        pred_raw_list = []
        for j in range(len(lf_ev)):
            filt_idx = int(lf_ev[j])
            local_idx = filtered_to_local.get(filt_idx)
            if local_idx is None:
                continue
            gev = local_idx + global_offset
            global_ev_list.append(gev)
            sid_list.append(int(lf_sid[j]))
            pred_raw_list.append(float(lf_pred_raw[j]))

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"localfit_file{file_index:04d}.npz")
        np.savez_compressed(
            out_path,
            global_event_idx=np.array(global_ev_list, dtype=np.int64),
            sensor_id=np.array(sid_list, dtype=np.int32),
            pred_npho_raw=np.array(pred_raw_list, dtype=np.float32),
        )
        print(f"[INFO] Saved {len(global_ev_list)} predictions to {out_path}")
        return out_path

    finally:
        for p in (dead_tmp.name, out_tmp.name, filtered_tmp.name):
            if os.path.exists(p):
                os.unlink(p)


def main():
    parser = argparse.ArgumentParser(
        description="Run LocalFitBaseline on one file for sensor-front validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", "-m", required=True,
                        help="Path to _sensorfront_manifest.npz")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-index", "-f", type=int,
                       help="Index of file to process (0-based)")
    group.add_argument("--all", "-a", action="store_true",
                       help="Process all files sequentially")
    parser.add_argument("--output-dir", "-o", type=str, default=None,
                        help="Output directory (default: <manifest_dir>/localfit_results/)")

    args = parser.parse_args()

    # Load manifest
    manifest = np.load(args.manifest, allow_pickle=True)
    matched_orig_idx = manifest["matched_orig_idx"]
    matched_sid = manifest["matched_sid"]
    file_list = list(manifest["file_list"])
    max_events_val = int(manifest["max_events"][0])
    max_events = max_events_val if max_events_val > 0 else None

    n_files = len(file_list)
    print(f"[INFO] Manifest: {len(matched_orig_idx)} matched events, "
          f"{n_files} files")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            os.path.dirname(args.manifest), "localfit_results")

    # Compute global_offset and n_in_file for each file
    file_info = []  # list of (global_offset, n_in_file)
    global_offset = 0
    total_events = 0
    for fi, root_file in enumerate(file_list):
        if max_events and total_events >= max_events:
            file_info.append((global_offset, 0))
            continue
        with uproot.open(root_file) as rf:
            n_in_file = rf["tree"].num_entries
        if max_events:
            n_in_file = min(n_in_file, max_events - total_events)
        file_info.append((global_offset, n_in_file))
        global_offset += n_in_file
        total_events += n_in_file

    if args.all:
        indices = list(range(n_files))
    else:
        if args.file_index < 0 or args.file_index >= n_files:
            print(f"[ERROR] --file-index {args.file_index} out of range "
                  f"[0, {n_files - 1}]")
            sys.exit(1)
        indices = [args.file_index]

    for fi in indices:
        offset, n_in = file_info[fi]
        if n_in == 0:
            print(f"[INFO] File {fi}: skipped (0 events)")
            continue
        run_localfit_one_file(
            root_file=file_list[fi],
            file_index=fi,
            matched_orig_idx=matched_orig_idx,
            matched_sid=matched_sid,
            global_offset=offset,
            n_in_file=n_in,
            output_dir=output_dir,
        )

    print("[INFO] Done.")


if __name__ == "__main__":
    main()
