#!/usr/bin/env python3
"""
Standalone baseline computation for inpainter validation.

Computes rule-based baselines (neighbor average, solid-angle weighted) for
dead/masked sensors, independent of any inpainter model or normalization
scheme.  All computation is done in raw photon space.  Produces a ROOT
file that can be passed to compare_inpainter.py via --baselines.

Usage:
    # MC pseudo-experiment with run 430000 dead pattern
    python macro/compute_inpainter_baselines.py \\
        --input data/E15to60_AngUni_PosSQ/val/ \\
        --run 430000 \\
        --output baselines_mc_run430000.root

    # With solid-angle-weighted baseline
    python macro/compute_inpainter_baselines.py \\
        --input data/E15to60_AngUni_PosSQ/val2/ \\
        --run 430000 \\
        --solid-angle-branch solidAngle \\
        --output baselines_mc_run430000.root

    # With dead channel file (no database needed)
    python macro/compute_inpainter_baselines.py \\
        --input data/val/ \\
        --dead-channel-file dead_channels_430000.txt \\
        --output baselines.root
"""

import os
import sys
import argparse
import numpy as np
import uproot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.inpainter_baselines import NeighborAverageBaseline, SolidAngleWeightedBaseline
from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_ALL_SENSOR_IDS,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows,
)
from lib.dataset import expand_path

# Import reusable functions from validate_inpainter
from validate_inpainter import (
    get_dead_channels, load_solid_angles,
    get_face_sensor_ids, FACE_NAME_TO_INT,
)

N_CHANNELS = 4760


def parse_n_artificial(value: str):
    """Parse n_artificial from string (same logic as validate_inpainter)."""
    try:
        return int(value)
    except ValueError:
        result = {}
        for part in value.split(','):
            face, count = part.strip().split(':')
            result[face.strip()] = int(count.strip())
        return result


def load_raw_data(input_path, tree_name="tree", max_events=None,
                  npho_branch="npho", time_branch="relative_time"):
    """Load raw (unnormalized) data from ROOT files.

    Returns dict with 'npho' (N, 4760), 'time' (N, 4760), and
    optional 'run', 'event' arrays.
    """
    file_list = expand_path(input_path)
    print(f"[INFO] Loading data from {len(file_list)} file(s)")
    if len(file_list) > 1:
        for f in file_list[:5]:
            print(f"  - {f}")
        if len(file_list) > 5:
            print(f"  ... and {len(file_list) - 5} more")

    all_data = {'npho': [], 'time': [], 'run': [], 'event': []}
    total = 0

    for fpath in file_list:
        if max_events and total >= max_events:
            break
        with uproot.open(fpath) as f:
            tree = f[tree_name]
            keys = tree.keys()

            actual_npho = npho_branch
            if npho_branch not in keys and "relative_npho" in keys:
                actual_npho = "relative_npho"

            npho = tree[actual_npho].array(library='np')
            time = tree[time_branch].array(library='np')

            n = len(npho)
            if max_events:
                remaining = max_events - total
                if n > remaining:
                    npho = npho[:remaining]
                    time = time[:remaining]
                    n = remaining

            all_data['npho'].append(npho)
            all_data['time'].append(time)

            for branch in ['run', 'event']:
                if branch in keys:
                    arr = tree[branch].array(library='np')
                    if max_events:
                        arr = arr[:n]
                    all_data[branch].append(arr)

            total += n

    data = {
        'npho': np.concatenate(all_data['npho']).astype(np.float32),
        'time': np.concatenate(all_data['time']).astype(np.float32),
    }
    if all_data['run']:
        data['run'] = np.concatenate(all_data['run'])
    if all_data['event']:
        data['event'] = np.concatenate(all_data['event'])

    print(f"[INFO] Loaded {len(data['npho']):,} events")
    return data


def create_artificial_mask_raw(raw_npho, n_artificial, dead_mask, seed=42):
    """Create artificial mask on healthy sensors using raw npho values.

    Sensors with raw_npho > 9e9 or NaN are considered invalid and excluded.

    Returns (artificial_mask, combined_mask).
    """
    rng = np.random.default_rng(seed)
    n_events = raw_npho.shape[0]
    artificial_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)

    # Invalid sensor mask (same as training pipeline)
    invalid = (raw_npho > 9e9) | np.isnan(raw_npho)

    if isinstance(n_artificial, dict):
        face_sensor_ids = {
            fname: get_face_sensor_ids(fname) for fname in n_artificial
        }
        for i in range(n_events):
            for fname, n_per_face in n_artificial.items():
                if n_per_face <= 0:
                    continue
                sids = face_sensor_ids[fname]
                valid = ~dead_mask[sids] & ~invalid[i, sids]
                valid_sids = sids[valid]
                if len(valid_sids) >= n_per_face:
                    chosen = rng.choice(valid_sids, size=n_per_face, replace=False)
                    artificial_mask[i, chosen] = True
    else:
        for i in range(n_events):
            valid = ~dead_mask & ~invalid[i]
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > n_artificial:
                chosen = rng.choice(valid_indices, size=n_artificial, replace=False)
                artificial_mask[i, chosen] = True

    combined_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
    combined_mask[:, dead_mask] = True
    combined_mask |= artificial_mask

    return artificial_mask, combined_mask


def main():
    parser = argparse.ArgumentParser(
        description="Compute rule-based baselines for inpainter comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--input", "-i", required=True,
                        help="Path to input ROOT file, directory, or glob pattern")
    parser.add_argument("--output", "-o", required=True,
                        help="Output ROOT file path")

    # Dead channels
    dead_group = parser.add_mutually_exclusive_group(required=True)
    dead_group.add_argument("--run", type=int,
                            help="Run number to fetch dead channels from database")
    dead_group.add_argument("--dead-channel-file", type=str,
                            help="Path to dead channel list file")

    # Mode
    parser.add_argument("--real-data", action="store_true",
                        help="Real data mode: dead channels have no truth (mask_type=1). "
                             "Default is MC mode where all sensors have truth (mask_type=0).")

    # Masking
    parser.add_argument("--n-artificial", type=str,
                        default="inner:10,us:1,ds:1,outer:1,top:1,bot:1",
                        help="Artificial masks per event (default: stratified 15 total)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for artificial masking (default: 42)")

    # Baselines
    parser.add_argument("--solid-angle-branch", type=str, default=None,
                        help="Branch name for solid angles (enables SA baseline)")
    parser.add_argument("--distance-threshold", type=float, default=20.0,
                        help="Distance threshold (cm) for SA neighbor selection (default: 20)")
    parser.add_argument("--npho-threshold", type=float, default=50.0,
                        help="Min total neighbor npho for SA weighting; "
                             "below this falls back to simple average (default: 50)")

    # Options
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum events to process")
    parser.add_argument("--tree-name", type=str, default="tree",
                        help="TTree name in ROOT files (default: tree)")

    args = parser.parse_args()

    # --- Load raw data (no normalization) ---
    data = load_raw_data(args.input, tree_name=args.tree_name,
                         max_events=args.max_events)

    raw_npho = data['npho']   # (N, 4760)
    raw_time = data['time']   # (N, 4760)
    n_events = raw_npho.shape[0]
    print(f"[INFO] {n_events} events, {N_CHANNELS} sensors")

    # --- Dead channels ---
    dead_channels = get_dead_channels(
        run_number=args.run,
        dead_channel_file=args.dead_channel_file,
    )
    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    dead_mask[dead_channels] = True
    print(f"[INFO] Dead channels: {len(dead_channels)}")

    # --- Artificial masking (on raw data) ---
    n_artificial = parse_n_artificial(args.n_artificial)
    artificial_mask, combined_mask = create_artificial_mask_raw(
        raw_npho, n_artificial, dead_mask, seed=args.seed,
    )
    print(f"[INFO] Artificial masks per event: {n_artificial}")

    # --- Solid angles ---
    solid_angles = None
    if args.solid_angle_branch:
        solid_angles = load_solid_angles(
            args.input, args.solid_angle_branch,
            tree_name=args.tree_name, max_events=args.max_events,
        )

    # --- Run baselines in raw photon space (no normalization needed) ---
    # Clamp invalid values to 0 for averaging
    npho_clean = raw_npho.copy()
    npho_invalid = (raw_npho > 9e9) | np.isnan(raw_npho) | (raw_npho < 0)
    npho_clean[npho_invalid] = 0.0

    print(f"[INFO] Running NeighborAverageBaseline (dist={args.distance_threshold} cm)...")
    avg_baseline = NeighborAverageBaseline(
        distance_threshold=args.distance_threshold,
    )
    baseline_preds = {
        'avg': avg_baseline.predict(npho_clean, combined_mask),
    }

    if solid_angles is not None:
        print(f"[INFO] Running SolidAngleWeightedBaseline "
              f"(dist={args.distance_threshold} cm, npho_thr={args.npho_threshold})...")
        sa_baseline = SolidAngleWeightedBaseline(
            distance_threshold=args.distance_threshold,
            npho_threshold=args.npho_threshold,
        )
        baseline_preds['sa'] = sa_baseline.predict(
            npho_clean, combined_mask, solid_angles=solid_angles,
        )

    # --- Collect per-sensor entries ---
    print("[INFO] Collecting per-sensor predictions...")

    sensor_face = np.full(N_CHANNELS, -1, dtype=np.int32)
    for fname, fint in FACE_NAME_TO_INT.items():
        sids = get_face_sensor_ids(fname)
        sensor_face[sids] = fint

    has_run = 'run' in data
    has_event = 'event' in data

    all_event_idx = []
    all_sensor_id = []
    all_face = []
    all_mask_type = []
    all_truth_npho = []
    all_run_number = []
    all_event_number = []
    all_baselines = {bname: [] for bname in baseline_preds}

    for i in range(n_events):
        masked_sensors = np.where(combined_mask[i])[0]
        n_masked = len(masked_sensors)
        if n_masked == 0:
            continue

        all_event_idx.append(np.full(n_masked, i, dtype=np.int32))
        all_sensor_id.append(masked_sensors.astype(np.int32))
        all_face.append(sensor_face[masked_sensors])

        # MC mode: all sensors have truth (mask_type=0)
        # Real data: only artificial masks have truth (0), dead channels don't (1)
        if args.real_data:
            mt = np.where(artificial_mask[i, masked_sensors], 0, 1).astype(np.int32)
        else:
            mt = np.zeros(n_masked, dtype=np.int32)
        all_mask_type.append(mt)

        # Store raw photon values as truth
        all_truth_npho.append(raw_npho[i, masked_sensors].astype(np.float32))

        run_val = int(data['run'][i]) if has_run else -1
        evt_val = int(data['event'][i]) if has_event else i
        all_run_number.append(np.full(n_masked, run_val, dtype=np.int64))
        all_event_number.append(np.full(n_masked, evt_val, dtype=np.int64))

        for bname, bpred_full in baseline_preds.items():
            all_baselines[bname].append(
                bpred_full[i, masked_sensors].astype(np.float32))

    # Concatenate
    event_idx = np.concatenate(all_event_idx)
    sensor_id = np.concatenate(all_sensor_id)
    face = np.concatenate(all_face)
    mask_type = np.concatenate(all_mask_type)
    truth_npho = np.concatenate(all_truth_npho)
    run_number = np.concatenate(all_run_number)
    event_number = np.concatenate(all_event_number)

    n_entries = len(event_idx)
    print(f"[INFO] {n_entries:,} masked-sensor entries")

    # Compute errors and build branches
    # For raw data: truth > 9e9 or NaN means no ground truth
    has_truth = (mask_type == 0) & (truth_npho < 9e9) & np.isfinite(truth_npho)

    branches = {
        'event_idx': event_idx,
        'sensor_id': sensor_id,
        'face': face,
        'mask_type': mask_type,
        'truth_npho': truth_npho,
        'run_number': run_number,
        'event_number': event_number,
    }

    for bname in baseline_preds:
        bpred = np.concatenate(all_baselines[bname])
        has_bpred = np.isfinite(bpred)
        berror = np.where(
            has_truth & has_bpred, bpred - truth_npho, np.float32(-999.0),
        )
        bpred = np.where(np.isfinite(bpred), bpred, -999.0).astype(np.float32)
        branches[f'baseline_{bname}_npho'] = bpred
        branches[f'baseline_{bname}_error_npho'] = berror

    # --- Save ---
    metadata = {
        'npho_scheme': np.array(['raw'], dtype='U16'),
        'distance_threshold': np.array([args.distance_threshold], dtype=np.float64),
        'seed': np.array([args.seed], dtype=np.int32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with uproot.recreate(args.output) as f:
        f.mktree('predictions', branches)
        f.mktree('metadata', metadata)

    print(f"[INFO] Saved baselines to {args.output}")
    baselines_str = ", ".join(f"baseline_{b}" for b in baseline_preds)
    print(f"[INFO] Baselines: {baselines_str}")
    print(f"[INFO] Values stored in raw photon space (no normalization)")
    print(f"[INFO] Use with: python macro/compare_inpainter.py --baselines {args.output}")


if __name__ == "__main__":
    main()
