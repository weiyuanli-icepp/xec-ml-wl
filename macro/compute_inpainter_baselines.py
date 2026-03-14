#!/usr/bin/env python3
"""
Standalone baseline computation for inpainter validation.

Computes rule-based baselines (neighbor average, solid-angle weighted) for
dead/masked sensors, independent of any inpainter model.  Produces a ROOT
file that can be passed to compare_inpainter.py via --baselines.

Usage:
    # MC pseudo-experiment with run 430000 dead pattern
    python macro/compute_inpainter_baselines.py \
        --input data/E15to60_AngUni_PosSQ/val/ \
        --run 430000 \
        --output baselines_mc_run430000.root

    # With solid-angle-weighted baseline
    python macro/compute_inpainter_baselines.py \
        --input data/E15to60_AngUni_PosSQ/val/ \
        --run 430000 \
        --solid-angle-branch solidAngle \
        --output baselines_mc_run430000.root

    # Custom normalization
    python macro/compute_inpainter_baselines.py \
        --input data/val/ --run 430000 \
        --npho-scheme sqrt \
        --output baselines_sqrt.root
"""

import os
import sys
import argparse
import numpy as np
import uproot

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.inpainter_baselines import NeighborAverageBaseline, SolidAngleWeightedBaseline
from lib.normalization import NphoTransform
from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_ALL_SENSOR_IDS,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows,
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_TIME, DEFAULT_NPHO_THRESHOLD,
)

# Import reusable functions from validate_inpainter
from validate_inpainter import (
    load_data, normalize_data, get_dead_channels,
    create_artificial_mask, load_solid_angles,
    get_face_sensor_ids, FACE_NAME_TO_INT,
)

N_CHANNELS = 4760
MODEL_SENTINEL_NPHO = -1.0


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

    # Masking
    parser.add_argument("--n-artificial", type=str,
                        default="inner:10,us:1,ds:1,outer:1,top:1,bot:1",
                        help="Artificial masks per event (default: stratified 15 total)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for artificial masking (default: 42)")

    # Normalization
    parser.add_argument("--npho-scheme", type=str, default="log1p",
                        choices=["log1p", "sqrt", "anscombe", "linear"],
                        help="Npho normalization scheme (default: log1p)")
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE)
    parser.add_argument("--npho-scale2", type=float, default=DEFAULT_NPHO_SCALE2)
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT)

    # Baselines
    parser.add_argument("--baseline-k", type=int, default=1,
                        help="k-hop parameter for neighbor search (default: 1)")
    parser.add_argument("--solid-angle-branch", type=str, default=None,
                        help="Branch name for solid angles (enables SA baseline)")

    # Options
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum events to process")
    parser.add_argument("--tree-name", type=str, default="tree",
                        help="TTree name in ROOT files (default: tree)")

    args = parser.parse_args()

    # --- Load and normalize data ---
    print("[INFO] Loading data...")
    data = load_data(args.input, tree_name=args.tree_name,
                     max_events=args.max_events)

    npho_scheme = args.npho_scheme
    npho_scale = args.npho_scale
    npho_scale2 = args.npho_scale2
    time_scale = args.time_scale
    time_shift = args.time_shift

    print(f"[INFO] Normalizing with scheme={npho_scheme}")
    x = normalize_data(data['npho'], data['time'],
                       npho_scheme=npho_scheme,
                       npho_scale=npho_scale,
                       npho_scale2=npho_scale2,
                       time_scale=time_scale,
                       time_shift=time_shift)

    n_events = x.shape[0]
    print(f"[INFO] {n_events} events, {N_CHANNELS} sensors")

    # --- Dead channels ---
    dead_channels = get_dead_channels(
        run_number=args.run,
        dead_channel_file=args.dead_channel_file,
    )
    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    dead_mask[dead_channels] = True
    print(f"[INFO] Dead channels: {len(dead_channels)}")

    # --- Artificial masking ---
    n_artificial = parse_n_artificial(args.n_artificial)
    artificial_mask, combined_mask = create_artificial_mask(
        x, n_artificial, dead_mask, seed=args.seed,
    )
    print(f"[INFO] Artificial masks per event: {n_artificial}")

    # --- Solid angles ---
    solid_angles = None
    if args.solid_angle_branch:
        solid_angles = load_solid_angles(
            args.input, args.solid_angle_branch,
            tree_name=args.tree_name, max_events=args.max_events,
        )

    # --- Run baselines ---
    npho_transform = NphoTransform(
        scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2,
    )

    print(f"[INFO] Running NeighborAverageBaseline (k={args.baseline_k})...")
    avg_baseline = NeighborAverageBaseline(k=args.baseline_k)
    baseline_preds = {
        'avg': avg_baseline.predict(x[:, :, 0], combined_mask,
                                    npho_transform=npho_transform),
    }

    if solid_angles is not None:
        print(f"[INFO] Running SolidAngleWeightedBaseline (k={args.baseline_k})...")
        sa_baseline = SolidAngleWeightedBaseline(k=args.baseline_k)
        baseline_preds['sa'] = sa_baseline.predict(
            x[:, :, 0], combined_mask, solid_angles=solid_angles,
            npho_transform=npho_transform,
        )

    # --- Collect per-sensor entries ---
    print("[INFO] Collecting per-sensor predictions...")

    # Build sensor_id -> face mapping
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
    all_truth_time = []
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

        # mask_type: 0 = artificial (has truth), 1 = dead
        mt = np.where(artificial_mask[i, masked_sensors], 0, 1).astype(np.int32)
        all_mask_type.append(mt)

        all_truth_npho.append(x[i, masked_sensors, 0].astype(np.float32))
        all_truth_time.append(x[i, masked_sensors, 1].astype(np.float32))

        run_val = int(data['run'][i]) if has_run else -1
        evt_val = int(data['event'][i]) if has_event else i
        all_run_number.append(np.full(n_masked, run_val, dtype=np.int64))
        all_event_number.append(np.full(n_masked, evt_val, dtype=np.int64))

        for bname, bpred_full in baseline_preds.items():
            bpred = bpred_full[i, masked_sensors].astype(np.float32)
            all_baselines[bname].append(bpred)

    # Concatenate
    event_idx = np.concatenate(all_event_idx)
    sensor_id = np.concatenate(all_sensor_id)
    face = np.concatenate(all_face)
    mask_type = np.concatenate(all_mask_type)
    truth_npho = np.concatenate(all_truth_npho)
    truth_time = np.concatenate(all_truth_time)
    run_number = np.concatenate(all_run_number)
    event_number = np.concatenate(all_event_number)

    n_entries = len(event_idx)
    print(f"[INFO] {n_entries:,} masked-sensor entries")

    # Compute errors and build branches
    has_truth = (mask_type == 0) & (truth_npho != MODEL_SENTINEL_NPHO)

    branches = {
        'event_idx': event_idx,
        'sensor_id': sensor_id,
        'face': face,
        'mask_type': mask_type,
        'truth_npho': truth_npho,
        'truth_time': truth_time,
        'run_number': run_number,
        'event_number': event_number,
    }

    for bname in baseline_preds:
        bpred = np.concatenate(all_baselines[bname])
        has_bpred = np.isfinite(bpred) & (bpred > -900)
        berror = np.where(
            has_truth & has_bpred, bpred - truth_npho, -999.0,
        ).astype(np.float32)
        bpred = np.where(np.isfinite(bpred), bpred, -999.0).astype(np.float32)
        branches[f'baseline_{bname}_npho'] = bpred
        branches[f'baseline_{bname}_error_npho'] = berror

    # --- Save ---
    metadata = {
        'npho_scheme': np.array([npho_scheme], dtype='U16'),
        'npho_scale': np.array([npho_scale], dtype=np.float64),
        'npho_scale2': np.array([npho_scale2], dtype=np.float64),
        'time_scale': np.array([time_scale], dtype=np.float64),
        'time_shift': np.array([time_shift], dtype=np.float64),
        'baseline_k': np.array([args.baseline_k], dtype=np.int32),
        'seed': np.array([args.seed], dtype=np.int32),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with uproot.recreate(args.output) as f:
        f.mktree('predictions', branches)
        f.mktree('metadata', metadata)

    print(f"[INFO] Saved baselines to {args.output}")
    baselines_str = ", ".join(f"baseline_{b}" for b in baseline_preds)
    print(f"[INFO] Baselines: {baselines_str}")
    print(f"[INFO] Use with: python macro/compare_inpainter.py --baselines {args.output}")


if __name__ == "__main__":
    main()
