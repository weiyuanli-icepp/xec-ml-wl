#!/usr/bin/env python3
"""
Generate pseudo prediction files for testing show_mae_comparison.py and show_inpainter_comparison.py.

Usage:
    # Generate MAE predictions file
    python macro/generate_test_predictions.py data/large_val.root --output test_mae_predictions.root --mode mae

    # Generate inpainter predictions file
    python macro/generate_test_predictions.py data/large_val.root --output test_inpainter_predictions.root --mode inpainter

    # Generate both
    python macro/generate_test_predictions.py data/large_val.root --mode both

The script:
- Reads input data and applies normalization
- Creates random 60% mask
- Generates pseudo predictions: pred = truth + random[0,1] * truth * 0.1
"""

import sys
import os
import argparse
import numpy as np
import uproot

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE,
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    flatten_hex_rows
)


def normalize_input(raw_npho, raw_time,
                    npho_scale=DEFAULT_NPHO_SCALE,
                    npho_scale2=DEFAULT_NPHO_SCALE2,
                    time_scale=DEFAULT_TIME_SCALE,
                    time_shift=DEFAULT_TIME_SHIFT,
                    sentinel_value=DEFAULT_SENTINEL_VALUE):
    """Apply normalization to raw input data."""
    mask_npho_bad = (raw_npho <= 0.0) | (raw_npho > 9e9) | np.isnan(raw_npho)
    mask_time_bad = mask_npho_bad | (np.abs(raw_time) > 9e9) | np.isnan(raw_time)

    raw_npho_safe = np.where(mask_npho_bad, 0.0, raw_npho)
    npho_norm = np.log1p(raw_npho_safe / npho_scale) / npho_scale2
    time_norm = (raw_time / time_scale) - time_shift

    npho_norm[mask_npho_bad] = 0.0
    time_norm[mask_time_bad] = sentinel_value

    return npho_norm, time_norm


def generate_random_mask(n_sensors, mask_ratio=0.6, sentinel_value=DEFAULT_SENTINEL_VALUE, time_norm=None):
    """Generate random mask, excluding already-invalid sensors."""
    mask = np.zeros(n_sensors, dtype=np.float32)

    # Identify valid sensors (not already sentinel)
    if time_norm is not None:
        valid = time_norm != sentinel_value
    else:
        valid = np.ones(n_sensors, dtype=bool)

    valid_indices = np.where(valid)[0]
    n_to_mask = int(len(valid_indices) * mask_ratio)

    if n_to_mask > 0:
        masked_indices = np.random.choice(valid_indices, size=n_to_mask, replace=False)
        mask[masked_indices] = 1.0

    return mask


def generate_pseudo_prediction(truth, mask, error_scale=0.1):
    """Generate pseudo prediction: pred = truth + random * truth * error_scale."""
    noise = np.random.rand(*truth.shape).astype(np.float32)
    # Only add error where masked
    error = noise * np.abs(truth) * error_scale * mask
    pred = truth + error
    return pred


def get_face_for_sensor(sensor_id):
    """Determine which face a sensor belongs to."""
    inner_flat = INNER_INDEX_MAP.flatten()
    us_flat = US_INDEX_MAP.flatten()
    ds_flat = DS_INDEX_MAP.flatten()
    outer_flat = OUTER_COARSE_FULL_INDEX_MAP.flatten()
    top_flat = flatten_hex_rows(TOP_HEX_ROWS)
    bot_flat = flatten_hex_rows(BOTTOM_HEX_ROWS)

    if sensor_id in inner_flat:
        return 0  # inner
    elif sensor_id in us_flat:
        return 1  # us
    elif sensor_id in ds_flat:
        return 2  # ds
    elif sensor_id in outer_flat:
        return 3  # outer
    elif sensor_id in top_flat:
        return 4  # top
    elif sensor_id in bot_flat:
        return 5  # bot
    return -1


# Pre-compute face lookup for efficiency
def build_face_lookup():
    """Build a lookup array for sensor_id -> face."""
    lookup = np.full(4760, -1, dtype=np.int32)

    for sid in INNER_INDEX_MAP.flatten():
        lookup[sid] = 0
    for sid in US_INDEX_MAP.flatten():
        lookup[sid] = 1
    for sid in DS_INDEX_MAP.flatten():
        lookup[sid] = 2
    for sid in OUTER_COARSE_FULL_INDEX_MAP.flatten():
        lookup[sid] = 3
    for sid in flatten_hex_rows(TOP_HEX_ROWS):
        lookup[sid] = 4
    for sid in flatten_hex_rows(BOTTOM_HEX_ROWS):
        lookup[sid] = 5

    return lookup

FACE_LOOKUP = build_face_lookup()


def generate_mae_predictions(input_file, output_file, tree_name="tree",
                             npho_branch="relative_npho", time_branch="relative_time",
                             mask_ratio=0.6, error_scale=0.1, max_events=100,
                             npho_scale=DEFAULT_NPHO_SCALE, npho_scale2=DEFAULT_NPHO_SCALE2,
                             time_scale=DEFAULT_TIME_SCALE, time_shift=DEFAULT_TIME_SHIFT,
                             sentinel_value=DEFAULT_SENTINEL_VALUE):
    """Generate MAE-style predictions ROOT file."""
    print(f"Generating MAE predictions from: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Mask ratio: {mask_ratio}, Error scale: {error_scale}")

    with uproot.open(input_file) as f:
        tree = f[tree_name]
        n_entries = min(tree.num_entries, max_events)
        print(f"  Processing {n_entries} events...")

        # Load data
        arrays = tree.arrays([npho_branch, time_branch], library="np",
                            entry_stop=n_entries)

        raw_npho_all = arrays[npho_branch].astype("float32")
        raw_time_all = arrays[time_branch].astype("float32")

    # Prepare output arrays
    truth_npho_out = []
    truth_time_out = []
    masked_npho_out = []
    masked_time_out = []
    pred_npho_out = []
    pred_time_out = []
    mask_out = []

    for i in range(n_entries):
        raw_npho = raw_npho_all[i]
        raw_time = raw_time_all[i]

        # Normalize
        npho_norm, time_norm = normalize_input(
            raw_npho, raw_time,
            npho_scale=npho_scale, npho_scale2=npho_scale2,
            time_scale=time_scale, time_shift=time_shift,
            sentinel_value=sentinel_value
        )

        # Generate mask
        mask = generate_random_mask(len(npho_norm), mask_ratio, sentinel_value, time_norm)

        # Create masked input
        masked_npho = npho_norm.copy()
        masked_time = time_norm.copy()
        masked_npho[mask > 0.5] = 0.0
        masked_time[mask > 0.5] = sentinel_value

        # Generate predictions
        pred_npho = generate_pseudo_prediction(npho_norm, mask, error_scale)
        pred_time = generate_pseudo_prediction(time_norm, mask, error_scale)

        truth_npho_out.append(npho_norm)
        truth_time_out.append(time_norm)
        masked_npho_out.append(masked_npho)
        masked_time_out.append(masked_time)
        pred_npho_out.append(pred_npho)
        pred_time_out.append(pred_time)
        mask_out.append(mask)

    # Write to ROOT file
    # Use explicit type specification to avoid awkward import issues
    truth_npho_arr = np.array(truth_npho_out, dtype=np.float32)
    truth_time_arr = np.array(truth_time_out, dtype=np.float32)
    masked_npho_arr = np.array(masked_npho_out, dtype=np.float32)
    masked_time_arr = np.array(masked_time_out, dtype=np.float32)
    pred_npho_arr = np.array(pred_npho_out, dtype=np.float32)
    pred_time_arr = np.array(pred_time_out, dtype=np.float32)
    mask_arr = np.array(mask_out, dtype=np.float32)

    def _dtype_with_shape(arr):
        if arr.ndim == 1:
            return arr.dtype
        return np.dtype((arr.dtype, arr.shape[1:]))

    with uproot.recreate(output_file) as f:
        # Define tree with explicit numpy dtype specifications (including fixed shapes)
        f.mktree("tree", {
            "truth_npho": _dtype_with_shape(truth_npho_arr),
            "truth_time": _dtype_with_shape(truth_time_arr),
            "masked_npho": _dtype_with_shape(masked_npho_arr),
            "masked_time": _dtype_with_shape(masked_time_arr),
            "pred_npho": _dtype_with_shape(pred_npho_arr),
            "pred_time": _dtype_with_shape(pred_time_arr),
            "mask": _dtype_with_shape(mask_arr),
        })
        # Extend with actual data
        f["tree"].extend({
            "truth_npho": truth_npho_arr,
            "truth_time": truth_time_arr,
            "masked_npho": masked_npho_arr,
            "masked_time": masked_time_arr,
            "pred_npho": pred_npho_arr,
            "pred_time": pred_time_arr,
            "mask": mask_arr,
        })

    print(f"  Written {n_entries} events to {output_file}")


def generate_inpainter_predictions(input_file, output_file, tree_name="tree",
                                   npho_branch="relative_npho", time_branch="relative_time",
                                   mask_ratio=0.6, error_scale=0.1, max_events=100,
                                   npho_scale=DEFAULT_NPHO_SCALE, npho_scale2=DEFAULT_NPHO_SCALE2,
                                   time_scale=DEFAULT_TIME_SCALE, time_shift=DEFAULT_TIME_SHIFT,
                                   sentinel_value=DEFAULT_SENTINEL_VALUE):
    """Generate inpainter-style predictions ROOT file."""
    print(f"Generating inpainter predictions from: {input_file}")
    print(f"  Output: {output_file}")
    print(f"  Mask ratio: {mask_ratio}, Error scale: {error_scale}")

    with uproot.open(input_file) as f:
        tree = f[tree_name]
        n_entries = min(tree.num_entries, max_events)
        print(f"  Processing {n_entries} events...")

        # Check for run/event branches
        available = tree.keys()
        has_run_event = "run" in available and "event" in available

        branches = [npho_branch, time_branch]
        if has_run_event:
            branches.extend(["run", "event"])

        arrays = tree.arrays(branches, library="np", entry_stop=n_entries)

        raw_npho_all = arrays[npho_branch].astype("float32")
        raw_time_all = arrays[time_branch].astype("float32")

        if has_run_event:
            run_all = arrays["run"]
            event_all = arrays["event"]
        else:
            run_all = np.zeros(n_entries, dtype=np.int32)
            event_all = np.arange(n_entries, dtype=np.int32)

    # Prepare output arrays (per-masked-sensor format)
    event_idx_out = []
    sensor_id_out = []
    face_out = []
    pred_npho_out = []
    pred_time_out = []
    truth_npho_out = []
    truth_time_out = []
    run_number_out = []
    event_number_out = []

    for i in range(n_entries):
        raw_npho = raw_npho_all[i]
        raw_time = raw_time_all[i]

        # Normalize
        npho_norm, time_norm = normalize_input(
            raw_npho, raw_time,
            npho_scale=npho_scale, npho_scale2=npho_scale2,
            time_scale=time_scale, time_shift=time_shift,
            sentinel_value=sentinel_value
        )

        # Generate mask
        mask = generate_random_mask(len(npho_norm), mask_ratio, sentinel_value, time_norm)

        # Get masked sensor indices
        masked_indices = np.where(mask > 0.5)[0]

        for sid in masked_indices:
            # Generate prediction with error
            noise_npho = np.random.rand() * error_scale
            noise_time = np.random.rand() * error_scale

            pred_npho = npho_norm[sid] + noise_npho * abs(npho_norm[sid])
            pred_time = time_norm[sid] + noise_time * abs(time_norm[sid])

            event_idx_out.append(i)
            sensor_id_out.append(sid)
            face_out.append(FACE_LOOKUP[sid])
            pred_npho_out.append(pred_npho)
            pred_time_out.append(pred_time)
            truth_npho_out.append(npho_norm[sid])
            truth_time_out.append(time_norm[sid])
            run_number_out.append(int(run_all[i]))
            event_number_out.append(int(event_all[i]))

    # Write to ROOT file
    # Use explicit type specification to avoid awkward import issues
    event_idx_arr = np.array(event_idx_out, dtype=np.int32)
    sensor_id_arr = np.array(sensor_id_out, dtype=np.int32)
    face_arr = np.array(face_out, dtype=np.int32)
    pred_npho_arr = np.array(pred_npho_out, dtype=np.float32)
    pred_time_arr = np.array(pred_time_out, dtype=np.float32)
    truth_npho_arr = np.array(truth_npho_out, dtype=np.float32)
    truth_time_arr = np.array(truth_time_out, dtype=np.float32)
    run_number_arr = np.array(run_number_out, dtype=np.int32)
    event_number_arr = np.array(event_number_out, dtype=np.int32)

    # Metadata arrays
    meta_npho_scale = np.array([npho_scale], dtype=np.float32)
    meta_npho_scale2 = np.array([npho_scale2], dtype=np.float32)
    meta_time_scale = np.array([time_scale], dtype=np.float32)
    meta_time_shift = np.array([time_shift], dtype=np.float32)
    meta_sentinel = np.array([sentinel_value], dtype=np.float32)

    with uproot.recreate(output_file) as f:
        # Main predictions tree - define with types, then extend
        f.mktree("tree", {
            "event_idx": np.int32,
            "sensor_id": np.int32,
            "face": np.int32,
            "pred_npho": np.float32,
            "pred_time": np.float32,
            "truth_npho": np.float32,
            "truth_time": np.float32,
            "run_number": np.int32,
            "event_number": np.int32,
        })
        f["tree"].extend({
            "event_idx": event_idx_arr,
            "sensor_id": sensor_id_arr,
            "face": face_arr,
            "pred_npho": pred_npho_arr,
            "pred_time": pred_time_arr,
            "truth_npho": truth_npho_arr,
            "truth_time": truth_time_arr,
            "run_number": run_number_arr,
            "event_number": event_number_arr,
        })

        # Metadata tree
        f.mktree("metadata", {
            "npho_scale": np.float32,
            "npho_scale2": np.float32,
            "time_scale": np.float32,
            "time_shift": np.float32,
            "sentinel_value": np.float32,
        })
        f["metadata"].extend({
            "npho_scale": meta_npho_scale,
            "npho_scale2": meta_npho_scale2,
            "time_scale": meta_time_scale,
            "time_shift": meta_time_shift,
            "sentinel_value": meta_sentinel,
        })

    print(f"  Written {len(event_idx_out)} predictions ({n_entries} events) to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate pseudo prediction files for testing visualization macros",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate MAE predictions
  python macro/generate_test_predictions.py data/large_val.root --mode mae

  # Generate inpainter predictions
  python macro/generate_test_predictions.py data/large_val.root --mode inpainter

  # Generate both with custom output paths
  python macro/generate_test_predictions.py data/large_val.root --mode both \\
      --mae_output test_mae.root --inpainter_output test_inpainter.root

  # Test the outputs
  python macro/show_mae_comparison.py 0 --file test_mae.root --channel npho
  python macro/show_inpainter_comparison.py 0 --predictions test_inpainter.root --original data/large_val.root
        """
    )

    parser.add_argument("input_file", help="Input ROOT file (e.g., data/large_val.root)")
    parser.add_argument("--mode", choices=["mae", "inpainter", "both"], default="both",
                        help="Output mode: mae, inpainter, or both (default: both)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path (for single mode)")
    parser.add_argument("--mae_output", type=str, default="test_mae_predictions.root",
                        help="MAE output file (default: test_mae_predictions.root)")
    parser.add_argument("--inpainter_output", type=str, default="test_inpainter_predictions.root",
                        help="Inpainter output file (default: test_inpainter_predictions.root)")
    parser.add_argument("--tree", type=str, default="tree", help="Input tree name")
    parser.add_argument("--npho_branch", type=str, default="relative_npho")
    parser.add_argument("--time_branch", type=str, default="relative_time")
    parser.add_argument("--mask_ratio", type=float, default=0.6, help="Mask ratio (default: 0.6)")
    parser.add_argument("--error_scale", type=float, default=0.1,
                        help="Error scale: pred = truth + random*truth*scale (default: 0.1)")
    parser.add_argument("--max_events", type=int, default=100,
                        help="Maximum events to process (default: 100)")

    # Normalization parameters
    parser.add_argument("--npho_scale", type=float, default=DEFAULT_NPHO_SCALE)
    parser.add_argument("--npho_scale2", type=float, default=DEFAULT_NPHO_SCALE2)
    parser.add_argument("--time_scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--time_shift", type=float, default=DEFAULT_TIME_SHIFT)
    parser.add_argument("--sentinel_value", type=float, default=DEFAULT_SENTINEL_VALUE)

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # Determine output files
    if args.output:
        if args.mode == "mae":
            args.mae_output = args.output
        elif args.mode == "inpainter":
            args.inpainter_output = args.output

    # Generate requested outputs
    if args.mode in ["mae", "both"]:
        generate_mae_predictions(
            args.input_file, args.mae_output,
            tree_name=args.tree,
            npho_branch=args.npho_branch,
            time_branch=args.time_branch,
            mask_ratio=args.mask_ratio,
            error_scale=args.error_scale,
            max_events=args.max_events,
            npho_scale=args.npho_scale,
            npho_scale2=args.npho_scale2,
            time_scale=args.time_scale,
            time_shift=args.time_shift,
            sentinel_value=args.sentinel_value,
        )

    if args.mode in ["inpainter", "both"]:
        generate_inpainter_predictions(
            args.input_file, args.inpainter_output,
            tree_name=args.tree,
            npho_branch=args.npho_branch,
            time_branch=args.time_branch,
            mask_ratio=args.mask_ratio,
            error_scale=args.error_scale,
            max_events=args.max_events,
            npho_scale=args.npho_scale,
            npho_scale2=args.npho_scale2,
            time_scale=args.time_scale,
            time_shift=args.time_shift,
            sentinel_value=args.sentinel_value,
        )

    print("\nDone! Test with:")
    if args.mode in ["mae", "both"]:
        print(f"  python macro/show_mae_comparison.py 0 --file {args.mae_output} --channel npho")
        print(f"  python macro/show_mae_comparison.py 0 --file {args.mae_output} --channel time")
    if args.mode in ["inpainter", "both"]:
        print(f"  python macro/show_inpainter_comparison.py 0 --predictions {args.inpainter_output} --original {args.input_file}")


if __name__ == "__main__":
    main()
