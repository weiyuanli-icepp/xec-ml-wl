# Usage:
# python macro/show_inpainter_comparison.py 0 \
#     --predictions artifacts/inpainter/inpainter_predictions_epoch_50.root \
#     --original data/large_val.root \
#     --channel npho --save outputs/inpainter_event_0.pdf
#
# Note: event_idx in predictions file corresponds to entry index in original file
# Note: Outer face predictions are excluded when using finegrid mode (sensor_id is grid index, not flat index)
# Note: Normalization factors are read from predictions file metadata (if available)

import sys
import os
import argparse
import numpy as np
import uproot

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.event_display import plot_mae_comparison
    from lib.geom_defs import (
        DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
        DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE,
        DEFAULT_NPHO_THRESHOLD
    )
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Run from repo root (xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)


def load_metadata_from_predictions(pred_file):
    """
    Load normalization metadata from predictions file if available.
    Returns dict with normalization factors, or empty dict if not found.
    """
    metadata = {}
    try:
        with uproot.open(pred_file) as f:
            if "metadata" in f:
                meta_tree = f["metadata"]
                for key in ["npho_scale", "npho_scale2", "time_scale", "time_shift", "sentinel_value"]:
                    if key in meta_tree:
                        val = meta_tree[key].array(library="np")[0]
                        if not np.isnan(val):
                            metadata[key] = float(val)
    except Exception as e:
        print(f"  Warning: Could not read metadata from predictions file: {e}")
    return metadata


def normalize_input(raw_npho, raw_time,
                    npho_scale=DEFAULT_NPHO_SCALE,
                    npho_scale2=DEFAULT_NPHO_SCALE2,
                    time_scale=DEFAULT_TIME_SCALE,
                    time_shift=DEFAULT_TIME_SHIFT,
                    sentinel_value=DEFAULT_SENTINEL_VALUE):
    """
    Apply the same normalization as training to raw input data.
    Returns normalized (npho, time) arrays.
    """
    # Identify bad values
    mask_npho_bad = (raw_npho <= 0.0) | (raw_npho > 9e9) | np.isnan(raw_npho)
    mask_time_bad = mask_npho_bad | (np.abs(raw_time) > 9e9) | np.isnan(raw_time)

    # Normalize
    raw_npho_safe = np.where(mask_npho_bad, 0.0, raw_npho)
    npho_norm = np.log1p(raw_npho_safe / npho_scale) / npho_scale2
    time_norm = (raw_time / time_scale) - time_shift

    npho_norm[mask_npho_bad] = 0.0
    time_norm[mask_time_bad] = sentinel_value

    return npho_norm, time_norm


def main():
    parser = argparse.ArgumentParser(
        description="Display inpainter truth/masked/prediction comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python macro/show_inpainter_comparison.py 0 \\
      --predictions artifacts/inpainter/inpainter_predictions_epoch_50.root \\
      --original data/large_val.root \\
      --channel npho

  python macro/show_inpainter_comparison.py 42 \\
      --predictions artifacts/inpainter/inpainter_predictions_epoch_50.root \\
      --original data/large_val.root \\
      --channel time --save event_42.pdf
        """
    )
    parser.add_argument("event_idx", type=int, help="Event index (0-based, matches entry in original file)")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to inpainter predictions ROOT file")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original validation ROOT file")
    parser.add_argument("--tree", type=str, default="tree", help="TTree name in original file")
    parser.add_argument("--channel", type=str, choices=["npho", "time"], default="npho")
    parser.add_argument("--include_top_bottom", action="store_true",
                        help="Include top/bottom hex faces in the comparison grid")
    parser.add_argument("--save", type=str, default=None, help="Save path (PDF recommended)")

    # Normalization parameters (will be loaded from predictions file if available)
    parser.add_argument("--npho_scale", type=float, default=None,
                        help="Override npho_scale (default: read from predictions file or use default)")
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    # Input branch names
    parser.add_argument("--npho_branch", type=str, default="relative_npho")
    parser.add_argument("--time_branch", type=str, default="relative_time")

    # Time-invalid threshold (for visualization)
    parser.add_argument("--npho_threshold", type=float, default=None,
                        help=f"Npho threshold for time validity (raw scale, default: {DEFAULT_NPHO_THRESHOLD})")

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    if not os.path.exists(args.original):
        print(f"Error: Original file not found: {args.original}")
        sys.exit(1)

    # --- Load normalization metadata from predictions file ---
    print(f"Loading metadata from: {args.predictions}")
    metadata = load_metadata_from_predictions(args.predictions)

    # Use metadata values, CLI overrides, or defaults (in that order)
    npho_scale = args.npho_scale if args.npho_scale is not None else metadata.get("npho_scale", DEFAULT_NPHO_SCALE)
    npho_scale2 = args.npho_scale2 if args.npho_scale2 is not None else metadata.get("npho_scale2", DEFAULT_NPHO_SCALE2)
    time_scale = args.time_scale if args.time_scale is not None else metadata.get("time_scale", DEFAULT_TIME_SCALE)
    time_shift = args.time_shift if args.time_shift is not None else metadata.get("time_shift", DEFAULT_TIME_SHIFT)
    sentinel_value = args.sentinel_value if args.sentinel_value is not None else metadata.get("sentinel_value", DEFAULT_SENTINEL_VALUE)

    if metadata:
        print(f"  Using normalization from predictions file:")
        print(f"    npho_scale={npho_scale}, npho_scale2={npho_scale2}")
        print(f"    time_scale={time_scale}, time_shift={time_shift}")
        print(f"    sentinel_value={sentinel_value}")
    else:
        print(f"  Warning: No metadata found in predictions file, using defaults")
        print(f"    npho_scale={npho_scale}, npho_scale2={npho_scale2}")
        print(f"    time_scale={time_scale}, time_shift={time_shift}")
        print(f"    sentinel_value={sentinel_value}")

    # --- Load original data ---
    print(f"Loading original data from: {args.original}")
    print(f"  Using branches: npho='{args.npho_branch}', time='{args.time_branch}'")
    with uproot.open(args.original) as f:
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found. Available: {list(f.keys())}")
            sys.exit(1)

        tree = f[args.tree]
        n_entries = tree.num_entries
        if args.event_idx < 0 or args.event_idx >= n_entries:
            print(f"Error: Event index {args.event_idx} out of bounds (max {n_entries - 1})")
            sys.exit(1)

        # Load single event with input data and truth info
        input_branches = [args.npho_branch, args.time_branch]
        truth_branches = ["energyTruth", "xyzVTX", "emiAng"]

        # Check which truth branches exist
        available_keys = tree.keys()
        truth_branches = [b for b in truth_branches if b in available_keys]

        arrays = tree.arrays(
            input_branches + truth_branches,
            library="np",
            entry_start=args.event_idx,
            entry_stop=args.event_idx + 1
        )

        raw_npho = arrays[args.npho_branch][0].astype("float32")
        raw_time = arrays[args.time_branch][0].astype("float32")

        # Extract truth info for display
        event_info = {}
        if "energyTruth" in arrays:
            event_info["energy"] = float(arrays["energyTruth"][0])
        if "xyzVTX" in arrays:
            xyz = arrays["xyzVTX"][0]
            event_info["xyz"] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        if "emiAng" in arrays:
            ang = arrays["emiAng"][0]
            event_info["theta"] = float(ang[0])
            event_info["phi"] = float(ang[1])

    # Normalize
    npho_norm, time_norm = normalize_input(
        raw_npho, raw_time,
        npho_scale=npho_scale,
        npho_scale2=npho_scale2,
        time_scale=time_scale,
        time_shift=time_shift,
        sentinel_value=sentinel_value
    )

    # x_truth: full normalized sensor values
    x_truth = np.stack([npho_norm, time_norm], axis=-1)  # (4760, 2)
    num_sensors = x_truth.shape[0]

    # Debug: Show overall event data distribution
    print(f"  Raw npho: non-zero={np.sum(raw_npho > 0)}/{len(raw_npho)}, min={raw_npho.min():.2f}, max={raw_npho.max():.2f}, mean={raw_npho.mean():.2f}")
    print(f"  Normalized npho: min={npho_norm.min():.4f}, max={npho_norm.max():.4f}, mean={npho_norm.mean():.4f}")

    # --- Load inpainter predictions ---
    print(f"Loading predictions from: {args.predictions}")
    with uproot.open(args.predictions) as f:
        pred_tree = f["tree"]

        # Check available branches
        available_branches = pred_tree.keys()
        base_branches = ["event_idx", "sensor_id", "face", "pred_npho", "pred_time", "truth_npho", "truth_time"]
        branches_to_load = [b for b in base_branches if b in available_branches]

        # Try to load run_number and event_number if available
        has_run_event = "run_number" in available_branches and "event_number" in available_branches
        if has_run_event:
            branches_to_load.extend(["run_number", "event_number"])

        all_arrays = pred_tree.arrays(branches_to_load, library="np")

        # Filter for this event
        event_mask = all_arrays["event_idx"] == args.event_idx
        n_total_masked = event_mask.sum()

        if n_total_masked == 0:
            print(f"Warning: No predictions found for event_idx={args.event_idx}")
            print("This could mean:")
            print("  - The event index doesn't exist in predictions file")
            print("  - No sensors were masked for this event")
            sys.exit(1)

        # Extract run/event numbers if available (take first value for this event)
        run_number = None
        event_number = None
        if has_run_event:
            run_numbers = all_arrays["run_number"][event_mask]
            event_numbers = all_arrays["event_number"][event_mask]
            if len(run_numbers) > 0:
                run_number = int(run_numbers[0])
                event_number = int(event_numbers[0])
                print(f"  Found run={run_number}, event={event_number}")

        # Face map: 0=inner, 1=us, 2=ds, 3=outer, 4=top, 5=bot
        # For outer face in finegrid mode, sensor_id is a grid index (h*W + w), NOT a flat sensor index
        # This causes collision with actual sensor indices from other faces (e.g., inner starts at 0)
        # We must exclude face=3 (outer) predictions to avoid corrupting the visualization
        face_ids = all_arrays["face"][event_mask]

        # Check if outer face predictions exist (face=3)
        n_outer = (face_ids == 3).sum()
        if n_outer > 0:
            print(f"  Note: Excluding {n_outer} outer face predictions (finegrid mode uses grid indices)")

        # Filter to only include faces with valid flat sensor indices (exclude outer=3)
        valid_face_mask = face_ids != 3
        sensor_ids = all_arrays["sensor_id"][event_mask][valid_face_mask]
        pred_npho = all_arrays["pred_npho"][event_mask][valid_face_mask]
        pred_time = all_arrays["pred_time"][event_mask][valid_face_mask]
        truth_npho_pred = all_arrays["truth_npho"][event_mask][valid_face_mask]
        truth_time_pred = all_arrays["truth_time"][event_mask][valid_face_mask]

        n_masked = len(sensor_ids)

        # Validate sensor_ids are in valid range
        invalid_ids = (sensor_ids < 0) | (sensor_ids >= num_sensors)
        if invalid_ids.any():
            print(f"  Warning: {invalid_ids.sum()} sensor_ids out of range [0, {num_sensors})")
            # Filter out invalid indices
            valid_idx_mask = ~invalid_ids
            sensor_ids = sensor_ids[valid_idx_mask]
            pred_npho = pred_npho[valid_idx_mask]
            pred_time = pred_time[valid_idx_mask]
            truth_npho_pred = truth_npho_pred[valid_idx_mask]
            truth_time_pred = truth_time_pred[valid_idx_mask]
            n_masked = len(sensor_ids)

    print(f"Event {args.event_idx}: {n_masked} valid masked sensors ({100*n_masked/num_sensors:.1f}%)")

    # Validate normalization consistency between original file and predictions file
    truth_from_original = x_truth[sensor_ids]
    npho_diff = np.abs(truth_from_original[:, 0] - truth_npho_pred).mean()
    time_diff = np.abs(truth_from_original[:, 1] - truth_time_pred).mean()
    if npho_diff > 0.05 or time_diff > 0.05:
        print(f"  Warning: Normalization mismatch detected!")
        print(f"    npho diff: {npho_diff:.4f}, time diff: {time_diff:.4f}")
        print(f"    This may indicate different normalization parameters were used.")
    elif npho_diff > 0.01 or time_diff > 0.01:
        print(f"  Note: Small normalization difference (npho: {npho_diff:.4f}, time: {time_diff:.4f})")
        print(f"    Consider regenerating predictions with: python macro/generate_test_predictions.py")

    # Debug: Show sample masked sensor values
    print(f"\n  === Debug: Masked sensor values (first 10) ===")
    print(f"  {'sensor_id':>10} | {'truth_npho':>12} | {'pred_npho':>12} | {'diff_npho':>12} | {'truth_time':>12} | {'pred_time':>12} | {'diff_time':>12}")
    print(f"  {'-'*10} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12} | {'-'*12}")
    for i in range(min(10, n_masked)):
        sid = sensor_ids[i]
        t_npho = truth_from_original[i, 0]
        p_npho = pred_npho[i]
        t_time = truth_from_original[i, 1]
        p_time = pred_time[i]
        print(f"  {sid:10d} | {t_npho:12.6f} | {p_npho:12.6f} | {p_npho - t_npho:+12.6f} | {t_time:12.6f} | {p_time:12.6f} | {p_time - t_time:+12.6f}")

    # Summary statistics
    print(f"\n  === Summary statistics ===")
    print(f"  Truth npho:  min={truth_from_original[:, 0].min():.4f}, max={truth_from_original[:, 0].max():.4f}, mean={truth_from_original[:, 0].mean():.4f}")
    print(f"  Pred npho:   min={pred_npho.min():.4f}, max={pred_npho.max():.4f}, mean={pred_npho.mean():.4f}")
    print(f"  Truth time:  min={truth_from_original[:, 1].min():.4f}, max={truth_from_original[:, 1].max():.4f}, mean={truth_from_original[:, 1].mean():.4f}")
    print(f"  Pred time:   min={pred_time.min():.4f}, max={pred_time.max():.4f}, mean={pred_time.mean():.4f}")

    # --- Reconstruct mask, x_masked, x_pred ---
    # mask: 1 where masked, 0 where visible
    mask = np.zeros(num_sensors, dtype=np.float32)
    mask[sensor_ids] = 1.0

    # x_masked: input with masked sensors set to sentinel
    x_masked = x_truth.copy()
    x_masked[sensor_ids, 0] = 0.0  # npho = 0 for masked
    x_masked[sensor_ids, 1] = sentinel_value  # time = sentinel for masked

    # x_pred: truth with masked sensors replaced by predictions
    x_pred = x_truth.copy()
    x_pred[sensor_ids, 0] = pred_npho
    x_pred[sensor_ids, 1] = pred_time

    # --- Plot ---
    # Build title with event features
    if run_number is not None and event_number is not None:
        title_parts = [f"Inpainter - Run {run_number} Event {event_number} (idx={args.event_idx})"]
    else:
        title_parts = [f"Inpainter - Event {args.event_idx}"]

    if "energy" in event_info:
        energy_mev = event_info['energy'] * 1000  # Convert GeV to MeV
        title_parts.append(f"E={energy_mev:.1f} MeV")
    if "xyz" in event_info:
        x, y, z = event_info["xyz"]
        title_parts.append(f"VTX=({x:.0f}, {y:.0f}, {z:.0f})")
    if "theta" in event_info and "phi" in event_info:
        title_parts.append(f"θ={event_info['theta']:.2f}, φ={event_info['phi']:.2f}")

    title = " | ".join(title_parts)

    savepath = args.save
    if savepath and "." not in os.path.basename(savepath):
        savepath = f"{savepath}.pdf"

    # Determine npho_threshold for visualization (convert from raw to normalized space)
    raw_threshold = args.npho_threshold if args.npho_threshold is not None else DEFAULT_NPHO_THRESHOLD
    npho_threshold_norm = np.log1p(raw_threshold / npho_scale) / npho_scale2

    plot_mae_comparison(
        x_truth,
        x_masked,
        mask,
        x_pred=x_pred,
        channel=args.channel,
        title=title,
        savepath=savepath,
        include_top_bottom=args.include_top_bottom,
        npho_threshold=npho_threshold_norm,
    )

    if savepath:
        print(f"Saved to: {savepath}")


if __name__ == "__main__":
    main()
