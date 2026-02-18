# Usage:
# python macro/show_inpainter_comparison.py 0 \
#     --predictions artifacts/inpainter/inpainter_predictions_epoch_50.root \
#     --original data/large_val.root \
#     --channel npho --save outputs/inpainter_event_0.pdf
#
# With directory as original (requires --run and --event):
# python macro/show_inpainter_comparison.py \
#     --predictions artifacts/inpainter/predictions_mc.root \
#     --original data/mc_samples/single_run/ \
#     --run 42 --event 1234 \
#     --channel npho
#
# Note: event_idx in predictions file corresponds to entry index in original file
# Note: Outer face predictions are included when using sensor-level mode (valid sensor IDs 4092-4759)
# Note: Normalization factors are read from predictions file metadata (if available)

import sys
import os
import argparse
import glob
import numpy as np
import uproot

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.event_display import plot_mae_comparison
    from lib.geom_defs import (
        DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
        DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME,
        DEFAULT_NPHO_THRESHOLD, OUTER_ALL_SENSOR_IDS
    )
    from lib.dataset import expand_path
    from lib.normalization import NphoTransform
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
                # Read npho_scheme (string type)
                if "npho_scheme" in meta_tree:
                    val = meta_tree["npho_scheme"].array(library="np")[0]
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    metadata["npho_scheme"] = str(val)
    except Exception as e:
        print(f"  Warning: Could not read metadata from predictions file: {e}")
    return metadata


def normalize_input(raw_npho, raw_time,
                    npho_scale=DEFAULT_NPHO_SCALE,
                    npho_scale2=DEFAULT_NPHO_SCALE2,
                    time_scale=DEFAULT_TIME_SCALE,
                    time_shift=DEFAULT_TIME_SHIFT,
                    sentinel_time=DEFAULT_SENTINEL_TIME,
                    sentinel_npho=-1.0,
                    npho_scheme="log1p"):
    """
    Apply the same normalization as training to raw input data.
    Returns normalized (npho, time) arrays.
    """
    # Identify bad values
    mask_npho_bad = (raw_npho <= 0.0) | (raw_npho > 9e9) | np.isnan(raw_npho)
    mask_time_bad = mask_npho_bad | (np.abs(raw_time) > 9e9) | np.isnan(raw_time)

    # Normalize npho using NphoTransform
    raw_npho_safe = np.where(mask_npho_bad, 0.0, raw_npho)
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
    npho_norm = npho_transform.forward(raw_npho_safe)

    # Normalize time
    time_norm = (raw_time / time_scale) - time_shift

    npho_norm[mask_npho_bad] = sentinel_npho
    time_norm[mask_time_bad] = sentinel_time

    return npho_norm, time_norm


def main():
    parser = argparse.ArgumentParser(
        description="Display inpainter truth/masked/prediction comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # By event index (single file)
  python macro/show_inpainter_comparison.py 0 \\
      --predictions artifacts/inpainter/inpainter_predictions_epoch_50.root \\
      --original data/large_val.root \\
      --channel npho

  # By run and event number (directory of files)
  python macro/show_inpainter_comparison.py \\
      --predictions artifacts/inpainter/predictions_mc.root \\
      --original data/mc_samples/single_run/ \\
      --run 42 --event 1234 \\
      --channel npho
        """
    )
    parser.add_argument("event_idx", type=int, nargs='?', default=None,
                        help="Event index (0-based). Not needed if --run and --event are specified.")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to inpainter predictions ROOT file")
    parser.add_argument("--original", type=str, required=True,
                        help="Path to original ROOT file or directory")
    parser.add_argument("--run", type=int, default=None,
                        help="Run number (required when --original is a directory)")
    parser.add_argument("--event", type=int, default=None,
                        help="Event number (used with --run to find specific event)")
    parser.add_argument("--tree", type=str, default="tree", help="TTree name in original file")
    parser.add_argument("--channel", type=str, choices=["npho", "time", "both"], default="npho")
    parser.add_argument("--include_top_bottom", action="store_true",
                        help="Include top/bottom hex faces in the comparison grid")
    parser.add_argument("--save", type=str, default=None, help="Save path (PDF recommended)")

    # Normalization parameters (will be loaded from predictions file if available)
    parser.add_argument("--npho_scale", type=float, default=None,
                        help="Override npho_scale (default: read from predictions file or use default)")
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_time", type=float, default=None)

    # Input branch names
    parser.add_argument("--npho_branch", type=str, default="npho")
    parser.add_argument("--time_branch", type=str, default="relative_time")

    # Time-invalid threshold (for visualization)
    parser.add_argument("--npho_threshold", type=float, default=None,
                        help=f"Npho threshold for time validity (raw scale, default: {DEFAULT_NPHO_THRESHOLD})")

    # Residual mode
    parser.add_argument("--relative_residual", action="store_true",
                        help="Show relative residual (pred-truth)/|truth| instead of absolute (pred-truth)")

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    if not os.path.exists(args.original):
        print(f"Error: Original file/directory not found: {args.original}")
        sys.exit(1)

    # --- Handle directory vs single file for --original ---
    is_directory = os.path.isdir(args.original)

    if is_directory:
        # Directory mode: require --run and --event
        if args.run is None or args.event is None:
            print("Error: When --original is a directory, --run and --event are required.")
            sys.exit(1)

        # Find ROOT files in directory
        file_list = expand_path(args.original)
        if not file_list:
            print(f"Error: No ROOT files found in directory: {args.original}")
            sys.exit(1)

        print(f"Directory mode: searching {len(file_list)} files for run={args.run}, event={args.event}")

        # Find the file containing the specified run number
        original_file = None
        event_idx_in_file = None

        for fpath in file_list:
            try:
                with uproot.open(fpath) as f:
                    if args.tree not in f:
                        continue
                    tree = f[args.tree]

                    # Check if file has run/event branches
                    available = tree.keys()
                    run_branch = None
                    event_branch = None
                    for rb in ["run", "run_number", "runNumber", "Info.run"]:
                        if rb in available:
                            run_branch = rb
                            break
                    for eb in ["event", "event_number", "eventNumber", "Info.event"]:
                        if eb in available:
                            event_branch = eb
                            break

                    if run_branch is None or event_branch is None:
                        # Try to infer run from filename (e.g., MCGamma_00042.root)
                        import re
                        match = re.search(r'(\d{5})\.root$', os.path.basename(fpath))
                        if match:
                            file_run = int(match.group(1))
                            if file_run == args.run:
                                # Found matching file, search by event only
                                if event_branch:
                                    events = tree[event_branch].array(library="np")
                                    event_matches = np.where(events == args.event)[0]
                                    if len(event_matches) > 0:
                                        original_file = fpath
                                        event_idx_in_file = int(event_matches[0])
                                        break
                        continue

                    # Read run/event arrays
                    runs = tree[run_branch].array(library="np")
                    events = tree[event_branch].array(library="np")

                    # Find matching entry
                    matches = np.where((runs == args.run) & (events == args.event))[0]
                    if len(matches) > 0:
                        original_file = fpath
                        event_idx_in_file = int(matches[0])
                        break
            except Exception as e:
                print(f"  Warning: Could not read {fpath}: {e}")
                continue

        if original_file is None:
            print(f"Error: Could not find run={args.run}, event={args.event} in any file")
            sys.exit(1)

        print(f"  Found in: {original_file} at index {event_idx_in_file}")
        args.original = original_file
        args.event_idx = event_idx_in_file
    else:
        # Single file mode: require event_idx
        if args.event_idx is None:
            if args.run is not None and args.event is not None:
                # Try to find event by run/event in single file
                print(f"Searching for run={args.run}, event={args.event} in {args.original}")
                with uproot.open(args.original) as f:
                    if args.tree not in f:
                        print(f"Error: Tree '{args.tree}' not found.")
                        sys.exit(1)
                    tree = f[args.tree]
                    available = tree.keys()

                    run_branch = None
                    event_branch = None
                    for rb in ["run", "run_number", "runNumber", "Info.run"]:
                        if rb in available:
                            run_branch = rb
                            break
                    for eb in ["event", "event_number", "eventNumber", "Info.event"]:
                        if eb in available:
                            event_branch = eb
                            break

                    if run_branch is None or event_branch is None:
                        print(f"Error: Could not find run/event branches in file")
                        print(f"  Available branches: {available}")
                        sys.exit(1)

                    runs = tree[run_branch].array(library="np")
                    events = tree[event_branch].array(library="np")
                    matches = np.where((runs == args.run) & (events == args.event))[0]

                    if len(matches) == 0:
                        print(f"Error: Could not find run={args.run}, event={args.event}")
                        sys.exit(1)

                    args.event_idx = int(matches[0])
                    print(f"  Found at index {args.event_idx}")
            else:
                print("Error: event_idx is required when --original is a single file.")
                print("  Alternatively, provide both --run and --event to search by run/event number.")
                sys.exit(1)

    # --- Load normalization metadata from predictions file ---
    print(f"Loading metadata from: {args.predictions}")
    metadata = load_metadata_from_predictions(args.predictions)

    # Use metadata values, CLI overrides, or defaults (in that order)
    npho_scale = args.npho_scale if args.npho_scale is not None else metadata.get("npho_scale", DEFAULT_NPHO_SCALE)
    npho_scale2 = args.npho_scale2 if args.npho_scale2 is not None else metadata.get("npho_scale2", DEFAULT_NPHO_SCALE2)
    time_scale = args.time_scale if args.time_scale is not None else metadata.get("time_scale", DEFAULT_TIME_SCALE)
    time_shift = args.time_shift if args.time_shift is not None else metadata.get("time_shift", DEFAULT_TIME_SHIFT)
    sentinel_time = args.sentinel_time if args.sentinel_time is not None else metadata.get("sentinel_value", DEFAULT_SENTINEL_TIME)
    npho_scheme = metadata.get("npho_scheme", "log1p")

    if metadata:
        print(f"  Using normalization from predictions file:")
        print(f"    npho_scale={npho_scale}, npho_scale2={npho_scale2}")
        print(f"    time_scale={time_scale}, time_shift={time_shift}")
        print(f"    sentinel_time={sentinel_time}, npho_scheme={npho_scheme}")
    else:
        print(f"  Warning: No metadata found in predictions file, using defaults")
        print(f"    npho_scale={npho_scale}, npho_scale2={npho_scale2}")
        print(f"    time_scale={time_scale}, time_shift={time_shift}")
        print(f"    sentinel_time={sentinel_time}, npho_scheme={npho_scheme}")

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
        sentinel_time=sentinel_time,
        npho_scheme=npho_scheme
    )

    # x_truth: full normalized sensor values
    x_truth = np.stack([npho_norm, time_norm], axis=-1)  # (4760, 2)
    num_sensors = x_truth.shape[0]

    # --- Load inpainter predictions ---
    print(f"Loading predictions from: {args.predictions}")
    with uproot.open(args.predictions) as f:
        # Try common tree names for predictions
        if "predictions" in f:
            pred_tree = f["predictions"]
        elif "tree" in f:
            pred_tree = f["tree"]
        else:
            available = [k.split(";")[0] for k in f.keys()]
            print(f"Error: No 'predictions' or 'tree' found. Available: {available}")
            sys.exit(1)

        # Check available branches
        available_branches = pred_tree.keys()
        base_branches = ["event_idx", "sensor_id", "face", "pred_npho", "pred_time", "truth_npho", "truth_time"]
        branches_to_load = [b for b in base_branches if b in available_branches]

        # Check which prediction channels are available (npho-only or npho+time)
        has_pred_npho = "pred_npho" in available_branches
        has_pred_time = "pred_time" in available_branches

        # Warn if user requested time but it's not available
        if args.channel == "time" and not has_pred_time:
            print(f"Error: Time channel requested but pred_time not found in predictions file.")
            print(f"  This model may have been trained with predict_channels: ['npho']")
            print(f"  Available prediction branches: {[k for k in available_branches if 'pred' in k.lower()]}")
            sys.exit(1)
        if args.channel == "both" and not has_pred_time:
            print(f"Warning: Time channel not available (npho-only model), plotting npho only.")
            args.channel = "npho"

        # Try to load run_number and event_number if available
        has_run_event = "run_number" in available_branches and "event_number" in available_branches
        if has_run_event:
            branches_to_load.extend(["run_number", "event_number"])

        all_arrays = pred_tree.arrays(branches_to_load, library="np")

        # Filter for this event - prefer run/event matching if available
        if args.run is not None and args.event is not None and has_run_event:
            # Match by run/event numbers
            event_mask = (all_arrays["run_number"] == args.run) & (all_arrays["event_number"] == args.event)
            match_desc = f"run={args.run}, event={args.event}"
        else:
            # Match by event_idx
            event_mask = all_arrays["event_idx"] == args.event_idx
            match_desc = f"event_idx={args.event_idx}"

        n_total_masked = event_mask.sum()

        if n_total_masked == 0:
            print(f"Warning: No predictions found for {match_desc}")
            print("This could mean:")
            print("  - The event doesn't exist in predictions file")
            print("  - No sensors were masked for this event")
            if args.run is not None and args.event is not None and not has_run_event:
                print("  - Predictions file doesn't have run_number/event_number branches")
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
        face_ids = all_arrays["face"][event_mask]
        sensor_ids_all = all_arrays["sensor_id"][event_mask]

        # Handle outer face (face=3) predictions:
        # - Sensor-level mode: sensor_id contains actual flat sensor IDs (4092-4759)
        # - Legacy grid-level mode: sensor_id is grid index (h*W + w), which collides with other faces
        # We check if outer face sensor IDs are in the valid outer sensor range
        outer_mask = face_ids == 3
        n_outer = outer_mask.sum()

        if n_outer > 0:
            outer_sensor_ids = sensor_ids_all[outer_mask]
            outer_min_valid = OUTER_ALL_SENSOR_IDS.min()
            outer_max_valid = OUTER_ALL_SENSOR_IDS.max()

            # Check if outer sensor IDs are in valid range
            outer_valid = (outer_sensor_ids >= outer_min_valid) & (outer_sensor_ids <= outer_max_valid)
            n_outer_valid = outer_valid.sum()

            if n_outer_valid == n_outer:
                print(f"  Found {n_outer} outer face predictions (sensor-level mode, IDs in {outer_min_valid}-{outer_max_valid})")
                valid_face_mask = np.ones(len(face_ids), dtype=bool)  # Include all faces
            elif n_outer_valid > 0:
                print(f"  Warning: Mixed outer face modes detected ({n_outer_valid}/{n_outer} valid)")
                # Include non-outer faces + valid outer predictions
                valid_face_mask = ~outer_mask | (outer_mask & np.isin(sensor_ids_all, OUTER_ALL_SENSOR_IDS))
            else:
                print(f"  Note: Excluding {n_outer} outer face predictions (legacy grid-level mode)")
                valid_face_mask = ~outer_mask
        else:
            valid_face_mask = np.ones(len(face_ids), dtype=bool)

        sensor_ids = sensor_ids_all[valid_face_mask]
        pred_npho = all_arrays["pred_npho"][event_mask][valid_face_mask]
        # Use predicted time if available, otherwise use truth time
        if has_pred_time:
            pred_time = all_arrays["pred_time"][event_mask][valid_face_mask]
        else:
            pred_time = all_arrays["truth_time"][event_mask][valid_face_mask]  # Placeholder
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

    if args.run is not None and args.event is not None:
        print(f"Run {args.run} Event {args.event} (idx={args.event_idx}): {n_masked} valid masked sensors ({100*n_masked/num_sensors:.1f}%)")
    else:
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

    # --- Reconstruct mask, x_masked, x_pred ---
    # mask: 1 where masked, 0 where visible
    mask = np.zeros(num_sensors, dtype=np.float32)
    mask[sensor_ids] = 1.0

    # x_masked: input with masked sensors set to sentinel
    x_masked = x_truth.copy()
    x_masked[sensor_ids, 0] = -1.0  # npho = sentinel_npho for masked
    x_masked[sensor_ids, 1] = sentinel_time  # time = sentinel for masked

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
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
    npho_threshold_norm = npho_transform.convert_threshold(raw_threshold)

    # Determine which channels to plot
    channels_to_plot = ["npho", "time"] if args.channel == "both" else [args.channel]

    for ch in channels_to_plot:
        # Handle save path for multiple channels
        if savepath and args.channel == "both":
            base, ext = os.path.splitext(savepath)
            ch_savepath = f"{base}_{ch}{ext}"
        else:
            ch_savepath = savepath

        plot_mae_comparison(
            x_truth,
            x_masked,
            mask,
            x_pred=x_pred,
            channel=ch,
            title=title,
            savepath=ch_savepath,
            include_top_bottom=args.include_top_bottom,
            npho_threshold=npho_threshold_norm,
            relative_residual=args.relative_residual,
        )

        if ch_savepath:
            print(f"Saved {ch} plot to: {ch_savepath}")


if __name__ == "__main__":
    main()
