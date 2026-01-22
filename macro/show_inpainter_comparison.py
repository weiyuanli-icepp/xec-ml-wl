# Usage:
# python macro/show_inpainter_comparison.py 0 \
#     --predictions artifacts/inpainter/inpainter_predictions_epoch_50.root \
#     --original data/large_val.root \
#     --channel npho --save outputs/inpainter_event_0.pdf
#
# Note: event_idx in predictions file corresponds to entry index in original file

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
        DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE
    )
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Run from repo root (xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)


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

    # Normalization parameters (should match training)
    parser.add_argument("--npho_scale", type=float, default=DEFAULT_NPHO_SCALE)
    parser.add_argument("--npho_scale2", type=float, default=DEFAULT_NPHO_SCALE2)
    parser.add_argument("--time_scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--time_shift", type=float, default=DEFAULT_TIME_SHIFT)
    parser.add_argument("--sentinel_value", type=float, default=DEFAULT_SENTINEL_VALUE)

    # Input branch names
    parser.add_argument("--npho_branch", type=str, default="relative_npho")
    parser.add_argument("--time_branch", type=str, default="relative_time")

    args = parser.parse_args()

    # Validate files exist
    if not os.path.exists(args.predictions):
        print(f"Error: Predictions file not found: {args.predictions}")
        sys.exit(1)
    if not os.path.exists(args.original):
        print(f"Error: Original file not found: {args.original}")
        sys.exit(1)

    # --- Load original data ---
    print(f"Loading original data from: {args.original}")
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
        npho_scale=args.npho_scale,
        npho_scale2=args.npho_scale2,
        time_scale=args.time_scale,
        time_shift=args.time_shift,
        sentinel_value=args.sentinel_value
    )

    # x_truth: full normalized sensor values
    x_truth = np.stack([npho_norm, time_norm], axis=-1)  # (4760, 2)
    num_sensors = x_truth.shape[0]

    # --- Load inpainter predictions ---
    print(f"Loading predictions from: {args.predictions}")
    with uproot.open(args.predictions) as f:
        pred_tree = f["tree"]

        # Load all predictions for this event
        all_arrays = pred_tree.arrays(
            ["event_idx", "sensor_id", "pred_npho", "pred_time"],
            library="np"
        )

        # Filter for this event
        event_mask = all_arrays["event_idx"] == args.event_idx
        n_masked = event_mask.sum()

        if n_masked == 0:
            print(f"Warning: No predictions found for event_idx={args.event_idx}")
            print("This could mean:")
            print("  - The event index doesn't exist in predictions file")
            print("  - No sensors were masked for this event")
            sys.exit(1)

        sensor_ids = all_arrays["sensor_id"][event_mask]
        pred_npho = all_arrays["pred_npho"][event_mask]
        pred_time = all_arrays["pred_time"][event_mask]

    print(f"Event {args.event_idx}: {n_masked} masked sensors ({100*n_masked/num_sensors:.1f}%)")

    # --- Reconstruct mask, x_masked, x_pred ---
    # mask: 1 where masked, 0 where visible
    mask = np.zeros(num_sensors, dtype=np.float32)
    mask[sensor_ids] = 1.0

    # x_masked: input with masked sensors set to sentinel
    x_masked = x_truth.copy()
    x_masked[sensor_ids, 0] = 0.0  # npho = 0 for masked
    x_masked[sensor_ids, 1] = args.sentinel_value  # time = sentinel for masked

    # x_pred: truth with masked sensors replaced by predictions
    x_pred = x_truth.copy()
    x_pred[sensor_ids, 0] = pred_npho
    x_pred[sensor_ids, 1] = pred_time

    # --- Plot ---
    # Build title with event features
    title_parts = [f"Inpainter - Event {args.event_idx}"]

    if "energy" in event_info:
        title_parts.append(f"E={event_info['energy']:.1f} MeV")
    if "xyz" in event_info:
        x, y, z = event_info["xyz"]
        title_parts.append(f"VTX=({x:.0f}, {y:.0f}, {z:.0f})")
    if "theta" in event_info and "phi" in event_info:
        title_parts.append(f"θ={event_info['theta']:.2f}, φ={event_info['phi']:.2f}")

    title = " | ".join(title_parts)

    savepath = args.save
    if savepath and "." not in os.path.basename(savepath):
        savepath = f"{savepath}.pdf"

    plot_mae_comparison(
        x_truth,
        x_masked,
        mask,
        x_pred=x_pred,
        channel=args.channel,
        title=title,
        savepath=savepath,
        include_top_bottom=args.include_top_bottom,
    )

    if savepath:
        print(f"Saved to: {savepath}")


if __name__ == "__main__":
    main()
