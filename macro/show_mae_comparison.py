# Usage:
# python macro/show_mae_comparison.py 0 --file artifacts/<RUN_NAME>/mae_predictions_epoch_1.root --channel npho --include_top_bottom --save outputs/event_0.pdf
import sys
import os
import argparse
import numpy as np
import uproot

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.event_display import plot_mae_comparison
    from lib.geom_defs import (
        DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2, DEFAULT_NPHO_THRESHOLD
    )
    from lib.normalization import NphoTransform
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Run from repo root (xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)


def _decode_run_id(val):
    if isinstance(val, bytes):
        return val.decode(errors="ignore")
    return str(val)


def load_metadata_from_predictions(pred_file):
    """
    Load normalization metadata from MAE predictions file if available.
    Returns dict with normalization factors, or empty dict if not found.
    """
    metadata = {}
    try:
        with uproot.open(pred_file) as f:
            if "metadata" in f:
                meta_tree = f["metadata"]
                for key in ["npho_scale", "npho_scale2", "time_scale", "time_shift"]:
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


def main():
    parser = argparse.ArgumentParser(description="Display MAE truth/masked/prediction comparison from ROOT")
    parser.add_argument("event_id", type=int, help="Entry index (0-based)")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to MAE predictions ROOT file (mae_predictions_epoch_*.root)",
    )
    parser.add_argument("--tree", type=str, default="tree", help="TTree name")
    parser.add_argument("--channel", type=str, choices=["npho", "time"], default="npho")
    parser.add_argument("--include_top_bottom", action="store_true",
                        help="Include top/bottom hex faces in the comparison grid")
    parser.add_argument("--save", type=str, default=None, help="Save path (PDF recommended)")
    parser.add_argument("--npho_threshold", type=float, default=None,
                        help=f"Npho threshold for time validity (raw scale, default: {DEFAULT_NPHO_THRESHOLD})")
    parser.add_argument("--relative_residual", action="store_true",
                        help="Show relative residual (pred-truth)/|truth| instead of absolute (pred-truth)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        sys.exit(1)

    # Load metadata from predictions file
    metadata = load_metadata_from_predictions(args.file)
    npho_scale = metadata.get("npho_scale", DEFAULT_NPHO_SCALE)
    npho_scale2 = metadata.get("npho_scale2", DEFAULT_NPHO_SCALE2)
    npho_scheme = metadata.get("npho_scheme", "log1p")

    if metadata:
        print(f"  Using normalization from predictions file:")
        print(f"    npho_scale={npho_scale}, npho_scale2={npho_scale2}, npho_scheme={npho_scheme}")

    with uproot.open(args.file) as f:
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found. Available: {f.keys()}")
            sys.exit(1)

        tree = f[args.tree]
        n_entries = tree.num_entries
        if args.event_id < 0 or args.event_id >= n_entries:
            print(f"Error: Index {args.event_id} out of bounds (max {n_entries - 1})")
            sys.exit(1)

        branches = ["truth_npho", "truth_time", "mask", "masked_npho", "masked_time"]

        # Check which prediction channels are available (npho-only or npho+time)
        has_pred_npho = "pred_npho" in tree.keys()
        has_pred_time = "pred_time" in tree.keys()
        has_pred = has_pred_npho  # At least npho must be present

        if has_pred_npho:
            branches.append("pred_npho")
        if has_pred_time:
            branches.append("pred_time")

        # Warn if user requested time but it's not available
        if args.channel == "time" and not has_pred_time:
            print(f"Error: Time channel requested but pred_time not found in file.")
            print(f"  This model may have been trained with predict_channels: ['npho']")
            print(f"  Available prediction branches: {[k for k in tree.keys() if 'pred' in k.lower()]}")
            sys.exit(1)

        # Check for metadata branches (similar to inpainter)
        available_keys = tree.keys()
        has_run_id = "run_id" in available_keys
        has_run_number = "run_number" in available_keys
        has_event_number = "event_number" in available_keys
        has_energy = "energyTruth" in available_keys or "energy" in available_keys
        has_position = "xyzVTX" in available_keys
        has_angle = "emiAng" in available_keys

        if has_run_id:
            branches.append("run_id")
        if has_run_number:
            branches.append("run_number")
        if has_event_number:
            branches.append("event_number")
        if "energyTruth" in available_keys:
            branches.append("energyTruth")
        elif "energy" in available_keys:
            branches.append("energy")
        if has_position:
            branches.append("xyzVTX")
        if has_angle:
            branches.append("emiAng")

        arrays = tree.arrays(branches, library="np",
                             entry_start=args.event_id, entry_stop=args.event_id + 1)

        truth_npho = arrays["truth_npho"][0]
        truth_time = arrays["truth_time"][0]
        masked_npho = arrays["masked_npho"][0]
        masked_time = arrays["masked_time"][0]
        mask = arrays["mask"][0]

        if has_pred:
            pred_npho = arrays["pred_npho"][0]
            # Use predicted time if available, otherwise use truth time
            if has_pred_time:
                pred_time = arrays["pred_time"][0]
            else:
                pred_time = arrays["truth_time"][0]  # Use truth time as placeholder
            x_pred = np.stack([pred_npho, pred_time], axis=-1)
        else:
            x_pred = None

        x_truth = np.stack([truth_npho, truth_time], axis=-1)
        x_masked = np.stack([masked_npho, masked_time], axis=-1)

        # Build title similar to inpainter comparison
        # Try to get run/event numbers first
        run_number = None
        event_number = None
        if has_run_number and has_event_number:
            run_number = int(arrays["run_number"][0])
            event_number = int(arrays["event_number"][0])
            title_parts = [f"MAE - Run {run_number} Event {event_number} (idx={args.event_id})"]
        elif has_run_id:
            run_id_val = _decode_run_id(arrays["run_id"][0])
            title_parts = [f"MAE - Entry {args.event_id} | run_id {run_id_val}"]
        else:
            title_parts = [f"MAE - Entry {args.event_id}"]

        # Add energy info
        if "energyTruth" in arrays:
            energy_mev = float(arrays["energyTruth"][0]) * 1000  # GeV to MeV
            title_parts.append(f"E={energy_mev:.1f} MeV")
        elif "energy" in arrays:
            energy_mev = float(arrays["energy"][0]) * 1000
            title_parts.append(f"E={energy_mev:.1f} MeV")

        # Add position info
        if has_position:
            xyz = arrays["xyzVTX"][0]
            title_parts.append(f"VTX=({float(xyz[0]):.0f}, {float(xyz[1]):.0f}, {float(xyz[2]):.0f})")

        # Add angle info
        if has_angle:
            ang = arrays["emiAng"][0]
            title_parts.append(f"θ={float(ang[0]):.2f}, φ={float(ang[1]):.2f}")
        title = " | ".join(title_parts)

        savepath = args.save
        if savepath and "." not in os.path.basename(savepath):
            savepath = f"{savepath}.pdf"

        # Determine npho_threshold for visualization (convert from raw to normalized space)
        raw_threshold = args.npho_threshold if args.npho_threshold is not None else DEFAULT_NPHO_THRESHOLD
        npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
        npho_threshold_norm = npho_transform.convert_threshold(raw_threshold)

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
            relative_residual=args.relative_residual,
        )


if __name__ == "__main__":
    main()
