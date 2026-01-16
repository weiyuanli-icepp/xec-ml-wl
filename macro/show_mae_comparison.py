# Usage:
# python macro/show_mae_comparison.py 0 --file artifacts/<RUN_NAME>/mae_predictions_epoch_1.root --channel npho --save outputs/event_0.pdf
import sys
import os
import argparse
import numpy as np
import uproot

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.event_display import plot_mae_comparison
except ImportError:
    print("Error: Could not import 'plot_mae_comparison'.")
    print("Run from repo root (xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)


def _decode_run_id(val):
    if isinstance(val, bytes):
        return val.decode(errors="ignore")
    return str(val)


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
    parser.add_argument("--save", type=str, default=None, help="Save path (PDF recommended)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        sys.exit(1)

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
        has_pred = "pred_npho" in tree.keys() and "pred_time" in tree.keys()
        if has_pred:
            branches += ["pred_npho", "pred_time"]
        has_run_id = "run_id" in tree.keys()
        if has_run_id:
            branches.append("run_id")

        arrays = tree.arrays(branches, library="np",
                             entry_start=args.event_id, entry_stop=args.event_id + 1)

        truth_npho = arrays["truth_npho"][0]
        truth_time = arrays["truth_time"][0]
        masked_npho = arrays["masked_npho"][0]
        masked_time = arrays["masked_time"][0]
        mask = arrays["mask"][0]

        if has_pred:
            pred_npho = arrays["pred_npho"][0]
            pred_time = arrays["pred_time"][0]
            x_pred = np.stack([pred_npho, pred_time], axis=-1)
        else:
            x_pred = None

        x_truth = np.stack([truth_npho, truth_time], axis=-1)
        x_masked = np.stack([masked_npho, masked_time], axis=-1)

        title_parts = [f"Entry {args.event_id}"]
        if has_run_id:
            run_id_val = _decode_run_id(arrays["run_id"][0])
            title_parts.append(f"run_id {run_id_val}")
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
        )


if __name__ == "__main__":
    main()
