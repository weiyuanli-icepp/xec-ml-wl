# Usage:
# python macro/show_mae_masked_err.py --file artifacts/<RUN_NAME>/mae_predictions_epoch_10.root --channel npho --mode masked --bins 200
import sys
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.geom_defs import (
        INNER_INDEX_MAP,
        US_INDEX_MAP,
        DS_INDEX_MAP,
        OUTER_COARSE_FULL_INDEX_MAP,
        OUTER_CENTER_INDEX_MAP,
        TOP_HEX_ROWS,
        BOTTOM_HEX_ROWS,
        flatten_hex_rows,
    )
except ImportError:
    print("Error: Could not import geometry helpers.")
    print("Run from repo root (xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)


def _face_indices(name):
    if name == "inner":
        return INNER_INDEX_MAP.reshape(-1)
    if name == "us":
        return US_INDEX_MAP.reshape(-1)
    if name == "ds":
        return DS_INDEX_MAP.reshape(-1)
    if name == "outer":
        return OUTER_COARSE_FULL_INDEX_MAP.reshape(-1)
    if name == "outer_center":
        return OUTER_CENTER_INDEX_MAP.reshape(-1)
    if name == "outer_union":
        coarse = OUTER_COARSE_FULL_INDEX_MAP.reshape(-1)
        center = OUTER_CENTER_INDEX_MAP.reshape(-1)
        return np.unique(np.concatenate([coarse, center]))
    if name == "top":
        return flatten_hex_rows(TOP_HEX_ROWS)
    if name == "bottom":
        return flatten_hex_rows(BOTTOM_HEX_ROWS)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Plot err_npho/err_time for masked/visible channels from MAE predictions ROOT."
    )
    parser.add_argument("--file", type=str, required=True, help="Path to mae_predictions_epoch_*.root")
    parser.add_argument("--tree", type=str, default="tree", help="TTree name")
    parser.add_argument("--channel", type=str, choices=["npho", "time"], default="npho")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["masked", "visible", "all"],
        default="masked",
        help="Use masked channels, visible channels, or all channels.",
    )
    parser.add_argument(
        "--face",
        type=str,
        choices=["all", "inner", "us", "ds", "outer", "outer_center", "outer_union", "top", "bottom"],
        default="all",
        help="Restrict to a detector face (default: all).",
    )
    parser.add_argument("--event", type=int, default=None, help="Single event index (0-based)")
    parser.add_argument("--max_events", type=int, default=None, help="Max events to read")
    parser.add_argument("--bins", type=int, default=200, help="Histogram bins")
    parser.add_argument("--range", type=float, nargs=2, default=None, help="Histogram range min max")
    parser.add_argument("--abs", dest="use_abs", action="store_true", help="Plot |err| instead of err")
    parser.add_argument("--save", type=str, default=None, help="Save figure path (e.g., plots/err.pdf)")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        sys.exit(1)

    err_branch = f"err_{args.channel}"
    pred_branch = f"pred_{args.channel}"
    truth_branch = f"truth_{args.channel}"

    with uproot.open(args.file) as f:
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found. Available: {f.keys()}")
            sys.exit(1)

        tree = f[args.tree]
        keys = set(tree.keys())

        if err_branch in keys:
            branches = [err_branch, "mask"]
        elif pred_branch in keys and truth_branch in keys:
            branches = [pred_branch, truth_branch, "mask"]
        else:
            print(f"Error: Missing '{err_branch}' or prediction/truth branches.")
            print(f"Available branches: {sorted(keys)}")
            sys.exit(1)

        entry_start = None
        entry_stop = None
        if args.event is not None:
            entry_start = args.event
            entry_stop = args.event + 1
        elif args.max_events is not None:
            entry_stop = args.max_events

        arrays = tree.arrays(branches, library="np", entry_start=entry_start, entry_stop=entry_stop)

    mask = arrays["mask"].astype("float32")
    if err_branch in arrays:
        err = arrays[err_branch].astype("float32")
    else:
        err = (arrays[pred_branch] - arrays[truth_branch]).astype("float32")

    idx = _face_indices(args.face)
    if idx is not None:
        err = err[:, idx]
        mask = mask[:, idx]

    if args.mode == "masked":
        sel = mask > 0.5
    elif args.mode == "visible":
        sel = mask <= 0.5
    else:
        sel = np.ones_like(mask, dtype=bool)

    values = err[sel]
    if args.use_abs:
        values = np.abs(values)

    if values.size == 0:
        print("No values selected (check mode/face or mask content).")
        sys.exit(1)

    mean = float(np.mean(values))
    rms = float(np.sqrt(np.mean(values * values)))
    count = int(values.size)

    title_bits = [
        f"err_{args.channel}",
        f"mode={args.mode}",
        f"face={args.face}",
        f"n={count}",
        f"mean={mean:.4g}",
        f"rms={rms:.4g}",
    ]
    if args.event is not None:
        title_bits.append(f"event={args.event}")

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=args.bins, range=args.range, histtype="step", linewidth=1.5)
    plt.title(" | ".join(title_bits))
    plt.xlabel(f"err_{args.channel}" + (" (abs)" if args.use_abs else ""))
    plt.ylabel("count")
    plt.grid(alpha=0.2)

    if args.save:
        plt.savefig(args.save, bbox_inches="tight", dpi=120)
    else:
        plt.show()


if __name__ == "__main__":
    main()
