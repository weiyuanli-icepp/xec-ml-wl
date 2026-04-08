#!/usr/bin/env python3
"""
Apples-to-apples identity check: our SolidAngleWeightedBaseline vs
meganalyzer's xecenepmweight.npho dead-channel recovery.

Unlike check_sa_consistency.py, this script reads ONLY the meganalyzer rec
file and uses exactly the same inputs meganalyzer used:
  - Raw npho from xeccl.npho at gamma 0
  - Reconstructed position from xecposlfit.xyz at gamma 0
  - Per-event dead mask from xecenepmweight.nphorecovered == 1

This bypasses PrepareRealData.C, compute_inpainter_baselines.py, and the
static dead-channel database entirely. If the algorithms are identical,
residuals should collapse to float32 precision.

Usage:
    python macro/check_sa_identity.py \\
        --rec /data/project/meg/offline/run/559xxx/rec559261.root \\
        --output sa_identity_559261.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import awkward as ak
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.inpainter_baselines import SolidAngleWeightedBaseline
from lib.solid_angle import SolidAngleComputer
from lib.geom_defs import FACE_SENSOR_IDS


N_SENSORS = 4760
KXEC_INVALID_ABS = 900.0  # cm; guard for ±1000 cm kXECInvalid* sentinel

# Face order for display
FACE_ORDER = ["inner", "outer", "us", "ds", "top", "bot"]


def _sensor_to_face_map():
    """Build (4760,) array mapping sensor_id → face name."""
    face_of = np.full(N_SENSORS, "unknown", dtype=object)
    for face_name in FACE_ORDER:
        if face_name in FACE_SENSOR_IDS:
            face_of[FACE_SENSOR_IDS[face_name].astype(int)] = face_name
    return face_of


# ---------------------------------------------------------------------------
# Branch discovery helpers
# ---------------------------------------------------------------------------

def _find_branch(tree, candidates):
    """Return the first branch name from candidates that exists in tree."""
    keys = set(tree.keys())
    for c in candidates:
        if c in keys:
            return c
    return None


# ---------------------------------------------------------------------------
# Vectorization helpers
# ---------------------------------------------------------------------------

def vectorize_per_pm(raw_ak, fill_value, dtype):
    """Collapse (N, var PMs, var gammas) → (N, 4760) by taking gamma 0."""
    first = ak.firsts(raw_ak, axis=-1)          # (N, var PMs)
    padded = ak.pad_none(first, N_SENSORS, clip=True, axis=1)
    return ak.to_numpy(ak.fill_none(padded, fill_value)).astype(dtype)


def vectorize_gamma0_xyz(raw_ak):
    """Collapse xecposlfit.xyz from (N, var nGamma, 3) → (N, 3) float64.

    Events with no gammas get NaN rows (filtered downstream).
    """
    # Take first gamma of each event
    first_gamma = ak.firsts(raw_ak, axis=1)     # (N, var 3) or None
    # Pad to exactly 3 components
    padded = ak.pad_none(first_gamma, 3, clip=True, axis=1)
    return ak.to_numpy(ak.fill_none(padded, np.nan)).astype(np.float64)


# ---------------------------------------------------------------------------
# Rec file reader
# ---------------------------------------------------------------------------

def load_rec(rec_path, max_events=None):
    """Read all required branches from meganalyzer rec file.

    Returns a dict with:
      run (N,), event (N,), xyz (N, 3),
      x_npho (N, 4760), meg_pred (N, 4760), recovered (N, 4760)
    """
    print(f"[INFO] Loading rec file: {rec_path}")
    with uproot.open(rec_path) as f:
        # Find the tree
        tree = None
        for name in ("rec", "rec;1"):
            if name in f:
                tree = f[name]
                break
        if tree is None:
            raise RuntimeError(f"'rec' tree not found in {rec_path}")

        print(f"[INFO] Tree has {tree.num_entries} events")

        # --- branch discovery ---
        xccl_key = _find_branch(tree, [
            "xeccl/xeccl.npho", "xeccl.npho",
        ])
        meg_key = _find_branch(tree, [
            "xecenepmweight/xecenepmweight.npho",
            "xecenepmweight./xecenepmweight.npho",
            "xecenepmweight.npho",
        ])
        rec_key = _find_branch(tree, [
            "xecenepmweight/xecenepmweight.nphorecovered",
            "xecenepmweight./xecenepmweight.nphorecovered",
            "xecenepmweight.nphorecovered",
        ])
        xyz_key = _find_branch(tree, [
            "xecposlfit/xecposlfit.xyz[3]",
            "xecposlfit/xecposlfit.xyz",
            "xecposlfit./xecposlfit.xyz",
            "xecposlfit.xyz[3]",
            "xecposlfit.xyz",
        ])

        # Fallback discovery by suffix
        def _suffix_scan(needle_branch, suffix):
            for k in tree.keys():
                if needle_branch in k and k.endswith(suffix):
                    return k
            return None

        if xccl_key is None:
            xccl_key = _suffix_scan("xeccl", ".npho")
        if meg_key is None:
            meg_key = _suffix_scan("xecenepmweight", ".npho")
        if rec_key is None:
            rec_key = _suffix_scan("xecenepmweight", ".nphorecovered")
        if xyz_key is None:
            for k in tree.keys():
                if "xecposlfit" in k and (k.endswith(".xyz") or k.endswith(".xyz[3]")):
                    xyz_key = k
                    break

        # Discover run/event branches
        run_key = None
        event_key = None
        for k in tree.keys():
            kl = k.lower()
            if run_key is None and (
                "runnumber" in kl or kl.endswith(".run") or kl.endswith("/run")
            ):
                run_key = k
            if event_key is None and (
                "eventnumber" in kl or kl.endswith(".event") or kl.endswith("/event")
            ):
                event_key = k

        # Report and validate
        branches = {
            "xeccl.npho": xccl_key,
            "xecenepmweight.npho": meg_key,
            "nphorecovered": rec_key,
            "xecposlfit.xyz": xyz_key,
            "run": run_key,
            "event": event_key,
        }
        for name, key in branches.items():
            print(f"[INFO] {name:>22s}: {key}")
            if key is None:
                # Print available xecposlfit branches for debugging
                if name == "xecposlfit.xyz":
                    print("[DEBUG] Available xecposlfit branches:")
                    for k in tree.keys():
                        if "xecposlfit" in k:
                            print(f"   {k}")
                raise RuntimeError(f"Could not find '{name}' branch")

        entry_stop = max_events if max_events else None

        # --- read ---
        print("[INFO] Reading branches...")
        run_arr = tree[run_key].array(library="np", entry_stop=entry_stop)
        event_arr = tree[event_key].array(library="np", entry_stop=entry_stop)

        x_raw = tree[xccl_key].array(library="ak", entry_stop=entry_stop)
        meg_raw = tree[meg_key].array(library="ak", entry_stop=entry_stop)
        rec_raw = tree[rec_key].array(library="ak", entry_stop=entry_stop)
        xyz_raw = tree[xyz_key].array(library="ak", entry_stop=entry_stop)

    # --- vectorize ---
    print("[INFO] Vectorizing to (N, 4760) arrays...")
    x_npho = vectorize_per_pm(x_raw, np.nan, np.float32)
    meg_pred = vectorize_per_pm(meg_raw, np.nan, np.float32)
    recovered = vectorize_per_pm(rec_raw, 0, np.int8)
    xyz = vectorize_gamma0_xyz(xyz_raw)

    n = len(run_arr)
    print(f"[INFO] Loaded {n} events, "
          f"{int((recovered == 1).sum())} total recovered (PM,event) pairs")

    return {
        "run": np.asarray(run_arr, dtype=np.int64),
        "event": np.asarray(event_arr, dtype=np.int64),
        "xyz": xyz,
        "x_npho": x_npho,
        "meg_pred": meg_pred,
        "recovered": recovered,
    }


# ---------------------------------------------------------------------------
# Event filter
# ---------------------------------------------------------------------------

def filter_valid_events(data):
    """Drop events with missing or sentinel xyz (meganalyzer skips these)."""
    xyz = data["xyz"]
    finite = np.all(np.isfinite(xyz), axis=1)
    sane = np.all(np.abs(xyz) < KXEC_INVALID_ABS, axis=1)
    keep = finite & sane
    n_drop = int((~keep).sum())
    print(f"[INFO] Filtering events: {int(keep.sum())} kept, "
          f"{n_drop} dropped (invalid/missing gamma-0 position)")
    return {k: v[keep] for k, v in data.items()}


# ---------------------------------------------------------------------------
# Run our SA-wt baseline
# ---------------------------------------------------------------------------

def compute_our_predictions(data, distance_threshold, npho_threshold,
                             geom_dump_path=None):
    """Run SolidAngleWeightedBaseline on the rec-file-derived inputs.

    If geom_dump_path is provided, loads the static IsBad flag from it
    and uses it as the neighbor-exclusion mask (meganalyzer's semantics).
    Otherwise falls back to using the recovery mask as the exclusion.
    """
    print("[INFO] Computing solid angles from per-event xecposlfit.xyz ...")
    sac = SolidAngleComputer()  # auto-loads lib/sensor_directions.txt
    if not sac.geometry_loaded:
        raise RuntimeError(
            "SolidAngleComputer could not load geometry from "
            "lib/sensor_directions.txt"
        )
    omega = sac.compute(data["xyz"].astype(np.float64))  # (N, 4760) fractional
    if omega is None:
        raise RuntimeError("SolidAngleComputer.compute() returned None")

    # Dead-channel mask: meganalyzer's own per-event flag
    mask = (data["recovered"] == 1)

    # Load static IsBad for neighbor exclusion (meganalyzer's semantics)
    neighbor_exclude_mask = None
    if geom_dump_path is not None:
        print(f"[INFO] Loading static IsBad flag from {geom_dump_path}")
        gd = np.loadtxt(geom_dump_path, comments="#")
        if gd.shape[1] < 4:
            raise RuntimeError(
                f"Geometry dump {geom_dump_path} missing is_bad column; "
                "re-dump with latest dump_xec_geometry.C"
            )
        sensor_ids = gd[:, 0].astype(int)
        is_bad = gd[:, 2].astype(bool)  # column 2 = is_bad
        is_bad_arr = np.zeros(N_SENSORS, dtype=bool)
        is_bad_arr[sensor_ids] = is_bad
        neighbor_exclude_mask = is_bad_arr  # (4760,) broadcast per event
        n_is_bad = int(is_bad_arr.sum())
        n_mask_total = int(mask.sum())
        n_mask_per_event = mask.any(axis=0).sum()
        print(f"[INFO] Static IsBad count            : {n_is_bad}")
        print(f"[INFO] Union of recovered (any ev)   : {int(n_mask_per_event)}")
        print(f"[INFO] Total recovered (PM,event)    : {n_mask_total}")

    # Replace any residual NaNs in x_npho at padded positions (those positions
    # won't be in any real neighbor list anyway, but avoid NaN propagation)
    x_npho = np.nan_to_num(data["x_npho"], nan=0.0, posinf=0.0, neginf=0.0)

    print(
        f"[INFO] Running SolidAngleWeightedBaseline.predict() "
        f"(distance={distance_threshold}, npho_thr={npho_threshold})"
    )
    baseline = SolidAngleWeightedBaseline(
        distance_threshold=distance_threshold,
        npho_threshold=npho_threshold,
    )
    predictions = baseline.predict(
        x_npho=x_npho,
        mask=mask,
        solid_angles=omega,
        npho_transform=None,  # raw-photon space
        neighbor_exclude_mask=neighbor_exclude_mask,
    )
    return predictions, mask, omega


# ---------------------------------------------------------------------------
# Compare & stats
# ---------------------------------------------------------------------------

def compare(data, our_pred, mask):
    """Extract per-dead-channel comparison arrays."""
    meg_pred = data["meg_pred"]
    valid = mask & np.isfinite(our_pred) & np.isfinite(meg_pred)
    ev_idx, sid_idx = np.where(valid)

    our = our_pred[ev_idx, sid_idx].astype(np.float64)
    meg = meg_pred[ev_idx, sid_idx].astype(np.float64)
    diff = our - meg
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = diff / np.where(meg != 0, meg, 1.0)

    # Map each sensor_id to its face name
    face_map = _sensor_to_face_map()
    face = face_map[sid_idx]

    return {
        "n": len(our),
        "our": our,
        "meg": meg,
        "diff": diff,
        "rel": rel,
        "ev_idx": ev_idx,
        "sid_idx": sid_idx,
        "face": face,
    }


def print_statistics(stats, max_print):
    n = stats["n"]
    if n == 0:
        print("\n[WARN] Nothing to compare — rec file had no recovered channels")
        return

    our = stats["our"]
    meg = stats["meg"]
    diff = stats["diff"]
    rel = stats["rel"]

    print("\n" + "=" * 60)
    print("Identity Check Summary (all entries)")
    print("=" * 60)
    print(f"  N compared             : {n:,}")
    print(f"  Our pred [1,50,99]%    : "
          f"{np.percentile(our, [1, 50, 99])}")
    print(f"  Meg pred [1,50,99]%    : "
          f"{np.percentile(meg, [1, 50, 99])}")
    print(f"  |diff|   median / p99  : "
          f"{np.median(np.abs(diff)):.6g} / "
          f"{np.percentile(np.abs(diff), 99):.6g}")
    print(f"  |rel|    median / p99  : "
          f"{np.median(np.abs(rel)):.3%} / "
          f"{np.percentile(np.abs(rel), 99):.3%}")

    # Exact agreement buckets
    abs_rel = np.abs(rel)
    print("\n  Fraction with |rel| below:")
    for thr in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2):
        frac = float((abs_rel < thr).mean())
        print(f"    |rel| < {thr:<7g}    : {frac:.4%}")

    # Filtered stats (exclude outliers)
    sane = (np.abs(meg) < 1e5) & (np.abs(our) < 1e5)
    if sane.any():
        print(f"\n  Filtered |pred|<1e5 (N={int(sane.sum()):,}):")
        print(f"    diff  mean±std       : "
              f"{diff[sane].mean():+.6g} ± {diff[sane].std():.6g}")
        print(f"    rel   mean±std       : "
              f"{rel[sane].mean():+.3%} ± {rel[sane].std():.3%}")

    # Top-N worst
    face = stats["face"]
    k = min(max_print, n)
    worst = np.argsort(np.abs(diff))[::-1][:k]
    print(f"\n  Top {k} largest |diff|:")
    print(f"  {'ev':>6s} {'sid':>5s} {'face':>6s} {'our':>14s} {'meg':>14s} "
          f"{'diff':>14s} {'rel':>10s}")
    for i in worst:
        print(f"  {stats['ev_idx'][i]:>6d} {stats['sid_idx'][i]:>5d} "
              f"{str(face[i]):>6s} "
              f"{our[i]:>14.4g} {meg[i]:>14.4g} "
              f"{diff[i]:>+14.4g} {rel[i]:>+10.2%}")

    # --- Per-face summary (always print all faces, even if empty) ---
    print("\n" + "=" * 60)
    print("Per-Face Summary")
    print("=" * 60)
    header = (f"  {'Face':<8s} {'N':>9s} {'|diff|med':>11s} {'|rel|med':>10s} "
              f"{'<1e-6':>8s} {'<1e-4':>8s} {'<1e-2':>8s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for face_name in FACE_ORDER + ["unknown"]:
        sel = (face == face_name)
        nf = int(sel.sum())
        if nf == 0:
            print(f"  {face_name:<8s} {nf:>9,d} "
                  f"{'—':>11s} {'—':>10s} {'—':>8s} {'—':>8s} {'—':>8s}")
            continue
        fdiff = np.abs(diff[sel])
        frel = np.abs(rel[sel])
        med_d = np.median(fdiff)
        med_r = np.median(frel)
        f6 = float((frel < 1e-6).mean())
        f4 = float((frel < 1e-4).mean())
        f2 = float((frel < 1e-2).mean())
        print(f"  {face_name:<8s} {nf:>9,d} {med_d:>11.4g} {med_r:>9.3%} "
              f"{f6:>8.2%} {f4:>8.2%} {f2:>8.2%}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _plot_4panel(ax_grid, our, meg, diff, rel, title_suffix=""):
    """Fill a 2x2 axes grid with the standard 4-panel comparison."""
    # (0,0): linear scatter
    ax = ax_grid[0, 0]
    if len(meg) > 0:
        ax.scatter(meg, our, s=3, alpha=0.3, color="steelblue")
        lims = [min(meg.min(), our.min()), max(meg.max(), our.max())]
        ax.plot(lims, lims, "r--", lw=1, label="y = x")
        ax.set_aspect("equal", "box")
    ax.set_xlabel("Meganalyzer SA prediction [npho]")
    ax.set_ylabel("Our SA prediction [npho]")
    ax.set_title(f"Scatter (N={len(our):,}){title_suffix}")
    ax.legend(fontsize=8)

    # (0,1): absolute diff histogram
    ax = ax_grid[0, 1]
    if len(diff) > 0:
        lim = max(np.percentile(np.abs(diff), 99), 1e-6)
        ax.hist(diff, bins=100, range=(-lim, lim), color="steelblue",
                edgecolor="k", lw=0.3)
        ax.axvline(0, color="r", ls="--")
        ax.set_title(f"Absolute difference\n"
                     f"μ={diff.mean():+.4g}, σ={diff.std():.4g}")
    ax.set_xlabel("Our − Meg [npho]")
    ax.set_ylabel("Entries")

    # (1,0): relative diff histogram
    ax = ax_grid[1, 0]
    if len(rel) > 0:
        lim = max(np.percentile(np.abs(rel), 99), 1e-6)
        ax.hist(rel, bins=100, range=(-lim, lim), color="steelblue",
                edgecolor="k", lw=0.3)
        ax.axvline(0, color="r", ls="--")
        ax.set_title(f"Relative difference\n"
                     f"μ={rel.mean():+.3%}, σ={rel.std():.3%}")
    ax.set_xlabel("(Our − Meg) / Meg")
    ax.set_ylabel("Entries")

    # (1,1): log-log scatter
    ax = ax_grid[1, 1]
    positive = (meg > 0) & (our > 0)
    if positive.any():
        ax.loglog(meg[positive], our[positive], "o",
                  ms=2, alpha=0.3, color="steelblue")
        lims_log = [
            min(meg[positive].min(), our[positive].min()),
            max(meg[positive].max(), our[positive].max()),
        ]
        ax.plot(lims_log, lims_log, "r--", lw=1, label="y = x")
        ax.set_aspect("equal", "box")
        ax.legend(fontsize=8)
    ax.set_xlabel("Meganalyzer SA prediction [npho]")
    ax.set_ylabel("Our SA prediction [npho]")
    ax.set_title("Log-scale scatter")


def make_plots(stats, output_path, title_extra=""):
    if stats["n"] == 0:
        print("[WARN] No data to plot")
        return

    our = stats["our"]
    meg = stats["meg"]
    diff = stats["diff"]
    rel = stats["rel"]
    face = stats["face"]

    with PdfPages(output_path) as pdf:
        # --- Page 1: all entries ---
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        _plot_4panel(axes, our, meg, diff, rel)
        title = "SA-wt Identity Check: Our Python vs Meganalyzer — ALL"
        if title_extra:
            title += f"\n{title_extra}"
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # --- Pages 2+: per-face ---
        for face_name in FACE_ORDER + ["unknown"]:
            sel = (face == face_name)
            nf = int(sel.sum())
            if nf == 0:
                print(f"[INFO] Skipping face '{face_name}' (no dead channels)")
                continue
            print(f"[INFO] Face '{face_name}': {nf:,} entries")
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            _plot_4panel(
                axes, our[sel], meg[sel], diff[sel], rel[sel],
                title_suffix=f" [{face_name}]",
            )
            title = (f"SA-wt Identity Check — face: {face_name.upper()} "
                     f"(N={nf:,})")
            if title_extra:
                title += f"\n{title_extra}"
            fig.suptitle(title, fontsize=13)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\n[INFO] Saved: {output_path}")


# ---------------------------------------------------------------------------
# Debug CSV
# ---------------------------------------------------------------------------

def save_debug_csv(stats, data, path):
    """Dump per-comparison rows for post-analysis."""
    ev = stats["ev_idx"]
    sid = stats["sid_idx"]
    run = data["run"][ev]
    evt = data["event"][ev]
    arr = np.column_stack([
        run, evt, sid, stats["our"], stats["meg"], stats["diff"], stats["rel"],
    ])
    header = "run,event,sensor_id,our,meg,diff,rel"
    np.savetxt(path, arr, delimiter=",", header=header,
               fmt=["%d", "%d", "%d", "%.6g", "%.6g", "%.6g", "%.6g"])
    print(f"[INFO] Saved debug CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Apples-to-apples identity check: our SA-wt baseline "
                    "vs meganalyzer xecenepmweight.npho, using rec file alone."
    )
    parser.add_argument("--rec", required=True,
                        help="Meganalyzer rec file (e.g. rec559261.root)")
    parser.add_argument("--output", "-o", default="sa_identity.pdf",
                        help="Output PDF path (default: sa_identity.pdf)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Limit processing to first N events (for debug)")
    parser.add_argument("--distance-threshold", type=float, default=20.0,
                        help="Distance threshold (cm) for same-face neighbors "
                             "(default: 20.0, matches meganalyzer default)")
    parser.add_argument("--npho-threshold", type=float, default=50.0,
                        help="Npho threshold; below this the baseline falls "
                             "back to simple average. Pin to the DB value "
                             "used in the actual meganalyzer reconstruction "
                             "(default: 50.0)")
    parser.add_argument("--max-print", type=int, default=20,
                        help="Print top-N worst discrepancies (default: 20)")
    parser.add_argument("--save-debug-csv", default=None,
                        help="Optional CSV path to dump per-comparison rows")
    parser.add_argument("--geom-dump", default=None,
                        help="Path to dump_xec_geometry.C output (with is_bad "
                             "column). When provided, uses the static IsBad "
                             "flag as the neighbor-exclusion mask to match "
                             "meganalyzer's fInvalid/GetIsBad split exactly.")
    args = parser.parse_args()

    data = load_rec(args.rec, max_events=args.max_events)
    data = filter_valid_events(data)

    if len(data["run"]) == 0:
        print("[ERROR] No valid events remain after filtering")
        sys.exit(1)

    n_dead_total = int((data["recovered"] == 1).sum())
    print(f"[INFO] {n_dead_total:,} recovered (PM,event) pairs after filtering")

    our_pred, mask, _ = compute_our_predictions(
        data, args.distance_threshold, args.npho_threshold,
        geom_dump_path=args.geom_dump,
    )

    stats = compare(data, our_pred, mask)
    print_statistics(stats, args.max_print)

    title_extra = (f"distance={args.distance_threshold} cm, "
                   f"npho_thr={args.npho_threshold}")
    make_plots(stats, args.output, title_extra)

    if args.save_debug_csv:
        save_debug_csv(stats, data, args.save_debug_csv)


if __name__ == "__main__":
    main()
