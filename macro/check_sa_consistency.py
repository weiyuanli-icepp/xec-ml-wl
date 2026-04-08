#!/usr/bin/env python3
"""
Consistency check: compare our SA-wt dead channel recovery against
meganalyzer's xecenepmweight output.

Reads:
  - Our baseline output (from compute_inpainter_baselines.py) with per-sensor
    predictions (baseline_sa_npho) for dead channels.
  - The meganalyzer rec file with xecenepmweight.npho and
    xecenepmweight.nphorecovered.

For each dead channel (nphorecovered == 1), compares our SA prediction to
the meganalyzer recovered value, matched by (run, event, sensor_id).

Usage:
    python macro/check_sa_consistency.py \\
        --baselines baselines_consistency_559261.root \\
        --rec /data/project/meg/offline/run/559xxx/rec559261.root \\
        --output sa_consistency_559261.pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import uproot

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _find_branch(tree, candidates):
    """Return the first branch name from candidates that exists in tree."""
    keys = set(tree.keys())
    for c in candidates:
        if c in keys:
            return c
    return None


def _find_branch_prefix(tree, prefix_candidates):
    """Return the first key in tree that starts with any of the candidates."""
    keys = list(tree.keys())
    for p in prefix_candidates:
        for k in keys:
            if k.startswith(p):
                return k
    return None


def load_meganalyzer_recovery(rec_path):
    """Load (run, event, npho[4760], nphorecovered[4760]) from rec file.

    Returns dict with keys: run, event, npho, nphorecovered.
    Shapes: run/event are (N,), npho/nphorecovered are (N, 4760).
    """
    print(f"[INFO] Loading meganalyzer rec file: {rec_path}")
    with uproot.open(rec_path) as f:
        # Find the tree
        tree_name = None
        for name in ("rec", "rec;1"):
            if name in f:
                tree_name = name
                break
        if tree_name is None:
            raise RuntimeError(f"Could not find 'rec' tree in {rec_path}")
        tree = f[tree_name]

        print(f"[INFO] Tree '{tree_name}' has {tree.num_entries} events")

        # Find xecenepmweight branches
        npho_candidates = [
            "xecenepmweight/xecenepmweight.npho",
            "xecenepmweight.npho",
        ]
        recovered_candidates = [
            "xecenepmweight/xecenepmweight.nphorecovered",
            "xecenepmweight.nphorecovered",
        ]

        npho_key = _find_branch(tree, npho_candidates)
        recovered_key = _find_branch(tree, recovered_candidates)

        if npho_key is None:
            # Fallback: search by suffix
            for k in tree.keys():
                if k.endswith(".npho") and "xecenepmweight" in k:
                    npho_key = k
                    break
        if recovered_key is None:
            for k in tree.keys():
                if k.endswith(".nphorecovered") and "xecenepmweight" in k:
                    recovered_key = k
                    break

        if npho_key is None or recovered_key is None:
            print(f"[ERROR] Could not find xecenepmweight branches. Available:")
            for k in tree.keys():
                if "xecenepmweight" in k:
                    print(f"   {k}")
            raise RuntimeError("xecenepmweight branches not found")

        print(f"[INFO] npho branch      : {npho_key}")
        print(f"[INFO] recovered branch : {recovered_key}")

        # Discover run/event branches by scanning all keys
        all_keys = list(tree.keys())
        run_key = None
        event_key = None
        for k in all_keys:
            kl = k.lower()
            if run_key is None and ("runnumber" in kl or kl.endswith(".run") or kl.endswith("/run")):
                run_key = k
            if event_key is None and ("eventnumber" in kl or kl.endswith(".event") or kl.endswith("/event")):
                event_key = k

        print(f"[INFO] run branch       : {run_key}")
        print(f"[INFO] event branch     : {event_key}")

        if run_key is None or event_key is None:
            print("[DEBUG] Available keys containing 'run' or 'event' or 'Info':")
            for k in all_keys:
                kl = k.lower()
                if "run" in kl or "event" in kl or "info" in kl:
                    print(f"   {k}")
            raise RuntimeError("Could not find run/event number branches")

        run_arr = tree[run_key].array(library="np")
        event_arr = tree[event_key].array(library="np")

        # xecenepmweight is a TClonesArray of per-PM results.
        # The shape is typically (n_events, 4760, n_gamma) — since real data
        # usually has one gamma per event, we take the first gamma (index 0).
        npho_raw = tree[npho_key].array(library="ak")
        recovered_raw = tree[recovered_key].array(library="ak")

    import awkward as ak
    print(f"[INFO] npho raw type: {npho_raw.type}")
    print("[INFO] Vectorizing to (n_events, 4760) numpy arrays...")

    n_events = len(npho_raw)

    # Shape: (n_events, var PMs, var gammas). Take first gamma per PM.
    # ak.firsts collapses the innermost axis by taking the first element
    # (None if empty).
    npho_first = ak.firsts(npho_raw, axis=-1)           # (n_events, var PMs)
    recovered_first = ak.firsts(recovered_raw, axis=-1)  # (n_events, var PMs)

    # Pad to exactly 4760 PMs per event (clip if longer) so we can convert
    # to a rectangular numpy array.
    npho_padded = ak.pad_none(npho_first, 4760, clip=True, axis=1)
    recovered_padded = ak.pad_none(recovered_first, 4760, clip=True, axis=1)

    # Fill missing with sentinels and convert to numpy
    npho_np = ak.to_numpy(ak.fill_none(npho_padded, np.nan)).astype(np.float32)
    recovered_np = ak.to_numpy(ak.fill_none(recovered_padded, 0)).astype(np.int8)

    print(f"[INFO] Loaded {n_events} events, "
          f"{int(recovered_np.sum())} total dead-channel entries")

    return {
        "run": np.asarray(run_arr, dtype=np.int64),
        "event": np.asarray(event_arr, dtype=np.int64),
        "npho": npho_np,
        "nphorecovered": recovered_np,
    }


def load_baselines(baselines_path):
    """Load per-sensor SA predictions from compute_inpainter_baselines.py output."""
    print(f"[INFO] Loading baselines file: {baselines_path}")
    with uproot.open(baselines_path) as f:
        tree = f["predictions"]
        data = {
            "run": tree["run_number"].array(library="np"),
            "event": tree["event_number"].array(library="np"),
            "sensor_id": tree["sensor_id"].array(library="np"),
            "mask_type": tree["mask_type"].array(library="np"),
            "sa_pred": tree["baseline_sa_npho"].array(library="np"),
        }
    # Keep only dead channels (mask_type == 1 in real data mode)
    dead = data["mask_type"] == 1
    for k in data:
        data[k] = data[k][dead]
    print(f"[INFO] Loaded {len(data['sensor_id']):,} dead-channel predictions")
    return data


def main():
    parser = argparse.ArgumentParser(
        description="Compare our SA-wt recovery against meganalyzer"
    )
    parser.add_argument("--baselines", required=True,
                        help="Output from compute_inpainter_baselines.py")
    parser.add_argument("--rec", required=True,
                        help="Meganalyzer rec file (e.g. rec559261.root)")
    parser.add_argument("--output", "-o", default="sa_consistency.pdf",
                        help="Output PDF path")
    parser.add_argument("--max-print", type=int, default=20,
                        help="Print first N mismatches for debugging")
    args = parser.parse_args()

    # --- Load both files ---
    baselines = load_baselines(args.baselines)
    meg = load_meganalyzer_recovery(args.rec)

    # --- Build (run, event) -> event index lookup for meganalyzer ---
    meg_key = {
        (int(r), int(e)): i
        for i, (r, e) in enumerate(zip(meg["run"], meg["event"]))
    }

    # --- For each baseline entry, find matching meganalyzer value ---
    n_total = len(baselines["sensor_id"])
    meg_vals = np.full(n_total, np.nan, dtype=np.float32)
    meg_recovered = np.zeros(n_total, dtype=np.int8)

    n_matched = 0
    n_unmatched_event = 0
    for k in range(n_total):
        key = (int(baselines["run"][k]), int(baselines["event"][k]))
        idx = meg_key.get(key)
        if idx is None:
            n_unmatched_event += 1
            continue
        sid = int(baselines["sensor_id"][k])
        meg_vals[k] = meg["npho"][idx, sid]
        meg_recovered[k] = meg["nphorecovered"][idx, sid]
        n_matched += 1

    print(f"\n[INFO] Matched {n_matched:,} / {n_total:,} entries")
    if n_unmatched_event > 0:
        print(f"[WARN] {n_unmatched_event:,} entries unmatched (missing event)")

    # --- Only keep entries where meganalyzer also marks this sensor as recovered ---
    both_recovered = (meg_recovered == 1) & np.isfinite(meg_vals) & np.isfinite(baselines["sa_pred"])
    print(f"[INFO] Both methods recovered: {int(both_recovered.sum()):,}")

    if not both_recovered.any():
        print("[ERROR] No overlapping recovered entries!")
        sys.exit(1)

    our_pred = baselines["sa_pred"][both_recovered]
    meg_pred = meg_vals[both_recovered]
    diff = our_pred - meg_pred
    rel_diff = diff / np.where(meg_pred != 0, meg_pred, 1.0)

    # --- Statistics ---
    print("\n" + "=" * 60)
    print("Consistency Check Summary (all entries)")
    print("=" * 60)
    print(f"  N compared             : {len(our_pred):,}")
    print(f"  Our pred  median       : {np.median(our_pred):.3f}")
    print(f"  Our pred  [1,50,99]%   : {np.percentile(our_pred, [1, 50, 99])}")
    print(f"  Meg pred  median       : {np.median(meg_pred):.3f}")
    print(f"  Meg pred  [1,50,99]%   : {np.percentile(meg_pred, [1, 50, 99])}")
    print(f"  |diff|    median / p99 : {np.median(np.abs(diff)):.4f} / "
          f"{np.percentile(np.abs(diff), 99):.4f}")
    print(f"  |rel|     median / p99 : {np.median(np.abs(rel_diff)):.2%} / "
          f"{np.percentile(np.abs(rel_diff), 99):.2%}")

    # --- Robust summary: filter to physically reasonable range ---
    # Meganalyzer's xecenepmweight.npho can contain outliers (e.g. huge values
    # when the sum of neighbors is large and the solid-angle ratio inflates it).
    # Use the central 95% to assess "typical" agreement.
    both_sane = (np.abs(meg_pred) < 1e5) & (np.abs(our_pred) < 1e5)
    n_sane = int(both_sane.sum())
    if n_sane > 0:
        od = our_pred[both_sane]
        md = meg_pred[both_sane]
        dd = diff[both_sane]
        rd = rel_diff[both_sane]
        print("\n" + "=" * 60)
        print(f"Filtered to |pred| < 1e5 photons (N={n_sane:,}, "
              f"{n_sane/len(our_pred)*100:.1f}%)")
        print("=" * 60)
        print(f"  Our pred  mean ± std   : {od.mean():+.3f} ± {od.std():.3f}")
        print(f"  Meg pred  mean ± std   : {md.mean():+.3f} ± {md.std():.3f}")
        print(f"  diff      mean ± std   : {dd.mean():+.4f} ± {dd.std():.4f}")
        print(f"  |diff|    median / p99 : {np.median(np.abs(dd)):.4f} / "
              f"{np.percentile(np.abs(dd), 99):.4f}")
        print(f"  rel diff  mean ± std   : {rd.mean():+.3%} ± {rd.std():.3%}")
        print(f"  |rel|     median / p99 : {np.median(np.abs(rd)):.3%} / "
              f"{np.percentile(np.abs(rd), 99):.3%}")

    # Closest/farthest
    worst = np.argsort(np.abs(diff))[::-1][:args.max_print]
    print(f"\nTop {args.max_print} largest discrepancies:")
    print(f"  {'idx':>8s} {'our':>12s} {'meg':>12s} {'diff':>12s} {'rel':>10s}")
    for i in worst:
        print(f"  {i:>8d} {our_pred[i]:>12.4f} {meg_pred[i]:>12.4f} "
              f"{diff[i]:>+12.4f} {rel_diff[i]:>+10.2%}")

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # (0,0): scatter our vs meg
    ax = axes[0, 0]
    ax.scatter(meg_pred, our_pred, s=3, alpha=0.3, color="steelblue")
    lims = [
        min(meg_pred.min(), our_pred.min()),
        max(meg_pred.max(), our_pred.max()),
    ]
    ax.plot(lims, lims, "r--", lw=1, label="y = x")
    ax.set_xlabel("Meganalyzer SA prediction [npho]")
    ax.set_ylabel("Our SA prediction [npho]")
    ax.set_title(f"Scatter (N={len(our_pred):,})")
    ax.legend()
    ax.set_aspect("equal", "box")

    # (0,1): absolute difference histogram
    ax = axes[0, 1]
    lim = max(np.percentile(np.abs(diff), 99), 1e-6)
    ax.hist(diff, bins=100, range=(-lim, lim), color="steelblue", edgecolor="k", lw=0.3)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("Our - Meganalyzer [npho]")
    ax.set_ylabel("Entries")
    ax.set_title(f"Absolute difference\nμ={diff.mean():+.4f}, σ={diff.std():.4f}")

    # (1,0): relative difference histogram
    ax = axes[1, 0]
    lim = max(np.percentile(np.abs(rel_diff), 99), 1e-6)
    ax.hist(rel_diff, bins=100, range=(-lim, lim), color="steelblue", edgecolor="k", lw=0.3)
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("(Our - Meg) / Meg")
    ax.set_ylabel("Entries")
    ax.set_title(f"Relative difference\nμ={rel_diff.mean():+.2%}, σ={rel_diff.std():.2%}")

    # (1,1): log scatter for wide dynamic range
    ax = axes[1, 1]
    positive = (meg_pred > 0) & (our_pred > 0)
    if positive.any():
        ax.loglog(meg_pred[positive], our_pred[positive], "o",
                  ms=2, alpha=0.3, color="steelblue")
        lims_log = [
            min(meg_pred[positive].min(), our_pred[positive].min()),
            max(meg_pred[positive].max(), our_pred[positive].max()),
        ]
        ax.plot(lims_log, lims_log, "r--", lw=1, label="y = x")
        ax.set_xlabel("Meganalyzer SA prediction [npho]")
        ax.set_ylabel("Our SA prediction [npho]")
        ax.set_title("Log-scale scatter")
        ax.legend()
        ax.set_aspect("equal", "box")

    fig.suptitle(f"SA-wt Dead Channel Recovery: Consistency Check\n"
                 f"{Path(args.baselines).name} vs {Path(args.rec).name}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"\n[INFO] Saved: {args.output}")


if __name__ == "__main__":
    main()
