#!/usr/bin/env python3
"""
Compare Python solid-angle predictions against C++ LocalFitBaseline output.

Verifies that the Python port produces identical results to the C++ macro
for dead channel predictions.

Usage:
    python macro/debug_localfit_solidangle.py \
        --localfit-root localfit_output.root \
        [--event-idx 0] [--max-events 5]
"""

import argparse
import sys
import numpy as np
import uproot
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from show_localfit_event import (
    compute_inner_pm_geometry, compute_predicted_npho, uvw_to_xyz,
    N_INNER,
)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Python vs C++ solid-angle predictions")
    parser.add_argument("--localfit-root", required=True,
                        help="LocalFitBaseline output ROOT file")
    parser.add_argument("--event-idx", type=int, default=None,
                        help="Specific event index (default: first event)")
    parser.add_argument("--max-events", type=int, default=5,
                        help="Number of events to check (default: 5)")
    args = parser.parse_args()

    # Load C++ results
    with uproot.open(args.localfit_root) as f:
        pos_tree = f["position"]
        pos = {
            "event_idx": pos_tree["event_idx"].array(library="np"),
            "uvwFitNoDead": pos_tree["uvwFitNoDead"].array(library="np"),
            "fitScale": pos_tree["fitScale"].array(library="np"),
            "fitChisq": pos_tree["fitChisq"].array(library="np"),
        }
        pred_tree = f["predictions"]
        pred = {
            "event_idx": pred_tree["event_idx"].array(library="np"),
            "sensor_id": pred_tree["sensor_id"].array(library="np"),
            "pred_npho": pred_tree["pred_npho"].array(library="np"),
            "truth_npho": pred_tree["truth_npho"].array(library="np"),
        }

    # Compute PM geometry once
    pm_pos, pm_dir, _, _ = compute_inner_pm_geometry()

    # Select events
    if args.event_idx is not None:
        event_indices = [args.event_idx]
    else:
        event_indices = pos["event_idx"][:args.max_events]

    all_ratios = []

    for ev_idx in event_indices:
        ev_idx = int(ev_idx)
        pos_mask = pos["event_idx"] == ev_idx
        if not pos_mask.any():
            print(f"Event {ev_idx}: not found in position tree, skipping")
            continue

        pi = np.where(pos_mask)[0][0]
        uvw_fit = pos["uvwFitNoDead"][pi]
        scale = float(pos["fitScale"][pi])
        chi2 = float(pos["fitChisq"][pi])

        # Python prediction
        fit_xyz = uvw_to_xyz(uvw_fit)
        py_pred_all = compute_predicted_npho(fit_xyz, scale, pm_pos, pm_dir)

        # C++ predictions for this event's dead channels
        pred_mask = pred["event_idx"] == ev_idx
        if not pred_mask.any():
            print(f"Event {ev_idx}: no dead channel predictions, skipping")
            continue

        cpp_sids = pred["sensor_id"][pred_mask]
        cpp_preds = pred["pred_npho"][pred_mask]
        cpp_truths = pred["truth_npho"][pred_mask]

        print(f"\nEvent {ev_idx}  |  chi2/ndf={chi2:.2f}  |  "
              f"fitScale={scale:.3e}")
        print(f"  FitNoDead UVW = ({uvw_fit[0]:.2f}, {uvw_fit[1]:.2f}, "
              f"{uvw_fit[2]:.2f}) cm")
        print(f"  FitNoDead XYZ = ({fit_xyz[0]:.2f}, {fit_xyz[1]:.2f}, "
              f"{fit_xyz[2]:.2f}) cm")
        print(f"  {'sensor':>6s} {'truth':>10s} {'C++ pred':>10s} "
              f"{'Py pred':>10s} {'ratio':>8s}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

        for sid, cpred, ctruth in zip(cpp_sids, cpp_preds, cpp_truths):
            sid = int(sid)
            if sid >= N_INNER:
                continue
            ppred = py_pred_all[sid]
            if cpred > 0:
                ratio = ppred / cpred
                all_ratios.append(ratio)
            else:
                ratio = float("nan")
            print(f"  {sid:6d} {ctruth:10.2f} {cpred:10.2f} "
                  f"{ppred:10.2f} {ratio:8.4f}")

    # Summary
    if all_ratios:
        ratios = np.array(all_ratios)
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(ratios)} dead channel predictions compared")
        print(f"  Py/C++ ratio: mean={ratios.mean():.6f}  "
              f"std={ratios.std():.6f}  "
              f"min={ratios.min():.6f}  max={ratios.max():.6f}")
        if np.allclose(ratios, 1.0, atol=1e-3):
            print("  PASS: Python matches C++ to within 0.1%")
        else:
            print("  FAIL: Significant discrepancy detected")
    else:
        print("\nNo predictions to compare.")


if __name__ == "__main__":
    main()
