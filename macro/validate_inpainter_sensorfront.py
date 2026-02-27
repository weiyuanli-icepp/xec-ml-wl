#!/usr/bin/env python3
"""
Sensor-Front Validation: mask the single inner-face sensor whose front
region contains the MC truth first interaction point (FIP), then run the
ML inpainter + baselines to see if the peak npho can be recovered.

Motivation: neighbor-average baselines underestimate npho when the FIP is
directly in front of the masked sensor, because the peak is lost. This
script tests whether the ML inpainter can recover it.

Usage:
    python macro/validate_inpainter_sensorfront.py \\
        --checkpoint artifacts/.../checkpoint_best.pth \\
        --input data/E15to60_AngUni_PosSQ/val2/MCGamma_0.root \\
        --output artifacts/sensorfront_validation/ \\
        [--solid-angle-branch solid_angle] \\
        [--local-fit-baseline] \\
        [--max-events N] \\
        [--batch-size 64] \\
        [--device cpu|cuda]
"""

from __future__ import annotations

import gc
import os
import sys
import argparse
import csv
import numpy as np
import torch
import uproot
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_TIME,
)
from lib.normalization import NphoTransform
from lib.dataset import expand_path
from lib.sensor_geometry import load_sensor_positions

# Reuse functions from validate_inpainter
from macro.validate_inpainter import (
    load_model,
    normalize_data,
    run_inference,
    collect_predictions,
    compute_metrics,
    compute_baseline_metrics,
    save_predictions,
    N_CHANNELS,
    MODEL_SENTINEL_TIME,
    MODEL_SENTINEL_NPHO,
)

# Geometry constants (from LocalFitBaseline.C)
K_XERIN = 64.84       # cm, inner radius of spacer
K_MPPC_HEIGHT = 0.13  # cm, offset from spacer to photoelectric surface
K_REFF = K_XERIN + K_MPPC_HEIGHT  # 64.97 cm

N_INNER = 4092  # inner face sensors: IDs 0..4091


# =========================================================
#  XYZ → UVW transform (matches LocalFitBaseline.C)
# =========================================================

def xyz_to_uvw(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ coordinates to UVW.  Vectorized.

    Args:
        xyz: (..., 3) array with columns (x, y, z).

    Returns:
        uvw: (..., 3) array with columns (u, v, w).
    """
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2)
    phi = np.where(
        (x == 0.0) & (y == 0.0),
        0.0,
        np.pi + np.arctan2(-y, -x),
    )
    u = z
    v = -(phi - np.pi) * K_REFF
    w = r - K_REFF
    return np.stack([u, v, w], axis=-1)


def build_inner_sensor_uvw() -> np.ndarray:
    """Load sensor positions and return UVW for inner face (0..4091).

    Returns:
        sensor_uvw: (4092, 3) float64 array.
    """
    pos_file = os.path.join(
        os.path.dirname(__file__), '..', 'lib', 'sensor_positions.txt'
    )
    pos_file = os.path.abspath(pos_file)
    xyz = load_sensor_positions(pos_file)[:N_INNER]  # (4092, 3)
    return xyz_to_uvw(xyz)


# =========================================================
#  Event ↔ sensor matching
# =========================================================

def match_events_to_sensors(
    uvw_truth: np.ndarray,
    sensor_uvw: np.ndarray,
    du: float = 0.5,
    dv: float = 0.5,
    w_max: float = 1.0,
) -> tuple:
    """Find the inner sensor whose front region contains the FIP.

    For each event, checks whether uvwTruth falls within the box
    |u_fip - u_sensor| < du, |v_fip - v_sensor| < dv, 0 < w_fip < w_max.

    Args:
        uvw_truth: (N, 3) FIP positions in UVW.
        sensor_uvw: (4092, 3) inner-sensor UVW positions.
        du, dv: half-widths of the matching box in u and v (cm).
        w_max: maximum depth from inner face (cm).

    Returns:
        matched_event_idx: (M,) int indices into the event array.
        matched_sensor_id: (M,) int sensor IDs (0..4091).
    """
    N = uvw_truth.shape[0]
    S = sensor_uvw.shape[0]

    # Depth cut (only events with w in range)
    w_fip = uvw_truth[:, 2]
    depth_ok = (w_fip > 0) & (w_fip < w_max)
    candidate_idx = np.where(depth_ok)[0]

    if len(candidate_idx) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    # Vectorized u,v matching: (N_cand, 1) vs (1, S)
    u_fip = uvw_truth[candidate_idx, 0][:, None]  # (N_cand, 1)
    v_fip = uvw_truth[candidate_idx, 1][:, None]
    u_s = sensor_uvw[:, 0][None, :]  # (1, S)
    v_s = sensor_uvw[:, 1][None, :]

    u_match = np.abs(u_fip - u_s) < du  # (N_cand, S)
    v_match = np.abs(v_fip - v_s) < dv
    match = u_match & v_match  # (N_cand, S)

    # For events with exactly one match take it; for multiple take nearest
    matched_event_idx = []
    matched_sensor_id = []
    for i, ev_idx in enumerate(candidate_idx):
        hits = np.where(match[i])[0]
        if len(hits) == 0:
            continue
        if len(hits) == 1:
            sid = hits[0]
        else:
            # Pick the closest sensor in u-v
            dist2 = (uvw_truth[ev_idx, 0] - sensor_uvw[hits, 0])**2 + \
                     (uvw_truth[ev_idx, 1] - sensor_uvw[hits, 1])**2
            sid = hits[np.argmin(dist2)]
        matched_event_idx.append(ev_idx)
        matched_sensor_id.append(sid)

    return (
        np.array(matched_event_idx, dtype=np.int64),
        np.array(matched_sensor_id, dtype=np.int64),
    )


# =========================================================
#  Memory-efficient streaming loader
# =========================================================

def load_and_match_streaming(
    input_path: str,
    sensor_uvw: np.ndarray,
    npho_scheme: str,
    tree_name: str = "tree",
    max_events: Optional[int] = None,
    du: float = 0.5,
    dv: float = 0.5,
    w_max: float = 1.0,
    sa_branch: Optional[str] = None,
) -> dict:
    """Load ROOT files one at a time, match events, keep only matched.

    This avoids loading all files into memory simultaneously.  Peak memory
    is proportional to the largest single file, not the total.

    Returns dict with keys: x_input, x_original, uvw_matched, matched_sid,
    run_numbers, event_numbers, solid_angles (or None), n_total,
    matched_orig_idx (global indices into concatenated files, for local-fit).
    """
    file_list = expand_path(input_path)
    print(f"[INFO] Loading data from {len(file_list)} file(s) (streaming)")
    for f in file_list[:5]:
        print(f"  - {f}")
    if len(file_list) > 5:
        print(f"  ... and {len(file_list) - 5} more")

    accum = {
        "x_norm": [], "uvw_matched": [],
        "matched_sid": [], "run": [], "event": [],
        "sa": [], "matched_orig_idx": [],
        "truth_at_mask": [],  # (M, 2) — truth values at the masked sensor
    }
    global_offset = 0  # running event counter across files
    n_total = 0

    for fp in file_list:
        if max_events and n_total >= max_events:
            break

        with uproot.open(fp) as f:
            tree = f[tree_name]

            # --- read branches ---
            npho_branch = "npho" if "npho" in tree.keys() else "relative_npho"
            npho = tree[npho_branch].array(library="np")
            time = tree["relative_time"].array(library="np")
            uvw = tree["uvwTruth"].array(library="np")

            n_file = len(npho)
            if max_events:
                remaining = max_events - n_total
                if n_file > remaining:
                    npho = npho[:remaining]
                    time = time[:remaining]
                    uvw = uvw[:remaining]
                    n_file = remaining

            # optional branches
            run_arr = tree["run"].array(library="np")[:n_file] if "run" in tree.keys() else None
            event_arr = tree["event"].array(library="np")[:n_file] if "event" in tree.keys() else None

            sa_arr = None
            if sa_branch and sa_branch in tree.keys():
                sa_arr = tree[sa_branch].array(library="np")[:n_file]

        # --- normalize (one file at a time) ---
        x_norm = normalize_data(npho, time, npho_scheme=npho_scheme)
        del npho, time  # free raw arrays

        # --- match events ---
        m_idx, m_sid = match_events_to_sensors(
            uvw, sensor_uvw, du=du, dv=dv, w_max=w_max,
        )

        if len(m_idx) > 0:
            matched_x = x_norm[m_idx]
            # Save truth at the masked sensor BEFORE sentinel is applied
            truth = matched_x[np.arange(len(m_idx)), m_sid, :].copy()  # (M, 2)
            accum["truth_at_mask"].append(truth)
            accum["x_norm"].append(matched_x)
            accum["uvw_matched"].append(uvw[m_idx])
            accum["matched_sid"].append(m_sid)
            accum["matched_orig_idx"].append(m_idx + global_offset)
            if run_arr is not None:
                accum["run"].append(run_arr[m_idx])
            if event_arr is not None:
                accum["event"].append(event_arr[m_idx])
            if sa_arr is not None:
                accum["sa"].append(sa_arr[m_idx])

        n_total += n_file
        global_offset += n_file
        del x_norm, uvw, sa_arr, run_arr, event_arr
        gc.collect()

    print(f"[INFO] Loaded {n_total:,} events total (streaming)")

    if not accum["x_norm"]:
        return {"n_total": n_total, "n_matched": 0}

    result = {
        "x_norm": np.concatenate(accum["x_norm"]),
        "truth_at_mask": np.concatenate(accum["truth_at_mask"]),
        "uvw_matched": np.concatenate(accum["uvw_matched"]),
        "matched_sid": np.concatenate(accum["matched_sid"]),
        "matched_orig_idx": np.concatenate(accum["matched_orig_idx"]),
        "n_total": n_total,
    }
    result["run_numbers"] = (np.concatenate(accum["run"])
                             if accum["run"] else None)
    result["event_numbers"] = (np.concatenate(accum["event"])
                               if accum["event"] else None)
    result["solid_angles"] = (np.concatenate(accum["sa"])
                              if accum["sa"] else None)
    result["n_matched"] = len(result["x_norm"])
    return result


# =========================================================
#  Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(
        description="Sensor-front validation: mask the sensor in front of "
                    "the FIP and test inpainter recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Model
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", "-c", type=str,
                             help="Path to inpainter checkpoint (.pth)")
    model_group.add_argument("--torchscript", "-t", type=str,
                             help="Path to TorchScript model (.pt)")

    # Data
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input ROOT file (MC with uvwTruth)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    # Options
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"])
    parser.add_argument("--tree-name", type=str, default="tree")
    parser.add_argument("--npho-scheme", type=str, default=None,
                        choices=["log1p", "sqrt", "anscombe", "linear"],
                        help="Override npho normalization scheme")

    # Matching geometry
    parser.add_argument("--du", type=float, default=0.5,
                        help="Half-width in u for FIP matching (cm, default 0.5)")
    parser.add_argument("--dv", type=float, default=0.5,
                        help="Half-width in v for FIP matching (cm, default 0.5)")
    parser.add_argument("--w-max", type=float, default=1.0,
                        help="Max depth from inner face (cm, default 1.0)")

    # Baselines
    parser.add_argument("--solid-angle-branch", type=str, default=None,
                        help="Branch name for solid angles (enables SA baseline)")
    parser.add_argument("--local-fit-baseline", action="store_true",
                        help="Enable LocalFitBaseline via ROOT macro")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, model_type, predict_channels, model_npho_scheme = load_model(
        checkpoint_path=args.checkpoint,
        torchscript_path=args.torchscript,
        device=args.device,
    )

    if args.npho_scheme is not None:
        npho_scheme = args.npho_scheme
    elif model_npho_scheme is not None:
        npho_scheme = model_npho_scheme
    else:
        npho_scheme = "log1p"
        print(f"[WARN] Npho scheme unknown, using default: {npho_scheme}")

    # ------------------------------------------------------------------
    # 2. Build inner-sensor UVW lookup
    # ------------------------------------------------------------------
    sensor_uvw = build_inner_sensor_uvw()  # (4092, 3)

    # ------------------------------------------------------------------
    # 3. Stream-load files: normalize, match, keep only matched events
    #    Peak memory ≈ one file (~13K events × 4760 × 2 × 4 bytes ≈ 470 MB)
    #    instead of all files (~4.7 GB + copies).
    # ------------------------------------------------------------------
    loaded = load_and_match_streaming(
        input_path=args.input,
        sensor_uvw=sensor_uvw,
        npho_scheme=npho_scheme,
        tree_name=args.tree_name,
        max_events=args.max_events,
        du=args.du, dv=args.dv, w_max=args.w_max,
        sa_branch=args.solid_angle_branch,
    )

    n_total = loaded["n_total"]
    n_matched = loaded.get("n_matched", 0)
    print(f"\n[INFO] Events total:   {n_total:>8,}")
    print(f"[INFO] Events matched: {n_matched:>8,} "
          f"({n_matched / n_total * 100:.1f}%)")

    if n_matched == 0:
        print("[ERROR] No events matched any sensor front region. Exiting.")
        sys.exit(1)

    x_norm = loaded["x_norm"]           # (M, 4760, 2) — unmasked
    truth_at_mask = loaded["truth_at_mask"]  # (M, 2) — truth at masked sensor
    uvw_matched = loaded["uvw_matched"]
    matched_sid = loaded["matched_sid"]
    matched_orig_idx = loaded["matched_orig_idx"]
    run_numbers = loaded["run_numbers"]
    event_numbers = loaded["event_numbers"]
    solid_angles = loaded["solid_angles"]
    del loaded
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Build per-event mask & run baselines BEFORE applying sentinels
    #    (baselines need the unmasked data)
    # ------------------------------------------------------------------
    ev_range = np.arange(n_matched)
    combined_mask = np.zeros((n_matched, N_CHANNELS), dtype=bool)
    combined_mask[ev_range, matched_sid] = True

    print(f"[INFO] Masked sensors: {n_matched:,} (1 per event)")

    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    artificial_mask = combined_mask

    baseline_results = {}

    from lib.inpainter_baselines import (
        NeighborAverageBaseline, SolidAngleWeightedBaseline,
    )

    # Baselines run on unmasked npho (x_norm still has original values)
    x_npho_orig = x_norm[:, :, 0]

    print("[INFO] Running NeighborAverageBaseline...")
    avg_baseline = NeighborAverageBaseline(k=1)
    avg_preds = avg_baseline.predict(x_npho_orig, combined_mask)
    avg_results = _collect_baseline_fast(
        avg_preds, truth_at_mask, matched_sid, n_matched,
    )
    del avg_preds
    baseline_results["avg"] = avg_results

    if solid_angles is not None:
        print("[INFO] Running SolidAngleWeightedBaseline...")
        sa_baseline = SolidAngleWeightedBaseline(k=1)
        sa_preds = sa_baseline.predict(x_npho_orig, combined_mask,
                                       solid_angles=solid_angles)
        sa_results = _collect_baseline_fast(
            sa_preds, truth_at_mask, matched_sid, n_matched,
        )
        del sa_preds, solid_angles
        baseline_results["sa"] = sa_results
    del x_npho_orig
    gc.collect()

    # ------------------------------------------------------------------
    # 5. Apply sentinels & run inference (with cache)
    # ------------------------------------------------------------------
    cache_file = os.path.join(args.output, "_inference_cache.npz")
    predictions = None

    if os.path.isfile(cache_file):
        try:
            cached = np.load(cache_file)
            if (cached["predictions"].shape[0] == n_matched and
                    np.array_equal(cached["matched_sid"], matched_sid)):
                predictions = cached["predictions"]
                print(f"[INFO] Loaded cached inference ({cache_file})")
        except Exception:
            pass

    if predictions is None:
        x_norm[ev_range, matched_sid, 0] = MODEL_SENTINEL_NPHO
        x_norm[ev_range, matched_sid, 1] = MODEL_SENTINEL_TIME

        print(f"[INFO] Running inference on {args.device}...")
        predictions = run_inference(
            model, model_type, x_norm, combined_mask,
            batch_size=args.batch_size, device=args.device,
            predict_channels=predict_channels,
        )

        # Save cache for re-runs
        np.savez_compressed(cache_file,
                            predictions=predictions,
                            matched_sid=matched_sid)
        print(f"[INFO] Saved inference cache to {cache_file}")

        # Restore truth at masked sensors
        x_norm[ev_range, matched_sid, :] = truth_at_mask

    x_original = x_norm  # alias — no copy needed
    del x_norm
    gc.collect()

    # Local fit
    if args.local_fit_baseline:
        lf_results = _run_local_fit_lightweight(
            args.input, matched_orig_idx, matched_sid, truth_at_mask,
            npho_scheme, max_events=args.max_events,
        )
        if lf_results:
            baseline_results["localfit"] = lf_results

    # ------------------------------------------------------------------
    # 9. Compute baseline metrics
    # ------------------------------------------------------------------
    baseline_metrics_dict = {}
    for bname, bpreds in baseline_results.items():
        baseline_metrics_dict[bname] = compute_baseline_metrics(bpreds)

    # ------------------------------------------------------------------
    # 10. Collect ML predictions
    # ------------------------------------------------------------------
    print("[INFO] Collecting predictions...")
    pred_list = collect_predictions(
        predictions, x_original, combined_mask,
        artificial_mask, dead_mask,
        run_numbers=run_numbers,
        event_numbers=event_numbers,
        predict_channels=predict_channels,
    )

    # Add w_fip to each prediction entry
    _add_w_fip(pred_list, uvw_matched, matched_sid)

    ml_metrics = compute_metrics(pred_list, predict_channels=predict_channels)

    # ------------------------------------------------------------------
    # 11. Print summary
    # ------------------------------------------------------------------
    _print_sensorfront_summary(
        n_total, n_matched, ml_metrics, baseline_metrics_dict, npho_scheme,
    )

    # ------------------------------------------------------------------
    # 12. Save outputs
    # ------------------------------------------------------------------
    # Add w_fip to baseline results too (for ROOT output)
    for bname in baseline_results:
        _add_w_fip(baseline_results[bname], uvw_matched, matched_sid)

    pred_file = os.path.join(args.output, "predictions_sensorfront.root")
    save_predictions(
        pred_list, pred_file,
        predict_channels=predict_channels,
        npho_scheme=npho_scheme,
        baseline_results=baseline_results,
    )

    # Extra: save w_fip branch into the ROOT file
    _append_w_fip_branch(pred_file, pred_list)

    metrics_file = os.path.join(args.output, "metrics_sensorfront.csv")
    with open(metrics_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for k, v in sorted(ml_metrics.items()):
            writer.writerow([k, v])
        for bname, bm in baseline_metrics_dict.items():
            for k, v in sorted(bm.items()):
                writer.writerow([f"baseline_{bname}_{k}", v])
    print(f"[INFO] Saved metrics to {metrics_file}")

    print(f"\n[INFO] Done! Output in {args.output}")


# =========================================================
#  Helpers
# =========================================================

def _run_local_fit_lightweight(
    input_path: str,
    matched_orig_idx: np.ndarray,
    matched_sid: np.ndarray,
    truth_at_mask: np.ndarray,
    npho_scheme: str,
    max_events: Optional[int] = None,
) -> list:
    """Run LocalFitBaseline macro on each file and match results.

    Calls the ROOT macro once per input file with the unique dead channel
    list, then reads the macro output and matches (event_idx, sensor_id) to
    our per-event mask.  Avoids allocating full n_total × 4760 arrays.
    """
    import subprocess
    import tempfile

    unique_sids = np.unique(matched_sid)

    macro_path = os.path.join(
        os.path.dirname(__file__), '..', 'others', 'LocalFitBaseline.C')
    macro_path = os.path.abspath(macro_path)
    if not os.path.isfile(macro_path):
        print(f"[WARNING] LocalFitBaseline.C not found at {macro_path}")
        return []

    file_list = expand_path(input_path)
    transform = NphoTransform(scheme=npho_scheme)

    # Run the macro on each file, accumulating a global lookup
    lf_lookup = {}  # (global_event_idx, sensor_id) -> norm prediction
    global_offset = 0
    total_events_seen = 0

    for fi, root_file in enumerate(file_list):
        if max_events and total_events_seen >= max_events:
            break

        # Count events in this file
        with uproot.open(root_file) as rf:
            n_in_file = rf["tree"].num_entries
        if max_events:
            n_in_file = min(n_in_file, max_events - total_events_seen)

        # Find matched events that fall in this file's range
        file_start = global_offset
        file_end = global_offset + n_in_file
        in_file = ((matched_orig_idx >= file_start) &
                   (matched_orig_idx < file_end))
        if not in_file.any():
            global_offset += n_in_file
            total_events_seen += n_in_file
            continue

        # Only pass the sensors needed for this file's matched events
        file_sids = np.unique(matched_sid[in_file])
        n_file_matched = int(in_file.sum())

        # Build set of (file-local event_idx, sensor_id) pairs we need
        file_need = set()
        for sub_i in np.where(in_file)[0]:
            ev_local = int(matched_orig_idx[sub_i]) - global_offset
            file_need.add((ev_local, int(matched_sid[sub_i])))

        dead_tmp = tempfile.NamedTemporaryFile(
            mode='w', suffix='.txt', delete=False)
        out_tmp = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        try:
            for ch in file_sids:
                dead_tmp.write(f"{ch}\n")
            dead_tmp.close()
            out_tmp.close()

            # Run macro on the original ROOT file (uproot-written files
            # have incompatible branch layout for ROOT C++ SetBranchAddress)
            cmd = (f'root -l -b -q \'{macro_path}("{root_file}", '
                   f'"{dead_tmp.name}", "{out_tmp.name}")\'')
            print(f"[INFO] Running LocalFitBaseline macro on "
                  f"{os.path.basename(root_file)} "
                  f"(file {fi+1}/{len(file_list)}, "
                  f"{n_file_matched} matched, {len(file_sids)} sensors)")
            sys.stdout.flush()
            result = subprocess.run(cmd, shell=True)

            ok = True
            if result.returncode != 0:
                try:
                    _t = uproot.open(out_tmp.name)
                    _t['predictions']
                    _t.close()
                except Exception:
                    print(f"[WARNING] macro failed on {root_file}, skipping")
                    ok = False

            if ok:
                with uproot.open(out_tmp.name) as f:
                    pt = f['predictions']
                    lf_ev = pt['event_idx'].array(library='np')
                    lf_sid = pt['sensor_id'].array(library='np')
                    lf_pred_raw = pt['pred_npho'].array(library='np')

                lf_pred_norm = transform.forward(
                    np.maximum(lf_pred_raw, transform.domain_min())
                ).astype(np.float32)

                # Only keep the (event, sensor) pairs we actually need
                for j in range(len(lf_ev)):
                    key_local = (int(lf_ev[j]), int(lf_sid[j]))
                    if key_local in file_need:
                        gev = key_local[0] + global_offset
                        lf_lookup[(gev, key_local[1])] = float(lf_pred_norm[j])

        finally:
            for p in (dead_tmp.name, out_tmp.name):
                if os.path.exists(p):
                    os.unlink(p)

        global_offset += n_in_file
        total_events_seen += n_in_file

    # Match to our per-event mask
    results = []
    n_found = 0
    for sub_i in range(len(matched_orig_idx)):
        ev_orig = int(matched_orig_idx[sub_i])
        sid = int(matched_sid[sub_i])
        truth_npho = float(truth_at_mask[sub_i, 0])

        key = (ev_orig, sid)
        if key in lf_lookup:
            pred_npho = lf_lookup[key]
            n_found += 1
        else:
            pred_npho = -999.0

        if truth_npho == MODEL_SENTINEL_NPHO or pred_npho == -999.0:
            error_npho = -999.0
        else:
            error_npho = pred_npho - truth_npho

        results.append({
            "event_idx": sub_i,
            "sensor_id": sid,
            "mask_type": 0,
            "pred_npho": pred_npho,
            "truth_npho": truth_npho,
            "error_npho": error_npho,
        })

    print(f"[INFO] LocalFitBaseline: {n_found}/{len(matched_orig_idx)} "
          f"matched predictions")
    return results


def _collect_baseline_fast(
    full_preds: np.ndarray,
    truth_at_mask: np.ndarray,
    matched_sid: np.ndarray,
    n_matched: int,
) -> list:
    """Collect per-sensor baseline results (1 masked sensor per event)."""
    results = []
    for i in range(n_matched):
        sid = int(matched_sid[i])
        pred_npho = float(full_preds[i, sid])
        truth_npho = float(truth_at_mask[i, 0])
        if truth_npho == MODEL_SENTINEL_NPHO:
            error_npho = -999.0
        else:
            error_npho = pred_npho - truth_npho
        results.append({
            "event_idx": i,
            "sensor_id": sid,
            "mask_type": 0,
            "pred_npho": pred_npho,
            "truth_npho": truth_npho,
            "error_npho": error_npho,
        })
    return results


def _add_w_fip(pred_list: list, uvw_matched: np.ndarray,
               matched_sid: np.ndarray):
    """Attach w_fip depth to each prediction dict (in-place)."""
    for entry in pred_list:
        ev = entry["event_idx"]
        entry["w_fip"] = float(uvw_matched[ev, 2])


def _append_w_fip_branch(root_path: str, pred_list: list):
    """Re-open the ROOT file and add the w_fip branch to predictions."""
    w_fip = np.array([p.get("w_fip", -999.0) for p in pred_list],
                     dtype=np.float32)
    # Read existing, add branch, rewrite
    with uproot.open(root_path) as f:
        existing = {k: f["predictions"][k].array(library="np")
                    for k in f["predictions"].keys()}
        meta = {k: f["metadata"][k].array(library="np")
                for k in f["metadata"].keys()}
    existing["w_fip"] = w_fip
    with uproot.recreate(root_path) as f:
        f.mktree("predictions", existing)
        f.mktree("metadata", meta)
    print(f"[INFO] Added w_fip branch to {root_path}")


def _print_sensorfront_summary(
    n_total: int,
    n_matched: int,
    ml_metrics: dict,
    baseline_metrics: dict,
    npho_scheme: str,
):
    """Print formatted summary table."""
    print("\n" + "=" * 70)
    print("SENSOR-FRONT VALIDATION")
    print("(inner face, FIP in front of masked sensor)")
    print("=" * 70)
    print(f"Events total:      {n_total:>8,}")
    print(f"Events matched:    {n_matched:>8,} ({n_matched / n_total * 100:.1f}%)")
    print(f"Npho scheme:       {npho_scheme}")

    if ml_metrics.get("npho_mae") is None:
        print("\n  No predictions with ground truth.")
        print("=" * 70)
        return

    # Denormalized MAE
    transform = NphoTransform(scheme=npho_scheme)

    # Approximate denormalized MAE: inverse(truth + mae) - inverse(truth)
    # For a rough table, use inverse(mae) as a scale indicator
    # More precise: compute from the per-sensor errors offline
    norm_mae_ml = ml_metrics["npho_mae"]

    print(f"\nPredictions with truth: {ml_metrics['n_with_truth']:,}")

    # Build comparison table
    methods = [("ML Inpainter", ml_metrics)]
    name_map = {"avg": "Neighbor Avg", "sa": "Solid Angle Wt",
                "localfit": "Local Fit (SA)"}
    for bname, bm in baseline_metrics.items():
        methods.append((name_map.get(bname, bname), bm))

    header = f"  {'Method':<22} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'68%':>8}"
    sep = f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}"
    print(f"\n  Metrics (normalized npho space):")
    print(header)
    print(sep)

    for name, m in methods:
        if m.get("npho_mae") is not None:
            print(f"  {name:<22} {m['npho_mae']:>8.4f} "
                  f"{m['npho_rmse']:>8.4f} {m['npho_bias']:>8.4f} "
                  f"{m['npho_68pct']:>8.4f}")
        else:
            print(f"  {name:<22}   (no predictions)")

    # Relative MAE (relative to ML)
    if len(methods) > 1 and norm_mae_ml > 0:
        print(f"\n  Relative MAE (vs ML Inpainter):")
        for name, m in methods:
            if m.get("npho_mae") is not None:
                rel = m["npho_mae"] / norm_mae_ml
                print(f"    {name:<22} {rel:.3f}x")

    print("=" * 70)


if __name__ == "__main__":
    main()
