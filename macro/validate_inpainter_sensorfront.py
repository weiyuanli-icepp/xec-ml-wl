#!/usr/bin/env python3
"""
Sensor-Front Validation: mask the single inner-face sensor whose front
region contains the MC truth first interaction point (FIP), then run the
ML inpainter + baselines to see if the peak npho can be recovered.

Motivation: neighbor-average baselines underestimate npho when the FIP is
directly in front of the masked sensor, because the peak is lost. This
script tests whether the ML inpainter can recover it.

Usage (parallel workflow for scan):
    # Step 1: Prepare manifest + baselines (no model needed, fast)
    python macro/validate_inpainter_sensorfront.py \\
        --manifest-only \\
        --input data/E15to60_AngUni_PosSQ/val2/ \\
        --output artifacts/sensorfront_shared/ \\
        --solid-angle-branch solid_angle

    # Step 2a: Submit LocalFit batch jobs (uses manifest)
    bash macro/submit_localfit_sensorfront.sh \\
        artifacts/sensorfront_shared/_sensorfront_manifest.npz

    # Step 2b: Run ML inference per scan step (parallel with 2a)
    python macro/validate_inpainter_sensorfront.py \\
        --checkpoint artifacts/inp_scan_s1_baseline/inpainter_checkpoint_best.pth \\
        --load-manifest artifacts/sensorfront_shared/ \\
        --baselines-from artifacts/sensorfront_shared/ \\
        --local-fit-results artifacts/sensorfront_shared/localfit_results/ \\
        --output artifacts/inp_scan_s1_baseline/validation_sensorfront/

Usage (standalone, all-in-one):
    python macro/validate_inpainter_sensorfront.py \\
        --checkpoint artifacts/.../checkpoint_best.pth \\
        --input data/E15to60_AngUni_PosSQ/val2/MCGamma_0.root \\
        --output artifacts/sensorfront_validation/ \\
        [--solid-angle-branch solid_angle] \\
        [--max-events N] [--batch-size 64] [--device cpu|cuda]
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
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    tree_name: str = "tree",
    max_events: Optional[int] = None,
    du: float = 0.5,
    dv: float = 0.5,
    w_max: float = 1.0,
    sa_branch: Optional[str] = None,
    save_raw: bool = False,
) -> dict:
    """Load ROOT files one at a time, match events, keep only matched.

    This avoids loading all files into memory simultaneously.  Peak memory
    is proportional to the largest single file, not the total.

    Returns dict with keys: x_norm, uvw_matched, matched_sid,
    run_numbers, event_numbers, solid_angles (or None), n_total,
    matched_orig_idx (global indices into concatenated files, for local-fit).
    If save_raw=True, also includes npho_raw and time_raw (pre-normalization).
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
    if save_raw:
        accum["npho_raw"] = []
        accum["time_raw"] = []
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

        # --- match events (before normalization — uses uvw only) ---
        m_idx, m_sid = match_events_to_sensors(
            uvw, sensor_uvw, du=du, dv=dv, w_max=w_max,
        )

        if len(m_idx) > 0:
            npho_m = npho[m_idx]
            time_m = time[m_idx]

            # Save raw matched data if requested (before normalization)
            if save_raw:
                accum["npho_raw"].append(npho_m.copy())
                accum["time_raw"].append(time_m.copy())

            # Normalize only matched events (element-wise, same result)
            x_norm_m = normalize_data(npho_m, time_m, npho_scheme=npho_scheme,
                                      npho_scale=npho_scale, npho_scale2=npho_scale2)
            # Save truth at the masked sensor BEFORE sentinel is applied
            truth = x_norm_m[np.arange(len(m_idx)), m_sid, :].copy()  # (M, 2)
            accum["truth_at_mask"].append(truth)
            accum["x_norm"].append(x_norm_m)
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
        del npho, time, uvw, sa_arr, run_arr, event_arr
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
    if save_raw:
        result["npho_raw"] = np.concatenate(accum["npho_raw"])
        result["time_raw"] = np.concatenate(accum["time_raw"])
    return result


# =========================================================
#  Manifest & prepared-data helpers
# =========================================================

def save_prepared_data(
    output_dir: str,
    loaded: dict,
    input_path: str,
    max_events: Optional[int],
    npho_scheme: str,
    npho_scale: float,
    npho_scale2: float,
    baseline_raw: Optional[dict] = None,
):
    """Save manifest, raw matched data, and raw baselines to *output_dir*.

    Files created:
      _sensorfront_manifest.npz — matching info + file list (for LocalFit)
      _sensorfront_data.npz     — raw matched data (for ML inference reload)
      _baselines_raw.npz        — raw baseline predictions (scheme-independent)
    """
    os.makedirs(output_dir, exist_ok=True)
    file_list = expand_path(input_path)

    # --- manifest (small, for LocalFit batch jobs) ---
    manifest_path = os.path.join(output_dir, "_sensorfront_manifest.npz")
    np.savez_compressed(
        manifest_path,
        matched_orig_idx=loaded["matched_orig_idx"],
        matched_sid=loaded["matched_sid"],
        truth_at_mask=loaded["truth_at_mask"],
        npho_scheme=np.array([npho_scheme]),
        file_list=np.array(file_list),
        max_events=np.array([max_events if max_events else -1]),
    )
    print(f"[INFO] Saved manifest to {manifest_path}")

    # --- raw matched data (for ML inference without re-reading ROOT) ---
    data_path = os.path.join(output_dir, "_sensorfront_data.npz")
    data_dict = {
        "npho_raw": loaded["npho_raw"],
        "time_raw": loaded["time_raw"],
        "matched_sid": loaded["matched_sid"],
        "matched_orig_idx": loaded["matched_orig_idx"],
        "uvw_matched": loaded["uvw_matched"],
        "n_total": np.array([loaded["n_total"]]),
    }
    if loaded.get("run_numbers") is not None:
        data_dict["run_numbers"] = loaded["run_numbers"]
    if loaded.get("event_numbers") is not None:
        data_dict["event_numbers"] = loaded["event_numbers"]
    if loaded.get("solid_angles") is not None:
        data_dict["solid_angles"] = loaded["solid_angles"]
    np.savez_compressed(data_path, **data_dict)
    print(f"[INFO] Saved raw matched data to {data_path}")

    # --- raw baseline predictions (scheme-independent) ---
    if baseline_raw is not None:
        bl_path = os.path.join(output_dir, "_baselines_raw.npz")
        np.savez_compressed(bl_path, **baseline_raw)
        print(f"[INFO] Saved raw baselines to {bl_path}")

    print(f"       Use run_localfit_sensorfront.py --manifest {manifest_path} "
          f"--file-index <0..{len(file_list)-1}> for LocalFit batch jobs")


def load_prepared_data(
    prepared_dir: str,
    npho_scheme: str,
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
) -> dict:
    """Load raw matched data from a prepared directory and normalize.

    Returns a dict with the same keys as load_and_match_streaming().
    """
    data_path = os.path.join(prepared_dir, "_sensorfront_data.npz")
    print(f"[INFO] Loading prepared data from {data_path}")
    data = np.load(data_path, allow_pickle=True)

    npho_raw = data["npho_raw"]
    time_raw = data["time_raw"]
    matched_sid = data["matched_sid"]

    # Normalize with the target scheme
    x_norm = normalize_data(npho_raw, time_raw, npho_scheme=npho_scheme,
                            npho_scale=npho_scale, npho_scale2=npho_scale2)
    n_matched = len(x_norm)
    ev_range = np.arange(n_matched)
    truth_at_mask = x_norm[ev_range, matched_sid, :].copy()

    result = {
        "x_norm": x_norm,
        "truth_at_mask": truth_at_mask,
        "uvw_matched": data["uvw_matched"],
        "matched_sid": matched_sid,
        "matched_orig_idx": data["matched_orig_idx"],
        "n_total": int(data["n_total"][0]),
        "n_matched": n_matched,
    }
    result["run_numbers"] = (data["run_numbers"]
                             if "run_numbers" in data else None)
    result["event_numbers"] = (data["event_numbers"]
                               if "event_numbers" in data else None)
    result["solid_angles"] = (data["solid_angles"]
                              if "solid_angles" in data else None)
    print(f"[INFO] Loaded {n_matched:,} matched events, normalized with {npho_scheme}")
    return result


def run_baselines_raw(
    x_npho_norm: np.ndarray,
    combined_mask: np.ndarray,
    npho_xf: 'NphoTransform',
    solid_angles: Optional[np.ndarray],
    matched_sid: np.ndarray,
) -> dict:
    """Run neighbor-avg and solid-angle baselines, return raw predictions.

    The baselines internally denormalize to raw space for averaging,
    so the raw predictions are scheme-independent.

    Returns dict with keys: avg_pred_raw, and optionally sa_pred_raw.
    Values are (M,) float32 arrays of raw npho predictions at the masked sensor.
    """
    from lib.inpainter_baselines import (
        NeighborAverageBaseline, SolidAngleWeightedBaseline,
    )
    n_matched = len(x_npho_norm)
    ev_range = np.arange(n_matched)
    result = {}

    print("[INFO] Running NeighborAverageBaseline...")
    avg_baseline = NeighborAverageBaseline(k=1)
    avg_preds = avg_baseline.predict(x_npho_norm, combined_mask,
                                     npho_transform=npho_xf)
    # Extract prediction at masked sensor, convert to raw
    avg_at_mask = avg_preds[ev_range, matched_sid]
    result["avg_pred_raw"] = npho_xf.inverse(avg_at_mask).astype(np.float32)
    del avg_preds

    if solid_angles is not None:
        print("[INFO] Running SolidAngleWeightedBaseline...")
        sa_baseline = SolidAngleWeightedBaseline(k=1)
        sa_preds = sa_baseline.predict(x_npho_norm, combined_mask,
                                       solid_angles=solid_angles,
                                       npho_transform=npho_xf)
        sa_at_mask = sa_preds[ev_range, matched_sid]
        result["sa_pred_raw"] = npho_xf.inverse(sa_at_mask).astype(np.float32)
        del sa_preds

    return result


def load_baselines_raw(
    baselines_path: str,
    truth_at_mask: np.ndarray,
    matched_sid: np.ndarray,
    npho_xf: 'NphoTransform',
) -> dict:
    """Load raw baseline predictions and normalize to the target scheme.

    Returns dict mapping baseline name -> list-of-dicts (same format as
    _collect_baseline_fast).
    """
    data = np.load(baselines_path)
    baseline_results = {}
    n_matched = len(matched_sid)

    for key, name in [("avg_pred_raw", "avg"), ("sa_pred_raw", "sa")]:
        if key not in data:
            continue
        pred_raw = data[key]
        # Normalize raw predictions to the target scheme
        pred_norm = npho_xf.forward(
            np.maximum(pred_raw, npho_xf.domain_min())
        ).astype(np.float32)

        results = []
        for i in range(n_matched):
            sid = int(matched_sid[i])
            p = float(pred_norm[i])
            t = float(truth_at_mask[i, 0])
            err = p - t if t != MODEL_SENTINEL_NPHO else -999.0
            results.append({
                "event_idx": i,
                "sensor_id": sid,
                "mask_type": 0,
                "pred_npho": p,
                "truth_npho": t,
                "error_npho": err,
            })
        baseline_results[name] = results
        print(f"[INFO] Loaded {name} baseline: {n_matched} predictions "
              f"(normalized to {npho_xf.scheme})")

    return baseline_results


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

    # Model (not required for --manifest-only)
    model_group = parser.add_mutually_exclusive_group(required=False)
    model_group.add_argument("--checkpoint", "-c", type=str,
                             help="Path to inpainter checkpoint (.pth)")
    model_group.add_argument("--torchscript", "-t", type=str,
                             help="Path to TorchScript model (.pt)")

    # Data
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to input ROOT file (MC with uvwTruth)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    # Preparation modes
    parser.add_argument("--manifest-only", action="store_true",
                        help="Create manifest + raw baselines only (no model "
                             "needed). Use --input and --output. Subsequent "
                             "inference jobs use --load-manifest.")
    parser.add_argument("--load-manifest", type=str, default=None,
                        help="Load prepared data from this directory instead "
                             "of reading ROOT files (created by --manifest-only)")
    parser.add_argument("--baselines-from", type=str, default=None,
                        help="Load pre-computed raw baselines from this "
                             "directory (created by --manifest-only)")

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
    parser.add_argument("--local-fit-results", type=str, default=None,
                        help="Directory with pre-computed local-fit results "
                             "(from run_localfit_sensorfront.py)")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip saving manifest (for local-fit batch jobs)")

    args = parser.parse_args()

    # Validate argument combinations
    if args.manifest_only:
        if not args.input:
            parser.error("--manifest-only requires --input")
        if args.checkpoint or args.torchscript:
            parser.error("--manifest-only does not use --checkpoint/--torchscript")
    elif args.load_manifest:
        if not (args.checkpoint or args.torchscript):
            parser.error("--load-manifest requires --checkpoint or --torchscript")
    else:
        if not args.input:
            parser.error("--input is required (or use --load-manifest)")
        if not (args.checkpoint or args.torchscript):
            parser.error("--checkpoint or --torchscript is required "
                         "(or use --manifest-only)")

    os.makedirs(args.output, exist_ok=True)

    # ==================================================================
    #  MODE A: --manifest-only  (no model, prepare data + baselines)
    # ==================================================================
    if args.manifest_only:
        npho_scheme = args.npho_scheme or "log1p"
        npho_scale = DEFAULT_NPHO_SCALE
        npho_scale2 = DEFAULT_NPHO_SCALE2

        sensor_uvw = build_inner_sensor_uvw()

        loaded = load_and_match_streaming(
            input_path=args.input,
            sensor_uvw=sensor_uvw,
            npho_scheme=npho_scheme,
            npho_scale=npho_scale, npho_scale2=npho_scale2,
            tree_name=args.tree_name,
            max_events=args.max_events,
            du=args.du, dv=args.dv, w_max=args.w_max,
            sa_branch=args.solid_angle_branch,
            save_raw=True,
        )

        n_total = loaded["n_total"]
        n_matched = loaded.get("n_matched", 0)
        print(f"\n[INFO] Events total:   {n_total:>8,}")
        print(f"[INFO] Events matched: {n_matched:>8,} "
              f"({n_matched / n_total * 100:.1f}%)")

        if n_matched == 0:
            print("[ERROR] No events matched any sensor front region. Exiting.")
            sys.exit(1)

        # --- Run baselines and extract raw predictions ---
        x_norm = loaded["x_norm"]
        matched_sid = loaded["matched_sid"]
        n_matched = len(x_norm)
        ev_range = np.arange(n_matched)
        combined_mask = np.zeros((n_matched, N_CHANNELS), dtype=bool)
        combined_mask[ev_range, matched_sid] = True

        npho_xf = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale,
                                npho_scale2=npho_scale2)
        baseline_raw = run_baselines_raw(
            x_norm[:, :, 0], combined_mask, npho_xf,
            loaded.get("solid_angles"), matched_sid,
        )
        del x_norm
        gc.collect()

        # --- Save everything ---
        save_prepared_data(
            output_dir=args.output,
            loaded=loaded,
            input_path=args.input,
            max_events=args.max_events,
            npho_scheme=npho_scheme,
            npho_scale=npho_scale,
            npho_scale2=npho_scale2,
            baseline_raw=baseline_raw,
        )

        print(f"\n[INFO] Manifest-only mode complete. Output in {args.output}")
        print(f"[INFO] Next steps:")
        print(f"  1. Submit LocalFit:  bash macro/submit_localfit_sensorfront.sh "
              f"{os.path.join(args.output, '_sensorfront_manifest.npz')}")
        print(f"  2. Run ML inference: python macro/validate_inpainter_sensorfront.py "
              f"--checkpoint <ckpt> --load-manifest {args.output} "
              f"--baselines-from {args.output} --output <out_dir>")
        return

    # ==================================================================
    #  MODE B & C: Inference mode (with --load-manifest or --input)
    # ==================================================================

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, model_type, predict_channels, model_npho_scheme, \
        model_npho_scale, model_npho_scale2 = load_model(
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

    npho_scale = model_npho_scale
    npho_scale2 = model_npho_scale2

    # ------------------------------------------------------------------
    # 2. Load data: from prepared manifest or from ROOT files
    # ------------------------------------------------------------------
    if args.load_manifest:
        loaded = load_prepared_data(
            prepared_dir=args.load_manifest,
            npho_scheme=npho_scheme,
            npho_scale=npho_scale,
            npho_scale2=npho_scale2,
        )
    else:
        sensor_uvw = build_inner_sensor_uvw()
        loaded = load_and_match_streaming(
            input_path=args.input,
            sensor_uvw=sensor_uvw,
            npho_scheme=npho_scheme,
            npho_scale=npho_scale,
            npho_scale2=npho_scale2,
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
    # 3. Build per-event mask & run baselines BEFORE applying sentinels
    # ------------------------------------------------------------------
    ev_range = np.arange(n_matched)
    combined_mask = np.zeros((n_matched, N_CHANNELS), dtype=bool)
    combined_mask[ev_range, matched_sid] = True

    print(f"[INFO] Masked sensors: {n_matched:,} (1 per event)")

    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    artificial_mask = combined_mask

    npho_xf = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale,
                            npho_scale2=npho_scale2)

    baseline_results = {}

    if args.baselines_from:
        # Load pre-computed raw baselines (scheme-independent)
        bl_path = os.path.join(args.baselines_from, "_baselines_raw.npz")
        if os.path.isfile(bl_path):
            baseline_results = load_baselines_raw(
                bl_path, truth_at_mask, matched_sid, npho_xf,
            )
        else:
            print(f"[WARN] Baselines file not found: {bl_path}, "
                  f"computing inline")
            args.baselines_from = None

    if not args.baselines_from:
        # Compute baselines inline (current behavior)
        from lib.inpainter_baselines import (
            NeighborAverageBaseline, SolidAngleWeightedBaseline,
        )

        x_npho_orig = x_norm[:, :, 0]

        print("[INFO] Running NeighborAverageBaseline...")
        avg_baseline = NeighborAverageBaseline(k=1)
        avg_preds = avg_baseline.predict(x_npho_orig, combined_mask,
                                         npho_transform=npho_xf)
        avg_results = _collect_baseline_fast(
            avg_preds, truth_at_mask, matched_sid, n_matched,
        )
        del avg_preds
        baseline_results["avg"] = avg_results

        if solid_angles is not None:
            print("[INFO] Running SolidAngleWeightedBaseline...")
            sa_baseline = SolidAngleWeightedBaseline(k=1)
            sa_preds = sa_baseline.predict(x_npho_orig, combined_mask,
                                           solid_angles=solid_angles,
                                           npho_transform=npho_xf)
            sa_results = _collect_baseline_fast(
                sa_preds, truth_at_mask, matched_sid, n_matched,
            )
            del sa_preds
            baseline_results["sa"] = sa_results
        del x_npho_orig
    del solid_angles
    gc.collect()

    # ------------------------------------------------------------------
    # 4. Apply sentinels & run inference (with cache)
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

    # ------------------------------------------------------------------
    # 5. Save manifest for local-fit batch jobs (legacy path)
    # ------------------------------------------------------------------
    if not args.no_manifest and not args.load_manifest and args.input:
        file_list = expand_path(args.input)
        manifest_path = os.path.join(args.output, "_sensorfront_manifest.npz")
        np.savez_compressed(
            manifest_path,
            matched_orig_idx=matched_orig_idx,
            matched_sid=matched_sid,
            truth_at_mask=truth_at_mask,
            npho_scheme=np.array([npho_scheme]),
            file_list=np.array(file_list),
            max_events=np.array([args.max_events if args.max_events else -1]),
        )
        print(f"[INFO] Saved manifest to {manifest_path}")
        print(f"       Use run_localfit_sensorfront.py --manifest {manifest_path} "
              f"--file-index <0..{len(file_list)-1}> for batch jobs")

    # ------------------------------------------------------------------
    # 6. Load pre-computed local-fit results (if provided)
    # ------------------------------------------------------------------
    if args.local_fit_results:
        lf_results = _load_local_fit_results(
            args.local_fit_results, matched_orig_idx, matched_sid,
            truth_at_mask, npho_scheme,
            npho_scale=npho_scale, npho_scale2=npho_scale2,
        )
        if lf_results:
            baseline_results["localfit"] = lf_results

    # ------------------------------------------------------------------
    # 7. Compute baseline metrics
    # ------------------------------------------------------------------
    baseline_metrics_dict = {}
    for bname, bpreds in baseline_results.items():
        baseline_metrics_dict[bname] = compute_baseline_metrics(bpreds)

    # ------------------------------------------------------------------
    # 8. Collect ML predictions
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
    # 9. Print summary
    # ------------------------------------------------------------------
    _print_sensorfront_summary(
        n_total, n_matched, ml_metrics, baseline_metrics_dict, npho_scheme,
        npho_scale=npho_scale, npho_scale2=npho_scale2,
    )

    # ------------------------------------------------------------------
    # 10. Save outputs
    # ------------------------------------------------------------------
    # Add w_fip to baseline results too (for ROOT output)
    for bname in baseline_results:
        _add_w_fip(baseline_results[bname], uvw_matched, matched_sid)

    pred_file = os.path.join(args.output, "predictions_sensorfront.root")
    save_predictions(
        pred_list, pred_file,
        predict_channels=predict_channels,
        npho_scheme=npho_scheme,
        npho_scale=npho_scale,
        npho_scale2=npho_scale2,
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

def _load_local_fit_results(
    results_dir: str,
    matched_orig_idx: np.ndarray,
    matched_sid: np.ndarray,
    truth_at_mask: np.ndarray,
    npho_scheme: str,
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
) -> list:
    """Load pre-computed local-fit results from batch jobs.

    Reads all localfit_file*.npz files from results_dir, matches them
    to the current manifest's (matched_orig_idx, matched_sid), and
    returns a prediction list compatible with compute_baseline_metrics().
    """
    from glob import glob as _glob

    result_files = sorted(_glob(os.path.join(results_dir, "localfit_file*.npz")))
    if not result_files:
        print(f"[WARNING] No localfit_file*.npz found in {results_dir}")
        return []

    print(f"[INFO] Loading local-fit results from {len(result_files)} file(s)")

    transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)

    # Build lookup: (global_event_idx, sensor_id) -> normalized prediction
    lf_lookup = {}
    for rf in result_files:
        data = np.load(rf)
        gev_arr = data["global_event_idx"]
        sid_arr = data["sensor_id"]
        pred_raw = data["pred_npho_raw"]
        pred_norm = transform.forward(
            np.maximum(pred_raw, transform.domain_min())
        ).astype(np.float32)
        for j in range(len(gev_arr)):
            lf_lookup[(int(gev_arr[j]), int(sid_arr[j]))] = float(pred_norm[j])

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
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
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
    transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)

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
