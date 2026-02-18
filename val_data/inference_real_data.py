#!/usr/bin/env python3
"""
Run inference on real data using ONNX model (single-task or multi-task).

Supports both:
- Legacy single-task angle model (XECEncoder)
- Multi-task model (XECMultiHeadModel) with angle, energy, timing, uvwFI

Usage:
    # Single-task angle model
    python val_data/inference_real_data.py \\
        --onnx model.onnx \\
        --input DataGammaAngle_RunXXXX.root \\
        --output Output_RunXXXX.root

    # Multi-task model (auto-detects outputs from ONNX)
    python val_data/inference_real_data.py \\
        --onnx model_multitask.onnx \\
        --input DataGammaAngle_RunXXXX.root \\
        --output Output_RunXXXX.root
"""
import os
import sys
import argparse
import numpy as np
import uproot
import onnxruntime as ort
import time
import re

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.geom_defs import (
    DEFAULT_NPHO_SCALE,
    DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE,
    DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_TIME,
    DEFAULT_NPHO_THRESHOLD,
)


def get_opening_angle_deg(theta1, phi1, theta2, phi2):
    """
    Calculates the opening angle (in degrees) between two vectors defined by (theta, phi).
    Uses MEG II coordinate convention: z = cos(theta), with theta measured from +z axis.
    """
    t1 = np.deg2rad(theta1)
    p1 = np.deg2rad(phi1)
    t2 = np.deg2rad(theta2)
    p2 = np.deg2rad(phi2)

    # Convert to unit vectors (MEG II convention)
    x1 = -np.sin(t1) * np.cos(p1)
    y1 = np.sin(t1) * np.sin(p1)
    z1 = np.cos(t1)

    x2 = -np.sin(t2) * np.cos(p2)
    y2 = np.sin(t2) * np.sin(p2)
    z2 = np.cos(t2)

    dot = x1*x2 + y1*y2 + z1*z2
    dot = np.clip(dot, -1.0, 1.0)

    return np.rad2deg(np.arccos(dot))


def normalize_input(npho_raw, time_raw,
                    npho_scale, npho_scale2, time_scale, time_shift,
                    sentinel_time, npho_threshold, sentinel_npho=-1.0):
    """
    Normalize input data matching the training preprocessing in dataset.py.

    Normalization scheme:
    - npho > 9e9 or isnan: truly invalid (dead/missing sensor) → sentinel_npho
    - npho below domain minimum: domain-breaking → 0.0
    - npho < npho_threshold: npho valid, time set to sentinel (timing unreliable)
    - otherwise: normal normalization for both (negative npho values allowed)

    npho_norm = log1p(npho / npho_scale) / npho_scale2
    time_norm = time / time_scale - time_shift
    """
    # True invalids: dead/missing sensors, corrupted data
    mask_npho_invalid = (npho_raw > 9e9) | np.isnan(npho_raw)
    # Domain-breaking values for log1p
    domain_min = -npho_scale * 0.999
    mask_domain_break = (~mask_npho_invalid) & (npho_raw < domain_min)

    # Identify invalid time values
    mask_time_invalid = mask_npho_invalid | (npho_raw < npho_threshold) | (np.abs(time_raw) > 9e9) | np.isnan(time_raw)

    # Normalize npho: log1p transform (allow negatives through)
    npho_safe = np.where(mask_npho_invalid | mask_domain_break, 0.0, npho_raw)
    npho_norm = np.log1p(npho_safe / npho_scale) / npho_scale2
    npho_norm[mask_npho_invalid] = sentinel_npho  # dead channel → npho sentinel
    npho_norm[mask_domain_break] = 0.0                  # domain break → zero signal

    # Normalize time: linear transform
    time_norm = (time_raw / time_scale) - time_shift
    time_norm[mask_time_invalid] = sentinel_time

    return np.stack([npho_norm, time_norm], axis=-1).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Run Inference on Real Data (ONNX)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--onnx", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--input", type=str, required=True, help="Input ROOT file (Real Data)")
    parser.add_argument("--output", type=str, default=None, help="Output ROOT file")
    parser.add_argument("--tree", type=str, default="tree", help="TTree name")
    parser.add_argument("--chunksize", type=int, default=1024, help="Inference chunk size")

    # Input branch names
    parser.add_argument("--npho_branch", type=str, default="npho",
                        help="Branch name for photon counts (default: npho)")
    parser.add_argument("--time_branch", type=str, default="relative_time",
                        help="Branch name for timing (default: relative_time)")

    # Preprocessing Params (MUST MATCH TRAINING!)
    parser.add_argument("--npho_scale", type=float, default=DEFAULT_NPHO_SCALE,
                        help=f"Npho scale for log1p (default: {DEFAULT_NPHO_SCALE})")
    parser.add_argument("--npho_scale2", type=float, default=DEFAULT_NPHO_SCALE2,
                        help=f"Npho scale2 for log1p divisor (default: {DEFAULT_NPHO_SCALE2})")
    parser.add_argument("--time_scale", type=float, default=DEFAULT_TIME_SCALE,
                        help=f"Time scale (default: {DEFAULT_TIME_SCALE})")
    parser.add_argument("--time_shift", type=float, default=DEFAULT_TIME_SHIFT,
                        help=f"Time shift (default: {DEFAULT_TIME_SHIFT})")
    parser.add_argument("--sentinel_time", type=float, default=DEFAULT_SENTINEL_TIME,
                        help=f"Sentinel value for invalid time channels (default: {DEFAULT_SENTINEL_TIME})")
    parser.add_argument("--npho_threshold", type=float, default=DEFAULT_NPHO_THRESHOLD,
                        help=f"Npho threshold for valid timing (default: {DEFAULT_NPHO_THRESHOLD})")

    args = parser.parse_args()

    # Auto-generate output filename if not provided
    if args.output is None:
        m = re.match(r".*_(\d{6}-\d{6})\.root$", os.path.basename(args.input))
        if m:
            run_range = m.group(1)
            args.output = f"inference_results_Run{run_range}.root"
        else:
            args.output = "inference_results.root"

    # 1. Load ONNX Model
    print(f"[INFO] Loading Model: {args.onnx}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(args.onnx, providers=providers)
    except Exception as e:
        print(f"[WARN] Failed to load CUDA provider: {e}")
        session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])

    # Detect model outputs
    input_name = session.get_inputs()[0].name
    output_names = [out.name for out in session.get_outputs()]
    print(f"[INFO] Model outputs: {output_names}")

    # Determine if multi-task model
    is_multi_task = len(output_names) > 1 or any("output_" in name for name in output_names)

    # Map output names to task names
    task_outputs = {}
    for name in output_names:
        if name.startswith("output_"):
            task_name = name.replace("output_", "")
            task_outputs[task_name] = name
        elif name == "output":
            task_outputs["angle"] = name
        else:
            task_outputs[name] = name

    print(f"[INFO] Detected tasks: {list(task_outputs.keys())}")

    # 2. Open Input File
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"[INFO] Processing: {args.input}")
    print(f"[INFO] Normalization params: npho_scale={args.npho_scale}, npho_scale2={args.npho_scale2}, "
          f"time_scale={args.time_scale}, time_shift={args.time_shift}")

    model_branches = [args.npho_branch, args.time_branch]
    meta_branches = [
        "run", "event",
        "emiAng",      # For true_theta/phi
        "energyReco",  # For energy comparison
        "timeTruth",   # For timing comparison
        "xyzRecoFI",   # For position comparison
        "uvwRecoFI",   # For uvw position
        "xyzVTX",      # Vertex position
    ]

    read_branches = model_branches + meta_branches

    # Initialize results dictionary based on detected tasks
    results = {
        "run_id": [], "event_id": [],
    }

    # Add task-specific output columns
    if "angle" in task_outputs:
        results.update({
            "pred_theta": [], "pred_phi": [],
            "true_theta": [], "true_phi": [],
            "opening_angle": [],
        })
    if "energy" in task_outputs:
        results.update({
            "pred_energy": [],
            "true_energy": [],
        })
    if "timing" in task_outputs:
        results.update({
            "pred_timing": [],
            "true_timing": [],
        })
    if "uvwFI" in task_outputs:
        results.update({
            "pred_u": [], "pred_v": [], "pred_w": [],
            "true_u": [], "true_v": [], "true_w": [],
        })

    # Additional metadata columns
    results.update({
        "x_vtx": [], "y_vtx": [], "z_vtx": [],
    })

    # 3. Inference Loop
    total_events = 0
    start_time = time.time()

    with uproot.open(args.input) as f:
        if args.tree not in f:
            print(f"[ERROR] Tree '{args.tree}' not found in file.")
            return

        tree = f[args.tree]
        num_entries = tree.num_entries

        # Filter to only existing branches
        available_branches = set(tree.keys())
        read_branches = [b for b in read_branches if b in available_branches]

        for arrays in tree.iterate(read_branches, step_size=args.chunksize, library="np"):

            # --- A. Preprocessing (Model Input) ---
            npho_raw = arrays[args.npho_branch].astype("float32")
            time_raw = arrays[args.time_branch].astype("float32")

            X_batch = normalize_input(
                npho_raw, time_raw,
                args.npho_scale, args.npho_scale2,
                args.time_scale, args.time_shift,
                args.sentinel_time, args.npho_threshold
            )

            # --- B. Inference ---
            ort_inputs = {input_name: X_batch}
            outputs = session.run(output_names, ort_inputs)

            # Map outputs to task names
            output_dict = {name: out for name, out in zip(output_names, outputs)}

            batch_size = len(X_batch)

            # --- C. Extract Metadata/Truth ---
            results["run_id"].append(arrays["run"])
            results["event_id"].append(arrays["event"])

            # Angle task
            if "angle" in task_outputs:
                preds = output_dict[task_outputs["angle"]]  # (B, 2)

                if "emiAng" in arrays:
                    true_ang = arrays["emiAng"]
                    t_theta = true_ang[:, 0]
                    t_phi = true_ang[:, 1]
                else:
                    t_theta = np.zeros(batch_size)
                    t_phi = np.zeros(batch_size)

                oa = get_opening_angle_deg(preds[:, 0], preds[:, 1], t_theta, t_phi)

                results["pred_theta"].append(preds[:, 0])
                results["pred_phi"].append(preds[:, 1])
                results["true_theta"].append(t_theta)
                results["true_phi"].append(t_phi)
                results["opening_angle"].append(oa)

            # Energy task
            if "energy" in task_outputs:
                preds = output_dict[task_outputs["energy"]]  # (B, 1)
                if "energyReco" in arrays:
                    true_energy = arrays["energyReco"]
                else:
                    true_energy = np.zeros(batch_size)

                results["pred_energy"].append(preds.flatten())
                results["true_energy"].append(true_energy.flatten() if true_energy.ndim > 1 else true_energy)

            # Timing task
            if "timing" in task_outputs:
                preds = output_dict[task_outputs["timing"]]  # (B, 1)
                if "timeTruth" in arrays:
                    true_timing = arrays["timeTruth"]
                else:
                    true_timing = np.zeros(batch_size)

                results["pred_timing"].append(preds.flatten())
                results["true_timing"].append(true_timing.flatten() if true_timing.ndim > 1 else true_timing)

            # Position (uvwFI) task
            if "uvwFI" in task_outputs:
                preds = output_dict[task_outputs["uvwFI"]]  # (B, 3)
                if "uvwRecoFI" in arrays:
                    true_uvw = arrays["uvwRecoFI"]
                    tu, tv, tw = true_uvw[:, 0], true_uvw[:, 1], true_uvw[:, 2]
                else:
                    tu, tv, tw = np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)

                results["pred_u"].append(preds[:, 0])
                results["pred_v"].append(preds[:, 1])
                results["pred_w"].append(preds[:, 2])
                results["true_u"].append(tu)
                results["true_v"].append(tv)
                results["true_w"].append(tw)

            # Vertex position
            if "xyzVTX" in arrays:
                vtx = arrays["xyzVTX"]
                vx, vy, vz = vtx[:, 0], vtx[:, 1], vtx[:, 2]
            else:
                vx, vy, vz = np.zeros(batch_size), np.zeros(batch_size), np.zeros(batch_size)

            results["x_vtx"].append(vx)
            results["y_vtx"].append(vy)
            results["z_vtx"].append(vz)

            total_events += batch_size
            if total_events % 50000 == 0:
                print(f"   Processed {total_events}/{num_entries} events...")

    # 4. Save Output
    print(f"[INFO] Saving results to: {args.output}")

    # Concatenate all lists
    final_data = {}
    for k, v in results.items():
        if v:  # Only include non-empty lists
            final_data[k] = np.concatenate(v)

    with uproot.recreate(args.output) as f_out:
        branch_types = {
            "run_id": np.int32,
            "event_id": np.int32
        }
        for k in final_data.keys():
            if k not in branch_types:
                branch_types[k] = np.float32

        f_out.mktree("val_tree", branch_types)
        f_out["val_tree"].extend(final_data)

    duration = time.time() - start_time
    print(f"[DONE] Processed {total_events} events in {duration:.1f}s ({total_events/duration:.1f} evt/s)")
    print(f"[INFO] Output branches: {list(final_data.keys())}")


if __name__ == "__main__":
    main()
