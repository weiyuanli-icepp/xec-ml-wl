#!/usr/bin/env python3
"""
Run regressor on CEX real data with dead-channel recovery strategies.

Loads a CEX ROOT file (from CEXPreprocess.C), applies three dead-channel
recovery strategies — raw (no fill), neighbor averaging, and ML inpainting
— then runs the ONNX regressor on all variants and saves results.

Usage:
    # Raw + neighbor average only (no inpainter)
    python val_data/run_regressor_cex.py \
        --regressor-onnx regressor.onnx \
        --input CEX23_patch13_r557545_n100.root \
        --output regressor_cex_results.root

    # All three strategies
    python val_data/run_regressor_cex.py \
        --regressor-onnx regressor.onnx \
        --inpainter-torchscript inpainter.pt \
        --input CEX23_patch13_r557545_n100.root \
        --output regressor_cex_results.root

    # With checkpoint fallback instead of TorchScript
    python val_data/run_regressor_cex.py \
        --regressor-onnx regressor.onnx \
        --inpainter-checkpoint checkpoint_best.pth \
        --input CEX23_patch13_r557545_n100.root \
        --output regressor_cex_results.root
"""
import os
import sys
import argparse
import time
import numpy as np
import uproot
import onnxruntime as ort

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
from lib.inpainter_baselines import NeighborAverageBaseline

N_CHANNELS = 4760
SENTINEL_NPHO = -1.0


def normalize_input(npho_raw, time_raw,
                    npho_scale, npho_scale2, time_scale, time_shift,
                    sentinel_time, npho_threshold, sentinel_npho=SENTINEL_NPHO):
    """Normalize raw npho/time arrays to model input space.

    Identical to inference_real_data.normalize_input().
    """
    mask_npho_invalid = (npho_raw > 9e9) | np.isnan(npho_raw)
    domain_min = -npho_scale * 0.999
    mask_domain_break = (~mask_npho_invalid) & (npho_raw < domain_min)

    mask_time_invalid = (mask_npho_invalid
                         | (npho_raw < npho_threshold)
                         | (np.abs(time_raw) > 9e9)
                         | np.isnan(time_raw))

    npho_safe = np.where(mask_npho_invalid | mask_domain_break, 0.0, npho_raw)
    npho_norm = np.log1p(npho_safe / npho_scale) / npho_scale2
    npho_norm[mask_npho_invalid] = sentinel_npho
    npho_norm[mask_domain_break] = 0.0

    time_norm = (time_raw / time_scale) - time_shift
    time_norm[mask_time_invalid] = sentinel_time

    return np.stack([npho_norm, time_norm], axis=-1).astype(np.float32)


# ------------------------------------------------------------------
# Dead channel detection
# ------------------------------------------------------------------

def detect_dead_channels(dead_branch, npho_raw):
    """Get 1-D boolean dead mask from the ``dead`` branch or sentinel fallback.

    Parameters
    ----------
    dead_branch : np.ndarray or None
        ``dead`` branch from CEXPreprocess.C, shape (B, 4760), Bool_t.
        If present, uses first event's dead flags (constant per run).
    npho_raw : np.ndarray
        Raw npho array, shape (B, 4760).

    Returns
    -------
    dead_mask : np.ndarray, shape (4760,), bool
    """
    if dead_branch is not None:
        # dead is constant per run — use first event
        return dead_branch[0].astype(bool)

    # Fallback: channel is dead if sentinel in >90% of events
    invalid = npho_raw > 9e9
    frac = invalid.mean(axis=0)
    return frac > 0.9


# ------------------------------------------------------------------
# Dead channel fill strategies
# ------------------------------------------------------------------

def fill_dead_neighbor_avg(x_norm, dead_mask_1d, baseline):
    """Replace dead npho with neighbor average; time stays sentinel.

    Parameters
    ----------
    x_norm : np.ndarray, shape (B, 4760, 2)
        Normalized input (npho, time).
    dead_mask_1d : np.ndarray, shape (4760,), bool
    baseline : NeighborAverageBaseline

    Returns
    -------
    x_filled : np.ndarray, shape (B, 4760, 2)
    """
    x_filled = x_norm.copy()
    npho_norm = x_filled[:, :, 0]  # (B, 4760)
    mask_2d = np.broadcast_to(dead_mask_1d[None, :], npho_norm.shape).copy()
    npho_filled = baseline.predict(npho_norm, mask_2d)
    x_filled[:, :, 0] = npho_filled
    # Time stays as-is (sentinel for dead channels — consistent with training)
    return x_filled


def fill_dead_inpainter(x_norm, dead_mask_1d, model, device, batch_size=512):
    """Replace dead npho+time with inpainter predictions.

    Parameters
    ----------
    x_norm : np.ndarray, shape (B, 4760, 2)
    dead_mask_1d : np.ndarray, shape (4760,), bool
    model : TorchScript module or XEC_Inpainter
    device : str
    batch_size : int

    Returns
    -------
    x_filled : np.ndarray, shape (B, 4760, 2)
    """
    import torch

    x_filled = x_norm.copy()
    N = x_filled.shape[0]
    mask_2d = np.broadcast_to(dead_mask_1d[None, :], (N, N_CHANNELS)).copy()

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = torch.from_numpy(x_filled[start:end]).to(device)   # (b, 4760, 2)
        m_batch = torch.from_numpy(mask_2d[start:end]).to(device)    # (b, 4760)

        with torch.no_grad():
            pred = model(x_batch, m_batch)  # (b, 4760, C) where C=1 or 2

        pred_np = pred.cpu().numpy()

        if pred_np.shape[-1] >= 2:
            # Inpainter predicts both npho and time
            x_filled[start:end, dead_mask_1d, 0] = pred_np[:, dead_mask_1d, 0]
            x_filled[start:end, dead_mask_1d, 1] = pred_np[:, dead_mask_1d, 1]
        else:
            # Inpainter predicts npho only
            x_filled[start:end, dead_mask_1d, 0] = pred_np[:, dead_mask_1d, 0]

    return x_filled


# ------------------------------------------------------------------
# ONNX regressor inference
# ------------------------------------------------------------------

def run_regressor_onnx(session, x_norm, task_map, batch_size=1024):
    """Run batched ONNX regressor inference.

    Parameters
    ----------
    session : ort.InferenceSession
    x_norm : np.ndarray, shape (B, 4760, 2)
    task_map : dict
        {task_name: onnx_output_name}
    batch_size : int

    Returns
    -------
    results : dict
        {task_name: np.ndarray of shape (B,) or (B, D)}
    """
    input_name = session.get_inputs()[0].name
    output_names = list(task_map.values())
    N = x_norm.shape[0]

    accum = {task: [] for task in task_map}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ort_inputs = {input_name: x_norm[start:end]}
        outputs = session.run(output_names, ort_inputs)
        out_dict = dict(zip(output_names, outputs))
        for task, oname in task_map.items():
            accum[task].append(out_dict[oname])

    results = {}
    for task, chunks in accum.items():
        arr = np.concatenate(chunks, axis=0)
        results[task] = arr.flatten() if arr.shape[-1] == 1 else arr
    return results


# ------------------------------------------------------------------
# Auto-detect tasks from ONNX
# ------------------------------------------------------------------

def detect_tasks(session):
    """Detect tasks from ONNX output names.

    Output names follow ``output_{task}`` convention.

    Returns
    -------
    task_map : dict
        {task_name: onnx_output_name}
    """
    task_map = {}
    for out in session.get_outputs():
        name = out.name
        if name.startswith("output_"):
            task_name = name[len("output_"):]
            task_map[task_name] = name
        elif name == "output":
            task_map["angle"] = name
        else:
            task_map[name] = name
    return task_map


# ------------------------------------------------------------------
# Expand multi-dim task outputs into flat branch names
# ------------------------------------------------------------------

# For tasks that produce >1 value per event, define component names.
TASK_COMPONENTS = {
    "angle": ["theta", "phi"],
    "uvwFI": ["u", "v", "w"],
}


def expand_task_results(task_results):
    """Expand multi-dim task arrays into {branch_name: 1-D array}.

    Parameters
    ----------
    task_results : dict
        {task_name: np.ndarray}  — shape (B,) for scalar tasks, (B, D) for vector.

    Returns
    -------
    flat : dict
        {branch_name: np.ndarray shape (B,)}
    """
    flat = {}
    for task, arr in task_results.items():
        if arr.ndim == 1:
            flat[task] = arr
        elif task in TASK_COMPONENTS:
            for i, comp in enumerate(TASK_COMPONENTS[task]):
                flat[comp] = arr[:, i]
        else:
            # Generic: task_0, task_1, ...
            for i in range(arr.shape[1]):
                flat[f"{task}_{i}"] = arr[:, i]
    return flat


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run regressor on CEX real data with dead-channel recovery",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument("--regressor-onnx", required=True,
                        help="Path to regressor ONNX model")
    parser.add_argument("--input", required=True,
                        help="Input CEX ROOT file (from CEXPreprocess.C)")
    parser.add_argument("--output", default=None,
                        help="Output ROOT file (default: auto-generated)")

    # Inpainter (mutually exclusive)
    inp_group = parser.add_mutually_exclusive_group()
    inp_group.add_argument("--inpainter-torchscript",
                           help="Path to inpainter TorchScript (.pt)")
    inp_group.add_argument("--inpainter-checkpoint",
                           help="Path to inpainter checkpoint (.pth)")

    # Dead channel source overrides
    dead_group = parser.add_mutually_exclusive_group()
    dead_group.add_argument("--dead-from-db", type=int, metavar="RUN",
                            help="Query dead channels from DB for this run")
    dead_group.add_argument("--dead-from-file", type=str, metavar="PATH",
                            help="Load dead channel list from text file")

    # Task override
    parser.add_argument("--active-tasks", nargs="+", default=None,
                        help="Override auto-detected tasks (e.g. energy angle)")

    # Processing
    parser.add_argument("--chunksize", type=int, default=2048,
                        help="Events per chunk (default: 2048)")
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Batch size for inpainter inference (default: 1024)")
    parser.add_argument("--device", default="cpu",
                        help="Device for inpainter (default: cpu)")
    parser.add_argument("--neighbor-k", type=int, default=1,
                        help="Neighbor hops for averaging (default: 1)")
    parser.add_argument("--tree", default="tree",
                        help="TTree name in input file (default: tree)")

    # Normalization params
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE)
    parser.add_argument("--npho-scale2", type=float, default=DEFAULT_NPHO_SCALE2)
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE)
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT)
    parser.add_argument("--sentinel-time", type=float, default=DEFAULT_SENTINEL_TIME)
    parser.add_argument("--npho-threshold", type=float, default=DEFAULT_NPHO_THRESHOLD)

    args = parser.parse_args()

    # --- Auto-generate output filename ---
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.input))[0]
        args.output = f"regressor_{base}.root"

    # ======================================================================
    # 1. Load regressor ONNX
    # ======================================================================
    print(f"[INFO] Loading regressor: {args.regressor_onnx}")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        reg_session = ort.InferenceSession(args.regressor_onnx, providers=providers)
    except Exception:
        reg_session = ort.InferenceSession(args.regressor_onnx,
                                           providers=["CPUExecutionProvider"])

    # Detect tasks
    task_map = detect_tasks(reg_session)
    if args.active_tasks:
        task_map = {t: task_map[t] for t in args.active_tasks if t in task_map}
    print(f"[INFO] Active tasks: {list(task_map.keys())}")

    # ======================================================================
    # 2. Load inpainter (optional)
    # ======================================================================
    inpainter_model = None
    have_inpainter = False

    if args.inpainter_torchscript:
        import torch
        print(f"[INFO] Loading TorchScript inpainter: {args.inpainter_torchscript}")
        inpainter_model = torch.jit.load(args.inpainter_torchscript,
                                         map_location=args.device)
        inpainter_model.eval()
        have_inpainter = True
    elif args.inpainter_checkpoint:
        # Fallback: load from checkpoint
        from val_data.validate_inpainter_real import load_inpainter_model
        inpainter_model, _ = load_inpainter_model(args.inpainter_checkpoint,
                                                   device=args.device)
        have_inpainter = True

    if not have_inpainter:
        print("[INFO] No inpainter provided — inpainted strategy will output 1e10")

    # ======================================================================
    # 3. Build neighbor baseline
    # ======================================================================
    print(f"[INFO] Building neighbor map (k={args.neighbor_k})")
    baseline = NeighborAverageBaseline(k=args.neighbor_k)

    # ======================================================================
    # 4. Resolve dead channel mask
    # ======================================================================
    dead_mask_1d = None  # Will be set on first chunk or from override

    if args.dead_from_db is not None:
        from lib.db_utils import get_dead_channel_mask
        dead_mask_1d = get_dead_channel_mask(args.dead_from_db)
        print(f"[INFO] Dead channels from DB (run {args.dead_from_db}): "
              f"{dead_mask_1d.sum()}")
    elif args.dead_from_file is not None:
        from lib.db_utils import load_dead_channel_list
        dead_indices = load_dead_channel_list(args.dead_from_file)
        dead_mask_1d = np.zeros(N_CHANNELS, dtype=bool)
        dead_mask_1d[dead_indices] = True
        print(f"[INFO] Dead channels from file: {dead_mask_1d.sum()}")

    # ======================================================================
    # 5. Chunk-based processing
    # ======================================================================
    # Determine which branches to read
    model_branches = ["npho", "relative_time"]
    meta_branches = ["run", "event", "energyTruth", "energyReco",
                     "Ebgo", "Angle", "gstatus", "nDead"]
    dead_branches = ["dead"]
    all_requested = model_branches + meta_branches + dead_branches

    # Check available branches
    with uproot.open(args.input) as f_check:
        available = set(f_check[args.tree].keys())
    read_branches = [b for b in all_requested if b in available]
    has_dead_branch = "dead" in available

    if not has_dead_branch and dead_mask_1d is None:
        print("[WARN] No 'dead' branch and no --dead-from-db/--dead-from-file. "
              "Will detect dead channels from sentinel values.")

    # Accumulators for output branches
    out_data = {b: [] for b in meta_branches if b in available}

    # Per-task, per-strategy accumulators
    # Strategy suffixes: raw, neighavg, inpainted
    strategies = ["raw", "neighavg", "inpainted"]
    # We'll build flat branch names after first chunk (once we know task dims)
    strategy_accum = {s: [] for s in strategies}

    total_events = 0
    t0 = time.time()

    print(f"[INFO] Processing {args.input} (chunksize={args.chunksize})")

    for arrays in uproot.iterate(
        f"{args.input}:{args.tree}",
        expressions=read_branches,
        step_size=args.chunksize,
        library="np",
    ):
        B = len(arrays["run"])

        # --- Normalize ---
        npho_raw = arrays["npho"].astype(np.float32)
        time_raw = arrays["relative_time"].astype(np.float32)

        x_norm = normalize_input(
            npho_raw, time_raw,
            args.npho_scale, args.npho_scale2,
            args.time_scale, args.time_shift,
            args.sentinel_time, args.npho_threshold,
        )

        # --- Resolve dead mask (once) ---
        if dead_mask_1d is None:
            dead_br = arrays.get("dead", None)
            dead_mask_1d = detect_dead_channels(dead_br, npho_raw)
            n_dead = int(dead_mask_1d.sum())
            print(f"[INFO] Dead channels detected: {n_dead}")

        # --- Strategy 1: raw (no fill) ---
        x_raw = x_norm  # no copy needed, we don't modify it

        # --- Strategy 2: neighbor average ---
        x_neighavg = fill_dead_neighbor_avg(x_norm, dead_mask_1d, baseline)

        # --- Strategy 3: inpainted ---
        if have_inpainter:
            x_inpainted = fill_dead_inpainter(
                x_norm, dead_mask_1d, inpainter_model,
                args.device, batch_size=args.batch_size,
            )
        else:
            x_inpainted = None

        # --- Run regressor on each strategy ---
        res_raw = run_regressor_onnx(reg_session, x_raw, task_map,
                                     batch_size=args.chunksize)
        res_neighavg = run_regressor_onnx(reg_session, x_neighavg, task_map,
                                          batch_size=args.chunksize)
        if x_inpainted is not None:
            res_inpainted = run_regressor_onnx(reg_session, x_inpainted, task_map,
                                               batch_size=args.chunksize)
        else:
            # Fill with 1e10 sentinel
            res_inpainted = {}
            for task, arr in res_raw.items():
                res_inpainted[task] = np.full_like(arr, 1e10)

        # Expand multi-dim tasks into flat branches
        flat_raw = expand_task_results(res_raw)
        flat_neighavg = expand_task_results(res_neighavg)
        flat_inpainted = expand_task_results(res_inpainted)

        strategy_accum["raw"].append(flat_raw)
        strategy_accum["neighavg"].append(flat_neighavg)
        strategy_accum["inpainted"].append(flat_inpainted)

        # --- Accumulate metadata ---
        for b in out_data:
            out_data[b].append(arrays[b])

        total_events += B
        elapsed = time.time() - t0
        print(f"  {total_events} events ({elapsed:.1f}s, "
              f"{total_events / elapsed:.0f} evt/s)", end="\r")

    print()  # newline after \r

    # ======================================================================
    # 6. Concatenate and write output
    # ======================================================================
    # Metadata
    final = {}
    for b, chunks in out_data.items():
        arr = np.concatenate(chunks)
        if b in ("run", "event", "gstatus", "nDead"):
            final[b] = arr.astype(np.int32)
        else:
            final[b] = arr.astype(np.float32)

    # Strategy branches: {branch}_{strategy}
    for strategy in strategies:
        chunks = strategy_accum[strategy]
        if not chunks:
            continue
        # Get all branch names from first chunk
        branch_names = list(chunks[0].keys())
        for bname in branch_names:
            key = f"{bname}_{strategy}"
            final[key] = np.concatenate(
                [c[bname] for c in chunks]
            ).astype(np.float32)

    # Write
    print(f"[INFO] Writing {args.output} ({total_events} events, "
          f"{len(final)} branches)")

    with uproot.recreate(args.output) as f_out:
        branch_types = {}
        for k, v in final.items():
            if v.dtype == np.int32:
                branch_types[k] = np.int32
            else:
                branch_types[k] = np.float32
        f_out.mktree("tree", branch_types)
        f_out["tree"].extend(final)

    elapsed = time.time() - t0
    print(f"[DONE] {total_events} events in {elapsed:.1f}s "
          f"({total_events / elapsed:.0f} evt/s)")
    print(f"[INFO] Output: {args.output}")
    print(f"[INFO] Branches: {sorted(final.keys())}")


if __name__ == "__main__":
    main()
