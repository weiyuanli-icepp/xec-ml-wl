#!/usr/bin/env python3
"""
Run validation with an existing checkpoint (.pth) or ONNX model (.onnx).

Supports two modes:

1. **Standard validation** — generate predictions and resolution plots:

    python macro/validate_regressor.py checkpoint.pth --val_path data/val/
    python macro/validate_regressor.py model.onnx --val_path data/cex/ \\
        --config config.yaml --tasks energy

2. **Dead-channel recovery** — compare raw, neighbor-avg, and inpainted strategies:

    python macro/validate_regressor.py model.onnx --val_path data/cex/CEX_patch13.root \\
        --config config.yaml --tasks energy --dead-channel

    python macro/validate_regressor.py model.onnx --val_path data/cex/CEX_patch13.root \\
        --config config.yaml --tasks energy --dead-channel \\
        --inpainter-torchscript inpainter.pt
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
warnings.filterwarnings("ignore", message="Can't initialize NVML")
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.dataset import get_dataloader
from lib.config import load_config
from lib.train_regressor import save_validation_artifacts
from lib.geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME,
    DEFAULT_NPHO_THRESHOLD,
)
from lib.normalization import NphoTransform

N_CHANNELS = 4760
SENTINEL_NPHO = -1.0

# For tasks that produce >1 value per event, define component names.
TASK_COMPONENTS = {
    "angle": ["theta", "phi"],
    "uvwFI": ["u", "v", "w"],
}


def _print_timing(elapsed, n_events, backend, batch_times=None):
    """Print inference timing summary."""
    if n_events == 0:
        print(f"[INFO] {backend} inference completed in {elapsed:.1f}s (0 events)")
        return

    avg_us = elapsed / n_events * 1e6
    throughput = n_events / elapsed
    print(f"\n=== {backend} Inference Timing ===")
    print(f"  Total:      {elapsed:.1f}s  ({n_events} events)")
    print(f"  Throughput: {throughput:.0f} events/s")
    print(f"  Avg/event:  {avg_us:.1f} \u00b5s")

    if batch_times:
        per_event = np.concatenate([
            np.full(bs, dt / bs) for dt, bs in batch_times
        ]) * 1e6  # µs
        p50, p90, p99 = np.percentile(per_event, [50, 90, 99])
        print(f"  Median:     {p50:.1f} \u00b5s")
        print(f"  p90:        {p90:.1f} \u00b5s")
        print(f"  p99:        {p99:.1f} \u00b5s")
    print()


# ======================================================================
# Standard validation backends (PyTorch / ONNX)
# ======================================================================

def _run_pytorch(args, cfg, active_tasks, norm_params):
    """Run validation with a PyTorch checkpoint."""
    from lib.models import XECEncoder, XECMultiHeadModel
    from lib.engines import run_epoch_stream

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get model params
    if cfg:
        outer_mode = cfg.model.outer_mode
        outer_fine_pool = tuple(cfg.model.outer_fine_pool) if cfg.model.outer_fine_pool else None
        drop_path_rate = cfg.model.drop_path_rate
        encoder_dim = cfg.model.encoder_dim
        dim_feedforward = cfg.model.dim_feedforward
        num_fusion_layers = cfg.model.num_fusion_layers
    else:
        outer_mode = "finegrid"
        outer_fine_pool = (3, 3)
        drop_path_rate = 0.0
        encoder_dim = 1024
        dim_feedforward = None
        num_fusion_layers = 2

    # Build model
    print(f"[INFO] Building model: outer_mode={outer_mode}, outer_fine_pool={outer_fine_pool}, encoder_dim={encoder_dim}")
    base_regressor = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool,
        drop_path_rate=drop_path_rate,
        encoder_dim=encoder_dim,
        dim_feedforward=dim_feedforward,
        num_fusion_layers=num_fusion_layers,
        sentinel_time=norm_params["sentinel_time"],
    )
    # Determine nll_tasks for correct head dimensions
    nll_tasks = []
    config_meta = checkpoint.get("config", {})
    if isinstance(config_meta, dict):
        nll_tasks = config_meta.get("nll_tasks", [])
    if not nll_tasks and cfg and hasattr(cfg, 'tasks'):
        from lib.config import get_task_weights
        tw = get_task_weights(cfg)
        nll_tasks = [t for t, c in tw.items()
                     if isinstance(c, dict) and c.get("loss_fn") == "gaussian_nll"]

    model = XECMultiHeadModel(base_regressor, active_tasks=active_tasks, nll_tasks=nll_tasks)
    model.to(device)

    # Load weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[INFO] Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create dataloader
    print(f"[INFO] Loading validation data from: {args.val_path}")
    val_loader = get_dataloader(
        args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        **norm_params,
        load_truth_branches=True,
    )

    # Run validation
    task_weights = {task: 1.0 for task in active_tasks}
    print("[INFO] Running validation...")
    t0 = time.time()
    val_metrics, angle_pred, angle_true, extra_info, val_stats = run_epoch_stream(
        model, None, device, val_loader,
        scaler=None,
        train=False,
        amp=False,
        task_weights=task_weights,
        reweighter=None,
        channel_dropout_rate=0.0,
        grad_clip=0.0,
    )
    elapsed = time.time() - t0
    # Count events from collected root_data
    root_data_tmp = extra_info.get("root_data", {}) if extra_info else {}
    for v in root_data_tmp.values():
        if hasattr(v, '__len__') and len(v) > 0:
            n_total = len(v)
            break
    else:
        n_total = 0
    _print_timing(elapsed, n_total, "PyTorch")

    # Print metrics
    print("\n=== Validation Metrics ===")
    for k, v in sorted(val_metrics.items()):
        print(f"  {k}: {v:.4f}")

    root_data = extra_info.get("root_data", {}) if extra_info else {}
    worst_events = extra_info.get("worst_events", []) if extra_info else []

    return model, root_data, angle_pred, angle_true, worst_events


def _run_onnx(args, cfg, active_tasks, norm_params):
    """Run validation with an ONNX model using onnxruntime."""
    import onnxruntime as ort

    print(f"[INFO] Loading ONNX model: {args.checkpoint}")
    sess_opts = ort.SessionOptions()
    n_threads = int(os.environ.get("SLURM_CPUS_PER_TASK",
                    os.environ.get("OMP_NUM_THREADS", 4)))
    sess_opts.inter_op_num_threads = n_threads
    sess_opts.intra_op_num_threads = n_threads
    print(f"[INFO] ONNX threads: {n_threads}")
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(args.checkpoint, sess_options=sess_opts,
                                providers=providers)

    # Inspect model I/O
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()
    print(f"[INFO] ONNX inputs:  {[(i.name, i.shape) for i in input_info]}")
    print(f"[INFO] ONNX outputs: {[(o.name, o.shape) for o in output_info]}")

    input_name = input_info[0].name
    output_names = [o.name for o in output_info]

    # Create dataloader
    print(f"[INFO] Loading validation data from: {args.val_path}")
    val_loader = get_dataloader(
        args.val_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_threads=args.num_threads,
        **norm_params,
        load_truth_branches=True,
    )

    # Inference loop
    root_data = {
        "run_id": [], "event_id": [],
        "pred_theta": [], "pred_phi": [],
        "true_theta": [], "true_phi": [],
        "opening_angle": [],
        "pred_energy": [], "true_energy": [],
        "pred_timing": [], "true_timing": [],
        "pred_u": [], "pred_v": [], "pred_w": [],
        "true_u": [], "true_v": [], "true_w": [],
        "pos_angle": [],
        "x_truth": [], "y_truth": [], "z_truth": [],
        "x_vtx": [], "y_vtx": [], "z_vtx": [],
    }

    print("[INFO] Running ONNX inference...")
    t0 = time.time()
    n_events = 0
    n_batches = 0
    batch_times = []

    for input_tensor, target_dict in val_loader:
        # input_tensor: (batch, 4760, 2)
        x_np = input_tensor.numpy().astype(np.float32)
        t_batch = time.perf_counter()
        outputs = sess.run(output_names, {input_name: x_np})
        dt_batch = time.perf_counter() - t_batch

        # Map outputs to tasks
        output_map = dict(zip(output_names, outputs))

        batch_size = x_np.shape[0]
        n_events += batch_size
        n_batches += 1
        batch_times.append((dt_batch, batch_size))

        if n_batches % 50 == 1:
            print(f"  Batch {n_batches}: {n_events} events processed")

        # Collect run/event metadata
        if "run" in target_dict:
            root_data["run_id"].append(target_dict["run"].numpy())
        if "event" in target_dict:
            root_data["event_id"].append(target_dict["event"].numpy())

        # Energy task
        if "energy" in active_tasks:
            out_key = "output_energy"
            if out_key in output_map:
                raw = output_map[out_key]
                pred = raw[:, 0] if raw.ndim == 2 and raw.shape[1] > 1 else raw.flatten()
                root_data["pred_energy"].append(pred)
            true_e = target_dict.get("energy")
            if true_e is not None:
                root_data["true_energy"].append(true_e.numpy().flatten())

        # Angle task
        if "angle" in active_tasks:
            out_key = "output_angle"
            if out_key in output_map:
                pred = output_map[out_key]
                root_data["pred_theta"].append(pred[:, 0])
                root_data["pred_phi"].append(pred[:, 1])
            true_a = target_dict.get("angle")
            if true_a is not None:
                a = true_a.numpy()
                root_data["true_theta"].append(a[:, 0])
                root_data["true_phi"].append(a[:, 1])

        # Timing task
        if "timing" in active_tasks:
            out_key = "output_timing"
            if out_key in output_map:
                pred = output_map[out_key].flatten()
                root_data["pred_timing"].append(pred)
            true_t = target_dict.get("timing")
            if true_t is not None:
                root_data["true_timing"].append(true_t.numpy().flatten())

        # Position task (uvwFI)
        if "uvwFI" in active_tasks:
            out_key = "output_uvwFI"
            if out_key in output_map:
                pred = output_map[out_key]
                root_data["pred_u"].append(pred[:, 0])
                root_data["pred_v"].append(pred[:, 1])
                root_data["pred_w"].append(pred[:, 2])
            true_uvw = target_dict.get("uvwFI")
            if true_uvw is not None:
                u = true_uvw.numpy()
                root_data["true_u"].append(u[:, 0])
                root_data["true_v"].append(u[:, 1])
                root_data["true_w"].append(u[:, 2])

    elapsed = time.time() - t0
    _print_timing(elapsed, n_events, "ONNX", batch_times=batch_times)

    # Concatenate all arrays
    for k, v_list in root_data.items():
        if v_list:
            root_data[k] = np.concatenate(v_list, axis=0)
        else:
            root_data[k] = np.array([])

    # Print basic metrics for energy task
    if "energy" in active_tasks:
        pred_e = root_data.get("pred_energy", np.array([]))
        true_e = root_data.get("true_energy", np.array([]))
        if pred_e.size > 0 and true_e.size > 0:
            # Filter out sentinel truth values (1e10 = unavailable)
            valid = true_e < 1e9
            if valid.sum() > 0:
                err = pred_e[valid] - true_e[valid]
                print(f"\n=== Energy Metrics ({valid.sum()} events with truth) ===")
                print(f"  MAE:  {np.mean(np.abs(err))*1e3:.2f} MeV")
                print(f"  RMSE: {np.sqrt(np.mean(err**2))*1e3:.2f} MeV")
                print(f"  Bias: {np.mean(err)*1e3:+.2f} MeV")
            else:
                print(f"\n=== Energy: {pred_e.size} predictions, no truth available ===")
                print(f"  Pred mean: {np.mean(pred_e)*1e3:.2f} MeV")
                print(f"  Pred std:  {np.std(pred_e)*1e3:.2f} MeV")

    # Build angle arrays for save_validation_artifacts
    angle_pred = None
    angle_true = None
    if "angle" in active_tasks:
        pt = root_data.get("pred_theta", np.array([]))
        pp = root_data.get("pred_phi", np.array([]))
        if pt.size > 0 and pp.size > 0:
            angle_pred = np.stack([pt, pp], axis=1)
            angle_true = np.stack([
                root_data.get("true_theta", np.zeros_like(pt)),
                root_data.get("true_phi", np.zeros_like(pp))
            ], axis=1)

    return None, root_data, angle_pred, angle_true, []


# ======================================================================
# Dead-channel recovery helpers
# ======================================================================

def _normalize_input(npho_raw, time_raw, transform,
                     time_scale, time_shift, sentinel_time,
                     npho_threshold, sentinel_npho=SENTINEL_NPHO):
    """Normalize raw npho/time arrays to model input space.

    Uses NphoTransform for configurable npho normalization (log1p, sqrt, etc.).
    """
    mask_npho_invalid = (npho_raw > 9e9) | np.isnan(npho_raw)
    domain_min = transform.domain_min()
    mask_domain_break = (~mask_npho_invalid) & (npho_raw < domain_min)

    mask_time_invalid = (mask_npho_invalid
                         | (npho_raw < npho_threshold)
                         | (np.abs(time_raw) > 9e9)
                         | np.isnan(time_raw))

    npho_safe = np.where(mask_npho_invalid | mask_domain_break, 0.0, npho_raw)
    npho_norm = transform.forward(npho_safe)
    npho_norm[mask_npho_invalid] = sentinel_npho
    npho_norm[mask_domain_break] = 0.0

    time_norm = (time_raw / time_scale) - time_shift
    time_norm[mask_time_invalid] = sentinel_time

    return np.stack([npho_norm, time_norm], axis=-1).astype(np.float32)


def _detect_dead_channels(dead_branch, npho_raw):
    """Get 1-D boolean dead mask from ``dead`` branch or sentinel fallback."""
    if dead_branch is not None:
        return dead_branch[0].astype(bool)
    # Fallback: channel is dead if sentinel in >90% of events
    invalid = npho_raw > 9e9
    frac = invalid.mean(axis=0)
    return frac > 0.9


def _fill_dead_neighbor_avg(x_norm, dead_mask_1d, baseline, transform=None):
    """Replace dead npho with neighbor average; time stays sentinel.

    When ``transform`` is provided, averaging is done in raw (linear) npho
    space and the result is re-normalised — correct for nonlinear schemes.
    """
    x_filled = x_norm.copy()
    npho_norm = x_filled[:, :, 0]
    mask_2d = np.broadcast_to(dead_mask_1d[None, :], npho_norm.shape).copy()
    npho_filled = baseline.predict(npho_norm, mask_2d, npho_transform=transform)
    x_filled[:, :, 0] = npho_filled
    return x_filled


def _fill_dead_inpainter(x_norm, dead_mask_1d, model, device,
                         npho_raw, time_raw, transform,
                         is_torchscript=True,
                         sentinel_npho=SENTINEL_NPHO, batch_size=512):
    """Replace dead npho with inpainter predictions.

    Two model types are supported:

    - **TorchScript** (``is_torchscript=True``): The wrapper takes raw
      ``(B, 4760, 2)`` sensor values and returns raw npho ``(B, 4760)``.
      We re-normalize with the regressor's ``transform`` before placing
      into ``x_norm``.

    - **Checkpoint** (``is_torchscript=False``): The raw inpainter model
      takes normalised ``(B, 4760, 2)`` input and returns normalised
      ``(B, 4760, C)`` output (C=1 or 2).  Values are placed directly
      into ``x_norm``.
    """
    x_filled = x_norm.copy()
    N = x_filled.shape[0]
    mask_2d = np.broadcast_to(dead_mask_1d[None, :], (N, N_CHANNELS)).copy()

    if is_torchscript:
        # TorchScript wrapper: takes raw input, returns raw npho (2D)
        x_input = np.stack([npho_raw, time_raw], axis=-1).astype(np.float32)
    else:
        # Raw checkpoint model: takes normalised input
        x_input = x_norm

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_batch = torch.from_numpy(x_input[start:end]).to(device)
        m_batch = torch.from_numpy(mask_2d[start:end].astype(np.float32)).to(device)

        with torch.no_grad():
            pred = model(x_batch, m_batch)

        pred_np = pred.cpu().numpy()

        if is_torchscript:
            # Output: (batch, 4760) — raw npho; re-normalize for regressor
            inpainted_npho = pred_np[:, dead_mask_1d]
            inpainted_npho_safe = np.where(
                (inpainted_npho > 9e9) | np.isnan(inpainted_npho),
                0.0, inpainted_npho)
            domain_min = transform.domain_min()
            inpainted_npho_safe = np.where(
                inpainted_npho_safe < domain_min, 0.0, inpainted_npho_safe)
            inpainted_norm = transform.forward(inpainted_npho_safe)
            x_filled[start:end, dead_mask_1d, 0] = inpainted_norm
        else:
            # Output: (batch, 4760, C) — already normalised
            if pred_np.ndim == 3 and pred_np.shape[-1] >= 2:
                x_filled[start:end, dead_mask_1d, 0] = pred_np[:, dead_mask_1d, 0]
                x_filled[start:end, dead_mask_1d, 1] = pred_np[:, dead_mask_1d, 1]
            elif pred_np.ndim == 3:
                x_filled[start:end, dead_mask_1d, 0] = pred_np[:, dead_mask_1d, 0]
            else:
                x_filled[start:end, dead_mask_1d, 0] = pred_np[:, dead_mask_1d]

    return x_filled


def _run_regressor_onnx(session, x_norm, task_map, batch_size=1024):
    """Run batched ONNX regressor inference."""
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


def _detect_tasks_onnx(session):
    """Detect tasks from ONNX output names (``output_{task}`` convention)."""
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


def _expand_task_results(task_results):
    """Expand multi-dim task arrays into {branch_name: 1-D array}."""
    flat = {}
    for task, arr in task_results.items():
        if arr.ndim == 1:
            flat[task] = arr
        elif task in TASK_COMPONENTS:
            for i, comp in enumerate(TASK_COMPONENTS[task]):
                flat[comp] = arr[:, i]
        else:
            for i in range(arr.shape[1]):
                flat[f"{task}_{i}"] = arr[:, i]
    return flat


# ======================================================================
# Dead-channel recovery mode
# ======================================================================

def _run_dead_channel_recovery(args, norm_params):
    """Run ONNX regressor with dead-channel recovery strategies.

    Reads CEX ROOT files via uproot, applies three dead-channel recovery
    strategies (raw, neighbor averaging, ML inpainting), runs the ONNX
    regressor on each variant, and writes results to a ROOT file.
    """
    import uproot
    import onnxruntime as ort
    from lib.inpainter_baselines import NeighborAverageBaseline

    npho_scheme = norm_params.get("npho_scheme", "log1p")
    transform = NphoTransform(
        scheme=npho_scheme,
        npho_scale=norm_params["npho_scale"],
        npho_scale2=norm_params["npho_scale2"],
    )
    print(f"[INFO] Npho transform: {transform}")

    # --- Load regressor ONNX ---
    print(f"[INFO] Loading regressor: {args.checkpoint}")
    sess_opts = ort.SessionOptions()
    # Use 1 thread to minimise per-thread workspace memory (important when
    # both the regressor and inpainter models are loaded simultaneously).
    n_threads = 1
    sess_opts.inter_op_num_threads = n_threads
    sess_opts.intra_op_num_threads = n_threads
    print(f"[INFO] ONNX threads: {n_threads}")
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]
    reg_session = ort.InferenceSession(args.checkpoint, sess_options=sess_opts,
                                       providers=providers)

    # Detect tasks
    task_map = _detect_tasks_onnx(reg_session)
    if args.tasks:
        task_map = {t: task_map[t] for t in args.tasks if t in task_map}
    print(f"[INFO] Active tasks: {list(task_map.keys())}")

    # --- Load inpainter (optional) ---
    inpainter_model = None
    have_inpainter = False
    inpainter_is_torchscript = False

    if args.inpainter_torchscript:
        print(f"[INFO] Loading TorchScript inpainter: {args.inpainter_torchscript}")
        inpainter_model = torch.jit.load(args.inpainter_torchscript,
                                         map_location=args.device)
        inpainter_model.eval()
        have_inpainter = True
        inpainter_is_torchscript = True
    elif args.inpainter_checkpoint:
        from val_data.validate_inpainter_real import load_inpainter_model
        inpainter_model, _ = load_inpainter_model(args.inpainter_checkpoint,
                                                   device=args.device)
        have_inpainter = True

    if not have_inpainter:
        print("[INFO] No inpainter provided \u2014 inpainted strategy will output 1e10")

    # --- Build neighbor baseline ---
    print(f"[INFO] Building neighbor map (k={args.neighbor_k})")
    baseline = NeighborAverageBaseline(k=args.neighbor_k)

    # --- Resolve dead channel mask ---
    dead_mask_1d = None

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

    # --- Determine branches to read ---
    model_branches = ["npho", "relative_time"]
    meta_branches = ["run", "event", "energyTruth", "energyReco",
                     "Ebgo", "Angle", "gstatus", "nDead"]
    dead_branches = ["dead"]
    all_requested = model_branches + meta_branches + dead_branches

    tree_name = args.tree_name
    # Resolve glob patterns for branch discovery (uproot.open doesn't support globs)
    import glob as globmod
    if any(c in args.val_path for c in "*?["):
        discover_files = sorted(globmod.glob(args.val_path))
        if not discover_files:
            print(f"[ERROR] No files match: {args.val_path}")
            return
        discover_path = discover_files[0]
    else:
        discover_path = args.val_path
    with uproot.open(discover_path) as f_check:
        available = set(f_check[tree_name].keys())
    read_branches = [b for b in all_requested if b in available]
    has_dead_branch = "dead" in available

    if not has_dead_branch and dead_mask_1d is None:
        print("[WARN] No 'dead' branch and no --dead-from-db/--dead-from-file. "
              "Will detect dead channels from sentinel values.")

    strategies = ["raw", "neighavg", "inpainted"]

    total_events = 0
    t0 = time.time()
    npho_threshold = norm_params.get("npho_threshold", DEFAULT_NPHO_THRESHOLD)
    sentinel_npho = norm_params.get("sentinel_npho", SENTINEL_NPHO)
    int_branches = {"run", "event", "gstatus", "nDead"}
    meta_available = [b for b in meta_branches if b in available]

    # Determine output path
    output_path = args.output_dir
    if output_path is None:
        base = os.path.splitext(os.path.basename(args.val_path))[0]
        output_path = f"regressor_{base}.root"
    elif not output_path.endswith(".root"):
        os.makedirs(output_path, exist_ok=True)
        base = os.path.splitext(os.path.basename(args.val_path))[0]
        output_path = os.path.join(output_path, f"regressor_{base}.root")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Create output ROOT file before the loop — write chunks incrementally
    f_out = uproot.recreate(output_path)
    tree_created = False

    print(f"[INFO] Processing {args.val_path} (chunksize={args.chunksize})")
    print(f"[INFO] Output: {output_path}")

    for arrays in uproot.iterate(
        f"{args.val_path}:{tree_name}",
        expressions=read_branches,
        step_size=args.chunksize,
        library="np",
    ):
        B = len(arrays["run"])

        # --- Normalize ---
        npho_raw = arrays["npho"].astype(np.float32)
        time_raw = arrays["relative_time"].astype(np.float32)

        x_norm = _normalize_input(
            npho_raw, time_raw, transform,
            norm_params["time_scale"], norm_params["time_shift"],
            norm_params["sentinel_time"], npho_threshold,
            sentinel_npho=sentinel_npho,
        )

        # --- Resolve dead mask (once) ---
        if dead_mask_1d is None:
            dead_br = arrays.get("dead", None)
            dead_mask_1d = _detect_dead_channels(dead_br, npho_raw)
            n_dead = int(dead_mask_1d.sum())
            print(f"[INFO] Dead channels detected: {n_dead}")

        # --- Process strategies sequentially to reduce peak memory ---

        # Strategy 1: raw (no fill)
        res_raw = _run_regressor_onnx(reg_session, x_norm, task_map,
                                      batch_size=args.batch_size)
        flat_raw = _expand_task_results(res_raw)
        del res_raw

        # Strategy 2: neighbor average
        x_neighavg = _fill_dead_neighbor_avg(x_norm, dead_mask_1d, baseline,
                                             transform=transform)
        res_neighavg = _run_regressor_onnx(reg_session, x_neighavg, task_map,
                                           batch_size=args.batch_size)
        flat_neighavg = _expand_task_results(res_neighavg)
        del x_neighavg, res_neighavg

        # Strategy 3: inpainted
        if have_inpainter:
            x_inpainted = _fill_dead_inpainter(
                x_norm, dead_mask_1d, inpainter_model,
                args.device, npho_raw, time_raw, transform,
                is_torchscript=inpainter_is_torchscript,
                sentinel_npho=sentinel_npho, batch_size=args.batch_size,
            )
            res_inpainted = _run_regressor_onnx(reg_session, x_inpainted, task_map,
                                                batch_size=args.batch_size)
            flat_inpainted = _expand_task_results(res_inpainted)
            del x_inpainted, res_inpainted
        else:
            flat_inpainted = {}
            for bname, arr in flat_raw.items():
                flat_inpainted[bname] = np.full_like(arr, 1e10)

        del x_norm, npho_raw, time_raw

        # --- Build chunk output and write immediately ---
        chunk = {}
        for b in meta_available:
            arr = arrays[b]
            chunk[b] = arr.astype(np.int32) if b in int_branches else arr.astype(np.float32)

        flat_all = {"raw": flat_raw, "neighavg": flat_neighavg,
                    "inpainted": flat_inpainted}
        for strategy in strategies:
            for bname, arr in flat_all[strategy].items():
                chunk[f"{bname}_{strategy}"] = arr.astype(np.float32)

        if not tree_created:
            branch_types = {}
            for k, v in chunk.items():
                branch_types[k] = np.int32 if v.dtype == np.int32 else np.float32
            f_out.mktree("tree", branch_types)
            tree_created = True

        f_out["tree"].extend(chunk)
        del chunk, flat_raw, flat_neighavg, flat_inpainted, flat_all

        total_events += B
        elapsed = time.time() - t0
        print(f"  {total_events} events ({elapsed:.1f}s, "
              f"{total_events / elapsed:.0f} evt/s)", end="\r")

    print()  # newline after \r
    f_out.close()

    elapsed = time.time() - t0
    print(f"[DONE] {total_events} events in {elapsed:.1f}s "
          f"({total_events / elapsed:.0f} evt/s)")
    print(f"[INFO] Output: {output_path}")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run validation with an existing checkpoint or ONNX model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("checkpoint", type=str,
                        help="Path to checkpoint (.pth) or ONNX model (.onnx)")
    parser.add_argument("--val_path", type=str, required=True,
                        help="Path to validation data (directory or ROOT file)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config file (required for ONNX, optional for .pth)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: same as checkpoint)")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Tasks to evaluate (default: from checkpoint)")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_threads", type=int, default=4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # Dead-channel recovery mode
    dc_group = parser.add_argument_group("Dead-channel recovery",
        "Enable with --dead-channel to compare raw, neighbor-avg, and "
        "inpainted recovery strategies. Requires ONNX model.")
    dc_group.add_argument("--dead-channel", action="store_true",
                          help="Enable dead-channel recovery mode")
    dc_group.add_argument("--chunksize", type=int, default=2048,
                          help="Events per chunk for dead-channel mode (default: 2048)")
    dc_group.add_argument("--neighbor-k", type=int, default=1,
                          help="Neighbor hops for averaging (default: 1)")
    dc_group.add_argument("--tree-name", type=str, default="tree",
                          help="TTree name in input ROOT file (default: tree)")

    # Inpainter (mutually exclusive)
    inp_group = parser.add_mutually_exclusive_group()
    inp_group.add_argument("--inpainter-torchscript", type=str, default=None,
                           help="Path to inpainter TorchScript (.pt)")
    inp_group.add_argument("--inpainter-checkpoint", type=str, default=None,
                           help="Path to inpainter checkpoint (.pth)")

    # Dead channel source overrides
    dead_group = parser.add_mutually_exclusive_group()
    dead_group.add_argument("--dead-from-db", type=int, metavar="RUN", default=None,
                            help="Query dead channels from DB for this run")
    dead_group.add_argument("--dead-from-file", type=str, metavar="PATH", default=None,
                            help="Load dead channel list from text file")

    args = parser.parse_args()

    # Expand tilde in paths
    args.checkpoint = os.path.expanduser(args.checkpoint)
    args.val_path = os.path.expanduser(args.val_path)
    if args.inpainter_torchscript:
        args.inpainter_torchscript = os.path.expanduser(args.inpainter_torchscript)
    if args.inpainter_checkpoint:
        args.inpainter_checkpoint = os.path.expanduser(args.inpainter_checkpoint)
    if args.dead_from_file:
        args.dead_from_file = os.path.expanduser(args.dead_from_file)

    if not os.path.exists(args.checkpoint):
        print(f"Error: Model not found: {args.checkpoint}")
        sys.exit(1)

    is_onnx = args.checkpoint.endswith(".onnx")
    device = torch.device(args.device)

    # Dead-channel mode requires ONNX
    dead_channel_mode = args.dead_channel
    if dead_channel_mode and not is_onnx:
        print("[ERROR] Dead-channel recovery mode requires an ONNX model (.onnx)")
        sys.exit(1)

    print(f"[INFO] Model format: {'ONNX' if is_onnx else 'PyTorch'}")
    print(f"[INFO] Using device: {'cpu (ONNX)' if is_onnx else device}")
    if dead_channel_mode:
        print("[INFO] Mode: dead-channel recovery")

    # --- Load config ---
    cfg = None
    if not is_onnx:
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "config" in checkpoint:
            cfg = checkpoint["config"]
            print("[INFO] Using config from checkpoint")
        elif args.config:
            cfg = load_config(args.config)
            print(f"[INFO] Using config from: {args.config}")
        else:
            print("[INFO] No config found, using defaults")
    else:
        if args.config:
            cfg = load_config(args.config)
            print(f"[INFO] Using config from: {args.config}")
        else:
            print("[WARN] No --config provided for ONNX model. Using default normalization.")
            print("       This is likely wrong! Specify --config for correct normalization.")

    # --- Determine active tasks ---
    if args.tasks:
        active_tasks = args.tasks
    elif cfg and hasattr(cfg, 'tasks'):
        active_tasks = [t for t, tc in cfg.tasks.items() if tc.enabled]
    elif not is_onnx:
        # Infer from PyTorch checkpoint keys
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        active_tasks = []
        if any("angle" in k for k in state_dict.keys()):
            active_tasks.append("angle")
        if any("energy" in k for k in state_dict.keys()):
            active_tasks.append("energy")
        if any("timing" in k for k in state_dict.keys()):
            active_tasks.append("timing")
        if any("uvwFI" in k or "position" in k for k in state_dict.keys()):
            active_tasks.append("uvwFI")
    else:
        active_tasks = ["energy"]
        print("[INFO] No task info, defaulting to ['energy']")

    print(f"[INFO] Active tasks: {active_tasks}")

    # --- Normalization params ---
    if cfg:
        norm_params = {
            "npho_scale": cfg.normalization.npho_scale,
            "npho_scale2": cfg.normalization.npho_scale2,
            "time_scale": cfg.normalization.time_scale,
            "time_shift": cfg.normalization.time_shift,
            "sentinel_time": cfg.normalization.sentinel_time,
        }
        if hasattr(cfg.normalization, 'npho_scheme'):
            norm_params["npho_scheme"] = cfg.normalization.npho_scheme
        if hasattr(cfg.normalization, 'sentinel_npho'):
            norm_params["sentinel_npho"] = cfg.normalization.sentinel_npho
        if hasattr(cfg.normalization, 'npho_threshold'):
            norm_params["npho_threshold"] = cfg.normalization.npho_threshold
    else:
        norm_params = {
            "npho_scale": DEFAULT_NPHO_SCALE,
            "npho_scale2": DEFAULT_NPHO_SCALE2,
            "time_scale": DEFAULT_TIME_SCALE,
            "time_shift": DEFAULT_TIME_SHIFT,
            "sentinel_time": DEFAULT_SENTINEL_TIME,
        }

    print(f"[INFO] Normalization: npho_scheme={norm_params.get('npho_scheme', 'log1p')}, "
          f"npho_scale={norm_params.get('npho_scale')}, "
          f"npho_scale2={norm_params.get('npho_scale2')}")

    # --- Run inference ---
    if dead_channel_mode:
        _run_dead_channel_recovery(args, norm_params)
    elif is_onnx:
        model, root_data, angle_pred, angle_true, worst_events = \
            _run_onnx(args, cfg, active_tasks, norm_params)
    else:
        model, root_data, angle_pred, angle_true, worst_events = \
            _run_pytorch(args, cfg, active_tasks, norm_params)

    if dead_channel_mode:
        return

    # --- Check collected data ---
    print("\n=== Data Collection Check ===")
    for key in ["true_energy", "pred_energy", "true_u", "true_v", "true_w", "pred_u", "pred_v", "pred_w"]:
        arr = root_data.get(key, np.array([]))
        print(f"  {key}: {arr.shape if hasattr(arr, 'shape') else len(arr)} entries")

    # --- Save artifacts ---
    output_dir = args.output_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[INFO] Saving artifacts to: {output_dir}")
    run_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    save_validation_artifacts(
        model=model,
        angle_pred=angle_pred,
        angle_true=angle_true,
        root_data=root_data,
        active_tasks=active_tasks,
        artifact_dir=output_dir,
        run_name=run_name,
        epoch=None,
        worst_events=worst_events,
    )

    print("\n[INFO] Done!")


if __name__ == "__main__":
    main()
