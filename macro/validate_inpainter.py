#!/usr/bin/env python3
"""
Unified Inpainter Validation for MC and Real Data

This script handles both:
1. MC Validation (pseudo-experiment): Apply real dead channel pattern to MC
2. Real Data Validation: Use actual dead channels + artificial masking

Workflow:
    MC Data + Dead Pattern (from DB/file)
        → Apply mask → Inference → Full metrics (all have ground truth)

    Real Data + Dead Channels (from DB/file) + Artificial Masking
        → Inference → Partial metrics (only artificial masks have truth)

Usage:
    # MC pseudo-experiment: apply run 430000's dead pattern to MC
    # --input supports: single file, directory, or glob pattern
    python macro/validate_inpainter.py \\
        --checkpoint artifacts/inpainter/checkpoint_best.pth \\
        --input mc_validation.root \\
        --run 430000 \\
        --output validation_mc/

    # MC pseudo-experiment with directory of files
    python macro/validate_inpainter.py \\
        --checkpoint artifacts/inpainter/checkpoint_best.pth \\
        --input data/mc_samples/single_run/ \\
        --run 430000 \\
        --output validation_mc/

    # Real data validation: use dead channels + artificial masking
    python macro/validate_inpainter.py \\
        --checkpoint artifacts/inpainter/checkpoint_best.pth \\
        --input real_data.root \\
        --run 430000 \\
        --real-data \\
        --n-artificial 50 \\
        --output validation_real/

    # With TorchScript model (faster)
    python macro/validate_inpainter.py \\
        --torchscript artifacts/inpainter/inpainter.pt \\
        --input data.root \\
        --run 430000 \\
        --output validation/
"""

import gc
import os
import sys
import argparse
import csv
import subprocess
import tempfile
import numpy as np
import torch
import uproot
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    flatten_hex_rows, OUTER_ALL_SENSOR_IDS,
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_TIME, DEFAULT_NPHO_THRESHOLD
)
from lib.normalization import NphoTransform
from lib.dataset import expand_path
from lib.inpainter_baselines import NeighborAverageBaseline, SolidAngleWeightedBaseline

# Constants
N_CHANNELS = 4760
MODEL_SENTINEL_TIME = DEFAULT_SENTINEL_TIME
MODEL_SENTINEL_NPHO = -1.0

# Face index maps
TOP_HEX_FLAT = flatten_hex_rows(TOP_HEX_ROWS)
BOT_HEX_FLAT = flatten_hex_rows(BOTTOM_HEX_ROWS)

FACE_INDEX_MAPS = {
    'inner': INNER_INDEX_MAP,
    'us': US_INDEX_MAP,
    'ds': DS_INDEX_MAP,
    'outer': OUTER_COARSE_FULL_INDEX_MAP,
    'top': TOP_HEX_FLAT,
    'bot': BOT_HEX_FLAT,
}

FACE_NAME_TO_INT = {"inner": 0, "us": 1, "ds": 2, "outer": 3, "top": 4, "bot": 5}


def get_face_sensor_ids(face_name: str) -> np.ndarray:
    """Get flat sensor IDs for a face."""
    idx_map = FACE_INDEX_MAPS[face_name]
    if face_name in ['top', 'bot']:
        return idx_map.astype(np.int64)
    else:
        # Flatten 2D index map, filter out -1 (invalid)
        flat = idx_map.flatten()
        return flat[flat >= 0].astype(np.int64)


def load_data(input_path: str, tree_name: str = "tree",
              max_events: Optional[int] = None,
              npho_branch: str = "npho",
              time_branch: str = "relative_time") -> Dict[str, np.ndarray]:
    """Load data from ROOT file(s).

    Args:
        input_path: Path to ROOT file, directory, or glob pattern.
                   If directory, all .root files in it will be loaded.
        tree_name: Name of the tree in ROOT files.
        max_events: Maximum number of events to load (None = all).
        npho_branch: Branch name for npho data.
        time_branch: Branch name for time data.

    Returns:
        Dictionary with 'npho', 'time', and optional 'run', 'event' arrays.
    """
    # Expand path to list of files (handles directories, globs, single files)
    file_list = expand_path(input_path)
    print(f"[INFO] Loading data from {len(file_list)} file(s)")
    if len(file_list) > 1:
        for f in file_list[:5]:
            print(f"  - {f}")
        if len(file_list) > 5:
            print(f"  ... and {len(file_list) - 5} more")

    all_data = {'npho': [], 'time': [], 'run': [], 'event': []}
    total_events = 0

    for file_path in file_list:
        if max_events and total_events >= max_events:
            break

        with uproot.open(file_path) as f:
            tree = f[tree_name]

            # Try different branch names (check on first file)
            actual_npho_branch = npho_branch
            if npho_branch not in tree.keys():
                if "relative_npho" in tree.keys():
                    actual_npho_branch = "relative_npho"
                else:
                    raise ValueError(f"Cannot find npho branch in {file_path}. Available: {tree.keys()}")

            npho_arr = tree[actual_npho_branch].array(library='np')
            time_arr = tree[time_branch].array(library='np')

            # Limit events if needed
            n_in_file = len(npho_arr)
            if max_events:
                remaining = max_events - total_events
                if n_in_file > remaining:
                    npho_arr = npho_arr[:remaining]
                    time_arr = time_arr[:remaining]
                    n_in_file = remaining

            all_data['npho'].append(npho_arr)
            all_data['time'].append(time_arr)

            # Optional branches
            for branch in ['run', 'event']:
                if branch in tree.keys():
                    arr = tree[branch].array(library='np')
                    if max_events:
                        arr = arr[:n_in_file]
                    all_data[branch].append(arr)

            total_events += n_in_file

    # Concatenate all files
    data = {
        'npho': np.concatenate(all_data['npho']),
        'time': np.concatenate(all_data['time']),
    }

    # Add optional branches if present
    if all_data['run']:
        data['run'] = np.concatenate(all_data['run'])
    if all_data['event']:
        data['event'] = np.concatenate(all_data['event'])

    print(f"[INFO] Loaded {len(data['npho']):,} events total")
    return data


def load_solid_angles(input_path: str, branch_name: str,
                      tree_name: str = "tree",
                      max_events: Optional[int] = None) -> np.ndarray:
    """Load solid angles from ROOT file(s).

    Args:
        input_path: Path to ROOT file, directory, or glob pattern.
        branch_name: Branch name for solid angle data.
        tree_name: Name of the tree in ROOT files.
        max_events: Maximum number of events to load (None = all).

    Returns:
        solid_angles: (N_events, 4760) array of solid angles.
    """
    file_list = expand_path(input_path)
    all_sa = []
    total_events = 0

    for file_path in file_list:
        if max_events and total_events >= max_events:
            break

        with uproot.open(file_path) as f:
            tree = f[tree_name]
            if branch_name not in tree.keys():
                raise ValueError(
                    f"Solid angle branch '{branch_name}' not found in {file_path}. "
                    f"Available: {list(tree.keys())}"
                )
            sa_arr = tree[branch_name].array(library='np')

            n_in_file = len(sa_arr)
            if max_events:
                remaining = max_events - total_events
                if n_in_file > remaining:
                    sa_arr = sa_arr[:remaining]
                    n_in_file = remaining

            all_sa.append(sa_arr)
            total_events += n_in_file

    solid_angles = np.concatenate(all_sa)
    print(f"[INFO] Loaded solid angles from branch '{branch_name}': shape {solid_angles.shape}")
    return solid_angles


def normalize_data(npho: np.ndarray, time: np.ndarray,
                   npho_scheme: str = 'log1p',
                   npho_scale: float = DEFAULT_NPHO_SCALE,
                   npho_scale2: float = DEFAULT_NPHO_SCALE2,
                   time_scale: float = DEFAULT_TIME_SCALE,
                   time_shift: float = DEFAULT_TIME_SHIFT,
                   sentinel: float = MODEL_SENTINEL_TIME,
                   npho_threshold: float = DEFAULT_NPHO_THRESHOLD,
                   npho_sentinel: float = -1.0) -> np.ndarray:
    """Normalize data to model input format."""
    transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)
    domain_min = transform.domain_min()

    # True invalids: dead/missing sensors, corrupted data
    mask_npho_invalid = (npho > 9e9) | np.isnan(npho)
    mask_domain_break = (~mask_npho_invalid) & (npho < domain_min)
    mask_time_invalid = mask_npho_invalid | (npho < npho_threshold) | (np.abs(time) > 9e9) | np.isnan(time)

    # Normalize npho using configured scheme
    npho_safe = np.where(mask_npho_invalid | mask_domain_break, 0.0, npho)
    npho_norm = transform.forward(npho_safe)
    npho_norm[mask_npho_invalid] = npho_sentinel
    npho_norm[mask_domain_break] = 0.0

    # Normalize time: linear transform
    time_norm = (time / time_scale) - time_shift
    time_norm[mask_time_invalid] = sentinel

    return np.stack([npho_norm, time_norm], axis=-1).astype(np.float32)


def get_dead_channels(run_number: Optional[int] = None,
                      dead_channel_file: Optional[str] = None) -> np.ndarray:
    """Get dead channel list from database or file."""
    if run_number is not None:
        from lib.db_utils import get_dead_channels as db_get_dead_channels
        return np.array(db_get_dead_channels(run_number))
    elif dead_channel_file is not None:
        return np.loadtxt(dead_channel_file, dtype=np.int64)
    else:
        return np.array([], dtype=np.int64)


def create_artificial_mask(x: np.ndarray, n_artificial,
                           dead_mask: np.ndarray,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create artificial mask on healthy sensors for evaluation.

    Args:
        x: (N, 4760, 2) normalized sensor data.
        n_artificial: Total number of masks per event (int), or dict mapping
                      face name to count (e.g. {"inner": 10, "us": 1, ...}).
        dead_mask: (4760,) bool mask of dead channels.
        seed: Random seed.

    Returns:
        artificial_mask: (N, 4760) bool - artificially masked positions
        combined_mask: (N, 4760) bool - dead + artificial
    """
    rng = np.random.default_rng(seed)
    n_events = x.shape[0]
    artificial_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)

    if isinstance(n_artificial, dict):
        # Per-face stratified masking
        face_sensor_ids = {
            fname: get_face_sensor_ids(fname) for fname in n_artificial
        }
        for i in range(n_events):
            for fname, n_per_face in n_artificial.items():
                if n_per_face <= 0:
                    continue
                sids = face_sensor_ids[fname]
                valid = ~dead_mask[sids] & (x[i, sids, 0] != MODEL_SENTINEL_NPHO)
                valid_sids = sids[valid]
                if len(valid_sids) >= n_per_face:
                    chosen = rng.choice(valid_sids, size=n_per_face, replace=False)
                    artificial_mask[i, chosen] = True
    else:
        # Uniform random masking (legacy behavior)
        for i in range(n_events):
            valid = ~dead_mask & (x[i, :, 0] != MODEL_SENTINEL_NPHO)
            valid_indices = np.where(valid)[0]
            if len(valid_indices) > n_artificial:
                chosen = rng.choice(valid_indices, size=n_artificial, replace=False)
                artificial_mask[i, chosen] = True

    # Combined mask
    combined_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
    combined_mask[:, dead_mask] = True
    combined_mask |= artificial_mask

    return artificial_mask, combined_mask


def load_model(checkpoint_path: Optional[str] = None,
               torchscript_path: Optional[str] = None,
               device: str = 'cpu'):
    """Load inpainter model from checkpoint or TorchScript.

    Returns:
        model: The loaded model
        model_type: 'torchscript' or 'checkpoint'
        predict_channels: List of predicted channels (e.g., ['npho'] or ['npho', 'time'])
        npho_scheme: Normalization scheme (e.g., 'log1p', 'sqrt') or None if unknown
    """
    if torchscript_path:
        print(f"[INFO] Loading TorchScript model from {torchscript_path}")
        model = torch.jit.load(torchscript_path, map_location=device)
        model.eval()
        # Probe output shape with dummy input to detect predict_channels
        with torch.no_grad():
            dummy_x = torch.zeros(1, N_CHANNELS, 2, device=device)
            dummy_mask = torch.ones(1, N_CHANNELS, device=device)
            dummy_out = model(dummy_x, dummy_mask)
        out_channels = dummy_out.shape[-1]
        if out_channels == 1:
            predict_channels = ['npho']
        else:
            predict_channels = ['npho', 'time']
        print(f"[INFO] Detected output channels: {out_channels} → predict_channels={predict_channels}")
        # TorchScript has no metadata — npho_scheme must come from CLI
        return model, 'torchscript', predict_channels, None, DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2

    if checkpoint_path:
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        from lib.models import XECEncoder, XEC_Inpainter

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})

        # Get predict_channels from checkpoint config (default to both for legacy)
        predict_channels = config.get('predict_channels', ['npho', 'time'])
        npho_scheme = config.get('npho_scheme', 'log1p')
        npho_scale = config.get('npho_scale', DEFAULT_NPHO_SCALE)
        npho_scale2 = config.get('npho_scale2', DEFAULT_NPHO_SCALE2)
        print(f"[INFO] Predict channels: {predict_channels}")
        print(f"[INFO] Npho scheme: {npho_scheme} "
              f"(scale={npho_scale}, scale2={npho_scale2})")

        encoder = XECEncoder(
            outer_mode=config.get('outer_mode', 'finegrid'),
            outer_fine_pool=config.get('outer_fine_pool', None),
            encoder_dim=config.get('encoder_dim', 1024),
            dim_feedforward=config.get('dim_feedforward', None),
            num_fusion_layers=config.get('num_fusion_layers', 2),
            sentinel_time=config.get('sentinel_time', -1.0),
        )
        use_masked_attention = config.get('use_masked_attention', False)
        head_type = config.get('head_type', 'per_face')
        sensor_positions_file = config.get('sensor_positions_file', None)
        model = XEC_Inpainter(
            encoder=encoder,
            predict_channels=predict_channels,
            use_masked_attention=use_masked_attention,
            head_type=head_type,
            sensor_positions_file=sensor_positions_file,
            cross_attn_k=config.get('cross_attn_k', 16),
            cross_attn_hidden=config.get('cross_attn_hidden', 64),
            cross_attn_latent_dim=config.get('cross_attn_latent_dim', 128),
            cross_attn_pos_dim=config.get('cross_attn_pos_dim', 96),
            sentinel_npho=config.get('sentinel_npho', MODEL_SENTINEL_NPHO),
        )

        # Load weights (prefer EMA)
        if checkpoint.get('ema_state_dict') is not None:
            state_dict = checkpoint['ema_state_dict']
            print("[INFO] Using EMA weights")
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Handle DataParallel prefix
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model, 'checkpoint', predict_channels, npho_scheme, npho_scale, npho_scale2

    raise ValueError("Either --checkpoint or --torchscript must be specified")


def run_inference(model, model_type: str, x: np.ndarray, mask: np.ndarray,
                  batch_size: int = 64, device: str = 'cpu',
                  predict_channels: List[str] = None) -> np.ndarray:
    """
    Run inference and return predictions for all sensors.

    Args:
        predict_channels: List of channels being predicted (['npho'] or ['npho', 'time'])

    Returns:
        predictions: (N, 4760, out_channels) - predictions at masked positions, zeros elsewhere
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']
    out_channels = len(predict_channels)

    n_events = x.shape[0]
    all_preds = np.zeros((n_events, N_CHANNELS, out_channels), dtype=np.float32)

    with torch.no_grad():
        for start in tqdm(range(0, n_events, batch_size), desc="Inference"):
            end = min(start + batch_size, n_events)

            x_batch = torch.tensor(x[start:end], device=device)
            mask_batch = torch.tensor(mask[start:end], device=device)

            if model_type == 'torchscript':
                # TorchScript model returns (B, 4760, out_channels)
                pred_batch = model(x_batch, mask_batch)
                all_preds[start:end] = pred_batch.cpu().numpy()
            else:
                # Checkpoint model - use forward_full_output
                pred_batch = model.forward_full_output(x_batch, mask_batch)
                all_preds[start:end] = pred_batch.cpu().numpy()

    return all_preds


def run_baselines(x_original: np.ndarray, combined_mask: np.ndarray,
                  artificial_mask: Optional[np.ndarray],
                  dead_mask: np.ndarray,
                  baseline_k: int = 1,
                  solid_angles: Optional[np.ndarray] = None,
                  npho_transform: Optional[NphoTransform] = None,
                  ) -> Dict[str, np.ndarray]:
    """Run baseline predictions and return full prediction arrays.

    Args:
        x_original: (N, 4760, 2) normalized original data (before masking).
        combined_mask: (N, 4760) bool mask (dead + artificial).
        artificial_mask: (N, 4760) bool or None.
        dead_mask: (4760,) bool.
        baseline_k: k-hop parameter.
        solid_angles: (N, 4760) solid angles or None.
        npho_transform: NphoTransform for averaging in raw npho space.

    Returns:
        Dictionary mapping baseline name to (N, 4760) prediction arrays.
    """
    x_npho = x_original[:, :, 0]  # (N, 4760) normalized npho, unmasked

    print("[INFO] Running NeighborAverageBaseline...")
    avg_baseline = NeighborAverageBaseline()
    baseline_preds = {
        'avg': avg_baseline.predict(x_npho, combined_mask,
                                    npho_transform=npho_transform),
    }

    if solid_angles is not None:
        print("[INFO] Running SolidAngleWeightedBaseline...")
        sa_baseline = SolidAngleWeightedBaseline()
        baseline_preds['sa'] = sa_baseline.predict(
            x_npho, combined_mask, solid_angles=solid_angles,
            npho_transform=npho_transform,
        )

    return baseline_preds


def run_local_fit_baseline(
    input_path: str,
    dead_channels: np.ndarray,
    x_original: np.ndarray,
    combined_mask: np.ndarray,
    artificial_mask: Optional[np.ndarray],
    npho_scheme: str,
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    max_events: Optional[int] = None,
) -> List[Dict]:
    """Run LocalFitBaseline via ROOT macro and return normalized results.

    The macro operates on raw npho (reads directly from ROOT file).
    Predictions are normalized to match the ML model's normalized space.

    Only inner-face dead channels (0-4091) are predicted by the macro.

    Args:
        input_path: Path to the input ROOT file (single file only).
        dead_channels: Array of dead channel indices.
        x_original: (N, 4760, 2) normalized original data (for truth values).
        combined_mask: (N, 4760) bool mask.
        artificial_mask: (N, 4760) bool or None.
        npho_scheme: Normalization scheme for NphoTransform.
        npho_scale: Scale parameter for NphoTransform.
        npho_scale2: Scale2 parameter for NphoTransform.
        max_events: Max events (must match what was used to load data).

    Returns:
        List of per-sensor prediction dicts (same format as other baselines).
    """
    # Resolve to a single file
    file_list = expand_path(input_path)
    if len(file_list) != 1:
        print(f"[WARNING] LocalFitBaseline requires a single ROOT file, "
              f"got {len(file_list)}. Using first file only.")
    root_file = file_list[0]

    # Macro path (relative to repo root)
    macro_path = os.path.join(os.path.dirname(__file__), '..', 'others', 'LocalFitBaseline.C')
    macro_path = os.path.abspath(macro_path)
    if not os.path.isfile(macro_path):
        raise FileNotFoundError(f"LocalFitBaseline.C not found at {macro_path}")

    # Write dead channel list to temp file
    dead_tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    try:
        for ch in dead_channels:
            dead_tmp.write(f"{ch}\n")
        dead_tmp.close()

        # Output temp file for ROOT macro results
        out_tmp = tempfile.NamedTemporaryFile(suffix='.root', delete=False)
        out_tmp.close()

        # Call ROOT macro (stream output so user can see progress)
        cmd = (
            f'root -l -b -q \'{macro_path}("{root_file}", '
            f'"{dead_tmp.name}", "{out_tmp.name}")\''
        )
        n_events = x_original.shape[0]
        print(f"[INFO] Running LocalFitBaseline macro on {root_file}")
        print(f"[INFO]   {len(dead_channels)} dead channels, "
              f"~{n_events} events (this may take a while)...")
        sys.stdout.flush()
        result = subprocess.run(cmd, shell=True)
        if result.returncode != 0:
            # ROOT often segfaults during cleanup (TApplication::Terminate).
            # Check if the output file was written successfully before giving up.
            try:
                _test = uproot.open(out_tmp.name)
                _test['predictions']
                _test.close()
                print(f"[WARNING] LocalFitBaseline macro exited with code {result.returncode} "
                      f"(likely ROOT cleanup crash), but output file is valid — continuing.")
            except Exception:
                print(f"[ERROR] LocalFitBaseline macro failed (exit code {result.returncode})")
                return None

        # Load results from output ROOT file
        with uproot.open(out_tmp.name) as f:
            pred_tree = f['predictions']
            lf_event_idx = pred_tree['event_idx'].array(library='np')
            lf_sensor_id = pred_tree['sensor_id'].array(library='np')
            lf_truth_raw = pred_tree['truth_npho'].array(library='np')
            lf_pred_raw = pred_tree['pred_npho'].array(library='np')

        print(f"[INFO] LocalFitBaseline: loaded {len(lf_event_idx)} predictions from macro")

        # Normalize raw predictions using NphoTransform
        transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale,
                                  npho_scale2=npho_scale2)

        domain_min = transform.domain_min()
        lf_pred_safe = np.maximum(lf_pred_raw, domain_min)
        lf_pred_norm = transform.forward(lf_pred_safe).astype(np.float32)

        # Apply max_events filter (macro processes all events in the file)
        if max_events is not None:
            keep = lf_event_idx < max_events
            lf_event_idx = lf_event_idx[keep]
            lf_sensor_id = lf_sensor_id[keep]
            lf_pred_norm = lf_pred_norm[keep]

        # Build (N, 4760) prediction array (NaN = no prediction)
        n_events_total = x_original.shape[0]
        lf_pred_full = np.full((n_events_total, N_CHANNELS), np.nan,
                               dtype=np.float32)

        valid = lf_event_idx.astype(int) < n_events_total
        lf_pred_full[lf_event_idx[valid].astype(int),
                     lf_sensor_id[valid].astype(int)] = lf_pred_norm[valid]
        n_filled = int(valid.sum())
        print(f"[INFO] LocalFitBaseline: filled {n_filled} predictions into array")

    finally:
        # Clean up temp files
        if os.path.exists(dead_tmp.name):
            os.unlink(dead_tmp.name)
        if os.path.exists(out_tmp.name):
            os.unlink(out_tmp.name)

    return lf_pred_full


def collect_predictions(predictions: np.ndarray, x_original: np.ndarray,
                        mask: np.ndarray, artificial_mask: np.ndarray,
                        dead_mask: np.ndarray,
                        run_numbers: Optional[np.ndarray] = None,
                        event_numbers: Optional[np.ndarray] = None,
                        predict_channels: List[str] = None) -> Dict[str, np.ndarray]:
    """Collect ML predictions at masked positions into arrays.

    Returns:
        Dictionary of arrays, each with shape (n_masked_total,).
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']
    predict_time = 'time' in predict_channels
    pred_npho_idx = predict_channels.index('npho') if 'npho' in predict_channels else 0
    pred_time_idx = predict_channels.index('time') if predict_time else None

    # Pre-compute sensor -> face mapping
    sensor_face = np.full(N_CHANNELS, -1, dtype=np.int32)
    for fname, idx_map in FACE_INDEX_MAPS.items():
        face_int = FACE_NAME_TO_INT[fname]
        if fname in ['top', 'bot']:
            sensor_face[idx_map] = face_int
        else:
            flat = idx_map.flatten()
            sensor_face[flat[flat >= 0]] = face_int

    # Get all masked positions
    event_idxs, sensor_ids = np.where(mask)
    n_entries = len(event_idxs)

    # Vectorized extraction
    pred_npho = predictions[event_idxs, sensor_ids, pred_npho_idx].astype(np.float32)
    truth_npho = x_original[event_idxs, sensor_ids, 0].astype(np.float32)
    truth_time = x_original[event_idxs, sensor_ids, 1].astype(np.float32)
    face = sensor_face[sensor_ids]

    if predict_time and pred_time_idx is not None:
        pred_time = predictions[event_idxs, sensor_ids, pred_time_idx].astype(np.float32)
    else:
        pred_time = np.full(n_entries, -999.0, dtype=np.float32)

    # Mask type: 0=artificial (has truth), 1=dead (no truth)
    if artificial_mask is not None:
        is_artificial = artificial_mask[event_idxs, sensor_ids]
        mask_type = np.where(is_artificial, 0, 1).astype(np.int32)
    else:
        mask_type = np.zeros(n_entries, dtype=np.int32)

    # Error computation
    has_truth = (mask_type == 0) & (truth_npho != MODEL_SENTINEL_NPHO)
    error_npho = np.where(has_truth, pred_npho - truth_npho, -999.0).astype(np.float32)
    has_time_truth = has_truth & predict_time
    error_time = np.where(has_time_truth, pred_time - truth_time, -999.0).astype(np.float32)

    # Set truth to -999 for dead channels
    is_dead = mask_type == 1
    truth_npho = np.where(is_dead, -999.0, truth_npho).astype(np.float32)
    truth_time = np.where(is_dead, -999.0, truth_time).astype(np.float32)

    # Run/event numbers
    if run_numbers is not None:
        run_nums = run_numbers[event_idxs].astype(np.int64)
    else:
        run_nums = np.full(n_entries, -1, dtype=np.int64)
    if event_numbers is not None:
        event_nums = event_numbers[event_idxs].astype(np.int64)
    else:
        event_nums = np.full(n_entries, -1, dtype=np.int64)

    return {
        'event_idx': event_idxs.astype(np.int32),
        'sensor_id': sensor_ids.astype(np.int32),
        'run_number': run_nums,
        'event_number': event_nums,
        'face': face,
        'mask_type': mask_type,
        'truth_npho': truth_npho,
        'truth_time': truth_time,
        'pred_npho': pred_npho,
        'pred_time': pred_time,
        'error_npho': error_npho,
        'error_time': error_time,
    }


def compute_metrics(pred_data: Dict[str, np.ndarray],
                    predict_channels: List[str] = None) -> Dict:
    """Compute metrics from prediction arrays.

    Args:
        pred_data: Dictionary of arrays from collect_predictions().
        predict_channels: List of channels being predicted.
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']
    predict_time = 'time' in predict_channels

    mask_type = pred_data['mask_type']
    error_npho = pred_data['error_npho']

    artificial_with_truth = (mask_type == 0) & (error_npho > -999)

    metrics = {
        'n_total': len(mask_type),
        'n_artificial': int((mask_type == 0).sum()),
        'n_dead': int((mask_type == 1).sum()),
        'n_with_truth': int(artificial_with_truth.sum()),
    }

    if artificial_with_truth.any():
        err = error_npho[artificial_with_truth]

        metrics.update({
            'npho_mae': float(np.mean(np.abs(err))),
            'npho_rmse': float(np.sqrt(np.mean(err ** 2))),
            'npho_bias': float(np.mean(err)),
            'npho_68pct': float(np.percentile(np.abs(err), 68)),
        })

        if predict_time:
            err_time = pred_data['error_time'][artificial_with_truth]
            valid_time = err_time > -900
            if valid_time.any():
                et = err_time[valid_time]
                metrics.update({
                    'time_mae': float(np.mean(np.abs(et))),
                    'time_rmse': float(np.sqrt(np.mean(et ** 2))),
                    'time_bias': float(np.mean(et)),
                    'time_68pct': float(np.percentile(np.abs(et), 68)),
                })

        # Per-face metrics
        face = pred_data['face']
        for face_name, face_int in FACE_NAME_TO_INT.items():
            face_mask = artificial_with_truth & (face == face_int)
            if face_mask.any():
                face_err = error_npho[face_mask]
                metrics[f'{face_name}_n'] = int(face_mask.sum())
                metrics[f'{face_name}_npho_mae'] = float(np.mean(np.abs(face_err)))
                if predict_time:
                    face_err_time = pred_data['error_time'][face_mask]
                    valid_t = face_err_time > -900
                    if valid_t.any():
                        metrics[f'{face_name}_time_mae'] = float(np.mean(np.abs(face_err_time[valid_t])))

    return metrics


def compute_baseline_metrics(pred_full: np.ndarray, x_original: np.ndarray,
                              combined_mask: np.ndarray,
                              artificial_mask: Optional[np.ndarray] = None) -> Dict:
    """Compute metrics for a baseline from its (N, 4760) prediction array.

    Args:
        pred_full: (N, 4760) baseline predictions (NaN = no prediction).
        x_original: (N, 4760, 2) normalized original data.
        combined_mask: (N, 4760) bool mask.
        artificial_mask: (N, 4760) bool or None (None = MC mode, all have truth).

    Returns:
        Dictionary of metrics.
    """
    truth = x_original[:, :, 0]

    # In MC mode, all masked positions have truth
    if artificial_mask is None:
        eval_mask = combined_mask
    else:
        eval_mask = artificial_mask

    # Valid = has truth + baseline has prediction
    has_truth = eval_mask & (truth != MODEL_SENTINEL_NPHO)
    has_pred = np.isfinite(pred_full) & (pred_full > -900)
    valid = has_truth & has_pred

    pred_vals = pred_full[valid]
    truth_vals = truth[valid]
    error = pred_vals - truth_vals

    metrics = {'n_with_truth': int(valid.sum())}

    if len(error) > 0:
        metrics.update({
            'npho_mae': float(np.mean(np.abs(error))),
            'npho_rmse': float(np.sqrt(np.mean(error ** 2))),
            'npho_bias': float(np.mean(error)),
            'npho_68pct': float(np.percentile(np.abs(error), 68)),
        })

    return metrics


def save_predictions(pred_data: Dict[str, np.ndarray], output_path: str,
                     run_number: Optional[int] = None,
                     predict_channels: List[str] = None,
                     npho_scheme: str = 'log1p',
                     npho_scale: float = DEFAULT_NPHO_SCALE,
                     npho_scale2: float = DEFAULT_NPHO_SCALE2,
                     time_scale: float = DEFAULT_TIME_SCALE,
                     time_shift: float = DEFAULT_TIME_SHIFT,
                     baseline_preds: Optional[Dict[str, np.ndarray]] = None):
    """
    Save predictions to ROOT file with metadata.

    Args:
        pred_data: Dictionary of arrays from collect_predictions().
        output_path: Output ROOT file path
        run_number: Run number for dead channel pattern
        predict_channels: List of predicted channels (['npho'] or ['npho', 'time'])
        npho_scheme: Npho normalization scheme
        npho_scale, npho_scale2, time_scale, time_shift: Normalization parameters
        baseline_preds: Optional dict mapping baseline name to (N, 4760) arrays
    """
    n_entries = len(pred_data['event_idx'])
    if n_entries == 0:
        print("[WARNING] No predictions to save")
        return

    if predict_channels is None:
        predict_channels = ['npho', 'time']

    branches = {
        'event_idx': pred_data['event_idx'],
        'run_number': pred_data['run_number'],
        'event_number': pred_data['event_number'],
        'sensor_id': pred_data['sensor_id'],
        'face': pred_data['face'],
        'mask_type': pred_data['mask_type'],
        'truth_npho': pred_data['truth_npho'],
        'truth_time': pred_data['truth_time'],
        'pred_npho': pred_data['pred_npho'],
        'error_npho': pred_data['error_npho'],
    }

    if 'time' in predict_channels:
        branches['pred_time'] = pred_data['pred_time']
        branches['error_time'] = pred_data['error_time']

    if run_number is not None:
        branches['dead_pattern_run'] = np.full(n_entries, run_number, dtype=np.int32)

    # Add baseline branches by indexing into (N, 4760) arrays
    if baseline_preds:
        event_idxs = pred_data['event_idx']
        sensor_ids = pred_data['sensor_id']
        truth = pred_data['truth_npho']
        mask_type = pred_data['mask_type']

        for bname, bpred_full in baseline_preds.items():
            bpred = bpred_full[event_idxs, sensor_ids].astype(np.float32)
            # Compute error (valid when has truth and baseline has prediction)
            has_truth = (mask_type == 0) & (truth != MODEL_SENTINEL_NPHO)
            has_bpred = np.isfinite(bpred) & (bpred > -900)
            berror = np.where(
                has_truth & has_bpred, bpred - truth, -999.0
            ).astype(np.float32)
            # Replace NaN with -999 for ROOT compatibility
            bpred = np.where(np.isfinite(bpred), bpred, -999.0).astype(np.float32)
            branches[f'baseline_{bname}_npho'] = bpred
            branches[f'baseline_{bname}_error_npho'] = berror

    # Metadata for downstream analysis scripts
    metadata = {
        'predict_channels': np.array([','.join(predict_channels)], dtype='U32'),
        'npho_scheme': np.array([npho_scheme], dtype='U16'),
        'npho_scale': np.array([npho_scale], dtype=np.float64),
        'npho_scale2': np.array([npho_scale2], dtype=np.float64),
        'time_scale': np.array([time_scale], dtype=np.float64),
        'time_shift': np.array([time_shift], dtype=np.float64),
    }

    with uproot.recreate(output_path) as f:
        f.mktree('predictions', branches)
        f.mktree('metadata', metadata)

    print(f"[INFO] Saved {n_entries:,} predictions to {output_path}")
    print(f"[INFO] Metadata: predict_channels={predict_channels}")


def print_summary(metrics: Dict, is_real_data: bool, run_number: Optional[int],
                  baseline_metrics: Optional[Dict[str, Dict]] = None):
    """Print metrics summary.

    Args:
        metrics: ML model metrics from compute_metrics().
        is_real_data: Whether this is real data mode.
        run_number: Run number for dead channel pattern.
        baseline_metrics: Optional dict mapping baseline name ('avg', 'sa')
                          to their metric dicts from compute_baseline_metrics().
    """
    print("\n" + "=" * 70)
    print("INPAINTER VALIDATION RESULTS")
    print("=" * 70)

    mode = "Real Data" if is_real_data else "MC Pseudo-Experiment"
    print(f"Mode: {mode}")
    if run_number:
        print(f"Dead channel pattern from run: {run_number}")

    print(f"\nPredictions: {metrics['n_total']:,} total")
    if is_real_data:
        print(f"  - Artificial masks (has truth): {metrics['n_artificial']:,}")
        print(f"  - Dead channels (no truth): {metrics['n_dead']:,}")
    else:
        print(f"  - With ground truth: {metrics['n_with_truth']:,}")

    if metrics.get('npho_mae') is not None:
        print(f"\nGlobal Metrics (positions with ground truth):")
        print(f"  npho: MAE={metrics['npho_mae']:.4f}, RMSE={metrics['npho_rmse']:.4f}, "
              f"Bias={metrics['npho_bias']:.4f}, 68%={metrics['npho_68pct']:.4f}")
        # Time metrics only if available (npho-only models won't have them)
        if metrics.get('time_mae') is not None:
            print(f"  time: MAE={metrics['time_mae']:.4f}, RMSE={metrics['time_rmse']:.4f}, "
                  f"Bias={metrics['time_bias']:.4f}, 68%={metrics['time_68pct']:.4f}")

        print(f"\nPer-Face Metrics:")
        for face_name in ['inner', 'us', 'ds', 'outer', 'top', 'bot']:
            n = metrics.get(f'{face_name}_n', 0)
            if n > 0:
                line = f"  {face_name:>6}: n={n:6,}, npho_MAE={metrics[f'{face_name}_npho_mae']:.4f}"
                if metrics.get(f'{face_name}_time_mae') is not None:
                    line += f", time_MAE={metrics[f'{face_name}_time_mae']:.4f}"
                print(line)

    # --- Baseline comparison ---
    if baseline_metrics:
        print("\n" + "-" * 70)
        print("BASELINE COMPARISON (npho, normalized space)")
        print("-" * 70)

        # Print individual baseline metrics
        baseline_name_map = {
            'avg': 'Neighbor Avg (k-hop)',
            'sa': 'Solid Angle Weighted',
            'localfit': 'Local Fit (SA)',
        }
        for bname, bmetrics in baseline_metrics.items():
            label = baseline_name_map.get(bname, bname)
            if bmetrics.get('npho_mae') is not None:
                print(f"\n  {label}:")
                print(f"    MAE={bmetrics['npho_mae']:.4f}, "
                      f"RMSE={bmetrics['npho_rmse']:.4f}, "
                      f"Bias={bmetrics['npho_bias']:.4f}, "
                      f"68%={bmetrics['npho_68pct']:.4f}")
            else:
                print(f"\n  {label}: no predictions with ground truth")

        # Side-by-side comparison table
        if metrics.get('npho_mae') is not None:
            print(f"\n  {'Method':<26} {'MAE':>8} {'RMSE':>8} {'Bias':>8} {'68%':>8}")
            print(f"  {'-'*26} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
            print(f"  {'ML Model':<26} {metrics['npho_mae']:>8.4f} "
                  f"{metrics['npho_rmse']:>8.4f} {metrics['npho_bias']:>8.4f} "
                  f"{metrics['npho_68pct']:>8.4f}")
            for bname, bmetrics in baseline_metrics.items():
                label = baseline_name_map.get(bname, bname)
                if bmetrics.get('npho_mae') is not None:
                    print(f"  {label:<26} {bmetrics['npho_mae']:>8.4f} "
                          f"{bmetrics['npho_rmse']:>8.4f} {bmetrics['npho_bias']:>8.4f} "
                          f"{bmetrics['npho_68pct']:>8.4f}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Unified inpainter validation for MC and real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model (one required)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", "-c", type=str,
                             help="Path to inpainter checkpoint (.pth)")
    model_group.add_argument("--torchscript", "-t", type=str,
                             help="Path to TorchScript model (.pt)")

    # Data
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input ROOT file, directory, or glob pattern")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    # Dead channels (one required)
    dead_group = parser.add_mutually_exclusive_group(required=True)
    dead_group.add_argument("--run", type=int,
                            help="Run number to fetch dead channels from database")
    dead_group.add_argument("--dead-channel-file", type=str,
                            help="Path to dead channel list file")

    # Mode
    parser.add_argument("--real-data", action="store_true",
                        help="Real data mode: dead channels exist in data, add artificial masking")
    parser.add_argument("--n-artificial", type=str, default="inner:10,us:1,ds:1,outer:1,top:1,bot:1",
                        help="Artificial masks per event. Either a single int (uniform random) "
                             "or face:count pairs (e.g. 'inner:10,us:1,ds:1,outer:1,top:1,bot:1'). "
                             "Default: stratified 15 total (10 inner + 1 each other face)")

    # Options
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--num-threads", type=int, default=None,
                        help="Number of CPU threads for PyTorch (default: all available)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum events to process")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for artificial masking (default: 42)")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "cuda"],
                        help="Device (default: cpu)")
    parser.add_argument("--tree-name", type=str, default="tree",
                        help="TTree name in ROOT file (default: tree)")

    # Model options
    parser.add_argument("--predict-channels", type=str, nargs='+', default=None,
                        choices=["npho", "time"],
                        help="Override predict channels (e.g., --predict-channels npho). "
                             "Auto-detected from model output shape if not specified.")
    parser.add_argument("--npho-scheme", type=str, default=None,
                        choices=["log1p", "sqrt", "anscombe", "linear"],
                        help="Npho normalization scheme. Auto-detected from checkpoint; "
                             "REQUIRED for TorchScript models (default: log1p).")

    # Baselines
    parser.add_argument("--baselines", action="store_true",
                        help="Enable rule-based baseline computation alongside ML")
    parser.add_argument("--solid-angle-branch", type=str, default=None,
                        help="Branch name in ROOT file for solid angles "
                             "(enables solid-angle-weighted baseline)")
    parser.add_argument("--baseline-k", type=int, default=1,
                        help="k-hop parameter for baseline neighbor search (default: 1)")
    parser.add_argument("--local-fit-baseline", action="store_true",
                        help="Enable LocalFitBaseline via ROOT macro (requires ROOT in PATH)")

    args = parser.parse_args()

    # Set number of threads
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"[INFO] Using {args.num_threads} CPU threads")
    else:
        print(f"[INFO] Using {torch.get_num_threads()} CPU threads (default)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load model first to get npho_scheme before normalizing
    model, model_type, predict_channels, model_npho_scheme, \
        model_npho_scale, model_npho_scale2 = load_model(
        checkpoint_path=args.checkpoint,
        torchscript_path=args.torchscript,
        device=args.device
    )

    # Override predict_channels if specified via CLI
    if args.predict_channels is not None:
        print(f"[INFO] Overriding predict_channels: {predict_channels} → {args.predict_channels}")
        predict_channels = args.predict_channels

    # Resolve npho_scheme: CLI > checkpoint > default
    if args.npho_scheme is not None:
        npho_scheme = args.npho_scheme
        print(f"[INFO] Npho scheme (from CLI): {npho_scheme}")
    elif model_npho_scheme is not None:
        npho_scheme = model_npho_scheme
        print(f"[INFO] Npho scheme (from checkpoint): {npho_scheme}")
    else:
        npho_scheme = 'log1p'
        print(f"[WARN] Npho scheme unknown (TorchScript has no metadata), using default: {npho_scheme}")
        print(f"[WARN] Use --npho-scheme to specify if your model was trained with a different scheme")

    npho_scale = model_npho_scale
    npho_scale2 = model_npho_scale2

    # Load data
    data = load_data(args.input, tree_name=args.tree_name, max_events=args.max_events)
    n_events = len(data['npho'])

    # Normalize
    print(f"[INFO] Normalizing data (npho_scheme={npho_scheme}, "
          f"scale={npho_scale}, scale2={npho_scale2})...")
    x_input = normalize_data(data['npho'], data['time'], npho_scheme=npho_scheme,
                             npho_scale=npho_scale, npho_scale2=npho_scale2)
    x_original = x_input.copy()

    # Free raw arrays (no longer needed after normalization)
    del data['npho'], data['time']
    gc.collect()

    # Get dead channels
    dead_channels = get_dead_channels(
        run_number=args.run,
        dead_channel_file=args.dead_channel_file
    )
    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    if len(dead_channels) > 0:
        dead_mask[dead_channels] = True
    n_dead = dead_mask.sum()
    print(f"[INFO] Dead channels: {n_dead} ({n_dead/N_CHANNELS*100:.2f}%)")

    # Create masks
    if args.real_data:
        # Parse n_artificial: int or face:count pairs
        try:
            n_art = int(args.n_artificial)
        except ValueError:
            n_art = {}
            for part in args.n_artificial.split(','):
                fname, count = part.strip().split(':')
                n_art[fname.strip()] = int(count.strip())

        if isinstance(n_art, dict):
            total = sum(n_art.values())
            print(f"[INFO] Real data mode: stratified masking ({total} per event: "
                  + ", ".join(f"{f}={n}" for f, n in n_art.items()) + ")")
        else:
            print(f"[INFO] Real data mode: adding {n_art} uniform random masks per event")

        artificial_mask, combined_mask = create_artificial_mask(
            x_input, n_art, dead_mask, seed=args.seed
        )
        # Apply combined mask to input (per-channel sentinels)
        x_input[combined_mask, 0] = MODEL_SENTINEL_NPHO  # npho -> npho sentinel
        x_input[combined_mask, 1] = MODEL_SENTINEL_TIME        # time -> time sentinel
    else:
        # MC pseudo-experiment: apply dead pattern to clean MC
        print("[INFO] MC mode: applying dead channel pattern")
        artificial_mask = None
        combined_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
        combined_mask[:, dead_mask] = True
        x_input[combined_mask, 0] = MODEL_SENTINEL_NPHO  # npho -> npho sentinel
        x_input[combined_mask, 1] = MODEL_SENTINEL_TIME        # time -> time sentinel

    n_masked = combined_mask.sum()
    print(f"[INFO] Total masked sensors: {n_masked:,} ({n_masked/(n_events*N_CHANNELS)*100:.2f}%)")

    # Run inference
    print(f"[INFO] Running inference on {args.device}...")
    predictions = run_inference(
        model, model_type, x_input, combined_mask,
        batch_size=args.batch_size, device=args.device,
        predict_channels=predict_channels
    )

    # Free masked input and model (no longer needed)
    del x_input, model
    gc.collect()

    # --- Run baselines (if requested) ---
    baseline_preds = None  # Dict[str, np.ndarray] of (N, 4760) arrays
    baseline_metrics_dict = None
    if args.baselines:
        # Load solid angles if branch is provided
        solid_angles = None
        if args.solid_angle_branch:
            solid_angles = load_solid_angles(
                args.input, args.solid_angle_branch,
                tree_name=args.tree_name, max_events=args.max_events
            )

        npho_xf = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale,
                                npho_scale2=npho_scale2)
        baseline_preds = run_baselines(
            x_original, combined_mask, artificial_mask, dead_mask,
            baseline_k=args.baseline_k,
            solid_angles=solid_angles,
            npho_transform=npho_xf,
        )

        # Free solid angles (no longer needed)
        del solid_angles
        gc.collect()

    # --- LocalFitBaseline (if requested) ---
    if args.local_fit_baseline:
        if baseline_preds is None:
            baseline_preds = {}
        lf_pred_array = run_local_fit_baseline(
            input_path=args.input,
            dead_channels=dead_channels,
            x_original=x_original,
            combined_mask=combined_mask,
            artificial_mask=artificial_mask,
            npho_scheme=npho_scheme,
            max_events=args.max_events,
        )
        if lf_pred_array is not None:
            baseline_preds['localfit'] = lf_pred_array

    if baseline_preds:
        # Compute baseline metrics from (N, 4760) arrays
        baseline_metrics_dict = {}
        for bname, bpred_full in baseline_preds.items():
            baseline_metrics_dict[bname] = compute_baseline_metrics(
                bpred_full, x_original, combined_mask, artificial_mask
            )

    # Collect ML results
    print("[INFO] Collecting predictions...")
    pred_data = collect_predictions(
        predictions, x_original, combined_mask, artificial_mask, dead_mask,
        run_numbers=data.get('run'),
        event_numbers=data.get('event'),
        predict_channels=predict_channels
    )

    # Free large arrays no longer needed
    del predictions, x_original, combined_mask, data
    if artificial_mask is not None:
        del artificial_mask
    gc.collect()

    # Compute ML metrics
    metrics = compute_metrics(pred_data, predict_channels=predict_channels)

    # Print summary (with optional baseline comparison)
    print_summary(metrics, args.real_data, args.run,
                  baseline_metrics=baseline_metrics_dict)

    # Save outputs
    run_str = str(args.run) if args.run else "custom"
    mode_str = "real" if args.real_data else "mc"

    # Predictions ROOT file (with optional baseline branches)
    pred_file = os.path.join(args.output, f"predictions_{mode_str}_run{run_str}.root")
    save_predictions(pred_data, pred_file, run_number=args.run,
                     predict_channels=predict_channels,
                     npho_scheme=npho_scheme,
                     npho_scale=npho_scale,
                     npho_scale2=npho_scale2,
                     baseline_preds=baseline_preds)

    # Free remaining large data
    del pred_data
    if baseline_preds is not None:
        del baseline_preds
    gc.collect()

    # Metrics CSV (include baseline metrics if available)
    metrics_file = os.path.join(args.output, f"metrics_{mode_str}_run{run_str}.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, v])
        # Append baseline metrics with prefixed keys
        if baseline_metrics_dict:
            for bname, bmetrics in baseline_metrics_dict.items():
                for k, v in sorted(bmetrics.items()):
                    writer.writerow([f'baseline_{bname}_{k}', v])
    print(f"[INFO] Saved metrics to {metrics_file}")

    # Dead channel list
    dead_file = os.path.join(args.output, f"dead_channels_run{run_str}.txt")
    np.savetxt(dead_file, dead_channels, fmt='%d',
               header=f"Dead channels (n={n_dead})")
    print(f"[INFO] Saved dead channel list to {dead_file}")

    print(f"\n[INFO] Done! Use show_inpainter_comparison.py to visualize events:")
    print(f"  python macro/show_inpainter_comparison.py 0 \\")
    print(f"      --predictions {pred_file} \\")
    print(f"      --original {args.input}")


if __name__ == "__main__":
    main()
