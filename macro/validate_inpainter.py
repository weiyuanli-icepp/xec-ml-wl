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

import os
import sys
import argparse
import csv
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
    DEFAULT_SENTINEL_VALUE, DEFAULT_NPHO_THRESHOLD
)
from lib.dataset import expand_path
from lib.baselines import NeighborAverageBaseline, SolidAngleWeightedBaseline

# Constants
N_CHANNELS = 4760
MODEL_SENTINEL = DEFAULT_SENTINEL_VALUE

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
                   npho_scale: float = DEFAULT_NPHO_SCALE,
                   npho_scale2: float = DEFAULT_NPHO_SCALE2,
                   time_scale: float = DEFAULT_TIME_SCALE,
                   time_shift: float = DEFAULT_TIME_SHIFT,
                   sentinel: float = MODEL_SENTINEL,
                   npho_threshold: float = DEFAULT_NPHO_THRESHOLD) -> np.ndarray:
    """Normalize data to model input format."""
    # Invalid detection
    mask_npho_invalid = (npho > 9e9) | (npho < -npho_scale) | np.isnan(npho)
    mask_time_invalid = mask_npho_invalid | (npho < npho_threshold) | (np.abs(time) > 9e9) | np.isnan(time)

    # Normalize npho: log1p transform
    npho_safe = np.where(mask_npho_invalid, 0.0, np.maximum(npho, 0.0))
    npho_norm = np.log1p(npho_safe / npho_scale) / npho_scale2
    npho_norm[mask_npho_invalid] = sentinel

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


def create_artificial_mask(x: np.ndarray, n_artificial: int,
                           dead_mask: np.ndarray,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create artificial mask on healthy sensors for evaluation.

    Returns:
        artificial_mask: (N, 4760) bool - artificially masked positions
        combined_mask: (N, 4760) bool - dead + artificial
    """
    rng = np.random.default_rng(seed)
    n_events = x.shape[0]

    # Find valid sensors (not dead, not already sentinel)
    artificial_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)

    for i in range(n_events):
        # Valid = not dead AND has valid data (npho != sentinel)
        valid = ~dead_mask & (x[i, :, 0] != MODEL_SENTINEL)
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
    """
    if torchscript_path:
        print(f"[INFO] Loading TorchScript model from {torchscript_path}")
        model = torch.jit.load(torchscript_path, map_location=device)
        model.eval()
        # TorchScript models: assume default channels or detect from output
        predict_channels = getattr(model, 'predict_channels', ['npho', 'time'])
        return model, 'torchscript', predict_channels

    if checkpoint_path:
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        from lib.models import XECEncoder, XEC_Inpainter

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get('config', {})

        # Get predict_channels from checkpoint config (default to both for legacy)
        predict_channels = config.get('predict_channels', ['npho', 'time'])
        print(f"[INFO] Predict channels: {predict_channels}")

        encoder = XECEncoder(
            outer_mode=config.get('outer_mode', 'finegrid'),
            outer_fine_pool=config.get('outer_fine_pool', None),
        )
        use_masked_attention = config.get('use_masked_attention', False)
        model = XEC_Inpainter(encoder=encoder, predict_channels=predict_channels,
                              use_masked_attention=use_masked_attention)

        # Load weights (prefer EMA)
        if 'ema_state_dict' in checkpoint:
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
        return model, 'checkpoint', predict_channels

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
                  run_numbers: Optional[np.ndarray] = None,
                  event_numbers: Optional[np.ndarray] = None,
                  ) -> Dict[str, List[Dict]]:
    """Run baseline predictions and collect results per masked sensor.

    Args:
        x_original: (N, 4760, 2) normalized original data (before masking).
        combined_mask: (N, 4760) bool mask (dead + artificial).
        artificial_mask: (N, 4760) bool or None.
        dead_mask: (4760,) bool.
        baseline_k: k-hop parameter.
        solid_angles: (N, 4760) solid angles or None.
        run_numbers: optional run number per event.
        event_numbers: optional event number per event.

    Returns:
        Dictionary with keys 'avg' (and optionally 'sa') mapping to lists
        of per-sensor prediction dicts with keys:
        pred_npho, truth_npho, error_npho, event_idx, sensor_id
    """
    x_npho = x_original[:, :, 0]  # (N, 4760) normalized npho, unmasked
    n_events = x_npho.shape[0]

    # --- Neighbor Average Baseline ---
    print("[INFO] Running NeighborAverageBaseline...")
    avg_baseline = NeighborAverageBaseline(k=baseline_k)
    avg_preds_full = avg_baseline.predict(x_npho, combined_mask)  # (N, 4760)

    baseline_results = {}

    # Collect per-sensor results for avg baseline
    avg_results = []
    for i in range(n_events):
        masked_sensors = np.where(combined_mask[i])[0]
        for sensor_id in masked_sensors:
            is_artificial = artificial_mask[i, sensor_id] if artificial_mask is not None else True
            mask_type = 0 if is_artificial else 1

            pred_npho = float(avg_preds_full[i, sensor_id])
            truth_npho = float(x_original[i, sensor_id, 0])

            if truth_npho == MODEL_SENTINEL or mask_type == 1:
                error_npho = -999.0
                if mask_type == 1:
                    truth_npho = -999.0
            else:
                error_npho = pred_npho - truth_npho

            avg_results.append({
                'event_idx': i,
                'sensor_id': int(sensor_id),
                'mask_type': mask_type,
                'pred_npho': pred_npho,
                'truth_npho': truth_npho,
                'error_npho': error_npho,
            })
    baseline_results['avg'] = avg_results

    # --- Solid Angle Weighted Baseline ---
    if solid_angles is not None:
        print("[INFO] Running SolidAngleWeightedBaseline...")
        sa_baseline = SolidAngleWeightedBaseline(k=baseline_k)
        sa_preds_full = sa_baseline.predict(x_npho, combined_mask,
                                            solid_angles=solid_angles)

        sa_results = []
        for i in range(n_events):
            masked_sensors = np.where(combined_mask[i])[0]
            for sensor_id in masked_sensors:
                is_artificial = artificial_mask[i, sensor_id] if artificial_mask is not None else True
                mask_type = 0 if is_artificial else 1

                pred_npho = float(sa_preds_full[i, sensor_id])
                truth_npho = float(x_original[i, sensor_id, 0])

                if truth_npho == MODEL_SENTINEL or mask_type == 1:
                    error_npho = -999.0
                    if mask_type == 1:
                        truth_npho = -999.0
                else:
                    error_npho = pred_npho - truth_npho

                sa_results.append({
                    'event_idx': i,
                    'sensor_id': int(sensor_id),
                    'mask_type': mask_type,
                    'pred_npho': pred_npho,
                    'truth_npho': truth_npho,
                    'error_npho': error_npho,
                })
        baseline_results['sa'] = sa_results

    return baseline_results


def collect_predictions(predictions: np.ndarray, x_original: np.ndarray,
                        mask: np.ndarray, artificial_mask: np.ndarray,
                        dead_mask: np.ndarray,
                        run_numbers: Optional[np.ndarray] = None,
                        event_numbers: Optional[np.ndarray] = None,
                        predict_channels: List[str] = None) -> List[Dict]:
    """Collect predictions into a flat list.

    Args:
        predict_channels: List of channels being predicted (['npho'] or ['npho', 'time'])
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']
    predict_time = 'time' in predict_channels
    pred_npho_idx = predict_channels.index('npho') if 'npho' in predict_channels else 0
    pred_time_idx = predict_channels.index('time') if 'time' in predict_channels else None

    results = []
    n_events = predictions.shape[0]

    for i in range(n_events):
        masked_sensors = np.where(mask[i])[0]

        for sensor_id in masked_sensors:
            # Determine mask type: 0=artificial (has truth), 1=dead (no truth)
            is_artificial = artificial_mask[i, sensor_id] if artificial_mask is not None else True
            mask_type = 0 if is_artificial else 1

            # Get face name
            face_name = None
            for fname, idx_map in FACE_INDEX_MAPS.items():
                if fname in ['top', 'bot']:
                    if sensor_id in idx_map:
                        face_name = fname
                        break
                else:
                    if sensor_id in idx_map.flatten():
                        face_name = fname
                        break
            if face_name is None:
                face_name = 'unknown'

            # Get prediction
            pred_npho = float(predictions[i, sensor_id, pred_npho_idx])
            pred_time = float(predictions[i, sensor_id, pred_time_idx]) if predict_time else -999.0

            # Get truth (only valid for artificial masks or MC)
            truth_npho = float(x_original[i, sensor_id, 0])
            truth_time = float(x_original[i, sensor_id, 1])

            # Check if truth is valid
            if truth_npho == MODEL_SENTINEL or mask_type == 1:
                error_npho = -999.0
                error_time = -999.0
                if mask_type == 1:
                    truth_npho = -999.0
                    truth_time = -999.0
            else:
                error_npho = pred_npho - truth_npho
                error_time = (pred_time - truth_time) if predict_time else -999.0

            results.append({
                'event_idx': i,
                'run_number': int(run_numbers[i]) if run_numbers is not None else -1,
                'event_number': int(event_numbers[i]) if event_numbers is not None else -1,
                'sensor_id': int(sensor_id),
                'face': face_name,
                'mask_type': mask_type,
                'truth_npho': truth_npho,
                'truth_time': truth_time,
                'pred_npho': pred_npho,
                'pred_time': pred_time,
                'error_npho': error_npho,
                'error_time': error_time,
            })

    return results


def compute_metrics(predictions: List[Dict], predict_channels: List[str] = None) -> Dict:
    """Compute metrics from predictions.

    Args:
        predict_channels: List of channels being predicted (['npho'] or ['npho', 'time'])
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']
    predict_time = 'time' in predict_channels

    # Split by mask type
    artificial = [p for p in predictions if p['mask_type'] == 0 and p['error_npho'] > -999]
    dead = [p for p in predictions if p['mask_type'] == 1]

    metrics = {
        'n_total': len(predictions),
        'n_artificial': len([p for p in predictions if p['mask_type'] == 0]),
        'n_dead': len(dead),
        'n_with_truth': len(artificial),
    }

    if artificial:
        err_npho = np.array([p['error_npho'] for p in artificial])

        metrics.update({
            'npho_mae': np.mean(np.abs(err_npho)),
            'npho_rmse': np.sqrt(np.mean(err_npho ** 2)),
            'npho_bias': np.mean(err_npho),
            'npho_68pct': np.percentile(np.abs(err_npho), 68),
        })

        # Time metrics only if predicting time
        if predict_time:
            err_time = np.array([p['error_time'] for p in artificial])
            # Filter out -999 placeholder values
            valid_time = err_time > -900
            if valid_time.any():
                err_time_valid = err_time[valid_time]
                metrics.update({
                    'time_mae': np.mean(np.abs(err_time_valid)),
                    'time_rmse': np.sqrt(np.mean(err_time_valid ** 2)),
                    'time_bias': np.mean(err_time_valid),
                    'time_68pct': np.percentile(np.abs(err_time_valid), 68),
                })

        # Per-face metrics
        for face_name in ['inner', 'us', 'ds', 'outer', 'top', 'bot']:
            face_preds = [p for p in artificial if p['face'] == face_name]
            if face_preds:
                face_err_npho = np.array([p['error_npho'] for p in face_preds])
                metrics[f'{face_name}_n'] = len(face_preds)
                metrics[f'{face_name}_npho_mae'] = np.mean(np.abs(face_err_npho))
                if predict_time:
                    face_err_time = np.array([p['error_time'] for p in face_preds])
                    valid_time = face_err_time > -900
                    if valid_time.any():
                        metrics[f'{face_name}_time_mae'] = np.mean(np.abs(face_err_time[valid_time]))

    return metrics


def compute_baseline_metrics(baseline_predictions: List[Dict]) -> Dict:
    """Compute metrics for a single baseline from its prediction list.

    Args:
        baseline_predictions: List of dicts with keys error_npho, mask_type.

    Returns:
        Dictionary of metrics (npho_mae, npho_rmse, npho_bias).
    """
    with_truth = [p for p in baseline_predictions
                  if p['mask_type'] == 0 and p['error_npho'] > -999]

    metrics = {'n_with_truth': len(with_truth)}

    if with_truth:
        err = np.array([p['error_npho'] for p in with_truth])
        metrics.update({
            'npho_mae': np.mean(np.abs(err)),
            'npho_rmse': np.sqrt(np.mean(err ** 2)),
            'npho_bias': np.mean(err),
            'npho_68pct': np.percentile(np.abs(err), 68),
        })

    return metrics


def save_predictions(predictions: List[Dict], output_path: str,
                     run_number: Optional[int] = None,
                     predict_channels: List[str] = None,
                     npho_scale: float = DEFAULT_NPHO_SCALE,
                     npho_scale2: float = DEFAULT_NPHO_SCALE2,
                     time_scale: float = DEFAULT_TIME_SCALE,
                     time_shift: float = DEFAULT_TIME_SHIFT,
                     baseline_results: Optional[Dict[str, List[Dict]]] = None):
    """
    Save predictions to ROOT file with metadata.

    Args:
        predictions: List of prediction dictionaries
        output_path: Output ROOT file path
        run_number: Run number for dead channel pattern
        predict_channels: List of predicted channels (['npho'] or ['npho', 'time'])
        npho_scale, npho_scale2, time_scale, time_shift: Normalization parameters
        baseline_results: Optional dict from run_baselines() with keys 'avg' and/or 'sa'
    """
    if not predictions:
        print("[WARNING] No predictions to save")
        return

    if predict_channels is None:
        predict_channels = ['npho', 'time']

    branches = {
        'event_idx': np.array([p['event_idx'] for p in predictions], dtype=np.int32),
        'run_number': np.array([p['run_number'] for p in predictions], dtype=np.int64),
        'event_number': np.array([p['event_number'] for p in predictions], dtype=np.int64),
        'sensor_id': np.array([p['sensor_id'] for p in predictions], dtype=np.int32),
        'face': np.array([FACE_NAME_TO_INT.get(p['face'], -1) for p in predictions], dtype=np.int32),
        'mask_type': np.array([p['mask_type'] for p in predictions], dtype=np.int32),
        'truth_npho': np.array([p['truth_npho'] for p in predictions], dtype=np.float32),
        'truth_time': np.array([p['truth_time'] for p in predictions], dtype=np.float32),
        'pred_npho': np.array([p['pred_npho'] for p in predictions], dtype=np.float32),
        'error_npho': np.array([p['error_npho'] for p in predictions], dtype=np.float32),
    }

    # Only include time predictions if time was predicted
    if 'time' in predict_channels:
        branches['pred_time'] = np.array([p['pred_time'] for p in predictions], dtype=np.float32)
        branches['error_time'] = np.array([p['error_time'] for p in predictions], dtype=np.float32)

    if run_number is not None:
        branches['dead_pattern_run'] = np.full(len(predictions), run_number, dtype=np.int32)

    # Add baseline branches (same length as ML predictions, one entry per masked sensor)
    if baseline_results is not None:
        if 'avg' in baseline_results:
            avg_list = baseline_results['avg']
            branches['baseline_avg_npho'] = np.array(
                [p['pred_npho'] for p in avg_list], dtype=np.float32)
            branches['baseline_avg_error_npho'] = np.array(
                [p['error_npho'] for p in avg_list], dtype=np.float32)
        if 'sa' in baseline_results:
            sa_list = baseline_results['sa']
            branches['baseline_sa_npho'] = np.array(
                [p['pred_npho'] for p in sa_list], dtype=np.float32)
            branches['baseline_sa_error_npho'] = np.array(
                [p['error_npho'] for p in sa_list], dtype=np.float32)

    # Metadata for downstream analysis scripts
    metadata = {
        'predict_channels': np.array([','.join(predict_channels)], dtype='U32'),
        'npho_scale': np.array([npho_scale], dtype=np.float64),
        'npho_scale2': np.array([npho_scale2], dtype=np.float64),
        'time_scale': np.array([time_scale], dtype=np.float64),
        'time_shift': np.array([time_shift], dtype=np.float64),
    }

    with uproot.recreate(output_path) as f:
        f['predictions'] = branches
        f['metadata'] = metadata

    print(f"[INFO] Saved {len(predictions):,} predictions to {output_path}")
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
    parser.add_argument("--n-artificial", type=int, default=50,
                        help="Number of artificial masks per event (real data mode, default: 50)")

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

    # Baselines
    parser.add_argument("--baselines", action="store_true",
                        help="Enable rule-based baseline computation alongside ML")
    parser.add_argument("--solid-angle-branch", type=str, default=None,
                        help="Branch name in ROOT file for solid angles "
                             "(enables solid-angle-weighted baseline)")
    parser.add_argument("--baseline-k", type=int, default=1,
                        help="k-hop parameter for baseline neighbor search (default: 1)")

    args = parser.parse_args()

    # Set number of threads
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
        print(f"[INFO] Using {args.num_threads} CPU threads")
    else:
        print(f"[INFO] Using {torch.get_num_threads()} CPU threads (default)")

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load data
    data = load_data(args.input, tree_name=args.tree_name, max_events=args.max_events)
    n_events = len(data['npho'])

    # Normalize
    print("[INFO] Normalizing data...")
    x_input = normalize_data(data['npho'], data['time'])
    x_original = x_input.copy()

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
        # Real data: dead channels already in data + artificial masking
        print(f"[INFO] Real data mode: adding {args.n_artificial} artificial masks per event")
        artificial_mask, combined_mask = create_artificial_mask(
            x_input, args.n_artificial, dead_mask, seed=args.seed
        )
        # Apply combined mask to input
        x_input[combined_mask] = MODEL_SENTINEL
    else:
        # MC pseudo-experiment: apply dead pattern to clean MC
        print("[INFO] MC mode: applying dead channel pattern")
        artificial_mask = None
        combined_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
        combined_mask[:, dead_mask] = True
        x_input[combined_mask] = MODEL_SENTINEL

    n_masked = combined_mask.sum()
    print(f"[INFO] Total masked sensors: {n_masked:,} ({n_masked/(n_events*N_CHANNELS)*100:.2f}%)")

    # Load model
    model, model_type, predict_channels = load_model(
        checkpoint_path=args.checkpoint,
        torchscript_path=args.torchscript,
        device=args.device
    )

    # Run inference
    print(f"[INFO] Running inference on {args.device}...")
    predictions = run_inference(
        model, model_type, x_input, combined_mask,
        batch_size=args.batch_size, device=args.device,
        predict_channels=predict_channels
    )

    # --- Run baselines (if requested) ---
    baseline_results = None
    baseline_metrics_dict = None
    if args.baselines:
        # Load solid angles if branch is provided
        solid_angles = None
        if args.solid_angle_branch:
            solid_angles = load_solid_angles(
                args.input, args.solid_angle_branch,
                tree_name=args.tree_name, max_events=args.max_events
            )

        baseline_results = run_baselines(
            x_original, combined_mask, artificial_mask, dead_mask,
            baseline_k=args.baseline_k,
            solid_angles=solid_angles,
            run_numbers=data.get('run'),
            event_numbers=data.get('event'),
        )

        # Compute baseline metrics
        baseline_metrics_dict = {}
        for bname, bpreds in baseline_results.items():
            baseline_metrics_dict[bname] = compute_baseline_metrics(bpreds)

    # Collect ML results
    print("[INFO] Collecting predictions...")
    pred_list = collect_predictions(
        predictions, x_original, combined_mask, artificial_mask, dead_mask,
        run_numbers=data.get('run'),
        event_numbers=data.get('event'),
        predict_channels=predict_channels
    )

    # Compute ML metrics
    metrics = compute_metrics(pred_list, predict_channels=predict_channels)

    # Print summary (with optional baseline comparison)
    print_summary(metrics, args.real_data, args.run,
                  baseline_metrics=baseline_metrics_dict)

    # Save outputs
    run_str = str(args.run) if args.run else "custom"
    mode_str = "real" if args.real_data else "mc"

    # Predictions ROOT file (with optional baseline branches)
    pred_file = os.path.join(args.output, f"predictions_{mode_str}_run{run_str}.root")
    save_predictions(pred_list, pred_file, run_number=args.run,
                     predict_channels=predict_channels,
                     baseline_results=baseline_results)

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
