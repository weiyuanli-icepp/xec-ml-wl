#!/usr/bin/env python3
"""
Pseudo-Experiment: Apply Real Data Dead Channels to MC

This script applies the dead channel pattern from a real data run to MC data,
allowing evaluation of inpainter performance on dead channel positions where
we have ground truth (unlike real data).

Workflow:
1. Load MC data (same format as training data)
2. Fetch dead channel pattern from database (for a specific run)
3. Apply dead channel mask to MC (mark as sentinel)
4. Run inpainter inference
5. Compare predictions with ground truth

This provides a baseline to understand how well the inpainter can recover
dead channels, using MC data where ground truth is available for all sensors.

Usage:
    # Basic usage
    python macro/pseudo_experiment_mc.py \\
        --checkpoint artifacts/inpainter/checkpoint_best.pth \\
        --input mc_data.root \\
        --run 430000 \\
        --output pseudo_experiment/

    # Compare multiple runs
    for run in 430000 431000 432000; do
        python macro/pseudo_experiment_mc.py \\
            --checkpoint checkpoint.pth \\
            --input mc_data.root \\
            --run $run \\
            --output pseudo_experiment_run${run}/
    done
"""

import os
import sys
import argparse
import numpy as np
import torch
import uproot
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models import XECEncoder, XEC_Inpainter
from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    flatten_hex_rows,
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2, DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME,
    DEFAULT_NPHO_THRESHOLD,
)

# Constants
N_CHANNELS = 4760
MODEL_SENTINEL_TIME = DEFAULT_SENTINEL_TIME  # -1.0
MODEL_SENTINEL_NPHO = -1.0  # Npho sentinel (matches dataset.py default)

# Flatten hex rows to get sensor indices
TOP_HEX_FLAT_INDICES = flatten_hex_rows(TOP_HEX_ROWS)
BOT_HEX_FLAT_INDICES = flatten_hex_rows(BOTTOM_HEX_ROWS)

# Face index maps
FACE_INDEX_MAPS = {
    'inner': INNER_INDEX_MAP,
    'us': US_INDEX_MAP,
    'ds': DS_INDEX_MAP,
    'outer': OUTER_COARSE_FULL_INDEX_MAP,
    'top': TOP_HEX_FLAT_INDICES,
    'bot': BOT_HEX_FLAT_INDICES,
}


def load_mc_data(input_path: str, tree_name: str = "tree",
                 max_events: int = None) -> Dict[str, np.ndarray]:
    """
    Load MC data from ROOT file (same format as training data).

    Args:
        input_path: Path to input ROOT file
        tree_name: Name of the tree
        max_events: Maximum number of events to load

    Returns:
        Dictionary with arrays for each branch
    """
    print(f"[INFO] Loading MC data from {input_path}")

    with uproot.open(input_path) as f:
        tree = f[tree_name]

        # Required branches (same as training)
        data = {
            'relative_npho': tree['relative_npho'].array(library='np'),
            'relative_time': tree['relative_time'].array(library='np'),
        }

        # Optional branches
        optional_branches = [
            'run', 'event', 'energy', 'timing',
            'xyzVTX', 'emiVec', 'angleVec',
        ]

        for branch in optional_branches:
            if branch in tree.keys():
                data[branch] = tree[branch].array(library='np')

    n_events = len(data['relative_npho'])

    if max_events and max_events < n_events:
        for key in data:
            data[key] = data[key][:max_events]
        n_events = max_events

    print(f"[INFO] Loaded {n_events:,} events")

    return data


def get_dead_channels(run_number: int = None,
                      dead_channel_file: str = None) -> np.ndarray:
    """
    Get dead channel list from database or file.

    Args:
        run_number: Run number to query (mutually exclusive with file)
        dead_channel_file: Path to dead channel list file

    Returns:
        Array of dead channel indices
    """
    if run_number is not None:
        from lib.db_utils import get_dead_channels as db_get_dead_channels
        return db_get_dead_channels(run_number)
    elif dead_channel_file is not None:
        from lib.db_utils import load_dead_channel_list
        return load_dead_channel_list(dead_channel_file)
    else:
        raise ValueError("Either run_number or dead_channel_file must be specified")


def prepare_model_input(relative_npho: np.ndarray, relative_time: np.ndarray,
                        npho_scale: float = DEFAULT_NPHO_SCALE,
                        npho_scale2: float = DEFAULT_NPHO_SCALE2,
                        time_scale: float = DEFAULT_TIME_SCALE,
                        time_shift: float = DEFAULT_TIME_SHIFT,
                        sentinel_time: float = MODEL_SENTINEL_TIME,
                        sentinel_npho: float = MODEL_SENTINEL_NPHO,
                        npho_threshold: float = DEFAULT_NPHO_THRESHOLD) -> np.ndarray:
    """
    Prepare input tensor for the model.

    Applies the same normalization as the training pipeline (dataset.py):
    - npho: log1p(raw / npho_scale) / npho_scale2
    - time: raw / time_scale - time_shift

    Args:
        relative_npho: Raw npho values (N, 4760)
        relative_time: Raw time values (N, 4760)
        npho_scale, npho_scale2: Npho normalization parameters
        time_scale, time_shift: Time normalization parameters
        sentinel_time: Sentinel for invalid time channels
        sentinel_npho: Sentinel for invalid npho channels
        npho_threshold: Minimum npho for reliable timing

    Returns:
        Input array (N, 4760, 2)
    """
    # True invalids: dead/missing sensors, corrupted data
    mask_npho_invalid = (relative_npho > 9e9) | np.isnan(relative_npho)
    # Domain-breaking values for log1p
    domain_min = -npho_scale * 0.999
    mask_domain_break = (~mask_npho_invalid) & (relative_npho < domain_min)
    mask_time_invalid = (mask_npho_invalid | (relative_npho < npho_threshold) |
                         (np.abs(relative_time) > 9e9) | np.isnan(relative_time))

    # Normalize npho: log1p transform (allow negatives through)
    npho_safe = np.where(mask_npho_invalid | mask_domain_break, 0.0, relative_npho)
    npho_norm = np.log1p(npho_safe / npho_scale) / npho_scale2
    npho_norm[mask_npho_invalid] = sentinel_npho
    npho_norm[mask_domain_break] = 0.0

    # Normalize time: linear transform
    time_norm = (relative_time / time_scale) - time_shift
    time_norm[mask_time_invalid] = sentinel_time

    # Stack into (N, 4760, 2)
    x = np.stack([npho_norm, time_norm], axis=-1)

    return x.astype(np.float32)


def load_inpainter_model(checkpoint_path: str, device: str = 'cpu'):
    """
    Load inpainter model from checkpoint.

    Returns:
        Tuple of (model, predict_channels) where predict_channels is a list
        like ['npho'] or ['npho', 'time'].
    """
    print(f"[INFO] Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = checkpoint.get('config', {})

    # Get predict_channels from config (default to both for legacy checkpoints)
    predict_channels = config.get('predict_channels', ['npho', 'time'])
    print(f"[INFO] Predict channels: {predict_channels}")

    # Create encoder
    encoder = XECEncoder(
        outer_mode=config.get('outer_mode', 'finegrid'),
        outer_fine_pool=config.get('outer_fine_pool', None),
        encoder_dim=config.get('encoder_dim', 1024),
        dim_feedforward=config.get('dim_feedforward', None),
        num_fusion_layers=config.get('num_fusion_layers', 2),
        sentinel_time=config.get('sentinel_time', -1.0),
    )

    # Create inpainter with predict_channels
    head_type = config.get('head_type', 'per_face')
    model = XEC_Inpainter(
        encoder=encoder,
        predict_channels=predict_channels,
        use_masked_attention=config.get('use_masked_attention', False),
        head_type=head_type,
        sensor_positions_file=config.get('sensor_positions_file', None),
        cross_attn_k=config.get('cross_attn_k', 16),
        cross_attn_hidden=config.get('cross_attn_hidden', 64),
        cross_attn_latent_dim=config.get('cross_attn_latent_dim', 128),
        cross_attn_pos_dim=config.get('cross_attn_pos_dim', 96),
        sentinel_npho=config.get('sentinel_npho', -1.0),
    )

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_state_dict' in checkpoint:
        state_dict = checkpoint['ema_state_dict']
        print("[INFO] Using EMA weights")
    else:
        state_dict = checkpoint

    # Handle 'module.' prefix from DataParallel
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    print(f"[INFO] Model loaded successfully")
    return model, predict_channels


def run_inference(model: XEC_Inpainter, x: np.ndarray,
                  mask: np.ndarray, batch_size: int = 64,
                  device: str = 'cpu') -> List[Dict]:
    """
    Run inpainter inference on batches.

    Args:
        model: Inpainter model
        x: Input tensor (N, 4760, 2)
        mask: Boolean mask (N, 4760) of masked channels
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        List of batch prediction dictionaries
    """
    n_events = len(x)
    all_predictions = []

    head_type = getattr(model, 'head_type', 'per_face')

    model.eval()
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_events, batch_size), desc="Inference"):
            end_idx = min(start_idx + batch_size, n_events)

            x_batch = torch.tensor(x[start_idx:end_idx], device=device)
            mask_batch = torch.tensor(mask[start_idx:end_idx], device=device)

            if head_type == "cross_attention":
                # Cross-attention: use forward_full_output for flat (B, 4760, C) output
                pred_all = model.forward_full_output(x_batch, mask_batch)
                batch_preds = {
                    'batch_start': start_idx,
                    'batch_end': end_idx,
                    'pred_all': pred_all.cpu().numpy(),
                    'mask': mask_batch.cpu().numpy(),
                    'is_flat': True,
                }
            else:
                # Per-face: use forward() for results dict
                results, _, _ = model(x_batch, mask=mask_batch)
                batch_preds = {
                    'batch_start': start_idx,
                    'batch_end': end_idx,
                    'results': {k: {kk: vv.cpu().numpy() if torch.is_tensor(vv) else vv
                                   for kk, vv in v.items()}
                               for k, v in results.items()},
                    'is_flat': False,
                }
            all_predictions.append(batch_preds)

    return all_predictions


def collect_predictions(predictions: List[Dict], x_original: np.ndarray,
                        dead_mask: np.ndarray,
                        npho_scale: float = DEFAULT_NPHO_SCALE,
                        time_scale: float = DEFAULT_TIME_SCALE,
                        time_shift: float = DEFAULT_TIME_SHIFT,
                        predict_channels: List[str] = None) -> List[Dict]:
    """
    Collect predictions into a flat list for saving.

    Args:
        predictions: List of batch predictions from run_inference
        x_original: Original input before masking (N, 4760, 2)
        dead_mask: Boolean mask (4760,) of dead channels
        npho_scale, time_scale, time_shift: Normalization parameters
        predict_channels: List of predicted channels (['npho'] or ['npho', 'time'])

    Returns:
        List of prediction dictionaries
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']
    predict_time = 'time' in predict_channels
    # Map prediction channel names to indices in the prediction tensor
    pred_npho_idx = predict_channels.index('npho') if 'npho' in predict_channels else 0
    pred_time_idx = predict_channels.index('time') if 'time' in predict_channels else None

    all_preds = []

    # Build sensor_id → face_name mapping for flat predictions
    _face_id_to_name = None

    for batch in predictions:
        start_idx = batch['batch_start']
        end_idx = batch['batch_end']

        if batch.get('is_flat', False):
            # Cross-attention: flat (B, 4760, C) predictions
            pred_all = batch['pred_all']
            mask_np = batch['mask']
            B = pred_all.shape[0]

            # Lazily build face mapping
            if _face_id_to_name is None:
                from lib.sensor_geometry import build_sensor_face_ids
                _sensor_face_ids = build_sensor_face_ids()
                _face_id_to_name_map = {0: "inner", 1: "us", 2: "ds",
                                        3: "outer", 4: "top", 5: "bot"}

            for b in range(B):
                event_idx = start_idx + b
                masked_sensors = np.where(mask_np[b] > 0.5)[0]

                for sensor_id in masked_sensors:
                    sensor_id = int(sensor_id)
                    face_name = _face_id_to_name_map[int(_sensor_face_ids[sensor_id])]

                    pred_npho_norm = float(pred_all[b, sensor_id, pred_npho_idx])
                    pred_npho = pred_npho_norm / npho_scale if npho_scale != 0 else pred_npho_norm

                    if predict_time and pred_time_idx is not None and pred_all.shape[2] > pred_time_idx:
                        pred_time_norm = float(pred_all[b, sensor_id, pred_time_idx])
                        pred_time = (pred_time_norm - time_shift) / time_scale if time_scale != 0 else pred_time_norm
                    else:
                        pred_time = -999.0

                    truth_npho_norm = float(x_original[event_idx, sensor_id, 0])
                    truth_time_norm = float(x_original[event_idx, sensor_id, 1])

                    if truth_npho_norm == MODEL_SENTINEL_NPHO or abs(truth_npho_norm) > 1e9:
                        truth_npho = -999.0
                        error_npho = -999.0
                    else:
                        truth_npho = truth_npho_norm / npho_scale if npho_scale != 0 else truth_npho_norm
                        error_npho = pred_npho - truth_npho

                    if not predict_time:
                        truth_time = -999.0
                        error_time = -999.0
                    elif truth_time_norm == MODEL_SENTINEL_TIME or abs(truth_time_norm) > 1e9:
                        truth_time = -999.0
                        error_time = -999.0
                    else:
                        truth_time = (truth_time_norm - time_shift) / time_scale if time_scale != 0 else truth_time_norm
                        error_time = pred_time - truth_time

                    all_preds.append({
                        'event_idx': event_idx,
                        'sensor_id': sensor_id,
                        'face': face_name,
                        'truth_npho': truth_npho,
                        'truth_time': truth_time,
                        'pred_npho': pred_npho,
                        'pred_time': pred_time,
                        'error_npho': error_npho,
                        'error_time': error_time,
                    })
            continue

        # Per-face: results dict format
        results = batch['results']

        for face_name in ['inner', 'us', 'ds', 'outer', 'top', 'bot']:
            if face_name not in results:
                continue

            face_result = results[face_name]
            pred = face_result['pred']  # (B, max_masked, out_channels)
            valid = face_result['valid']  # (B, max_masked)

            B = pred.shape[0]

            for b in range(B):
                event_idx = start_idx + b

                n_valid = int(valid[b].sum())
                if n_valid == 0:
                    continue

                for i in range(n_valid):
                    if not valid[b, i]:
                        continue

                    # Get sensor ID - different faces have different index formats
                    if face_name == 'outer' and 'sensor_ids' in face_result:
                        # Outer face with sensor-level prediction (finegrid/split mode)
                        sensor_id = int(face_result['sensor_ids'][b, i])
                    elif face_name in ['top', 'bot']:
                        # Hex faces - indices are node indices
                        node_idx = int(face_result['indices'][b, i])
                        hex_indices = FACE_INDEX_MAPS[face_name]
                        sensor_id = int(hex_indices[node_idx])
                    else:
                        # Rectangular faces - indices are (h, w)
                        indices = face_result['indices']
                        h_idx = int(indices[b, i, 0])
                        w_idx = int(indices[b, i, 1])
                        idx_map = FACE_INDEX_MAPS[face_name]
                        sensor_id = int(idx_map[h_idx, w_idx])

                    # Get npho prediction (denormalize)
                    pred_npho_norm = float(pred[b, i, pred_npho_idx])
                    pred_npho = pred_npho_norm / npho_scale if npho_scale != 0 else pred_npho_norm

                    # Get time prediction only if predicting time
                    if predict_time and pred_time_idx is not None:
                        pred_time_norm = float(pred[b, i, pred_time_idx])
                        pred_time = (pred_time_norm - time_shift) / time_scale if time_scale != 0 else pred_time_norm
                    else:
                        pred_time = -999.0  # Sentinel for not predicted

                    # Get truth from original MC data
                    truth_npho_norm = float(x_original[event_idx, sensor_id, 0])
                    truth_time_norm = float(x_original[event_idx, sensor_id, 1])

                    # Check if truth is valid
                    if truth_npho_norm == MODEL_SENTINEL_NPHO or abs(truth_npho_norm) > 1e9:
                        truth_npho = -999.0
                        error_npho = -999.0
                    else:
                        truth_npho = truth_npho_norm / npho_scale if npho_scale != 0 else truth_npho_norm
                        error_npho = pred_npho - truth_npho

                    if not predict_time:
                        truth_time = -999.0
                        error_time = -999.0
                    elif truth_time_norm == MODEL_SENTINEL_TIME or abs(truth_time_norm) > 1e9:
                        truth_time = -999.0
                        error_time = -999.0
                    else:
                        truth_time = (truth_time_norm - time_shift) / time_scale if time_scale != 0 else truth_time_norm
                        error_time = pred_time - truth_time

                    pred_dict = {
                        'event_idx': event_idx,
                        'sensor_id': sensor_id,
                        'face': face_name,
                        'truth_npho': truth_npho,
                        'truth_time': truth_time,
                        'pred_npho': pred_npho,
                        'pred_time': pred_time,
                        'error_npho': error_npho,
                        'error_time': error_time,
                    }

                    all_preds.append(pred_dict)

    return all_preds


def save_predictions_to_root(predictions: List[Dict], output_path: str,
                             run_number: int = None,
                             predict_channels: List[str] = None,
                             npho_scale: float = DEFAULT_NPHO_SCALE,
                             npho_scale2: float = DEFAULT_NPHO_SCALE2,
                             time_scale: float = DEFAULT_TIME_SCALE,
                             time_shift: float = DEFAULT_TIME_SHIFT):
    """
    Save predictions to ROOT file with metadata.

    Args:
        predictions: List of prediction dictionaries
        output_path: Output ROOT file path
        run_number: Run number for dead channel pattern
        predict_channels: List of predicted channels (['npho'] or ['npho', 'time'])
        npho_scale, npho_scale2, time_scale, time_shift: Normalization parameters
    """
    if not predictions:
        print("[WARNING] No predictions to save")
        return

    if predict_channels is None:
        predict_channels = ['npho', 'time']

    predict_time = 'time' in predict_channels

    # Convert to arrays
    face_map = {"inner": 0, "us": 1, "ds": 2, "outer": 3, "top": 4, "bot": 5}

    branches = {
        'event_idx': np.array([p['event_idx'] for p in predictions], dtype=np.int32),
        'sensor_id': np.array([p['sensor_id'] for p in predictions], dtype=np.int32),
        'face': np.array([face_map.get(p['face'], -1) for p in predictions], dtype=np.int32),
        'truth_npho': np.array([p['truth_npho'] for p in predictions], dtype=np.float32),
        'truth_time': np.array([p['truth_time'] for p in predictions], dtype=np.float32),
        'pred_npho': np.array([p['pred_npho'] for p in predictions], dtype=np.float32),
        'error_npho': np.array([p['error_npho'] for p in predictions], dtype=np.float32),
    }

    # Only include time predictions if time was predicted
    if predict_time:
        branches['pred_time'] = np.array([p['pred_time'] for p in predictions], dtype=np.float32)
        branches['error_time'] = np.array([p['error_time'] for p in predictions], dtype=np.float32)

    # Add run number as metadata (same for all entries)
    if run_number is not None:
        branches['dead_pattern_run'] = np.full(len(predictions), run_number, dtype=np.int32)

    # Metadata for downstream analysis scripts
    metadata = {
        'predict_channels': np.array([','.join(predict_channels)], dtype='U32'),
        'npho_scale': np.array([npho_scale], dtype=np.float64),
        'npho_scale2': np.array([npho_scale2], dtype=np.float64),
        'time_scale': np.array([time_scale], dtype=np.float64),
        'time_shift': np.array([time_shift], dtype=np.float64),
    }

    with uproot.recreate(output_path) as f:
        f.mktree('predictions', branches)
        f.mktree('metadata', metadata)

    print(f"[INFO] Saved {len(predictions):,} predictions to {output_path}")
    print(f"[INFO] Metadata: predict_channels={predict_channels}")


def compute_metrics(predictions: List[Dict], predict_time: bool = True) -> Dict:
    """
    Compute metrics from predictions.

    Args:
        predictions: List of prediction dictionaries
        predict_time: Whether time channel was predicted

    Returns:
        Dictionary of metrics
    """
    # Filter valid predictions (where we have ground truth for npho)
    # For time, only require validity if predicting time
    valid_preds = [p for p in predictions if p['error_npho'] > -999]

    if not valid_preds:
        return {}

    error_npho = np.array([p['error_npho'] for p in valid_preds])

    metrics = {
        'n_predictions': len(valid_preds),
        'npho_mae': np.mean(np.abs(error_npho)),
        'npho_rmse': np.sqrt(np.mean(error_npho ** 2)),
        'npho_bias': np.mean(error_npho),
        'npho_res_68pct': np.percentile(np.abs(error_npho), 68),
    }

    # Time metrics only if predicting time
    if predict_time:
        # Filter to predictions with valid time error
        time_valid_preds = [p for p in valid_preds if p['error_time'] > -999]
        if time_valid_preds:
            error_time = np.array([p['error_time'] for p in time_valid_preds])
            metrics.update({
                'time_mae': np.mean(np.abs(error_time)),
                'time_rmse': np.sqrt(np.mean(error_time ** 2)),
                'time_bias': np.mean(error_time),
                'time_res_68pct': np.percentile(np.abs(error_time), 68),
            })

    # Per-face metrics
    for face_name in ['inner', 'us', 'ds', 'outer', 'top', 'bot']:
        face_preds = [p for p in valid_preds if p['face'] == face_name]
        if face_preds:
            face_err_npho = np.array([p['error_npho'] for p in face_preds])
            metrics[f'{face_name}_n'] = len(face_preds)
            metrics[f'{face_name}_npho_mae'] = np.mean(np.abs(face_err_npho))
            if predict_time:
                face_time_valid = [p for p in face_preds if p['error_time'] > -999]
                if face_time_valid:
                    face_err_time = np.array([p['error_time'] for p in face_time_valid])
                    metrics[f'{face_name}_time_mae'] = np.mean(np.abs(face_err_time))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Apply real data dead channels to MC for pseudo-experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Required arguments
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to inpainter checkpoint")
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input MC ROOT file")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    # Dead channel source (one required)
    dead_group = parser.add_mutually_exclusive_group(required=True)
    dead_group.add_argument("--run", type=int,
                            help="Run number to fetch dead channels from database")
    dead_group.add_argument("--dead-channel-file", type=str,
                            help="Path to dead channel list file")

    # Options
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of events to process (default: all)")
    parser.add_argument("--tree-name", type=str, default="tree",
                        help="Name of tree in ROOT file (default: tree)")

    # Normalization
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE,
                        help=f"Npho normalization scale (default: {DEFAULT_NPHO_SCALE})")
    parser.add_argument("--npho-scale2", type=float, default=DEFAULT_NPHO_SCALE2,
                        help=f"Npho normalization scale2 (default: {DEFAULT_NPHO_SCALE2})")
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE,
                        help=f"Time normalization scale (default: {DEFAULT_TIME_SCALE})")
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT,
                        help=f"Time normalization shift (default: {DEFAULT_TIME_SHIFT})")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load MC data
    data = load_mc_data(args.input, tree_name=args.tree_name, max_events=args.max_events)
    n_events = len(data['relative_npho'])

    # Get dead channels
    run_number = args.run
    if run_number:
        print(f"[INFO] Fetching dead channels for run {run_number} from database...")
    else:
        print(f"[INFO] Loading dead channels from {args.dead_channel_file}...")
        run_number = None

    dead_channels = get_dead_channels(run_number=args.run,
                                       dead_channel_file=args.dead_channel_file)

    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    dead_mask[dead_channels] = True
    n_dead = len(dead_channels)
    print(f"[INFO] Dead channels: {n_dead} ({n_dead/N_CHANNELS*100:.2f}%)")

    # Prepare model input (before masking)
    print("[INFO] Preparing model input...")
    x_input = prepare_model_input(
        data['relative_npho'], data['relative_time'],
        npho_scale=args.npho_scale,
        npho_scale2=args.npho_scale2,
        time_scale=args.time_scale,
        time_shift=args.time_shift,
    )

    # Store original for ground truth
    x_original = x_input.copy()

    # Create mask array (same dead pattern for all events)
    mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
    mask[:, dead_mask] = True

    # Apply dead channel mask to input (set each channel to its own sentinel)
    x_input[mask, 0] = MODEL_SENTINEL_NPHO  # npho channel
    x_input[mask, 1] = MODEL_SENTINEL_TIME   # time channel

    # Count how many valid sensors are being masked (have ground truth)
    n_valid_masked = 0
    for i in range(n_events):
        for sensor_id in np.where(dead_mask)[0]:
            if x_original[i, sensor_id, 0] != MODEL_SENTINEL_NPHO:
                n_valid_masked += 1
    print(f"[INFO] Total masked sensors with ground truth: {n_valid_masked:,}")

    # Load model
    model, predict_channels = load_inpainter_model(args.checkpoint, device='cpu')
    predict_time = 'time' in predict_channels

    # Run inference
    print("[INFO] Running inference (CPU)...")
    predictions = run_inference(
        model, x_input, mask,
        batch_size=args.batch_size, device='cpu'
    )

    # Collect predictions
    print("[INFO] Collecting predictions...")
    pred_list = collect_predictions(
        predictions, x_original, dead_mask,
        npho_scale=args.npho_scale,
        time_scale=args.time_scale,
        time_shift=args.time_shift,
        predict_channels=predict_channels
    )

    # Compute metrics
    metrics = compute_metrics(pred_list, predict_time=predict_time)

    # Print metrics
    print("\n" + "=" * 60)
    print("PSEUDO-EXPERIMENT RESULTS")
    print("=" * 60)
    if args.run:
        print(f"Dead channel pattern from run: {args.run}")
    else:
        print(f"Dead channel pattern from file: {args.dead_channel_file}")
    print(f"Number of dead channels: {n_dead}")
    print(f"Events processed: {n_events:,}")
    print(f"Total predictions: {len(pred_list):,}")
    print(f"Predictions with ground truth: {metrics.get('n_predictions', 0):,}")
    print(f"Predict channels: {predict_channels}")
    print()
    print("Global Metrics (all dead channel predictions):")
    print(f"  npho: MAE={metrics.get('npho_mae', np.nan):.4f}, "
          f"RMSE={metrics.get('npho_rmse', np.nan):.4f}, "
          f"Bias={metrics.get('npho_bias', np.nan):.4f}, "
          f"68%={metrics.get('npho_res_68pct', np.nan):.4f}")
    if predict_time:
        print(f"  time: MAE={metrics.get('time_mae', np.nan):.4f}, "
              f"RMSE={metrics.get('time_rmse', np.nan):.4f}, "
              f"Bias={metrics.get('time_bias', np.nan):.4f}, "
              f"68%={metrics.get('time_res_68pct', np.nan):.4f}")
    else:
        print("  time: (not predicted)")
    print()
    print("Per-Face Metrics:")
    for face_name in ['inner', 'us', 'ds', 'outer', 'top', 'bot']:
        n = metrics.get(f'{face_name}_n', 0)
        if n > 0:
            if predict_time:
                print(f"  {face_name:>6}: n={n:5d}, "
                      f"npho_MAE={metrics.get(f'{face_name}_npho_mae', np.nan):.4f}, "
                      f"time_MAE={metrics.get(f'{face_name}_time_mae', np.nan):.4f}")
            else:
                print(f"  {face_name:>6}: n={n:5d}, "
                      f"npho_MAE={metrics.get(f'{face_name}_npho_mae', np.nan):.4f}")
    print("=" * 60)

    # Save predictions to ROOT
    run_for_filename = args.run if args.run else "custom"
    output_file = os.path.join(args.output, f"pseudo_experiment_run{run_for_filename}.root")
    save_predictions_to_root(pred_list, output_file, run_number=args.run,
                             predict_channels=predict_channels,
                             npho_scale=args.npho_scale,
                             time_scale=args.time_scale,
                             time_shift=args.time_shift)

    # Save metrics to CSV
    import csv
    metrics_file = os.path.join(args.output, f"metrics_run{run_for_filename}.csv")
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        for k, v in sorted(metrics.items()):
            writer.writerow([k, v])
    print(f"[INFO] Saved metrics to {metrics_file}")

    # Save dead channel list
    dead_file = os.path.join(args.output, f"dead_channels_run{run_for_filename}.txt")
    np.savetxt(dead_file, dead_channels, fmt='%d',
               header=f"Dead channels from run {args.run if args.run else 'file'}\nTotal: {n_dead}")
    print(f"[INFO] Saved dead channel list to {dead_file}")

    print("\n[INFO] Done! Use analyze_inpainter.py on the output ROOT file for detailed analysis.")


if __name__ == "__main__":
    main()
