#!/usr/bin/env python3
"""
Validate Inpainter on Real Data with Dead Channels

This script evaluates the inpainter model on real detector data that already
contains dead channels. It artificially masks additional healthy sensors to
create ground truth for quantitative evaluation.

Workflow:
1. Load real data (from PrepareRealData.C output)
2. Fetch dead channel list from database (or file)
3. Identify dead channels in data (sentinel values)
4. Randomly mask healthy sensors (artificial masking)
5. Run inpainter inference
6. Save predictions with mask_type:
   - 0: Artificially masked (has ground truth)
   - 1: Originally dead (no ground truth)

Usage:
    # Using TorchScript model (recommended - faster)
    python macro/validate_inpainter_real.py \\
        --torchscript inpainter.pt \\
        --input real_data.root \\
        --run 430000 \\
        --output validation_output/

    # Using ONNX model
    python macro/validate_inpainter_real.py \\
        --onnx inpainter.onnx \\
        --input real_data.root \\
        --run 430000 \\
        --output validation_output/

    # Using checkpoint (slower, for debugging)
    python macro/validate_inpainter_real.py \\
        --checkpoint artifacts/inpainter/checkpoint_best.pth \\
        --input real_data.root \\
        --run 430000 \\
        --output validation_output/

    # Customize masking
    python macro/validate_inpainter_real.py \\
        --torchscript inpainter.pt \\
        --input real_data.root \\
        --run 430000 \\
        --n-mask-inner 10 \\
        --n-mask-other 1 \\
        --output validation_output/
"""

import os
import sys
import argparse
import numpy as np
import torch
import uproot
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.models import XECEncoder, XEC_Inpainter
from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    flatten_hex_rows,
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2, DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_TIME
)

# Constants
N_CHANNELS = 4760
REAL_DATA_SENTINEL = 1e10  # Sentinel value used in PrepareRealData.C
MODEL_SENTINEL_TIME = DEFAULT_SENTINEL_TIME  # Sentinel value expected by model (-5.0)
MODEL_SENTINEL_NPHO = -1.0  # Npho sentinel (matches dataset.py default)

# Flatten hex rows to get sensor indices
TOP_HEX_FLAT_INDICES = flatten_hex_rows(TOP_HEX_ROWS)
BOT_HEX_FLAT_INDICES = flatten_hex_rows(BOTTOM_HEX_ROWS)

# Face index maps for masking
FACE_INDEX_MAPS = {
    'inner': INNER_INDEX_MAP,
    'us': US_INDEX_MAP,
    'ds': DS_INDEX_MAP,
    'outer': OUTER_COARSE_FULL_INDEX_MAP,
    'top': TOP_HEX_FLAT_INDICES,
    'bot': BOT_HEX_FLAT_INDICES,
}


def load_real_data(input_path: str, tree_name: str = "tree") -> Dict[str, np.ndarray]:
    """
    Load real data from ROOT file produced by PrepareRealData.C.

    Args:
        input_path: Path to input ROOT file
        tree_name: Name of the tree

    Returns:
        Dictionary with arrays for each branch
    """
    print(f"[INFO] Loading real data from {input_path}")

    with uproot.open(input_path) as f:
        tree = f[tree_name]

        # Required branches
        # Note: Use raw 'npho' (not 'relative_npho') to match training normalization
        data = {
            'run': tree['run'].array(library='np'),
            'event': tree['event'].array(library='np'),
            'npho': tree['npho'].array(library='np'),
            'relative_time': tree['relative_time'].array(library='np'),
        }

        # Optional branches (for metadata)
        optional_branches = [
            'energyReco', 'timeReco',
            'xyzRecoFI', 'uvwRecoFI',
            'xyzRecoLC', 'uvwRecoLC',
            'xyzVTX', 'emiVec', 'emiAng',
            'npho_max_used', 'time_min_used',
        ]

        for branch in optional_branches:
            if branch in tree.keys():
                data[branch] = tree[branch].array(library='np')

    n_events = len(data['run'])
    print(f"[INFO] Loaded {n_events:,} events")

    return data


def get_dead_channels_from_db(run_number: int) -> np.ndarray:
    """
    Fetch dead channel list from MEG2 database.

    Args:
        run_number: Run number to query

    Returns:
        Array of dead channel indices
    """
    from lib.db_utils import get_dead_channels
    return get_dead_channels(run_number)


def get_dead_channels_from_file(file_path: str) -> np.ndarray:
    """
    Load dead channel list from text file.

    Args:
        file_path: Path to dead channel list file

    Returns:
        Array of dead channel indices
    """
    from lib.db_utils import load_dead_channel_list
    return load_dead_channel_list(file_path)


def detect_dead_from_data(npho: np.ndarray, relative_time: np.ndarray,
                          sentinel_threshold: float = 1e9) -> np.ndarray:
    """
    Detect dead channels from data by checking for sentinel values.

    Args:
        npho: Raw photon count array (N, 4760)
        relative_time: Time array (N, 4760)
        sentinel_threshold: Threshold to consider as sentinel

    Returns:
        Boolean mask (4760,) indicating consistently dead channels
    """
    # A channel is dead if it has sentinel values in most events
    npho_invalid = npho > sentinel_threshold  # (N, 4760)
    time_invalid = relative_time > sentinel_threshold  # (N, 4760)

    # Channel is dead if invalid in > 90% of events
    dead_fraction = (npho_invalid | time_invalid).mean(axis=0)
    dead_mask = dead_fraction > 0.9

    return dead_mask


def convert_sentinel_values(data: np.ndarray, from_sentinel: float = REAL_DATA_SENTINEL,
                            to_sentinel: float = MODEL_SENTINEL_TIME,
                            threshold: float = 1e9) -> np.ndarray:
    """
    Convert sentinel values from real data format to model format.

    Args:
        data: Input array
        from_sentinel: Original sentinel value (1e10)
        to_sentinel: Target sentinel value (-5.0)
        threshold: Threshold to detect sentinel

    Returns:
        Array with converted sentinel values
    """
    result = data.copy()
    mask = np.abs(data) > threshold
    result[mask] = to_sentinel
    return result


def get_healthy_sensors_per_face(dead_mask: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get list of healthy sensor indices for each face.

    Args:
        dead_mask: Boolean mask (4760,) indicating dead channels

    Returns:
        Dictionary mapping face name to array of healthy sensor indices
    """
    healthy = {}

    for face_name, index_map in FACE_INDEX_MAPS.items():
        if face_name in ['top', 'bot']:
            # Hex faces: index_map is flat array of sensor indices
            indices = np.array(index_map)
        else:
            # Rectangular faces: index_map is 2D array
            indices = index_map[index_map >= 0].flatten()

        # Filter out dead channels
        healthy_indices = indices[~dead_mask[indices]]
        healthy[face_name] = healthy_indices

    return healthy


def create_artificial_mask(n_events: int, dead_mask: np.ndarray,
                           n_mask_inner: int = 10, n_mask_other: int = 1,
                           seed: int = None) -> np.ndarray:
    """
    Create artificial masking for healthy sensors.

    Args:
        n_events: Number of events
        dead_mask: Boolean mask (4760,) indicating dead channels
        n_mask_inner: Number of healthy sensors to mask in inner face
        n_mask_other: Number of healthy sensors to mask in other faces
        seed: Random seed for reproducibility

    Returns:
        Boolean mask (N, 4760) indicating artificially masked channels
    """
    if seed is not None:
        np.random.seed(seed)

    healthy_per_face = get_healthy_sensors_per_face(dead_mask)

    # Define how many to mask per face
    n_mask_per_face = {
        'inner': n_mask_inner,
        'us': n_mask_other,
        'ds': n_mask_other,
        'outer': n_mask_other,
        'top': n_mask_other,
        'bot': n_mask_other,
    }

    artificial_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)

    for event_idx in range(n_events):
        for face_name, n_mask in n_mask_per_face.items():
            healthy = healthy_per_face[face_name]
            if len(healthy) < n_mask:
                print(f"[WARNING] Event {event_idx}: Only {len(healthy)} healthy sensors "
                      f"in {face_name}, requested {n_mask}")
                n_mask = len(healthy)

            if n_mask > 0:
                selected = np.random.choice(healthy, size=n_mask, replace=False)
                artificial_mask[event_idx, selected] = True

    return artificial_mask


def load_inpainter_model(checkpoint_path: str, device: str = 'cpu') -> Tuple[XEC_Inpainter, List[str]]:
    """
    Load inpainter model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (inpainter model, predict_channels list)
    """
    print(f"[INFO] Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use defaults
        config = {}

    # Get predict_channels from config (default to ["npho", "time"] for legacy)
    predict_channels = config.get('predict_channels', ['npho', 'time'])
    print(f"[INFO] predict_channels: {predict_channels}")

    # Create encoder
    encoder = XECEncoder(
        outer_mode=config.get('outer_mode', 'finegrid'),
        outer_fine_pool=config.get('outer_fine_pool', None),
    )

    # Create inpainter with predict_channels
    model = XEC_Inpainter(encoder=encoder, predict_channels=predict_channels)

    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'ema_state_dict' in checkpoint:
        # Prefer EMA weights if available
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


def load_torchscript_model(model_path: str, device: str = 'cpu'):
    """
    Load TorchScript model.

    Args:
        model_path: Path to .pt file
        device: Device to load model on

    Returns:
        TorchScript model
    """
    print(f"[INFO] Loading TorchScript model from {model_path}")
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print("[INFO] TorchScript model loaded successfully")
    return model


def load_onnx_model(model_path: str, device: str = 'cpu'):
    """
    Load ONNX model with ONNX Runtime.

    Args:
        model_path: Path to .onnx file
        device: Device to run on ('cpu' or 'cuda')

    Returns:
        ONNX Runtime session
    """
    try:
        import onnxruntime as ort
    except ImportError:
        raise RuntimeError("onnxruntime not installed. Install with: pip install onnxruntime")

    print(f"[INFO] Loading ONNX model from {model_path}")

    providers = ['CPUExecutionProvider']
    if device == 'cuda':
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    session = ort.InferenceSession(model_path, providers=providers)
    print(f"[INFO] ONNX model loaded (providers: {session.get_providers()})")
    return session


def prepare_model_input(npho: np.ndarray, relative_time: np.ndarray,
                        npho_scale: float = DEFAULT_NPHO_SCALE,
                        npho_scale2: float = DEFAULT_NPHO_SCALE2,
                        time_scale: float = DEFAULT_TIME_SCALE,
                        time_shift: float = DEFAULT_TIME_SHIFT,
                        sentinel: float = MODEL_SENTINEL_TIME) -> np.ndarray:
    """
    Prepare input tensor for the model.

    Converts real data format to model input format:
    - Apply normalization: npho_norm = log1p(npho / npho_scale) / npho_scale2
                          time_norm = time / time_scale + time_shift
    - Stack into (N, 4760, 2) tensor

    Args:
        npho: Raw photon counts from real data
        relative_time: Time relative to minimum (in seconds)
        npho_scale, npho_scale2: Npho normalization parameters (log1p transform)
        time_scale, time_shift: Time normalization parameters
        sentinel: Sentinel value for invalid channels

    Returns:
        Input array (N, 4760, 2)
    """
    n_events = len(npho)

    # Convert sentinel values
    npho_clean = convert_sentinel_values(npho, REAL_DATA_SENTINEL, sentinel)
    time_clean = convert_sentinel_values(relative_time, REAL_DATA_SENTINEL, sentinel)

    # Apply normalization (matching training preprocessing)
    # For valid values only
    npho_valid = npho_clean != sentinel
    time_valid = time_clean != sentinel

    # Npho: log1p(npho / npho_scale) / npho_scale2
    # Allow negatives through; only clamp domain-breaking values
    npho_sentinel = -1.0  # Separate sentinel for npho channel
    domain_min = -npho_scale * 0.999
    mask_domain_break = npho_valid & (npho_clean < domain_min)
    npho_safe = np.where(mask_domain_break, 0.0, npho_clean)
    npho_normalized = np.where(npho_valid,
                               np.log1p(npho_safe / npho_scale) / npho_scale2,
                               npho_sentinel)
    npho_normalized[mask_domain_break] = 0.0

    # Time: time / time_scale - time_shift (matching dataset.py normalization)
    # Note: The formula in geom_defs.py is: time_norm = (raw_time / TIME_SCALE) - TIME_SHIFT
    # With TIME_SHIFT = -0.46, this becomes: time_norm = raw_time / TIME_SCALE + 0.46
    time_normalized = np.where(time_valid,
                               time_clean / time_scale - time_shift,
                               sentinel)

    # Stack into (N, 4760, 2)
    x = np.stack([npho_normalized, time_normalized], axis=-1)

    return x.astype(np.float32)


def run_inference(model: XEC_Inpainter, x: np.ndarray,
                  combined_mask: np.ndarray, batch_size: int = 64,
                  device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Run inpainter inference on batches (checkpoint mode).

    Args:
        model: Inpainter model
        x: Input tensor (N, 4760, 2)
        combined_mask: Boolean mask (N, 4760) of all masked channels
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        Dictionary with predictions per face
    """
    n_events = len(x)
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for start_idx in tqdm(range(0, n_events, batch_size), desc="Inference"):
            end_idx = min(start_idx + batch_size, n_events)

            x_batch = torch.tensor(x[start_idx:end_idx], device=device)
            mask_batch = torch.tensor(combined_mask[start_idx:end_idx], device=device)

            # Run model - returns (results_dict, original_values, mask)
            results, _, _ = model(x_batch, mask=mask_batch)

            # Store results for this batch
            batch_preds = {
                'batch_start': start_idx,
                'batch_end': end_idx,
                'results': {k: {kk: vv.cpu().numpy() if torch.is_tensor(vv) else vv
                               for kk, vv in v.items()}
                           for k, v in results.items()}
            }
            all_predictions.append(batch_preds)

    return all_predictions


def run_inference_torchscript(model, x: np.ndarray,
                               combined_mask: np.ndarray, batch_size: int = 64,
                               device: str = 'cpu') -> np.ndarray:
    """
    Run inference with TorchScript model.

    Args:
        model: TorchScript model
        x: Input tensor (N, 4760, 2)
        combined_mask: Boolean mask (N, 4760) of all masked channels
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        Output array (N, 4760, 2) with predictions at masked positions
    """
    n_events = len(x)
    output = np.zeros_like(x)

    with torch.no_grad():
        for start_idx in tqdm(range(0, n_events, batch_size), desc="Inference (TorchScript)"):
            end_idx = min(start_idx + batch_size, n_events)

            x_batch = torch.tensor(x[start_idx:end_idx], device=device, dtype=torch.float32)
            mask_batch = torch.tensor(combined_mask[start_idx:end_idx], device=device, dtype=torch.float32)

            # Run model - returns (B, 4760, 2)
            result = model(x_batch, mask_batch)
            output[start_idx:end_idx] = result.cpu().numpy()

    return output


def run_inference_onnx(session, x: np.ndarray,
                       combined_mask: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """
    Run inference with ONNX Runtime.

    Args:
        session: ONNX Runtime session
        x: Input tensor (N, 4760, 2)
        combined_mask: Boolean mask (N, 4760) of all masked channels
        batch_size: Batch size for inference

    Returns:
        Output array (N, 4760, 2) with predictions at masked positions
    """
    n_events = len(x)
    output = np.zeros_like(x)

    for start_idx in tqdm(range(0, n_events, batch_size), desc="Inference (ONNX)"):
        end_idx = min(start_idx + batch_size, n_events)

        x_batch = x[start_idx:end_idx].astype(np.float32)
        mask_batch = combined_mask[start_idx:end_idx].astype(np.float32)

        # Run model
        ort_inputs = {
            "input": x_batch,
            "mask": mask_batch
        }
        result = session.run(None, ort_inputs)[0]
        output[start_idx:end_idx] = result

    return output


def collect_predictions_flat(output: np.ndarray, x_original: np.ndarray,
                              combined_mask: np.ndarray,
                              artificial_mask: np.ndarray, dead_mask: np.ndarray,
                              data: Dict[str, np.ndarray],
                              npho_scale: float = DEFAULT_NPHO_SCALE,
                              npho_scale2: float = DEFAULT_NPHO_SCALE2,
                              time_scale: float = DEFAULT_TIME_SCALE,
                              time_shift: float = DEFAULT_TIME_SHIFT,
                              predict_channels: List[str] = None) -> List[Dict]:
    """
    Collect predictions from flat output tensor (for TorchScript/ONNX).

    Args:
        output: Model output (N, 4760, out_channels) with predictions (in normalized space)
        x_original: Original input before masking (N, 4760, 2) - NOT USED, kept for API compat
        combined_mask: Boolean mask (N, 4760) of all masked channels
        artificial_mask: Boolean mask (N, 4760) of artificially masked
        dead_mask: Boolean mask (4760,) of dead channels
        data: Original data dictionary with raw npho/time and metadata
        npho_scale, npho_scale2: Npho normalization parameters (for denormalizing predictions)
        time_scale, time_shift: Time normalization parameters (for denormalizing predictions)
        predict_channels: List of predicted channels (e.g., ["npho"] or ["npho", "time"])

    Returns:
        List of prediction dictionaries

    Note:
        Truth values are taken DIRECTLY from raw data (data['npho'], data['relative_time']),
        NOT from normalized/denormalized values. This avoids any roundtrip errors.
        Only predictions need denormalization since they come from the model in normalized space.
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']

    predict_time = 'time' in predict_channels

    # Determine output channel indices
    pred_npho_idx = predict_channels.index('npho')
    pred_time_idx = predict_channels.index('time') if predict_time else -1

    all_preds = []
    n_events = len(output)

    # Raw data arrays for truth values (no normalization needed!)
    raw_npho = data['npho']
    raw_time = data['relative_time']

    def denorm_npho(npho_norm):
        """Denormalize npho: inverse of log1p(npho/scale)/scale2"""
        # npho_norm = log1p(npho / npho_scale) / npho_scale2
        # => npho = (expm1(npho_norm * npho_scale2)) * npho_scale
        return (np.expm1(npho_norm * npho_scale2)) * npho_scale

    def denorm_time(time_norm):
        """Denormalize time: inverse of time/scale - shift (matching dataset.py)"""
        # time_norm = time / time_scale - time_shift (from dataset.py)
        # => time = (time_norm + time_shift) * time_scale
        return (time_norm + time_shift) * time_scale

    # Determine face for each sensor
    sensor_to_face = {}
    for face_name, idx_map in FACE_INDEX_MAPS.items():
        if isinstance(idx_map, np.ndarray):
            if idx_map.ndim == 2:
                # Rectangular face
                for sensor_id in idx_map.flatten():
                    if sensor_id >= 0:
                        sensor_to_face[int(sensor_id)] = face_name
            else:
                # Hex face (1D array)
                for sensor_id in idx_map:
                    if sensor_id >= 0:
                        sensor_to_face[int(sensor_id)] = face_name

    for event_idx in tqdm(range(n_events), desc="Collecting predictions"):
        run = int(data['run'][event_idx])
        event = int(data['event'][event_idx])

        # Find masked sensors for this event
        masked_sensors = np.where(combined_mask[event_idx])[0]

        for sensor_id in masked_sensors:
            sensor_id = int(sensor_id)

            # Determine face
            face_name = sensor_to_face.get(sensor_id, 'unknown')

            # Determine mask type
            is_artificial = artificial_mask[event_idx, sensor_id]
            mask_type = 0 if is_artificial else 1

            # Get npho prediction (denormalize from model's normalized output)
            pred_npho_norm = float(output[event_idx, sensor_id, pred_npho_idx])
            pred_npho = denorm_npho(pred_npho_norm)

            # Get time prediction only if model predicts time
            if predict_time:
                pred_time_norm = float(output[event_idx, sensor_id, pred_time_idx])
                pred_time = denorm_time(pred_time_norm)
            else:
                pred_time = -999.0

            # Get truth DIRECTLY from raw data (no normalization roundtrip!)
            if is_artificial:
                truth_npho_raw = float(raw_npho[event_idx, sensor_id])
                truth_time_raw = float(raw_time[event_idx, sensor_id])

                # Check for invalid values (sentinel in raw data is 1e10)
                if truth_npho_raw > 1e9 or truth_npho_raw < 0:
                    truth_npho = -999.0
                    error_npho = -999.0
                else:
                    truth_npho = truth_npho_raw  # Use raw value directly!
                    error_npho = pred_npho - truth_npho

                if predict_time:
                    if truth_time_raw > 1e9:
                        truth_time = -999.0
                        error_time = -999.0
                    else:
                        truth_time = truth_time_raw  # Use raw value directly!
                        error_time = pred_time - truth_time
                else:
                    truth_time = -999.0
                    error_time = -999.0
            else:
                # Dead channel - no truth
                truth_npho = -999.0
                truth_time = -999.0
                error_npho = -999.0
                error_time = -999.0

            pred_dict = {
                'event_idx': event_idx,
                'run': run,
                'event': event,
                'sensor_id': sensor_id,
                'face': face_name,
                'mask_type': mask_type,
                'truth_npho': truth_npho,
                'pred_npho': pred_npho,
                'error_npho': error_npho,
            }

            # Only include time fields if model predicts time
            if predict_time:
                pred_dict['truth_time'] = truth_time
                pred_dict['pred_time'] = pred_time
                pred_dict['error_time'] = error_time

            all_preds.append(pred_dict)

    return all_preds


def collect_predictions(predictions: List[Dict], x_original: np.ndarray,
                        artificial_mask: np.ndarray, dead_mask: np.ndarray,
                        data: Dict[str, np.ndarray],
                        npho_scale: float = DEFAULT_NPHO_SCALE,
                        npho_scale2: float = DEFAULT_NPHO_SCALE2,
                        time_scale: float = DEFAULT_TIME_SCALE,
                        time_shift: float = DEFAULT_TIME_SHIFT,
                        predict_channels: List[str] = None) -> List[Dict]:
    """
    Collect predictions into a flat list for saving.

    Args:
        predictions: List of batch predictions from run_inference
        x_original: Original input before masking (N, 4760, 2) - NOT USED, kept for API compat
        artificial_mask: Boolean mask (N, 4760) of artificially masked
        dead_mask: Boolean mask (4760,) of dead channels
        data: Original data dictionary with raw npho/time and metadata
        npho_scale, npho_scale2: Npho normalization parameters (for denormalizing predictions)
        time_scale, time_shift: Time normalization parameters (for denormalizing predictions)
        predict_channels: List of predicted channels (e.g., ["npho"] or ["npho", "time"])

    Returns:
        List of prediction dictionaries

    Note:
        Truth values are taken DIRECTLY from raw data (data['npho'], data['relative_time']),
        NOT from normalized/denormalized values. This avoids any roundtrip errors.
        Only predictions need denormalization since they come from the model in normalized space.
    """
    if predict_channels is None:
        predict_channels = ['npho', 'time']

    predict_time = 'time' in predict_channels

    # Determine output channel indices
    pred_npho_idx = predict_channels.index('npho')
    pred_time_idx = predict_channels.index('time') if predict_time else -1

    all_preds = []

    # Raw data arrays for truth values (no normalization needed!)
    raw_npho = data['npho']
    raw_time = data['relative_time']

    def denorm_npho(npho_norm):
        """Denormalize npho: inverse of log1p(npho/scale)/scale2"""
        return (np.expm1(npho_norm * npho_scale2)) * npho_scale

    def denorm_time(time_norm):
        """Denormalize time: inverse of time/scale - shift (matching dataset.py)"""
        # time_norm = time / time_scale - time_shift (from dataset.py)
        # => time = (time_norm + time_shift) * time_scale
        return (time_norm + time_shift) * time_scale

    for batch in predictions:
        start_idx = batch['batch_start']
        end_idx = batch['batch_end']
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
                run = int(data['run'][event_idx])
                event = int(data['event'][event_idx])

                n_valid = int(valid[b].sum())
                if n_valid == 0:
                    continue

                for i in range(n_valid):
                    if not valid[b, i]:
                        continue

                    # Get sensor ID - different faces have different index formats
                    if face_name == 'outer' and 'sensor_ids' in face_result:
                        # Outer face with sensor-level prediction
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

                    # Determine mask type
                    is_artificial = artificial_mask[event_idx, sensor_id]
                    is_dead = dead_mask[sensor_id]
                    mask_type = 0 if is_artificial else 1

                    # Get npho prediction (denormalize from model's normalized output)
                    pred_npho_norm = float(pred[b, i, pred_npho_idx])
                    pred_npho = denorm_npho(pred_npho_norm)

                    # Get time prediction only if model predicts time
                    if predict_time:
                        pred_time_norm = float(pred[b, i, pred_time_idx])
                        pred_time = denorm_time(pred_time_norm)
                    else:
                        pred_time = -999.0

                    # Get truth DIRECTLY from raw data (no normalization roundtrip!)
                    if is_artificial:
                        truth_npho_raw = float(raw_npho[event_idx, sensor_id])
                        truth_time_raw = float(raw_time[event_idx, sensor_id])

                        # Check for invalid values (sentinel in raw data is 1e10)
                        if truth_npho_raw > 1e9 or truth_npho_raw < 0:
                            truth_npho = -999.0
                            error_npho = -999.0
                        else:
                            truth_npho = truth_npho_raw  # Use raw value directly!
                            error_npho = pred_npho - truth_npho

                        if predict_time:
                            if truth_time_raw > 1e9:
                                truth_time = -999.0
                                error_time = -999.0
                            else:
                                truth_time = truth_time_raw  # Use raw value directly!
                                error_time = pred_time - truth_time
                        else:
                            truth_time = -999.0
                            error_time = -999.0
                    else:
                        # Dead channel - no truth
                        truth_npho = -999.0
                        truth_time = -999.0
                        error_npho = -999.0
                        error_time = -999.0

                    pred_dict = {
                        'event_idx': event_idx,
                        'run': run,
                        'event': event,
                        'sensor_id': sensor_id,
                        'face': face_name,
                        'mask_type': mask_type,
                        'truth_npho': truth_npho,
                        'pred_npho': pred_npho,
                        'error_npho': error_npho,
                    }

                    # Only include time fields if model predicts time
                    if predict_time:
                        pred_dict['truth_time'] = truth_time
                        pred_dict['pred_time'] = pred_time
                        pred_dict['error_time'] = error_time

                    all_preds.append(pred_dict)

    return all_preds


def save_predictions_to_root(predictions: List[Dict], output_path: str,
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
        predict_channels: List of predicted channels (for metadata)
        npho_scale, npho_scale2, time_scale, time_shift: Normalization parameters (for metadata)
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
        'run': np.array([p['run'] for p in predictions], dtype=np.int32),
        'event': np.array([p['event'] for p in predictions], dtype=np.int32),
        'sensor_id': np.array([p['sensor_id'] for p in predictions], dtype=np.int32),
        'face': np.array([face_map.get(p['face'], -1) for p in predictions], dtype=np.int32),
        'mask_type': np.array([p['mask_type'] for p in predictions], dtype=np.int32),
        'truth_npho': np.array([p['truth_npho'] for p in predictions], dtype=np.float32),
        'pred_npho': np.array([p['pred_npho'] for p in predictions], dtype=np.float32),
        'error_npho': np.array([p['error_npho'] for p in predictions], dtype=np.float32),
    }

    # Only include time branches if model predicts time
    if predict_time:
        branches['truth_time'] = np.array([p.get('truth_time', -999.0) for p in predictions], dtype=np.float32)
        branches['pred_time'] = np.array([p.get('pred_time', -999.0) for p in predictions], dtype=np.float32)
        branches['error_time'] = np.array([p.get('error_time', -999.0) for p in predictions], dtype=np.float32)

    # Metadata tree
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
    print(f"[INFO]   predict_channels: {predict_channels}")

    # Print summary
    mask_types = branches['mask_type']
    n_artificial = (mask_types == 0).sum()
    n_dead = (mask_types == 1).sum()
    print(f"[INFO]   Artificial (mask_type=0): {n_artificial:,}")
    print(f"[INFO]   Dead (mask_type=1): {n_dead:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate inpainter on real data with dead channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Model source (one required)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--checkpoint", "-c",
                             help="Path to inpainter checkpoint (.pth)")
    model_group.add_argument("--torchscript", "-t",
                             help="Path to TorchScript model (.pt) - recommended for speed")
    model_group.add_argument("--onnx",
                             help="Path to ONNX model (.onnx)")

    # Required arguments
    parser.add_argument("--input", "-i", required=True,
                        help="Path to input ROOT file (from PrepareRealData.C)")
    parser.add_argument("--output", "-o", required=True,
                        help="Output directory")

    # Dead channel source (one required)
    dead_group = parser.add_mutually_exclusive_group(required=True)
    dead_group.add_argument("--run", type=int,
                            help="Run number to fetch dead channels from database")
    dead_group.add_argument("--dead-channel-file", type=str,
                            help="Path to dead channel list file")

    # Masking options
    parser.add_argument("--n-mask-inner", type=int, default=10,
                        help="Number of healthy sensors to mask in inner face (default: 10)")
    parser.add_argument("--n-mask-other", type=int, default=1,
                        help="Number of healthy sensors to mask in other faces (default: 1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")

    # Inference options
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for inference (default: 64)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for inference: 'cuda' or 'cpu' (default: cpu)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of events to process (default: all)")

    # Normalization (should match training)
    # Npho: npho_norm = log1p(npho / npho_scale) / npho_scale2
    # Time: time_norm = time / time_scale + time_shift
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE,
                        help=f"Npho normalization scale (default: {DEFAULT_NPHO_SCALE})")
    parser.add_argument("--npho-scale2", type=float, default=DEFAULT_NPHO_SCALE2,
                        help=f"Npho normalization scale2 for log1p (default: {DEFAULT_NPHO_SCALE2})")
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE,
                        help=f"Time normalization scale (default: {DEFAULT_TIME_SCALE})")
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT,
                        help=f"Time normalization shift (default: {DEFAULT_TIME_SHIFT})")
    parser.add_argument("--predict-channels", type=str, nargs='+', default=None,
                        help="Predicted channels, e.g., 'npho' or 'npho time' (default: auto-detect from checkpoint, or 'npho time' for TorchScript/ONNX)")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Load real data
    data = load_real_data(args.input)
    n_events = len(data['run'])

    if args.max_events:
        n_events = min(n_events, args.max_events)
        for key in data:
            data[key] = data[key][:n_events]
        print(f"[INFO] Limited to {n_events:,} events")

    # Get dead channels
    if args.run:
        print(f"[INFO] Fetching dead channels for run {args.run} from database...")
        dead_channels = get_dead_channels_from_db(args.run)
    else:
        print(f"[INFO] Loading dead channels from {args.dead_channel_file}...")
        dead_channels = get_dead_channels_from_file(args.dead_channel_file)

    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    dead_mask[dead_channels] = True
    print(f"[INFO] Dead channels from DB/file: {len(dead_channels)}")

    # Also detect dead from data
    data_dead_mask = detect_dead_from_data(data['npho'], data['relative_time'])
    n_data_dead = data_dead_mask.sum()
    print(f"[INFO] Dead channels detected from data: {n_data_dead}")

    # Combine: use union of DB dead and data-detected dead
    combined_dead_mask = dead_mask | data_dead_mask
    print(f"[INFO] Combined dead channels: {combined_dead_mask.sum()}")

    # Create artificial mask for healthy sensors
    print(f"[INFO] Creating artificial mask: {args.n_mask_inner} inner, {args.n_mask_other} others")
    artificial_mask = create_artificial_mask(
        n_events, combined_dead_mask,
        n_mask_inner=args.n_mask_inner,
        n_mask_other=args.n_mask_other,
        seed=args.seed
    )
    print(f"[INFO] Artificial masks per event: {artificial_mask.sum(axis=1).mean():.1f} avg")

    # Prepare model input
    print("[INFO] Preparing model input...")
    print(f"[INFO] Normalization parameters:")
    print(f"       npho_scale={args.npho_scale}, npho_scale2={args.npho_scale2}")
    print(f"       time_scale={args.time_scale}, time_shift={args.time_shift}")
    print(f"       (IMPORTANT: These must match the values used during training!)")
    x_input = prepare_model_input(
        data['npho'], data['relative_time'],
        npho_scale=args.npho_scale,
        npho_scale2=args.npho_scale2,
        time_scale=args.time_scale,
        time_shift=args.time_shift
    )

    # Diagnostic: Show input data statistics
    npho_valid = data['npho'][data['npho'] < 1e9]
    print(f"[INFO] Input npho statistics (raw, before normalization):")
    print(f"       min={npho_valid.min():.1f}, max={npho_valid.max():.1f}, "
          f"mean={npho_valid.mean():.1f}, median={np.median(npho_valid):.1f}")

    # Store original values before masking
    x_original = x_input.copy()

    # Apply combined mask (dead + artificial) to input
    combined_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
    combined_mask[:, combined_dead_mask] = True  # Dead channels
    combined_mask |= artificial_mask  # Plus artificial

    # Mask the input
    x_input[combined_mask, 0] = MODEL_SENTINEL_NPHO  # npho channel
    x_input[combined_mask, 1] = MODEL_SENTINEL_TIME   # time channel

    # Determine device (default: cpu for compatibility)
    device = args.device if args.device else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    print(f"[INFO] Using device: {device}")

    # Load model and run inference based on model type
    if args.torchscript:
        # TorchScript model (recommended)
        model = load_torchscript_model(args.torchscript, device=device)

        # Use CLI predict_channels or default to ['npho', 'time']
        predict_channels = args.predict_channels if args.predict_channels else ['npho', 'time']
        print(f"[INFO] predict_channels: {predict_channels}")

        print("[INFO] Running inference (TorchScript)...")
        output = run_inference_torchscript(
            model, x_input, combined_mask.astype(np.float32),
            batch_size=args.batch_size, device=device
        )

        # Collect predictions from flat output
        print("[INFO] Collecting predictions...")
        pred_list = collect_predictions_flat(
            output, x_original, combined_mask,
            artificial_mask, combined_dead_mask,
            data,
            npho_scale=args.npho_scale,
            npho_scale2=args.npho_scale2,
            time_scale=args.time_scale,
            time_shift=args.time_shift,
            predict_channels=predict_channels
        )

    elif args.onnx:
        # ONNX model
        session = load_onnx_model(args.onnx, device=device)

        # Use CLI predict_channels or default to ['npho', 'time']
        predict_channels = args.predict_channels if args.predict_channels else ['npho', 'time']
        print(f"[INFO] predict_channels: {predict_channels}")

        print("[INFO] Running inference (ONNX)...")
        output = run_inference_onnx(
            session, x_input, combined_mask.astype(np.float32),
            batch_size=args.batch_size
        )

        # Collect predictions from flat output
        print("[INFO] Collecting predictions...")
        pred_list = collect_predictions_flat(
            output, x_original, combined_mask,
            artificial_mask, combined_dead_mask,
            data,
            npho_scale=args.npho_scale,
            npho_scale2=args.npho_scale2,
            time_scale=args.time_scale,
            time_shift=args.time_shift,
            predict_channels=predict_channels
        )

    else:
        # Checkpoint model (slower, for debugging)
        model, predict_channels = load_inpainter_model(args.checkpoint, device=device)

        # Override with CLI if provided
        if args.predict_channels:
            predict_channels = args.predict_channels
            print(f"[INFO] Overriding predict_channels from CLI: {predict_channels}")

        print("[INFO] Running inference (checkpoint mode - consider using --torchscript for speed)...")
        predictions = run_inference(
            model, x_input, combined_mask,
            batch_size=args.batch_size, device=device
        )

        # Collect predictions from dict output
        print("[INFO] Collecting predictions...")
        pred_list = collect_predictions(
            predictions, x_original,
            artificial_mask, combined_dead_mask,
            data,
            npho_scale=args.npho_scale,
            npho_scale2=args.npho_scale2,
            time_scale=args.time_scale,
            time_shift=args.time_shift,
            predict_channels=predict_channels
        )

    # Diagnostic: Show normalization parameters used for PREDICTIONS ONLY
    # (Truth values come directly from raw data, no normalization applied)
    print(f"\n" + "=" * 60)
    print("NORMALIZATION DIAGNOSTICS")
    print("=" * 60)
    print(f"Parameters used for DENORMALIZING PREDICTIONS:")
    print(f"  npho_scale  = {args.npho_scale}")
    print(f"  npho_scale2 = {args.npho_scale2}")
    print(f"  time_scale  = {args.time_scale}")
    print(f"  time_shift  = {args.time_shift}")
    print(f"\nDenormalization formulas (applied to model output):")
    print(f"  pred_npho = expm1(model_output * {args.npho_scale2}) * {args.npho_scale}")
    print(f"  pred_time = (model_output + ({args.time_shift})) * {args.time_scale}")
    print(f"\nNote: truth_npho and truth_time are taken DIRECTLY from raw input data")
    print(f"      (no normalization roundtrip - avoids any transformation errors)")

    # Check pred vs truth
    artificial_preds = [p for p in pred_list if p['mask_type'] == 0 and p['truth_npho'] > 0]
    if artificial_preds:
        truth_npho_arr = np.array([p['truth_npho'] for p in artificial_preds])
        pred_npho_arr = np.array([p['pred_npho'] for p in artificial_preds])

        pred_vs_truth_ratio = pred_npho_arr / np.maximum(truth_npho_arr, 1e-6)
        print(f"\nPrediction vs Truth comparison (npho):")
        print(f"  truth_npho (raw): mean={truth_npho_arr.mean():.1f}, median={np.median(truth_npho_arr):.1f}")
        print(f"  pred_npho:        mean={pred_npho_arr.mean():.1f}, median={np.median(pred_npho_arr):.1f}")
        print(f"  pred/truth ratio: mean={pred_vs_truth_ratio.mean():.3f}, median={np.median(pred_vs_truth_ratio):.3f}")
        if abs(np.median(pred_vs_truth_ratio) - 1.0) > 0.1:
            print(f"\n  [WARNING] pred_npho systematically differs from truth_npho!")
            print(f"            This likely means the model was trained with DIFFERENT normalization parameters.")
            print(f"            Check what npho_scale/npho_scale2 were used during training.")
        else:
            print(f"  [OK] Predictions are close to truth values")
    print("=" * 60)

    # Save to ROOT
    output_file = os.path.join(args.output, "real_data_predictions.root")
    save_predictions_to_root(
        pred_list, output_file,
        predict_channels=predict_channels,
        npho_scale=args.npho_scale,
        npho_scale2=args.npho_scale2,
        time_scale=args.time_scale,
        time_shift=args.time_shift
    )

    # Save dead channel list
    dead_file = os.path.join(args.output, "dead_channels.txt")
    np.savetxt(dead_file, np.where(combined_dead_mask)[0], fmt='%d',
               header=f"Dead channels (combined DB + data detection)\nTotal: {combined_dead_mask.sum()}")
    print(f"[INFO] Saved dead channel list to {dead_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Events processed: {n_events:,}")
    print(f"Dead channels: {combined_dead_mask.sum()}")
    print(f"Artificial masks/event: {args.n_mask_inner} inner + {5 * args.n_mask_other} others = {args.n_mask_inner + 5 * args.n_mask_other}")
    print(f"Total predictions: {len(pred_list):,}")
    print(f"Output: {output_file}")
    print("=" * 60)

    print("\n[INFO] Done! Run analyze_inpainter.py on the output to compute metrics.")


if __name__ == "__main__":
    main()
