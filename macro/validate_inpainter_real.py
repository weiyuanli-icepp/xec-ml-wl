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
5. Run inpainter inference (CPU)
6. Save predictions with mask_type:
   - 0: Artificially masked (has ground truth)
   - 1: Originally dead (no ground truth)

Usage:
    # Basic usage
    python macro/validate_inpainter_real.py \\
        --checkpoint artifacts/inpainter/checkpoint_best.pth \\
        --input real_data.root \\
        --run 430000 \\
        --output validation_output/

    # With pre-saved dead channel list
    python macro/validate_inpainter_real.py \\
        --checkpoint checkpoint.pth \\
        --input real_data.root \\
        --dead-channel-file dead_channels.txt \\
        --output validation_output/

    # Customize masking
    python macro/validate_inpainter_real.py \\
        --checkpoint checkpoint.pth \\
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
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_FLAT_INDICES, BOT_HEX_FLAT_INDICES,
    DEFAULT_NPHO_SCALE, DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT, DEFAULT_SENTINEL_VALUE
)

# Constants
N_CHANNELS = 4760
REAL_DATA_SENTINEL = 1e10  # Sentinel value used in PrepareRealData.C
MODEL_SENTINEL = DEFAULT_SENTINEL_VALUE  # Sentinel value expected by model (-5.0)

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
        data = {
            'run': tree['run'].array(library='np'),
            'event': tree['event'].array(library='np'),
            'relative_npho': tree['relative_npho'].array(library='np'),
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


def detect_dead_from_data(relative_npho: np.ndarray, relative_time: np.ndarray,
                          sentinel_threshold: float = 1e9) -> np.ndarray:
    """
    Detect dead channels from data by checking for sentinel values.

    Args:
        relative_npho: Normalized npho array (N, 4760)
        relative_time: Normalized time array (N, 4760)
        sentinel_threshold: Threshold to consider as sentinel

    Returns:
        Boolean mask (4760,) indicating consistently dead channels
    """
    # A channel is dead if it has sentinel values in most events
    npho_invalid = relative_npho > sentinel_threshold  # (N, 4760)
    time_invalid = relative_time > sentinel_threshold  # (N, 4760)

    # Channel is dead if invalid in > 90% of events
    dead_fraction = (npho_invalid | time_invalid).mean(axis=0)
    dead_mask = dead_fraction > 0.9

    return dead_mask


def convert_sentinel_values(data: np.ndarray, from_sentinel: float = REAL_DATA_SENTINEL,
                            to_sentinel: float = MODEL_SENTINEL,
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


def load_inpainter_model(checkpoint_path: str, device: str = 'cpu') -> XEC_Inpainter:
    """
    Load inpainter model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Loaded inpainter model
    """
    print(f"[INFO] Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Use defaults
        config = {}

    # Create encoder
    encoder = XECEncoder(
        outer_mode=config.get('outer_mode', 'finegrid'),
        outer_fine_pool=config.get('outer_fine_pool', [3, 3]),
    )

    # Create inpainter
    model = XEC_Inpainter(encoder=encoder)

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

    return model


def prepare_model_input(relative_npho: np.ndarray, relative_time: np.ndarray,
                        npho_scale: float = DEFAULT_NPHO_SCALE,
                        time_scale: float = DEFAULT_TIME_SCALE,
                        time_shift: float = DEFAULT_TIME_SHIFT,
                        sentinel: float = MODEL_SENTINEL) -> np.ndarray:
    """
    Prepare input tensor for the model.

    Converts real data format to model input format:
    - Apply normalization (npho * scale, time * scale + shift)
    - Stack into (N, 4760, 2) tensor

    Args:
        relative_npho: Normalized npho from real data
        relative_time: Normalized time from real data
        npho_scale, time_scale, time_shift: Normalization parameters
        sentinel: Sentinel value for invalid channels

    Returns:
        Input array (N, 4760, 2)
    """
    n_events = len(relative_npho)

    # Convert sentinel values
    npho = convert_sentinel_values(relative_npho, REAL_DATA_SENTINEL, sentinel)
    time = convert_sentinel_values(relative_time, REAL_DATA_SENTINEL, sentinel)

    # Apply normalization (matching training preprocessing)
    # For valid values only
    npho_valid = npho != sentinel
    time_valid = time != sentinel

    npho_normalized = np.where(npho_valid, npho * npho_scale, sentinel)
    time_normalized = np.where(time_valid, time * time_scale + time_shift, sentinel)

    # Stack into (N, 4760, 2)
    x = np.stack([npho_normalized, time_normalized], axis=-1)

    return x.astype(np.float32)


def run_inference(model: XEC_Inpainter, x: np.ndarray,
                  combined_mask: np.ndarray, batch_size: int = 64,
                  device: str = 'cpu') -> Dict[str, np.ndarray]:
    """
    Run inpainter inference on batches.

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

            # Run model
            results = model(x_batch, external_mask=mask_batch)

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


def collect_predictions(predictions: List[Dict], x_original: np.ndarray,
                        artificial_mask: np.ndarray, dead_mask: np.ndarray,
                        data: Dict[str, np.ndarray],
                        npho_scale: float = DEFAULT_NPHO_SCALE,
                        time_scale: float = DEFAULT_TIME_SCALE,
                        time_shift: float = DEFAULT_TIME_SHIFT) -> List[Dict]:
    """
    Collect predictions into a flat list for saving.

    Args:
        predictions: List of batch predictions from run_inference
        x_original: Original input before masking (N, 4760, 2)
        artificial_mask: Boolean mask (N, 4760) of artificially masked
        dead_mask: Boolean mask (4760,) of dead channels
        data: Original data dictionary with metadata
        npho_scale, time_scale, time_shift: Normalization parameters

    Returns:
        List of prediction dictionaries
    """
    all_preds = []

    for batch in predictions:
        start_idx = batch['batch_start']
        end_idx = batch['batch_end']
        results = batch['results']

        for face_name in ['inner', 'us', 'ds', 'outer', 'top', 'bot']:
            if face_name not in results:
                continue

            face_result = results[face_name]
            pred = face_result['pred']  # (B, max_masked, 2)
            valid = face_result['valid']  # (B, max_masked)
            indices = face_result['indices']  # (B, max_masked) or (B, max_masked, 2)

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

                    # Get sensor ID
                    if face_name in ['top', 'bot']:
                        node_idx = int(indices[b, i])
                        hex_indices = FACE_INDEX_MAPS[face_name]
                        sensor_id = int(hex_indices[node_idx])
                    else:
                        h_idx = int(indices[b, i, 0])
                        w_idx = int(indices[b, i, 1])
                        idx_map = FACE_INDEX_MAPS[face_name]
                        sensor_id = int(idx_map[h_idx, w_idx])

                    # Determine mask type
                    is_artificial = artificial_mask[event_idx, sensor_id]
                    is_dead = dead_mask[sensor_id]
                    mask_type = 0 if is_artificial else 1

                    # Get prediction (denormalize)
                    pred_npho_norm = float(pred[b, i, 0])
                    pred_time_norm = float(pred[b, i, 1])

                    # Denormalize prediction
                    pred_npho = pred_npho_norm / npho_scale if npho_scale != 0 else pred_npho_norm
                    pred_time = (pred_time_norm - time_shift) / time_scale if time_scale != 0 else pred_time_norm

                    # Get truth (only valid for artificial mask)
                    if is_artificial:
                        truth_npho_norm = float(x_original[event_idx, sensor_id, 0])
                        truth_time_norm = float(x_original[event_idx, sensor_id, 1])

                        if truth_npho_norm > 1e9 or truth_npho_norm == MODEL_SENTINEL:
                            truth_npho = -999.0
                            error_npho = -999.0
                        else:
                            truth_npho = truth_npho_norm / npho_scale if npho_scale != 0 else truth_npho_norm
                            error_npho = pred_npho - truth_npho

                        if truth_time_norm > 1e9 or truth_time_norm == MODEL_SENTINEL:
                            truth_time = -999.0
                            error_time = -999.0
                        else:
                            truth_time = (truth_time_norm - time_shift) / time_scale if time_scale != 0 else truth_time_norm
                            error_time = pred_time - truth_time
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
                        'truth_time': truth_time,
                        'pred_npho': pred_npho,
                        'pred_time': pred_time,
                        'error_npho': error_npho,
                        'error_time': error_time,
                    }

                    all_preds.append(pred_dict)

    return all_preds


def save_predictions_to_root(predictions: List[Dict], output_path: str,
                             metadata: Dict = None):
    """
    Save predictions to ROOT file.

    Args:
        predictions: List of prediction dictionaries
        output_path: Output ROOT file path
        metadata: Optional metadata to store
    """
    if not predictions:
        print("[WARNING] No predictions to save")
        return

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
        'truth_time': np.array([p['truth_time'] for p in predictions], dtype=np.float32),
        'pred_npho': np.array([p['pred_npho'] for p in predictions], dtype=np.float32),
        'pred_time': np.array([p['pred_time'] for p in predictions], dtype=np.float32),
        'error_npho': np.array([p['error_npho'] for p in predictions], dtype=np.float32),
        'error_time': np.array([p['error_time'] for p in predictions], dtype=np.float32),
    }

    with uproot.recreate(output_path) as f:
        f['predictions'] = branches

    print(f"[INFO] Saved {len(predictions):,} predictions to {output_path}")

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

    # Required arguments
    parser.add_argument("--checkpoint", "-c", required=True,
                        help="Path to inpainter checkpoint")
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
    parser.add_argument("--max-events", type=int, default=None,
                        help="Maximum number of events to process (default: all)")

    # Normalization (should match training)
    parser.add_argument("--npho-scale", type=float, default=DEFAULT_NPHO_SCALE,
                        help=f"Npho normalization scale (default: {DEFAULT_NPHO_SCALE})")
    parser.add_argument("--time-scale", type=float, default=DEFAULT_TIME_SCALE,
                        help=f"Time normalization scale (default: {DEFAULT_TIME_SCALE})")
    parser.add_argument("--time-shift", type=float, default=DEFAULT_TIME_SHIFT,
                        help=f"Time normalization shift (default: {DEFAULT_TIME_SHIFT})")

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
    data_dead_mask = detect_dead_from_data(data['relative_npho'], data['relative_time'])
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
    x_input = prepare_model_input(
        data['relative_npho'], data['relative_time'],
        npho_scale=args.npho_scale,
        time_scale=args.time_scale,
        time_shift=args.time_shift
    )

    # Store original values before masking
    x_original = x_input.copy()

    # Apply combined mask (dead + artificial) to input
    combined_mask = np.zeros((n_events, N_CHANNELS), dtype=bool)
    combined_mask[:, combined_dead_mask] = True  # Dead channels
    combined_mask |= artificial_mask  # Plus artificial

    # Mask the input
    x_input[combined_mask] = MODEL_SENTINEL

    # Load model
    model = load_inpainter_model(args.checkpoint, device='cpu')

    # Run inference
    print("[INFO] Running inference (CPU)...")
    predictions = run_inference(
        model, x_input, combined_mask,
        batch_size=args.batch_size, device='cpu'
    )

    # Collect predictions
    print("[INFO] Collecting predictions...")
    pred_list = collect_predictions(
        predictions, x_original,
        artificial_mask, combined_dead_mask,
        data,
        npho_scale=args.npho_scale,
        time_scale=args.time_scale,
        time_shift=args.time_shift
    )

    # Save to ROOT
    output_file = os.path.join(args.output, "real_data_predictions.root")
    save_predictions_to_root(pred_list, output_file)

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
