#!/usr/bin/env python
"""
Sanity check script for MAE, Inpainter, and Regressor training pipelines.

Usage:
    python macro/sanity_check.py --data /path/to/test.root
    python macro/sanity_check.py --data /path/to/test.root --pipeline mae
    python macro/sanity_check.py --data /path/to/test.root --pipeline all --device cuda
"""

import argparse
import sys
import torch
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, ".")


def check_metrics(metrics, pipeline_name):
    """Check that metrics are reasonable."""
    issues = []

    # Check total loss
    loss = metrics.get("total_loss", metrics.get("loss", None))
    if loss is None:
        issues.append("No loss found in metrics")
    elif np.isnan(loss):
        issues.append(f"Loss is NaN")
    elif np.isinf(loss):
        issues.append(f"Loss is Inf")
    elif loss == 0:
        issues.append(f"Loss is exactly zero (suspicious)")
    elif loss < 0:
        issues.append(f"Loss is negative: {loss}")

    return issues


def test_mae(data_path, device, num_batches=5):
    """Test MAE training pipeline."""
    print("\n" + "="*60)
    print("Testing MAE Pipeline")
    print("="*60)

    from lib.models.mae import XEC_MAE
    from lib.models.regressor import XECEncoder
    from lib.engines.mae import run_epoch_mae
    from lib.geom_defs import (
        DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
        DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
        DEFAULT_SENTINEL_VALUE, DEFAULT_NPHO_THRESHOLD
    )

    # Create model with correct sentinel value
    print("Creating MAE model...")
    encoder = XECEncoder(outer_mode="finegrid", outer_fine_pool=(2, 2))
    model = XEC_MAE(encoder, mask_ratio=0.15, sentinel_value=DEFAULT_SENTINEL_VALUE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # Run training
    print(f"Running training on {data_path}...")
    try:
        metrics = run_epoch_mae(
            model=model,
            optimizer=optimizer,
            device=device,
            root_files=data_path,
            tree_name="tree",
            batch_size=256,
            step_size=num_batches * 256,  # Limit data
            amp=(device == 'cuda'),
            scaler=scaler,
            # Use correct normalization parameters from geom_defs
            npho_scale=DEFAULT_NPHO_SCALE,
            npho_scale2=DEFAULT_NPHO_SCALE2,
            time_scale=DEFAULT_TIME_SCALE,
            time_shift=DEFAULT_TIME_SHIFT,
            sentinel_value=DEFAULT_SENTINEL_VALUE,
            npho_threshold=DEFAULT_NPHO_THRESHOLD,
            dataloader_workers=0,
            dataset_workers=2,
            track_mae_rmse=True,
            track_train_metrics=True,
            profile=True,  # Enable time profiling
            log_invalid_npho=False,  # Suppress warnings for test
        )
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print metrics
    print("\nMetrics:")
    print(f"  Total Loss: {metrics['total_loss']:.6f}")
    if 'mae_npho' in metrics:
        print(f"  MAE Npho:   {metrics['mae_npho']:.6f}")
    if 'mae_time' in metrics:
        print(f"  MAE Time:   {metrics['mae_time']:.6f}")
    if 'actual_mask_ratio' in metrics:
        print(f"  Mask Ratio: {metrics['actual_mask_ratio']:.4f}")

    # Check metrics
    issues = check_metrics(metrics, "MAE")
    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nMAE: PASSED")
    return True


def test_inpainter(data_path, device, num_batches=5):
    """Test Inpainter training pipeline."""
    print("\n" + "="*60)
    print("Testing Inpainter Pipeline")
    print("="*60)

    from lib.models.inpainter import XEC_Inpainter
    from lib.models.regressor import XECEncoder
    from lib.engines.inpainter import run_epoch_inpainter
    from lib.geom_defs import (
        DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
        DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
        DEFAULT_SENTINEL_VALUE, DEFAULT_NPHO_THRESHOLD
    )

    # Create model with correct sentinel value
    print("Creating Inpainter model...")
    encoder = XECEncoder(outer_mode="finegrid", outer_fine_pool=(2, 2))
    model = XEC_Inpainter(encoder, freeze_encoder=False, sentinel_value=DEFAULT_SENTINEL_VALUE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # Run training
    print(f"Running training on {data_path}...")
    try:
        metrics = run_epoch_inpainter(
            model=model,
            optimizer=optimizer,
            device=device,
            root_files=data_path,
            tree_name="tree",
            batch_size=256,
            step_size=num_batches * 256,
            amp=(device == 'cuda'),
            scaler=scaler,
            # Use correct normalization parameters from geom_defs
            npho_scale=DEFAULT_NPHO_SCALE,
            npho_scale2=DEFAULT_NPHO_SCALE2,
            time_scale=DEFAULT_TIME_SCALE,
            time_shift=DEFAULT_TIME_SHIFT,
            sentinel_value=DEFAULT_SENTINEL_VALUE,
            npho_threshold=DEFAULT_NPHO_THRESHOLD,
            dataloader_workers=0,
            dataset_workers=2,
            dropout_rate=0.1,
            profile=True,  # Enable time profiling
            log_invalid_npho=False,
        )
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print metrics
    print("\nMetrics:")
    print(f"  Total Loss: {metrics['total_loss']:.6f}")
    if 'loss_npho' in metrics:
        print(f"  Npho Loss:  {metrics['loss_npho']:.6f}")
    if 'loss_time' in metrics:
        print(f"  Time Loss:  {metrics['loss_time']:.6f}")

    # Check metrics
    issues = check_metrics(metrics, "Inpainter")
    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nInpainter: PASSED")
    return True


def test_regressor(data_path, device, num_batches=5):
    """Test Regressor training pipeline."""
    print("\n" + "="*60)
    print("Testing Regressor Pipeline")
    print("="*60)

    from lib.models.regressor import XECMultiHeadModel
    from lib.engines.regressor import run_epoch_stream
    from lib.dataset import get_dataloader
    from lib.tasks import TASK_REGISTRY

    # Create model with angle task only (simplest)
    print("Creating Regressor model...")

    # Get task config
    task_config = {"angle": {"enabled": True, "loss_fn": "smooth_l1", "weight": 1.0}}
    active_tasks = {name: TASK_REGISTRY[name] for name in task_config if task_config[name].get("enabled", True)}

    # Task weights for training
    task_weights = {
        "angle": {"loss_fn": "smooth_l1", "weight": 1.0, "loss_beta": 1.0}
    }

    model = XECMultiHeadModel(
        active_tasks=active_tasks,
        task_config=task_config,
        embed_dim=1024,
        encoder_depth=2,
        encoder_heads=8,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=(device == 'cuda'))

    # Create dataloader
    print(f"Creating dataloader for {data_path}...")
    loader = get_dataloader(
        file_path=data_path,
        batch_size=256,
        num_workers=0,
        num_threads=2,
        step_size=num_batches * 256,
        load_truth_branches=True,
        log_invalid_npho=False,
    )

    # Run training
    print("Running training...")
    try:
        metrics, _, _, _, _ = run_epoch_stream(
            model=model,
            optimizer=optimizer,
            loader=loader,
            device=device,
            train=True,
            amp=(device == 'cuda'),
            scaler=scaler,
            task_weights=task_weights,
            grad_clip=1.0,
            profile=True,  # Enable time profiling
        )
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Print metrics
    print("\nMetrics:")
    print(f"  Total Loss: {metrics.get('total_opt', 'N/A')}")
    if 'smooth_l1' in metrics:
        print(f"  Smooth L1:  {metrics['smooth_l1']:.6f}")
    if 'l1' in metrics:
        print(f"  L1 Loss:    {metrics['l1']:.6f}")
    if 'mse' in metrics:
        print(f"  MSE Loss:   {metrics['mse']:.6f}")

    # Check metrics (use total_opt as the main loss)
    check_dict = {"total_loss": metrics.get("total_opt", None)}
    issues = check_metrics(check_dict, "Regressor")
    if issues:
        print("\nISSUES DETECTED:")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("\nRegressor: PASSED")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sanity check for training pipelines")
    parser.add_argument("--data", required=True, help="Path to test ROOT file")
    parser.add_argument("--pipeline", choices=["mae", "inpainter", "regressor", "all"],
                        default="all", help="Which pipeline to test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--num-batches", type=int, default=5,
                        help="Number of batches to process")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Data: {args.data}")
    print(f"Num batches: {args.num_batches}")

    results = {}

    if args.pipeline in ["mae", "all"]:
        results["MAE"] = test_mae(args.data, args.device, args.num_batches)

    if args.pipeline in ["inpainter", "all"]:
        results["Inpainter"] = test_inpainter(args.data, args.device, args.num_batches)

    if args.pipeline in ["regressor", "all"]:
        results["Regressor"] = test_regressor(args.data, args.device, args.num_batches)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll sanity checks passed!")
        return 0
    else:
        print("\nSome sanity checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
