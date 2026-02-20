"""
Test CDF-based flat masking vs uniform masking.

Generates synthetic npho data with a realistic exponential distribution,
runs both masking modes, and plots histograms of the masked sensors' npho.

Usage:
    python -m tests.test_mask_npho_flat
    python -m tests.test_mask_npho_flat --mask_ratio 0.10
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from lib.models.inpainter import XEC_Inpainter
from lib.models.regressor import XECEncoder


def make_synthetic_batch(batch_size: int = 256, n_sensors: int = 4760,
                         sentinel_npho: float = -1.0, sentinel_time: float = -1.0,
                         invalid_frac: float = 0.15, seed: int = 42):
    """Generate a synthetic batch with exponential npho distribution.

    Raw npho ~ Exponential(scale=500), so most sensors have low npho
    and a small tail extends to 10^4+. Normalization uses sqrt scheme:
    norm_npho = sqrt(raw) / sqrt(1000).
    """
    rng = np.random.RandomState(seed)

    raw_npho = rng.exponential(scale=500, size=(batch_size, n_sensors))
    # sqrt normalization: sqrt(x) / sqrt(npho_scale)
    npho_scale = 1000.0
    norm_npho = np.sqrt(raw_npho) / np.sqrt(npho_scale)

    # Time channel: simple uniform (not important for this test)
    norm_time = rng.uniform(-0.5, 0.5, size=(batch_size, n_sensors)).astype(np.float32)

    # Mark some sensors as invalid
    invalid_mask = rng.rand(batch_size, n_sensors) < invalid_frac
    norm_npho[invalid_mask] = sentinel_npho
    norm_time[invalid_mask] = sentinel_time

    x = np.stack([norm_npho, norm_time], axis=-1).astype(np.float32)
    return torch.from_numpy(x), torch.from_numpy(raw_npho.astype(np.float32))


def run_masking(mask_npho_flat: bool, x_flat: torch.Tensor,
                mask_ratio: float, sentinel_npho: float, sentinel_time: float):
    """Run random_masking with a minimal XEC_Inpainter."""
    encoder = XECEncoder(outer_mode="finegrid", encoder_dim=1024, num_fusion_layers=1)
    model = XEC_Inpainter(
        encoder,
        freeze_encoder=True,
        sentinel_time=sentinel_time,
        sentinel_npho=sentinel_npho,
        predict_channels=["npho"],
        mask_npho_flat=mask_npho_flat,
    )
    model.eval()
    with torch.no_grad():
        _, mask = model.random_masking(x_flat, mask_ratio=mask_ratio)
    return mask


def main():
    parser = argparse.ArgumentParser(description="Test CDF-based flat masking")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--mask_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default="tests/mask_npho_flat_test.pdf")
    args = parser.parse_args()

    sentinel_npho = -1.0
    sentinel_time = -1.0
    npho_scale = 1000.0

    print(f"Generating synthetic batch (B={args.batch_size}, N=4760, seed={args.seed})...")
    x_flat, raw_npho = make_synthetic_batch(
        batch_size=args.batch_size, sentinel_npho=sentinel_npho,
        sentinel_time=sentinel_time, seed=args.seed,
    )

    print(f"Running uniform masking (mask_ratio={args.mask_ratio})...")
    mask_uniform = run_masking(False, x_flat, args.mask_ratio, sentinel_npho, sentinel_time)

    print(f"Running CDF-based flat masking (mask_ratio={args.mask_ratio})...")
    mask_flat = run_masking(True, x_flat, args.mask_ratio, sentinel_npho, sentinel_time)

    # Collect raw npho of masked sensors
    valid = (x_flat[:, :, 0] != sentinel_npho)
    raw_all_valid = raw_npho[valid].numpy()

    raw_masked_uniform = raw_npho[mask_uniform.bool()].numpy()
    raw_masked_flat = raw_npho[mask_flat.bool()].numpy()

    print(f"\nAll valid sensors:      N={len(raw_all_valid):,}")
    print(f"Masked (uniform):      N={len(raw_masked_uniform):,}")
    print(f"Masked (CDF-flat):     N={len(raw_masked_flat):,}")
    print(f"\nRaw npho stats (all valid):   median={np.median(raw_all_valid):.0f}, "
          f"mean={np.mean(raw_all_valid):.0f}, max={np.max(raw_all_valid):.0f}")
    print(f"Raw npho stats (uniform):     median={np.median(raw_masked_uniform):.0f}, "
          f"mean={np.mean(raw_masked_uniform):.0f}, max={np.max(raw_masked_uniform):.0f}")
    print(f"Raw npho stats (CDF-flat):    median={np.median(raw_masked_flat):.0f}, "
          f"mean={np.mean(raw_masked_flat):.0f}, max={np.max(raw_masked_flat):.0f}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Use log-spaced bins for the exponential distribution
    bins = np.logspace(0, np.log10(max(raw_all_valid.max(), 1)), 50)

    axes[0].hist(raw_all_valid, bins=bins, alpha=0.7, color="gray", label="All valid")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Raw npho")
    axes[0].set_ylabel("Count")
    axes[0].set_title("All valid sensors")
    axes[0].legend()

    axes[1].hist(raw_masked_uniform, bins=bins, alpha=0.7, color="C0", label="Uniform mask")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Raw npho")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Uniform masking ({args.mask_ratio*100:.0f}%)")
    axes[1].legend()

    axes[2].hist(raw_masked_flat, bins=bins, alpha=0.7, color="C1", label="CDF-flat mask")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Raw npho")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"CDF-based flat masking ({args.mask_ratio*100:.0f}%)")
    axes[2].legend()

    fig.suptitle(f"Masked sensor npho distribution (B={args.batch_size}, mask_ratio={args.mask_ratio})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
