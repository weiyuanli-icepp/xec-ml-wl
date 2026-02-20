"""
Test CDF-based flat masking vs uniform masking.

Loads real MC data from a ROOT file (or generates synthetic data as fallback),
runs both masking modes, and plots histograms of the masked sensors' raw npho.

Usage:
    # With real MC data
    python -m tests.test_mask_npho_flat --root data/val.root

    # With config file (reads val_path from config)
    python -m tests.test_mask_npho_flat --config config/inp/inpainter_config.yaml

    # Synthetic data (no ROOT file needed)
    python -m tests.test_mask_npho_flat --synthetic

    # Options
    python -m tests.test_mask_npho_flat --root data/val.root --mask_ratio 0.10 --max_events 5000
"""

import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.models.inpainter import XEC_Inpainter
from lib.models.regressor import XECEncoder
from lib.normalization import NphoTransform
from lib.geom_defs import DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2, DEFAULT_SENTINEL_TIME


def load_from_root(root_path, npho_branch="npho", time_branch="relative_time",
                   npho_scale=DEFAULT_NPHO_SCALE, npho_scale2=DEFAULT_NPHO_SCALE2,
                   time_scale=1.14e-7, time_shift=-0.46,
                   sentinel_time=-1.0, sentinel_npho=-1.0,
                   npho_scheme="sqrt", npho_threshold=100,
                   max_events=10000, tree_name="tree"):
    """Load and normalize events from a ROOT file.

    Returns:
        x_flat: (N_events, 4760, 2) normalized tensor
        raw_npho: (N_events, 4760) raw npho array
    """
    import uproot

    path = os.path.expanduser(root_path)
    print(f"Loading ROOT file: {path}")

    transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)

    raw_npho_all = []
    x_all = []
    n_loaded = 0

    for arrays in uproot.iterate(
        f"{path}:{tree_name}",
        [npho_branch, time_branch],
        step_size=50000,
        library="np",
    ):
        raw_n = arrays[npho_branch].astype("float32")
        raw_t = arrays[time_branch].astype("float32")

        # Identify truly invalid sensors (dead/missing)
        mask_invalid = (raw_n > 9e9) | np.isnan(raw_n)

        # Normalize npho
        norm_n = transform.forward(np.clip(raw_n, 0, None))
        norm_n[mask_invalid] = sentinel_npho

        # Normalize time
        norm_t = raw_t / time_scale - time_shift
        norm_t[mask_invalid] = sentinel_time
        # Below-threshold sensors: valid npho, but unreliable time
        below_thresh = (~mask_invalid) & (raw_n < npho_threshold)
        norm_t[below_thresh] = sentinel_time

        x = np.stack([norm_n, norm_t], axis=-1).astype("float32")
        x_all.append(x)
        raw_npho_all.append(raw_n)

        n_loaded += len(raw_n)
        if n_loaded >= max_events:
            break

    x_cat = np.concatenate(x_all, axis=0)[:max_events]
    raw_cat = np.concatenate(raw_npho_all, axis=0)[:max_events]

    print(f"Loaded {len(x_cat):,} events from {root_path}")
    return torch.from_numpy(x_cat), raw_cat


def make_synthetic_batch(batch_size=256, n_sensors=4760,
                         sentinel_npho=-1.0, sentinel_time=-1.0,
                         invalid_frac=0.15, seed=42):
    """Generate a synthetic batch with exponential npho distribution."""
    rng = np.random.RandomState(seed)

    raw_npho = rng.exponential(scale=500, size=(batch_size, n_sensors))
    npho_scale = 1000.0
    norm_npho = np.sqrt(raw_npho) / np.sqrt(npho_scale)

    norm_time = rng.uniform(-0.5, 0.5, size=(batch_size, n_sensors)).astype(np.float32)

    invalid_mask = rng.rand(batch_size, n_sensors) < invalid_frac
    norm_npho[invalid_mask] = sentinel_npho
    norm_time[invalid_mask] = sentinel_time

    x = np.stack([norm_npho, norm_time], axis=-1).astype(np.float32)
    return torch.from_numpy(x), raw_npho.astype(np.float32)


def run_masking(mask_npho_flat, x_flat, mask_ratio, sentinel_npho, sentinel_time):
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

    # Process in chunks to avoid OOM on large batches
    chunk_size = 512
    masks = []
    for i in range(0, len(x_flat), chunk_size):
        chunk = x_flat[i:i + chunk_size]
        with torch.no_grad():
            _, mask = model.random_masking(chunk, mask_ratio=mask_ratio)
        masks.append(mask)
    return torch.cat(masks, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Test CDF-based flat masking")
    parser.add_argument("--root", type=str, default=None, help="Path to ROOT file with MC data")
    parser.add_argument("--config", type=str, default=None, help="Path to inpainter config YAML (reads val_path)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data instead of ROOT")
    parser.add_argument("--max_events", type=int, default=10000, help="Max events to load from ROOT")
    parser.add_argument("--mask_ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", type=str, default="tests/mask_npho_flat_test.pdf")
    args = parser.parse_args()

    sentinel_npho = -1.0
    sentinel_time = -1.0

    # Determine data source
    root_path = args.root
    npho_scheme = "sqrt"
    npho_scale = DEFAULT_NPHO_SCALE
    npho_scale2 = DEFAULT_NPHO_SCALE2
    time_scale = 1.14e-7
    time_shift = -0.46
    npho_branch = "npho"
    time_branch = "relative_time"
    tree_name = "tree"

    if args.config and not args.synthetic:
        from lib.config import load_inpainter_config
        cfg = load_inpainter_config(args.config, warn_missing=False, auto_update=False)
        if not root_path:
            root_path = cfg.data.val_path or cfg.data.train_path
        npho_scheme = cfg.normalization.npho_scheme
        npho_scale = cfg.normalization.npho_scale
        npho_scale2 = cfg.normalization.npho_scale2
        time_scale = cfg.normalization.time_scale
        time_shift = cfg.normalization.time_shift
        sentinel_time = cfg.normalization.sentinel_time
        sentinel_npho = cfg.normalization.sentinel_npho
        npho_branch = cfg.data.npho_branch
        time_branch = cfg.data.time_branch
        tree_name = cfg.data.tree_name

    if root_path and not args.synthetic:
        x_flat, raw_npho = load_from_root(
            root_path,
            npho_branch=npho_branch, time_branch=time_branch,
            npho_scale=npho_scale, npho_scale2=npho_scale2,
            time_scale=time_scale, time_shift=time_shift,
            sentinel_time=sentinel_time, sentinel_npho=sentinel_npho,
            npho_scheme=npho_scheme, max_events=args.max_events,
            tree_name=tree_name,
        )
        data_label = os.path.basename(root_path)
    else:
        print(f"Generating synthetic batch (B={args.max_events}, N=4760, seed={args.seed})...")
        x_flat, raw_npho = make_synthetic_batch(
            batch_size=args.max_events, sentinel_npho=sentinel_npho,
            sentinel_time=sentinel_time, seed=args.seed,
        )
        raw_npho = torch.from_numpy(raw_npho) if isinstance(raw_npho, np.ndarray) else raw_npho
        data_label = "synthetic"

    n_events = len(x_flat)
    print(f"\nRunning uniform masking (mask_ratio={args.mask_ratio})...")
    mask_uniform = run_masking(False, x_flat, args.mask_ratio, sentinel_npho, sentinel_time)

    print(f"Running CDF-based flat masking (mask_ratio={args.mask_ratio})...")
    mask_flat = run_masking(True, x_flat, args.mask_ratio, sentinel_npho, sentinel_time)

    # Collect raw npho of masked sensors
    if isinstance(raw_npho, torch.Tensor):
        raw_npho_np = raw_npho.numpy()
    else:
        raw_npho_np = raw_npho

    valid = (x_flat[:, :, 0] != sentinel_npho).numpy()
    raw_all_valid = raw_npho_np[valid]
    raw_masked_uniform = raw_npho_np[mask_uniform.bool().numpy()]
    raw_masked_flat = raw_npho_np[mask_flat.bool().numpy()]

    # Clip to positive for log binning (raw_npho should be >=0 for valid sensors)
    raw_all_valid = raw_all_valid[raw_all_valid > 0]
    raw_masked_uniform = raw_masked_uniform[raw_masked_uniform > 0]
    raw_masked_flat = raw_masked_flat[raw_masked_flat > 0]

    print(f"\nAll valid sensors:      N={len(raw_all_valid):,}")
    print(f"Masked (uniform):      N={len(raw_masked_uniform):,}")
    print(f"Masked (CDF-flat):     N={len(raw_masked_flat):,}")
    print(f"\nRaw npho stats (all valid):   median={np.median(raw_all_valid):.0f}, "
          f"mean={np.mean(raw_all_valid):.0f}, p99={np.percentile(raw_all_valid, 99):.0f}, "
          f"max={np.max(raw_all_valid):.0f}")
    print(f"Raw npho stats (uniform):     median={np.median(raw_masked_uniform):.0f}, "
          f"mean={np.mean(raw_masked_uniform):.0f}, p99={np.percentile(raw_masked_uniform, 99):.0f}, "
          f"max={np.max(raw_masked_uniform):.0f}")
    print(f"Raw npho stats (CDF-flat):    median={np.median(raw_masked_flat):.0f}, "
          f"mean={np.mean(raw_masked_flat):.0f}, p99={np.percentile(raw_masked_flat, 99):.0f}, "
          f"max={np.max(raw_masked_flat):.0f}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Use log-spaced bins
    log_max = np.log10(max(raw_all_valid.max(), 10))
    bins = np.logspace(0, log_max, 50)

    # Row 0: histograms (counts)
    axes[0, 0].hist(raw_all_valid, bins=bins, alpha=0.7, color="gray")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("Raw npho")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title("All valid sensors")

    axes[0, 1].hist(raw_masked_uniform, bins=bins, alpha=0.7, color="C0")
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Raw npho")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].set_title(f"Uniform masking ({args.mask_ratio*100:.0f}%)")

    axes[0, 2].hist(raw_masked_flat, bins=bins, alpha=0.7, color="C1")
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_xlabel("Raw npho")
    axes[0, 2].set_ylabel("Count")
    axes[0, 2].set_title(f"CDF-flat masking ({args.mask_ratio*100:.0f}%)")

    # Row 1: overlay comparison + ratio
    # Left: overlay both masks on same axes
    axes[1, 0].hist(raw_masked_uniform, bins=bins, alpha=0.5, color="C0", label="Uniform")
    axes[1, 0].hist(raw_masked_flat, bins=bins, alpha=0.5, color="C1", label="CDF-flat")
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("Raw npho")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Overlay comparison")
    axes[1, 0].legend()

    # Center: normalized (density) overlay
    axes[1, 1].hist(raw_masked_uniform, bins=bins, alpha=0.5, color="C0",
                    density=True, label="Uniform")
    axes[1, 1].hist(raw_masked_flat, bins=bins, alpha=0.5, color="C1",
                    density=True, label="CDF-flat")
    axes[1, 1].set_xscale("log")
    axes[1, 1].set_xlabel("Raw npho")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Normalized overlay")
    axes[1, 1].legend()

    # Right: CDF-flat / uniform ratio per bin
    h_uniform, _ = np.histogram(raw_masked_uniform, bins=bins)
    h_flat, _ = np.histogram(raw_masked_flat, bins=bins)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])  # geometric mean
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(h_uniform > 0, h_flat / h_uniform, np.nan)
    axes[1, 2].plot(bin_centers, ratio, "o-", color="C2", markersize=3)
    axes[1, 2].axhline(1.0, color="gray", ls="--", alpha=0.5)
    axes[1, 2].set_xscale("log")
    axes[1, 2].set_xlabel("Raw npho")
    axes[1, 2].set_ylabel("CDF-flat / Uniform")
    axes[1, 2].set_title("Ratio (flat / uniform)")
    axes[1, 2].set_ylim(0, max(np.nanmax(ratio) * 1.2, 3) if np.any(np.isfinite(ratio)) else 3)

    fig.suptitle(f"Masked npho distribution — {data_label}\n"
                 f"({n_events:,} events, mask_ratio={args.mask_ratio}, scheme={npho_scheme})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"\nPlot saved to {args.output}")


if __name__ == "__main__":
    main()
