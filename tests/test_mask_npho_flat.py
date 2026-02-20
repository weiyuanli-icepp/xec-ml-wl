"""
Test CDF-based flat masking vs uniform masking.

Loads real MC data from a ROOT file (or generates synthetic data as fallback),
runs both masking modes, and plots histograms of the masked sensors' raw npho.

Usage:
    # Single ROOT file
    python -m tests.test_mask_npho_flat --root data/val.root

    # Glob pattern (multiple files)
    python -m tests.test_mask_npho_flat --root "data/MCGamma_*.root"

    # Directory (loads all *.root files)
    python -m tests.test_mask_npho_flat --root data/

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
    import glob as globmod

    path = os.path.expanduser(root_path)

    # Resolve glob patterns, directories, or single files
    if '*' in path or '?' in path:
        files = sorted(globmod.glob(path))
        if not files:
            raise FileNotFoundError(f"No files matched pattern: {path}")
    elif os.path.isdir(path):
        files = sorted(globmod.glob(os.path.join(path, "*.root")))
        if not files:
            raise FileNotFoundError(f"No ROOT files found in directory: {path}")
    else:
        files = [path]

    print(f"Loading {len(files)} ROOT file(s): {files[0]}" +
          (f" ... {files[-1]}" if len(files) > 1 else ""))

    transform = NphoTransform(scheme=npho_scheme, npho_scale=npho_scale, npho_scale2=npho_scale2)

    raw_npho_all = []
    x_all = []
    n_loaded = 0

    files_input = {f: tree_name for f in files}
    for arrays in uproot.iterate(
        files_input,
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


def random_masking_standalone(x_flat, mask_ratio, sentinel_npho, sentinel_time,
                              mask_npho_flat=False, predict_time=False, debug=False):
    """Standalone random_masking (no model needed).

    Same logic as XEC_Inpainter.random_masking but without needing an encoder.
    """
    B, N, C = x_flat.shape
    device = x_flat.device

    if predict_time:
        already_invalid = (x_flat[:, :, 1] == sentinel_time)
    else:
        already_invalid = (x_flat[:, :, 0] == sentinel_npho)

    valid_count = (~already_invalid).sum(dim=1)  # (B,)
    num_to_mask = (valid_count.float() * mask_ratio).int()  # (B,)

    noise = torch.rand(B, N, device=device)
    noise[already_invalid] = float('inf')

    if mask_npho_flat:
        npho_vals = x_flat[:, :, 0].clone()
        npho_vals[already_invalid] = float('-inf')

        # Transform to log space; clamp zeros to eps
        eps = 0.01
        log_npho = torch.log(npho_vals.clamp(min=eps))
        log_npho[already_invalid] = float('-inf')

        sorted_indices = torch.argsort(log_npho, dim=1)  # ascending
        sorted_log = log_npho.gather(1, sorted_indices)  # (B, N)

        k_max = num_to_mask.max().item()
        k_f = num_to_mask.unsqueeze(1).float().clamp(min=1)  # (B, 1)

        # Log range: [log(eps), log(max) + tiny margin]
        log_lo = torch.tensor(float(torch.log(torch.tensor(eps))),
                              device=device).view(1, 1).expand(B, 1)
        log_hi = sorted_log[:, -1:] + 0.001  # (B, 1)

        # Equal-width bin edges in log space
        edge_idx = torch.arange(k_max + 1, device=device).unsqueeze(0).float()
        edges = log_lo + edge_idx * (log_hi - log_lo) / k_f  # (B, k_max+1)

        # Find sorted-position range for each bin via searchsorted
        edge_pos = torch.searchsorted(sorted_log, edges)  # (B, k_max+1)
        lo = edge_pos[:, :-1]  # (B, k_max)
        hi = edge_pos[:, 1:]   # (B, k_max)
        bin_sz = hi - lo       # (B, k_max)
        non_empty = (bin_sz > 0)

        # Pick one random sensor from each non-empty bin
        safe_sz = bin_sz.clamp(min=1)
        rand_off = (torch.rand(B, k_max, device=device) * safe_sz.float()).long()
        sel_sorted = (lo + rand_off).clamp(max=N - 1)
        sel_orig = sorted_indices.gather(1, sel_sorted)  # (B, k_max)

        # Assign low noise [j, j+1) to selected sensors; rest stay at inf
        bins_f = torch.arange(k_max, device=device).unsqueeze(0).expand(B, -1).float()
        noise = torch.full((B, N), float('inf'), device=device)
        sel_noise = bins_f + torch.rand(B, k_max, device=device)
        sel_noise[~non_empty] = float('inf')  # skip empty bins
        noise.scatter_(1, sel_orig, sel_noise)
        noise[already_invalid] = float('inf')

        if debug:
            n_non_empty = non_empty[0].sum().item()
            print(f"  [DEBUG] Flat-log: {k_max} log bins, {n_non_empty} non-empty "
                  f"(eff. mask ratio: {n_non_empty/N:.3f})")
            print(f"  [DEBUG] log range: [{log_lo[0,0]:.2f}, {log_hi[0,0]:.2f}], "
                  f"bin width: {(log_hi[0,0] - log_lo[0,0]).item()/k_max:.4f}")
            print(f"  [DEBUG] valid_count: min={valid_count.min()}, max={valid_count.max()}")
            print(f"  [DEBUG] num_to_mask: min={num_to_mask.min()}, max={num_to_mask.max()}")

    ids_shuffle = torch.argsort(noise, dim=1)

    position_in_sort = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
    should_mask = (position_in_sort < num_to_mask.unsqueeze(1)).float()
    mask = torch.zeros(B, N, device=device)
    mask.scatter_(1, ids_shuffle, should_mask)

    if debug:
        print(f"  [DEBUG] mask.sum()={mask.sum().item():.0f}, "
              f"per-event: min={mask.sum(1).min():.0f}, max={mask.sum(1).max():.0f}")

        if mask_npho_flat:
            ev0_mask_bool = mask[0].bool()
            ev0_npho_masked = x_flat[0, ev0_mask_bool, 0]
            n_zero = (x_flat[0, :, 0] == 0).sum().item()
            print(f"  [DEBUG] Event 0 masked norm_npho: ==0: {(ev0_npho_masked == 0).sum().item()}, "
                  f">0: {(ev0_npho_masked > 0).sum().item()}, "
                  f"min={ev0_npho_masked.min():.4f}, max={ev0_npho_masked.max():.4f}")
            print(f"  [DEBUG] Event 0: {n_zero} sensors with norm_npho=0 out of {N} "
                  f"({100*n_zero/N:.1f}%), expected masked ==0: ~{int(n_zero/N * num_to_mask[0].item())}")

    return mask


def run_masking(mask_npho_flat, x_flat, mask_ratio, sentinel_npho, sentinel_time):
    """Run random_masking on all data in chunks."""
    chunk_size = 512
    masks = []
    debug_first = True
    for i in range(0, len(x_flat), chunk_size):
        chunk = x_flat[i:i + chunk_size]
        with torch.no_grad():
            mask = random_masking_standalone(
                chunk, mask_ratio, sentinel_npho, sentinel_time,
                mask_npho_flat=mask_npho_flat,
                debug=debug_first,
            )
        masks.append(mask)
        debug_first = False
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

    # Detailed debug: check alignment and values
    raw_masked_flat_all = raw_npho_np[mask_flat.bool().numpy()]
    raw_masked_uniform_all = raw_npho_np[mask_uniform.bool().numpy()]
    print(f"\nMask totals (before >0 filter):")
    print(f"  mask_uniform.sum() = {mask_uniform.sum().item():.0f}")
    print(f"  mask_flat.sum()    = {mask_flat.sum().item():.0f}")
    print(f"  shapes: raw_npho={raw_npho_np.shape}, mask_flat={mask_flat.shape}, x_flat={x_flat.shape}")
    print(f"  raw_masked_uniform (incl <=0): N={len(raw_masked_uniform_all):,}, "
          f"==0: {(raw_masked_uniform_all == 0).sum():,}, <0: {(raw_masked_uniform_all < 0).sum():,}")
    print(f"  raw_masked_flat    (incl <=0): N={len(raw_masked_flat_all):,}, "
          f"==0: {(raw_masked_flat_all == 0).sum():,}, <0: {(raw_masked_flat_all < 0).sum():,}")

    # Check first event in detail
    ev0_mask_flat = mask_flat[0].bool().numpy()
    ev0_mask_uniform = mask_uniform[0].bool().numpy()
    ev0_flat_indices = np.where(ev0_mask_flat)[0]
    ev0_uni_indices = np.where(ev0_mask_uniform)[0]
    print(f"\n  Event 0 debug:")
    print(f"    uniform mask indices (first 10): {ev0_uni_indices[:10]}")
    print(f"    uniform raw_npho at those:       {raw_npho_np[0, ev0_uni_indices[:10]]}")
    print(f"    CDF-flat mask indices (first 10): {ev0_flat_indices[:10]}")
    print(f"    CDF-flat raw_npho at those:       {raw_npho_np[0, ev0_flat_indices[:10]]}")
    print(f"    CDF-flat norm_npho at those:      {x_flat[0, ev0_flat_indices[:10], 0].numpy()}")
    print(f"    raw_npho[0] stats: min={raw_npho_np[0].min():.1f}, "
          f"max={raw_npho_np[0].max():.1f}, ==0: {(raw_npho_np[0]==0).sum()}")
    print(f"    norm_npho[0] stats: min={x_flat[0,:,0].min():.4f}, "
          f"max={x_flat[0,:,0].max():.4f}")

    print(f"\nAll valid sensors:      N={len(raw_all_valid):,}")
    print(f"Masked (uniform):      N={len(raw_masked_uniform):,}")
    print(f"Masked (CDF-flat):     N={len(raw_masked_flat):,}")

    def _print_stats(label, arr):
        if len(arr) == 0:
            print(f"Raw npho stats ({label}):     (empty)")
            return
        print(f"Raw npho stats ({label}):     median={np.median(arr):.0f}, "
              f"mean={np.mean(arr):.0f}, p99={np.percentile(arr, 99):.0f}, "
              f"max={np.max(arr):.0f}")

    _print_stats("all valid", raw_all_valid)
    _print_stats("uniform", raw_masked_uniform)
    _print_stats("CDF-flat", raw_masked_flat)

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
