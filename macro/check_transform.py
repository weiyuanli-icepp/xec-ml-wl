# Usage:
# 1. make sure to activate xec-ml-wl conda environment
# 2. python macro/check_transform.py /path/to/data.root
# 3. python macro/check_transform.py /path/to/data.root --npho_threshold 10.0
# 4. python macro/check_transform.py /path/to/data.root --npho_scheme anscombe
# 5. python macro/check_transform.py /path/to/data.root --npho_scheme all  # compare all schemes
import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.normalization import NphoTransform

# ==========================================
#  CONFIGURATION
# ==========================================
# Time Scaling
# TIME_SCALE = 1e-6
# TIME_SHIFT = 0.
# SENTINEL_VAL = -0.1
TIME_SCALE = 1.14e-7
# TIME_SHIFT = 0.
SENTINEL_VAL = -1.0
# TIME_SCALE = 6.5e-8
TIME_SHIFT = -0.46
# SENTINEL_VAL = -0.5

# Npho Scaling
# NPHO_SCALE  = 0.58
# NPHO_SCALE  = 1
# NPHO_SCALE2 = 1.0 # for relative_npho
# NPHO_SCALE2 = 11.54 # for npho

NPHO_SCALE  = 1000
NPHO_SCALE2 = 4.08 
# NPHO_SCALE  = 100
# NPHO_SCALE2 = 6.4


# Branch Names
# BRANCH_NPHO = "relative_npho"
BRANCH_TIME = "relative_time"
BRANCH_NPHO = "npho"

# Npho threshold for meaningful time (raw scale)
# Time is only physically meaningful when npho > threshold
try:
    from lib.geom_defs import DEFAULT_NPHO_THRESHOLD
except ImportError:
    DEFAULT_NPHO_THRESHOLD = 10.0
# ==========================================

def analyze_file(file_path, npho_threshold=None, npho_scheme="log1p"):
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD

    print(f"--- Analyzing: {file_path} ---")
    print(f"Npho threshold for meaningful time: {npho_threshold}")
    print(f"Npho normalization scheme: {npho_scheme}")

    try:
        with uproot.open(file_path) as f:
            tree = f["tree"]
            df = tree.arrays([BRANCH_NPHO, BRANCH_TIME], library="np")
        # Flatten to treat all sensors as a single distribution
        raw_npho = df[BRANCH_NPHO].astype("float32").flatten()
        raw_time = df[BRANCH_TIME].astype("float32").flatten()
    except Exception as e:
        print(f"Error: {e}"); return

    # 1. Define Invalid Mask (Coupled Logic)
    # A pixel is invalid IF:
    #  a) It has no photons (Npho <= 0)  <-- Links Time to Npho
    #  b) Time is NaN
    #  c) Time is an error code (> 9e9)
    mask_inv = (raw_npho <= 0.0) | np.isnan(raw_time) | (np.abs(raw_time) > 9.0e9)

    # 1b. Define time-valid mask (npho > threshold for meaningful time)
    mask_time_valid = (raw_npho > npho_threshold) & ~mask_inv

    n_total = len(raw_time)
    n_inv = np.sum(mask_inv)
    n_time_valid = np.sum(mask_time_valid)
    print(f"Total Sensors: {n_total}")
    print(f"Invalid Sensors (Npho<=0 or ErrorCode): {n_inv} ({n_inv/n_total*100:.1f}%)")
    print(f"Time-valid Sensors (Npho>{npho_threshold}): {n_time_valid} ({n_time_valid/n_total*100:.1f}%)")

    # 2. Transform TIME
    # First, normalize everything
    trans_time = (raw_time / TIME_SCALE) - TIME_SHIFT
    # Then, FORCE invalid pixels to Sentinel
    trans_time[mask_inv] = SENTINEL_VAL
    
    # 3. Transform NPHO using NphoTransform
    clean_npho = np.maximum(raw_npho, 0.0)
    npho_transform = NphoTransform(scheme=npho_scheme, npho_scale=NPHO_SCALE, npho_scale2=NPHO_SCALE2)
    trans_npho = npho_transform.forward(clean_npho)
    # Ensure invalid Npho stays 0.0
    trans_npho[mask_inv] = 0.0

    # 4. Statistics (Signal Only)
    # Signal = anything NOT the sentinel value
    sig_time = trans_time[trans_time != SENTINEL_VAL]
    sig_npho = trans_npho[trans_npho > 0.0]

    # Time-valid signal (only sensors with npho > threshold)
    sig_time_valid = trans_time[mask_time_valid]

    print("\n" + "="*40)
    print("      TRANSFORMED STATISTICS      ")
    print("="*40)

    # Npho Stats
    if len(sig_npho) > 0:
        n_mean, n_std = np.mean(sig_npho), np.std(sig_npho)
        n_min, n_max = np.min(sig_npho), np.max(sig_npho)
        print(f"NPHO (transformed) | Mean: {n_mean:.4f}")
        print(f"NPHO (transformed) | Std:  {n_std:.4f}")
        print(f"NPHO (transformed) | Range: [{n_min:.2f}, {n_max:.2f}]")

    # Time Stats (all valid)
    print("-" * 40)
    if len(sig_time) > 0:
        t_mean, t_std = np.mean(sig_time), np.std(sig_time)
        t_min, t_max = np.min(sig_time), np.max(sig_time)
        print(f"TIME (all valid, npho>0)")
        print(f"  Count: {len(sig_time)}")
        print(f"  Mean:  {t_mean:.4f}  (Goal: ~0.0)")
        print(f"  Std:   {t_std:.4f}  (Goal: ~1.0)")
        print(f"  Range: [{t_min:.2f}, {t_max:.2f}]")
    else:
        print("[WARNING] No valid time signal found!")

    # Time Stats (time-valid only, npho > threshold)
    print("-" * 40)
    if len(sig_time_valid) > 0:
        tv_mean, tv_std = np.mean(sig_time_valid), np.std(sig_time_valid)
        tv_min, tv_max = np.min(sig_time_valid), np.max(sig_time_valid)
        print(f"TIME (time-valid, npho>{npho_threshold})")
        print(f"  Count: {len(sig_time_valid)} ({len(sig_time_valid)/len(sig_time)*100:.1f}% of all valid)")
        print(f"  Mean:  {tv_mean:.4f}  (Goal: ~0.0)")
        print(f"  Std:   {tv_std:.4f}  (Goal: ~1.0)")
        print(f"  Range: [{tv_min:.2f}, {tv_max:.2f}]")
    else:
        print(f"[WARNING] No time-valid sensors found (npho>{npho_threshold})!")

    # 5. Plotting (Side-by-Side)
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    # --- Left: Npho ---
    ax[0].hist(trans_npho, bins=200, color='blue', alpha=0.7, log=True)
    ax[0].set_title(f"Transformed Npho ({npho_scheme})\nScale={NPHO_SCALE}")
    ax[0].set_xlabel("Network Input Value")
    ax[0].set_ylabel("Count (Log Scale)")

    # --- Middle: Time (all) ---
    ax[1].hist(trans_time, bins=200, color='red', alpha=0.7, log=True)
    ax[1].set_title(f"Transformed Time (All)\nSentinel={SENTINEL_VAL}, Shift={TIME_SHIFT}")
    ax[1].set_xlabel("Network Input Value")
    # visual guides
    ax[1].axvline(SENTINEL_VAL, color='k', linestyle='-', linewidth=2, label=f"Invalid ({SENTINEL_VAL})")
    ax[1].axvline(0, color='k', linestyle='--', alpha=0.5, label="Signal Center (0.0)")
    ax[1].legend()

    # --- Right: Time (time-valid only, npho > threshold) ---
    if len(sig_time_valid) > 0:
        ax[2].hist(sig_time_valid, bins=200, color='green', alpha=0.7, log=True)
        ax[2].set_title(f"Transformed Time (npho>{npho_threshold})\n{len(sig_time_valid):,} sensors ({len(sig_time_valid)/n_total*100:.1f}%)")
        ax[2].set_xlabel("Network Input Value")
        ax[2].axvline(0, color='k', linestyle='--', alpha=0.5, label="Signal Center (0.0)")
        ax[2].legend()
    else:
        ax[2].text(0.5, 0.5, f"No time-valid sensors\n(npho>{npho_threshold})",
                   ha='center', va='center', transform=ax[2].transAxes)
        ax[2].set_title(f"Transformed Time (npho>{npho_threshold})")

    plt.tight_layout()
    plt.show()

def compare_all_schemes(file_path, npho_threshold=None):
    """Compare all npho normalization schemes side by side."""
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD

    print(f"--- Comparing all schemes: {file_path} ---")
    print(f"Npho threshold for meaningful time: {npho_threshold}")

    try:
        with uproot.open(file_path) as f:
            tree = f["tree"]
            df = tree.arrays([BRANCH_NPHO, BRANCH_TIME], library="np")
        raw_npho = df[BRANCH_NPHO].astype("float32").flatten()
    except Exception as e:
        print(f"Error: {e}"); return

    # Define invalid mask
    mask_inv = (raw_npho <= 0.0) | np.isnan(raw_npho) | (raw_npho > 9.0e9)
    clean_npho = np.maximum(raw_npho, 0.0)

    # Transform with each scheme
    schemes = ["log1p", "anscombe", "sqrt", "linear"]
    colors = ["blue", "green", "orange", "red"]
    transforms = {}

    for scheme in schemes:
        npho_transform = NphoTransform(scheme=scheme, npho_scale=NPHO_SCALE, npho_scale2=NPHO_SCALE2)
        trans = npho_transform.forward(clean_npho)
        trans[mask_inv] = 0.0
        transforms[scheme] = trans

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (scheme, color) in enumerate(zip(schemes, colors)):
        trans = transforms[scheme]
        sig = trans[trans > 0.0]

        ax = axes[i]
        ax.hist(trans, bins=200, color=color, alpha=0.7, log=True)
        ax.set_title(f"{scheme}\nScale={NPHO_SCALE}")
        ax.set_xlabel("Network Input Value")
        ax.set_ylabel("Count (Log Scale)")

        # Add statistics
        if len(sig) > 0:
            stats_text = f"Mean: {np.mean(sig):.3f}\nStd: {np.std(sig):.3f}\nRange: [{np.min(sig):.2f}, {np.max(sig):.2f}]"
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle(f"Npho Normalization Schemes Comparison\n{os.path.basename(file_path)}", fontsize=12)
    plt.tight_layout()
    plt.show()

    # Print comparison table
    print("\n" + "="*60)
    print("      NPHO NORMALIZATION SCHEMES COMPARISON      ")
    print("="*60)
    print(f"{'Scheme':<12} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-"*60)
    for scheme in schemes:
        trans = transforms[scheme]
        sig = trans[trans > 0.0]
        if len(sig) > 0:
            print(f"{scheme:<12} {np.mean(sig):>10.4f} {np.std(sig):>10.4f} {np.min(sig):>10.4f} {np.max(sig):>10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check transformed npho/time distributions from ROOT file"
    )
    parser.add_argument("file_path", help="Path to ROOT file")
    parser.add_argument("--npho_threshold", type=float, default=None,
                        help=f"Npho threshold for meaningful time (default: {DEFAULT_NPHO_THRESHOLD})")
    parser.add_argument("--npho_scheme", type=str, default="log1p",
                        choices=["log1p", "anscombe", "sqrt", "linear", "all"],
                        help="Npho normalization scheme (default: log1p, use 'all' to compare)")

    args = parser.parse_args()

    if args.npho_scheme == "all":
        compare_all_schemes(args.file_path, npho_threshold=args.npho_threshold)
    else:
        analyze_file(args.file_path, npho_threshold=args.npho_threshold, npho_scheme=args.npho_scheme)
