# Usage:
# 1. make sure to activate xec-ml-wl conda environment
# 2. python macro/check_transform.py /path/to/data.root
# 3. python macro/check_transform.py /path/to/data.root --npho_threshold 10.0
import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ==========================================
#  CONFIGURATION
# ==========================================
# Time Scaling
# TIME_SCALE = 1e-6
# TIME_SHIFT = 0.
# SENTINEL_VAL = -0.1
# TIME_SCALE = 1e-7
# TIME_SHIFT = 0.
# SENTINEL_VAL = -1.0
TIME_SCALE = 6.5e-8
TIME_SHIFT = 0.5
SENTINEL_VAL = -5.0

# Npho Scaling
NPHO_SCALE  = 0.58
# NPHO_SCALE  = 1
NPHO_SCALE2 = 1.0 # for relative_npho
# NPHO_SCALE2 = 11.54 # for npho

# Branch Names
BRANCH_NPHO = "relative_npho"
BRANCH_TIME = "relative_time"
# BRANCH_NPHO = "npho"

# Npho threshold for meaningful time (raw scale)
# Time is only physically meaningful when npho > threshold
try:
    from lib.geom_defs import DEFAULT_NPHO_THRESHOLD
except ImportError:
    DEFAULT_NPHO_THRESHOLD = 10.0
# ==========================================

def analyze_file(file_path, npho_threshold=None):
    if npho_threshold is None:
        npho_threshold = DEFAULT_NPHO_THRESHOLD

    print(f"--- Analyzing: {file_path} ---")
    print(f"Npho threshold for meaningful time: {npho_threshold}")

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
    
    # 3. Transform NPHO
    clean_npho = np.maximum(raw_npho, 0.0)
    trans_npho = np.log1p(clean_npho / NPHO_SCALE) / NPHO_SCALE2
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
    ax[0].set_title(f"Transformed Npho\nScale={NPHO_SCALE}")
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Check transformed npho/time distributions from ROOT file"
    )
    parser.add_argument("file_path", help="Path to ROOT file")
    parser.add_argument("--npho_threshold", type=float, default=None,
                        help=f"Npho threshold for meaningful time (default: {DEFAULT_NPHO_THRESHOLD})")

    args = parser.parse_args()
    analyze_file(args.file_path, npho_threshold=args.npho_threshold)
