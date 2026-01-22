# Usage:
# 1. make sure to activate xec-ml-wl conda environment
# 2. python macro/check_transform.py /path/to/data.root
import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

# ==========================================
#  CONFIGURATION
# ==========================================
# Time Scaling
# TIME_SCALE = 1e-6  
# TIME_SHIFT = 0.
# SENTINEL_VAL = -0.1
TIME_SCALE = 1e-7
TIME_SHIFT = 0.
SENTINEL_VAL = -1.0

# Npho Scaling
NPHO_SCALE  = 0.58
# NPHO_SCALE  = 1
# NPHO_SCALE2 = 1.0 # for relative_npho
NPHO_SCALE2 = 11.54 # for npho

# Branch Names
# BRANCH_NPHO = "relative_npho"
BRANCH_TIME = "relative_time"
BRANCH_NPHO = "npho"
# ==========================================

def analyze_file(file_path):
    print(f"--- Analyzing: {file_path} ---")
    
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
    
    n_total = len(raw_time)
    n_inv = np.sum(mask_inv)
    print(f"Total Sensors: {n_total}")
    print(f"Invalid Sensors (Npho<=0 or ErrorCode): {n_inv} ({n_inv/n_total*100:.1f}%)")

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

    print("\n" + "="*40)
    print("      TRANSFORMED STATISTICS      ")
    print("="*40)
    
    # Time Stats
    if len(sig_time) > 0:
        t_mean, t_std = np.mean(sig_time), np.std(sig_time)
        t_min, t_max = np.min(sig_time), np.max(sig_time)
        print(f"TIME (transformed) | Mean: {t_mean:.4f}  (Goal: ~0.0)")
        print(f"TIME (transformed) | Std:  {t_std:.4f}  (Goal: ~1.0)")
        print(f"TIME (transformed) | Range: [{t_min:.2f}, {t_max:.2f}]")
    else:
        print("[WARNING] No valid time signal found!")

    # Npho Stats
    if len(sig_npho) > 0:
        n_mean, n_std = np.mean(sig_npho), np.std(sig_npho)
        n_min, n_max = np.min(sig_npho), np.max(sig_npho)
        print(f"NPHO (transformed) | Mean: {n_mean:.4f}")
        print(f"NPHO (transformed) | Std:  {n_std:.4f}")
        print(f"NPHO (transformed) | Range: [{n_min:.2f}, {n_max:.2f}]")

    # 5. Plotting (Side-by-Side)
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- Left: Npho ---
    ax[0].hist(trans_npho, bins=200, color='blue', alpha=0.7, log=True)
    ax[0].set_title(f"Transformed Npho\nScale={NPHO_SCALE}")
    ax[0].set_xlabel("Network Input Value")
    ax[0].set_ylabel("Count (Log Scale)")
    # No grid
    
    # --- Right: Time ---
    ax[1].hist(trans_time, bins=200, color='red', alpha=0.7, log=True)
    ax[1].set_title(f"Transformed Time\nSentinel={SENTINEL_VAL}, Shift={TIME_SHIFT}")
    ax[1].set_xlabel("Network Input Value")
    # No grid
    
    # visual guides
    ax[1].axvline(SENTINEL_VAL, color='k', linestyle='-', linewidth=2, label=f"Invalid ({SENTINEL_VAL})")
    ax[1].axvline(0, color='k', linestyle='--', alpha=0.5, label="Signal Center (0.0)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1: analyze_file(sys.argv[1])
    else: print("Usage: python check_transform.py <file.root>")
