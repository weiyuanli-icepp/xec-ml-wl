import uproot
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import argparse

def analyze_events(file_path, start_event, num_events):
    print(f"--- Analyzing: {file_path} ---")
    print(f"--- Accumulating {num_events} events starting from index {start_event} ---")
    
    # 1. Load Data
    try:
        with uproot.open(file_path) as f:
            tree = f["tree"]
            df = tree.arrays(["relative_npho", "relative_time"], library="np")
            
        all_npho = df["relative_npho"].astype("float32")
        all_time = df["relative_time"].astype("float32")
        
        total_events = len(all_npho)
        end_event = start_event + num_events
        
        if start_event >= total_events:
            print(f"[ERROR] Start index {start_event} is out of bounds (Total: {total_events}).")
            return
        
        if end_event > total_events:
            print(f"[WARN] Requested range goes beyond file end. Clipping to {total_events}.")
            end_event = total_events

        # Slice and Flatten: Combined data from N events
        # Shape changes from (N, 4760) -> (N * 4760,)
        raw_npho = all_npho[start_event:end_event].flatten()
        raw_time = all_time[start_event:end_event].flatten()
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # 2. Preprocess (Same logic as engine)
    mask_inv = np.isnan(raw_time) | (raw_npho <= 0.0)
    
    # Check for outliers
    n_outliers = np.sum(np.abs(raw_time) > 9.0e9)
    mask_inv |= (np.abs(raw_time) > 9.0e9)
    
    # --- Handle Time ---
    proc_time = raw_time.copy()
    proc_time[mask_inv] = 0.0
    
    # --- Handle Npho ---
    proc_npho = raw_npho.copy()
    proc_npho = np.maximum(proc_npho, 0.0) 
    proc_npho = np.log1p(proc_npho)

    # 3. Statistics (Signal Only)
    non_zero_time = proc_time[proc_time != 0]
    
    print("\n--- STATISTICS (Signal Sensors Only) ---")
    print(f"Events Accumulating: {end_event - start_event}")
    print(f"Total Sensors Checked: {len(proc_time)}")
    print(f"Valid Signal Hits: {len(non_zero_time)}")
    
    if len(non_zero_time) > 0:
        mean_val = np.mean(non_zero_time)
        std_val = np.std(non_zero_time)
        median_val = np.median(non_zero_time)
        
        print(f"TIME (raw) | Mean: {mean_val:.4e} | Std: {std_val:.4e} | Median: {median_val:.4e}")
        print(f"\n--- SUGGESTED SCALING (Based on these {num_events} events) ---")
        print(f"Suggested TIME_SCALE: {std_val:.1e}")
    else:
        print("[WARNING] These events are empty!")

    # 4. Show Histograms
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Npho Plot
    ax[0].hist(proc_npho, bins=100, color='blue', alpha=0.7)
    ax[0].set_title(f"Events {start_event}-{end_event-1}: Npho (log1p)")
    ax[0].set_xlabel("log(1 + relative_npho)")
    ax[0].set_ylabel("Count")
    # No grid

    # Time Plot
    if len(non_zero_time) > 0:
        t_min, t_max = np.min(non_zero_time), np.max(non_zero_time)
        padding = (t_max - t_min) * 0.2 if t_max != t_min else 1.0
        ax[1].set_xlim(t_min - padding, t_max + padding)
    
    ax[1].hist(proc_time, bins=100, color='red', alpha=0.7)
    ax[1].set_title(f"Events {start_event}-{end_event-1}: Time")
    ax[1].set_xlabel("Time Value")
    # No grid
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze accumulated events from a ROOT file.")
    parser.add_argument("file_path", type=str, help="Path to ROOT file")
    parser.add_argument("--event", type=int, default=0, help="Start event index (default: 0)")
    parser.add_argument("--N", type=int, default=1, help="Number of events to accumulate (default: 1)")
    
    args = parser.parse_args()
    analyze_events(args.file_path, args.event, args.N)