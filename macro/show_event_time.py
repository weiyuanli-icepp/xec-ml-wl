import sys
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import torch

# Import the plotter from your updated geometry file
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from angle_model_geom import plot_event_time
except ImportError:
    print("Error: Could not import 'angle_model_geom.py'.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize LXe Event Time Distribution")
    parser.add_argument("event_id", type=int, help="The event index to visualize")
    parser.add_argument("--file", type=str, 
                        default="/data/user/ext-li_w1/meghome/xec-ml-wl/data/MCGammaAngle_0-49.root")
    parser.add_argument("--tree", type=str, default="tree")
    parser.add_argument("--npho_branch", type=str, default="relative_npho", help="Branch for photon counts (used for masking)")
    parser.add_argument("--time_branch", type=str, default="relative_time", help="Branch for timing (used for color)")
    parser.add_argument("--save", type=str, default=None, help="Path to save output PDF/PNG")

    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        sys.exit(1)

    print(f"Opening: {args.file}")
    with uproot.open(args.file) as f:
        tree = f[args.tree]
        
        # Load data
        print(f"Loading event {args.event_id}...")
        arrays = tree.arrays(
            [args.npho_branch, args.time_branch, "emiAng"], 
            library="np", 
            entry_start=args.event_id, 
            entry_stop=args.event_id+1
        )

        npho = arrays[args.npho_branch][0]  # (4760,)
        time = arrays[args.time_branch][0]  # (4760,)
        
        TimeScale = 1e-7
        time_scaled = time / TimeScale
        
        # STATS
        # 1. Identify garbage time values (consistent with angle_model_geom.py)
        # Using 1.0 as safe threshold (assuming valid times are < 1.0s)
        mask_garbage = np.abs(time) > 1.0 
        
        # 2. Identify valid hits: Must have photons AND valid time
        mask_valid = (npho > 0) & (~mask_garbage)
        
        n_hits = np.count_nonzero(mask_valid)
        
        if n_hits > 0:
            valid_times = time_scaled[mask_valid]
            t_min = valid_times.min()
            t_max = valid_times.max()
        else:
            t_min, t_max = 0.0, 0.0
        
        # Title
        angles = arrays["emiAng"][0]
        # Use scientific notation for time range if values are small (like 1e-7)
        title_str = (f"Event {args.event_id} Time Map | Hits: {n_hits}\n"
                     f"Time Range: [{t_min:.2e}, {t_max:.2e}] (scaled by 1e7) | "
                     f"Truth: θ={angles[0]:.1f}°, φ={angles[1]:.1f}°")

        print(f"Displaying {title_str}")

        # Plot
        plot_event_time(
            npho, 
            time_scaled, 
            title=title_str, 
            savepath=args.save
        )

if __name__ == "__main__":
    main()