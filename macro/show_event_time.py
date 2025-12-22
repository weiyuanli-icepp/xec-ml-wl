import sys
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import torch

# --- Import Layout ---
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.event_display import plot_event_time
except ImportError:
    print("Error: Could not import 'plot_event_time'.")
    print("Please ensure you are running this from the correct directory or set PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize LXe Event Time Distribution")
    
    # Arguments
    parser.add_argument("event_id", type=int, help="The Event Number (if --run_id is set) OR the file Index (0-based) otherwise.")
    parser.add_argument("--run_id", type=int, default=None, help="The Run Number. If provided, 'event_id' is treated as the physical event ID.")
    
    parser.add_argument("--file", type=str, 
                        default="/data/user/ext-li_w1/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/MCGammaAngle_0-99.root")
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
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found.")
            sys.exit(1)
            
        tree = f[args.tree]
        n_entries = tree.num_entries
        
        # --- LOGIC UPDATE: Determine Entry Index ---
        target_entry = None
        
        if args.run_id is not None:
            # Search Mode
            print(f"Searching for Run {args.run_id}, Event {args.event_id}...")
            
            if "run" not in tree.keys() or "event" not in tree.keys():
                print("Error: --run_id specified but 'run' or 'event' branches missing.")
                sys.exit(1)
            
            # Load ID arrays to find the match
            runs = tree["run"].array(library="np")
            events = tree["event"].array(library="np")
            
            mask = (runs == args.run_id) & (events == args.event_id)
            matches = np.where(mask)[0]
            
            if len(matches) == 0:
                print(f"Error: Event not found for Run {args.run_id}, Event {args.event_id}")
                sys.exit(1)
                
            target_entry = matches[0]
            print(f"-> Found at Entry Index: {target_entry}")
            
        else:
            # Index Mode
            target_entry = args.event_id
            if target_entry >= n_entries:
                print(f"Error: Index {target_entry} out of bounds (max {n_entries-1})")
                sys.exit(1)
            print(f"Loading Entry Index: {target_entry}")

        # --- Load Data ---
        branches = [args.npho_branch, args.time_branch]
        
        # Check for optional branches
        has_angles = "emiAng" in tree.keys()
        if has_angles: branches.append("emiAng")
            
        has_ids = "run" in tree.keys() and "event" in tree.keys()
        if has_ids: branches.extend(["run", "event"])

        print(f"Loading entry {target_entry}...")
        arrays = tree.arrays(
            branches, 
            library="np", 
            entry_start=target_entry, 
            entry_stop=target_entry+1
        )

        npho = arrays[args.npho_branch][0]  # (4760,)
        time = arrays[args.time_branch][0]  # (4760,)
        
        TimeScale = 1e-7
        time_scaled = time / TimeScale
        
        # STATS
        # 1. Identify garbage time values
        mask_garbage = np.abs(time) > 1.0 
        
        # 2. Identify valid hits
        mask_valid = (npho > 0) & (~mask_garbage)
        
        n_hits = np.count_nonzero(mask_valid)
        
        if n_hits > 0:
            valid_times = time_scaled[mask_valid]
            t_min = valid_times.min()
            t_max = valid_times.max()
        else:
            t_min, t_max = 0.0, 0.0
        
        # --- Construct Title ---
        title_str = ""
        
        # ID part
        if has_ids:
            title_str += f"Run {arrays['run'][0]} | Evt {arrays['event'][0]}"
        elif args.run_id is not None:
            title_str += f"Run {args.run_id} | Evt {args.event_id}"
        else:
            title_str += f"Entry {target_entry}"
            
        # Stats part
        title_str += f" | Hits: {n_hits}\n"
        title_str += f"Range: [{t_min:.2e}, {t_max:.2e}] (x1e7)"
        
        if has_angles:
            angles = arrays["emiAng"][0]
            title_str += f" | Truth: θ={angles[0]:.1f}°, φ={angles[1]:.1f}°"

        print(f"Displaying: {title_str.replace(chr(10), ' ')}")

        # Plot
        plot_event_time(
            npho, 
            time_scaled, 
            title=title_str, 
            savepath=args.save
        )

if __name__ == "__main__":
    main()