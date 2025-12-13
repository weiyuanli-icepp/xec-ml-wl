import sys
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import torch

# Import the geometry maps and the plotting function
# Make sure angle_model_geom.py is in the same directory or python path
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from angle_lib.event_display import plot_event_faces
except ImportError:
    print("Error: Could not import 'angle_model_geom.py'.")
    print("Please ensure the file is in the current directory.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize LXe Event from ROOT file")
    parser.add_argument("event_id", type=int, help="The event index to visualize (0-based)")
    parser.add_argument("--file", type=str, 
                        default="/data/user/ext-li_w1/meghome/xec-ml-wl/data/MCGammaAngle_0-49.root",
                        help="Path to the ROOT file")
    parser.add_argument("--tree", type=str, default="tree", help="Name of the TTree")
    parser.add_argument("--branch", type=str, default="relative_npho", 
                        help="Name of the photon branch (e.g. 'relative_npho' or 'npho')")
    parser.add_argument("--save", type=str, default=None, help="Path to save the output PNG (optional)")

    args = parser.parse_args()

    # 1. Open the ROOT file
    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        sys.exit(1)

    print(f"Opening: {args.file}")
    print(f"Tree:    {args.tree}")
    
    with uproot.open(args.file) as f:
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found in file.")
            print(f"Available keys: {f.keys()}")
            sys.exit(1)
            
        tree = f[args.tree]
        n_entries = tree.num_entries
        
        if args.event_id >= n_entries:
            print(f"Error: Event ID {args.event_id} is out of bounds (max {n_entries-1})")
            sys.exit(1)

        # 2. Load the specific event
        # We load: Photon counts, and Truth Angles (if available) for the title
        branches_to_load = [args.branch]
        has_angles = "emiAng" in tree.keys()
        if has_angles:
            branches_to_load.append("emiAng")

        print(f"Loading event {args.event_id}...")
        
        # entry_start/stop allows reading just one event without loading the whole file
        arrays = tree.arrays(branches_to_load, library="np", 
                             entry_start=args.event_id, entry_stop=args.event_id+1)

        # 3. Extract data
        npho = arrays[args.branch][0]  # Shape (4760,)
        
        title_str = f"Event ID: {args.event_id}"
        
        if has_angles:
            angles = arrays["emiAng"][0]
            # specific format for LXe angles
            title_str += f" | Truth: θ={angles[0]:.2f}°, φ={angles[1]:.2f}°"

        print(f"Displaying {title_str}")
        print(f"Total PMT Hits: {np.count_nonzero(npho)}")
        print(f"Max Npho: {np.max(npho):.1f}")

        # 4. Plot using the geometry aware plotter
        # We pass outer_mode='split' or 'finegrid'. 
        # The improved plotter handles 'finegrid' reconstruction automatically.
        plot_event_faces(
            npho, 
            title=title_str, 
            savepath=args.save,
            outer_mode="finegrid" 
        )

if __name__ == "__main__":
    main()