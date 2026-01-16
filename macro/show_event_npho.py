# Usage:
# python macro/show_event_npho.py 0 --file /path/to/data.root --branch relative_npho
import sys
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import torch

# --- Import Layout ---
try:
    # Adjust this path if your folder structure changes
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.event_display import plot_event_faces
except ImportError:
    print("Error: Could not import 'plot_event_faces'.")
    print("Please ensure you are running this from the correct directory (e.g., xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Visualize LXe Event from ROOT file")
    
    # Arguments
    parser.add_argument("event_id", type=int, help="The Event Number (if --run_id is set) OR the file Index (0-based) otherwise.")
    parser.add_argument("--run_id", type=int, default=None, help="The Run Number. If provided, 'event_id' is treated as the physical event ID.")
    
    parser.add_argument("--file", type=str, 
                        default="/data/user/ext-li_w1/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/MCGammaAngle_0-99.root",
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
    
    with uproot.open(args.file) as f:
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found. Available: {f.keys()}")
            sys.exit(1)
            
        tree = f[args.tree]
        n_entries = tree.num_entries
        
        # 2. Determine which Entry Index to load
        target_entry = None
        
        if args.run_id is not None:
            # --- SEARCH MODE (Physical Run/Event) ---
            print(f"Searching for Run {args.run_id}, Event {args.event_id}...")
            
            # Check if run/event branches exist
            if "run" not in tree.keys() or "event" not in tree.keys():
                print("Error: --run_id specified but 'run' or 'event' branches missing in TTree.")
                sys.exit(1)
            
            # Load only ID branches to find the index (fast)
            # Using library="np" returns numpy arrays
            runs = tree["run"].array(library="np")
            events = tree["event"].array(library="np")
            
            # Find matches
            mask = (runs == args.run_id) & (events == args.event_id)
            matches = np.where(mask)[0]
            
            if len(matches) == 0:
                print(f"Error: Event not found for Run {args.run_id}, Event {args.event_id}")
                sys.exit(1)
            elif len(matches) > 1:
                print(f"Warning: Multiple entries found for this Run/Event. Using the first one (Index {matches[0]}).")
                
            target_entry = matches[0]
            print(f"-> Found at Entry Index: {target_entry}")
            
        else:
            # --- INDEX MODE (0-based Index) ---
            target_entry = args.event_id
            if target_entry >= n_entries:
                print(f"Error: Index {target_entry} out of bounds (max {n_entries-1})")
                sys.exit(1)
            print(f"Loading Entry Index: {target_entry}")

        # 3. Load the Data
        # We load: Photon counts, and Truth Angles (if available) for the title
        branches_to_load = [args.branch]
        
        # Check for auxiliary branches to enhance the plot title
        has_angles = "emiAng" in tree.keys()
        if has_angles: branches_to_load.append("emiAng")
            
        has_ids = "run" in tree.keys() and "event" in tree.keys()
        if has_ids: branches_to_load.extend(["run", "event"])

        # Load specific entry
        arrays = tree.arrays(branches_to_load, library="np", 
                             entry_start=target_entry, entry_stop=target_entry+1)

        # 4. Extract data
        npho = arrays[args.branch][0]  # Shape (4760,)
        
        # Construct Title
        title_str = ""
        
        # Use actual Run/Event from file if available, otherwise use args
        if has_ids:
            r_val = arrays["run"][0]
            e_val = arrays["event"][0]
            title_str += f"Run {r_val} | Evt {e_val}"
        elif args.run_id is not None:
            title_str += f"Run {args.run_id} | Evt {args.event_id}"
        else:
            title_str += f"Entry Idx: {target_entry}"
        
        if has_angles:
            angles = arrays["emiAng"][0]
            title_str += f"\nTruth: θ={angles[0]:.2f}°, φ={angles[1]:.2f}°"

        print(f"Displaying: {title_str.replace(chr(10), ' ')}") # Replace newline for console print
        print(f"Total PMT Hits: {np.count_nonzero(npho)}")
        print(f"Max Npho: {np.max(npho):.1f}")

        # 5. Plot using the geometry aware plotter
        plot_event_faces(
            npho, 
            title=title_str, 
            savepath=args.save,
            outer_mode="finegrid" 
        )

if __name__ == "__main__":
    main()
