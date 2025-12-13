#!/usr/bin/env python3
import os
import argparse
import numpy as np
import uproot
import onnxruntime as ort
import time


### Key Features of this Script
# 1.  **Opening Angle Calculation:** I implemented `get_opening_angle_deg` using the standard MEG II coordinate definition ($z = \cos\theta$).
# 2.  **Missing Branch Handling:** The script checks if branches like `emiAng` or `xyzRecoFI` exist. If your "Real Data" file is actually *unlabelled* data (no truth info), it will fill those columns with zeros instead of crashing.
# 3.  **Strict Type Output:** It creates a `val_tree` with explicit types (`int32` for IDs, `float32` for physics vars) to match your training output format perfectly.
# 4.  **Memory Efficient:** It processes in chunks (`10000` events) but accumulates the *results* in memory. Since the results are just a few floats per event (not 4760-pixel images), this fits easily in RAM even for millions of events.

### Usage
# $ python inference_real_data.py \
#     --onnx onnx/meg2ang.onnx \
#     --input DataGammaAngle_RunXXXX.root \
#     --output Output_RunXXXX.root \
#     --NphoScale 1.0 --time_scale 2.32e6

def get_opening_angle_deg(theta1, phi1, theta2, phi2):
    """
    Calculates the opening angle (in degrees) between two vectors defined by (theta, phi).
    """
    # Convert to radians
    t1 = np.deg2rad(theta1)
    p1 = np.deg2rad(phi1)
    t2 = np.deg2rad(theta2)
    p2 = np.deg2rad(phi2)

    # Convert to unit vectors (MEG II convention: z=cos(theta))
    x1 = -np.sin(t1) * np.cos(p1)
    y1 = np.sin(t1) * np.sin(p1)
    z1 = np.cos(t1)

    x2 = -np.sin(t2) * np.cos(p2)
    y2 = np.sin(t2) * np.sin(p2)
    z2 = np.cos(t2)

    # Dot product
    dot = x1*x2 + y1*y2 + z1*z2
    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    
    return np.rad2deg(np.arccos(dot))

def main():
    parser = argparse.ArgumentParser(description="Run Inference on Real Data (ONNX)")
    parser.add_argument("--onnx", type=str, required=True, help="Path to .onnx model")
    parser.add_argument("--input", type=str, required=True, help="Input ROOT file (Real Data)")
    parser.add_argument("--output", type=str, required=True, help="Output ROOT file")
    parser.add_argument("--tree", type=str, default="tree", help="TTree name")
    parser.add_argument("--chunksize", type=int, default=10000, help="Inference chunk size")
    
    # Preprocessing Params (MUST MATCH TRAINING!)
    parser.add_argument("--npho_branch", type=str, default="relative_npho")
    parser.add_argument("--time_branch", type=str, default="relative_time")
    parser.add_argument("--NphoScale", type=float, default=1.0)
    parser.add_argument("--time_scale", type=float, default=2.32e6)
    parser.add_argument("--time_shift", type=float, default=0.0)
    
    args = parser.parse_args()

    # 1. Load ONNX Model
    print(f"[INFO] Loading Model: {args.onnx}")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(args.onnx, providers=providers)
    except Exception as e:
        print(f"[WARN] Failed to load CUDA provider: {e}")
        session = ort.InferenceSession(args.onnx, providers=['CPUExecutionProvider'])
        
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. Open Input File
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    print(f"[INFO] Processing: {args.input}")
    
    # Branches to read
    # Inputs for Model
    model_branches = [args.npho_branch, args.time_branch]
    
    # Metadata / Truth info to copy to output
    # Note: We assume these exist in the input file. If "emiAng" is missing, we can't compute opening angle.
    meta_branches = [
        "run", "event", 
        "emiAng",      # For true_theta/phi
        "energyReco",  # For energy_truth
        "xyzRecoFI",   # For x_truth, y_truth, z_truth
        "xyzVTX"       # For x_vtx, y_vtx, z_vtx
    ]
    
    read_branches = model_branches + meta_branches
    
    # Storage for all batches
    results = {
        "run_id": [], "event_id": [],
        "pred_theta": [], "pred_phi": [],
        "true_theta": [], "true_phi": [],
        "opening_angle": [],
        "energy_truth": [],
        "x_truth": [], "y_truth": [], "z_truth": [],
        "x_vtx": [], "y_vtx": [], "z_vtx": []
    }

    # 3. Inference Loop
    total_events = 0
    start_time = time.time()

    with uproot.open(args.input) as f:
        if args.tree not in f:
            print(f"[ERROR] Tree '{args.tree}' not found in file.")
            return
            
        tree = f[args.tree]
        num_entries = tree.num_entries
        
        # Iterate
        for arrays in tree.iterate(read_branches, step_size=args.chunksize, library="np"):
            
            # --- A. Preprocessing (Model Input) ---
            Npho = arrays[args.npho_branch].astype("float32")
            Time = arrays[args.time_branch].astype("float32")
            
            # Masking
            Npho = np.maximum(Npho, 0.0)
            mask_garbage = (np.abs(Time) > 1.0) | np.isnan(Time)
            mask_invalid = (Npho <= 0.0) | mask_garbage
            Time[mask_invalid] = 0.0
            
            # Scaling
            Npho_norm = np.log1p(Npho / args.NphoScale).astype("float32")
            Time_norm = (Time - args.time_shift) / args.time_scale
            Time_norm = Time_norm.astype("float32")
            
            # Input Tensor: (B, 4760, 2)
            X_batch = np.stack([Npho_norm, Time_norm], axis=-1)
            
            # --- B. Inference ---
            outputs = session.run([output_name], {input_name: X_batch})
            preds = outputs[0] # (B, 2) -> [Theta, Phi]
            
            # --- C. Extract Metadata/Truth ---
            # emiAng is (N, 2) -> [theta, phi]
            # If emiAng doesn't exist (real blind data), fill with NaN or 0
            if "emiAng" in arrays:
                true_ang = arrays["emiAng"]
                t_theta = true_ang[:, 0]
                t_phi   = true_ang[:, 1]
            else:
                t_theta = np.zeros(len(preds))
                t_phi   = np.zeros(len(preds))

            # Opening Angle
            oa = get_opening_angle_deg(preds[:,0], preds[:,1], t_theta, t_phi)
            
            # xyzRecoFI (N, 3)
            if "xyzRecoFI" in arrays:
                xyz = arrays["xyzRecoFI"]
                tx, ty, tz = xyz[:,0], xyz[:,1], xyz[:,2]
            else:
                tx, ty, tz = np.zeros(len(preds)), np.zeros(len(preds)), np.zeros(len(preds))

            # xyzVTX (N, 3)
            if "xyzVTX" in arrays:
                vtx = arrays["xyzVTX"]
                vx, vy, vz = vtx[:,0], vtx[:,1], vtx[:,2]
            else:
                vx, vy, vz = np.zeros(len(preds)), np.zeros(len(preds)), np.zeros(len(preds))
                
            # Energy
            if "energyReco" in arrays:
                en = arrays["energyReco"]
            else:
                en = np.zeros(len(preds))

            # --- D. Store ---
            results["run_id"].append(arrays["run"])
            results["event_id"].append(arrays["event"])
            results["pred_theta"].append(preds[:, 0])
            results["pred_phi"].append(preds[:, 1])
            results["true_theta"].append(t_theta)
            results["true_phi"].append(t_phi)
            results["opening_angle"].append(oa)
            results["energy_truth"].append(en)
            results["x_truth"].append(tx)
            results["y_truth"].append(ty)
            results["z_truth"].append(tz)
            results["x_vtx"].append(vx)
            results["y_vtx"].append(vy)
            results["z_vtx"].append(vz)
            
            total_events += len(preds)
            if total_events % 50000 == 0:
                print(f"   Processed {total_events}/{num_entries} events...")

    # 4. Save Output
    print(f"[INFO] Saving results to: {args.output}")
    
    # Concatenate all lists
    final_data = {k: np.concatenate(v) for k, v in results.items()}
    
    with uproot.recreate(args.output) as f_out:
        # Define types explicitly for robustness
        # Integers
        branch_types = {
            "run_id": np.int32, 
            "event_id": np.int32
        }
        # Floats for everything else
        for k in final_data.keys():
            if k not in branch_types:
                branch_types[k] = np.float32
                
        f_out.mktree("val_tree", branch_types)
        f_out["val_tree"].extend(final_data)
        
    duration = time.time() - start_time
    print(f"[DONE] Processed {total_events} events in {duration:.1f}s ({total_events/duration:.1f} evt/s)")

if __name__ == "__main__":
    main()
