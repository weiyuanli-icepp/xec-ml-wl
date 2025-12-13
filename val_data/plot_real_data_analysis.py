#!/usr/bin/env python3
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from angle_lib.angle_plotting import (
    plot_pred_truth_scatter,
    plot_resolution_profile,
    plot_profile,
    plot_face_weights
)
from angle_lib.model import AngleRegressorSharedFaces

def main():
    parser = argparse.ArgumentParser(description="Generate Analysis Plots from Real Data Inference")
    parser.add_argument("--input", type=str, required=True, help="Input ROOT file (from inference_real_data.py)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint.pth (for Face Weights)")
    parser.add_argument("--output_dir", type=str, default="plots_real_data", help="Output directory")
    parser.add_argument("--outer_mode", type=str, default="finegrid", help="Model architecture param for Face Weights")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    print(f"[INFO] Loading data from {args.input}...")
    with uproot.open(args.input) as f:
        tree = f["val_tree"]
        data = tree.arrays(["pred_theta", "pred_phi", "true_theta", "true_phi", "opening_angle"], library="np")
    
    pred = np.stack([data["pred_theta"], data["pred_phi"]], axis=1)
    true = np.stack([data["true_theta"], data["true_phi"]], axis=1)
    
    # 2. Scatter Plots
    print("[INFO] Plotting Scatter...")
    plot_pred_truth_scatter(pred, true, outfile=os.path.join(args.output_dir, "scatter.pdf"))
    
    # 3. Opening Angle Cosine Residual
    print("[INFO] Plotting Cosine Residual...")
    # Opening angle is in degrees. Cosine residual = 1 - cos(angle)
    # Be careful with deg->rad
    psi_rad = np.deg2rad(data["opening_angle"])
    cos_res = 1.0 - np.cos(psi_rad)
    
    plt.figure(figsize=(8, 6))
    plt.hist(cos_res, bins=100, range=(0, 0.05), log=True, histtype='stepfilled', alpha=0.7)
    plt.xlabel("1 - cos(Opening Angle)")
    plt.ylabel("Events")
    plt.title("Opening Angle Cosine Residual")
    plt.grid(True, which="both", alpha=0.3)
    plt.savefig(os.path.join(args.output_dir, "cos_residual.pdf"), bbox_inches='tight')
    plt.close()

    # 4. Resolution Profiles
    print("[INFO] Plotting Resolution Profiles...")
    plot_resolution_profile(pred, true, outfile=os.path.join(args.output_dir, "resolution_profile.pdf"))
    
    # 5. Bias Profiles
    print("[INFO] Plotting Bias Profiles...")
    plot_profile(data["pred_theta"], data["true_theta"], label="Theta", 
                 outfile=os.path.join(args.output_dir, "profile_theta.pdf"))
    plot_profile(data["pred_phi"], data["true_phi"], label="Phi", 
                 outfile=os.path.join(args.output_dir, "profile_phi.pdf"))

    # 6. Face Weights (Requires Model Checkpoint)
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"[INFO] Loading model from {args.checkpoint} for Face Weights...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Initialize model structure
            model = AngleRegressorSharedFaces(outer_mode=args.outer_mode).to(device)
            
            # Load weights
            ckpt = torch.load(args.checkpoint, map_location=device)
            
            # Handle EMA or standard
            state_dict = None
            if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
                print("   Loading EMA weights...")
                state_dict = ckpt["ema_state_dict"]
            elif "model_state_dict" in ckpt:
                print("   Loading standard weights...")
                state_dict = ckpt["model_state_dict"]
            else:
                # Assuming the file itself is the state dict
                state_dict = ckpt
            
            # Handle 'module.' prefix if present
            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    clean_state[k[7:]] = v
                else:
                    clean_state[k] = v
            
            try:
                model.load_state_dict(clean_state, strict=False)
                plot_face_weights(model, outfile=os.path.join(args.output_dir, "face_weights.pdf"))
            except Exception as e:
                print(f"[WARN] Failed to load model for face weights: {e}")
        else:
            print(f"[WARN] Checkpoint file not found: {args.checkpoint}")
    else:
        print("[INFO] No checkpoint provided. Skipping Face Weights plot.")

    print(f"[DONE] Plots saved to {args.output_dir}/")

if __name__ == "__main__":
    main()