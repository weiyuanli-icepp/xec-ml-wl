#!/usr/bin/env python3
"""
Generate analysis plots from real data inference results.

Supports both single-task (angle only) and multi-task model outputs.

Usage:
    python val_data/plot_real_data_analysis.py \\
        --input inference_results.root \\
        --output_dir plots_real_data/
"""
import os
import argparse
import numpy as np
import uproot
import matplotlib.pyplot as plt
import torch

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.plotting import (
    plot_pred_truth_scatter,
    plot_resolution_profile,
    plot_profile,
    plot_face_weights
)
from lib.models import XECEncoder, XECMultiHeadModel


def plot_1d_comparison(pred, true, label, unit="", outfile=None, bins=50):
    """Plot 1D histogram comparison of pred vs true."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Histogram of predictions
    axes[0].hist(pred, bins=bins, alpha=0.7, label="Predicted")
    axes[0].hist(true, bins=bins, alpha=0.7, label="Truth")
    axes[0].set_xlabel(f"{label} {unit}")
    axes[0].set_ylabel("Events")
    axes[0].legend()
    axes[0].set_title(f"{label} Distribution")

    # Scatter plot
    axes[1].scatter(true, pred, alpha=0.1, s=1)
    lims = [min(true.min(), pred.min()), max(true.max(), pred.max())]
    axes[1].plot(lims, lims, 'r--', linewidth=1)
    axes[1].set_xlabel(f"True {label} {unit}")
    axes[1].set_ylabel(f"Pred {label} {unit}")
    axes[1].set_title(f"{label}: Pred vs True")

    # Residual
    residual = pred - true
    axes[2].hist(residual, bins=bins, alpha=0.7)
    axes[2].axvline(0, color='r', linestyle='--')
    axes[2].set_xlabel(f"Pred - True {unit}")
    axes[2].set_ylabel("Events")
    mean_res = np.mean(residual)
    std_res = np.std(residual)
    axes[2].set_title(f"Residual: μ={mean_res:.3f}, σ={std_res:.3f}")

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, bbox_inches='tight')
        print(f"[INFO] Saved {outfile}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate Analysis Plots from Real Data Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input ROOT file (from inference_real_data.py)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint.pth (for Face Weights)")
    parser.add_argument("--output_dir", type=str, default="plots_real_data",
                        help="Output directory")
    parser.add_argument("--outer_mode", type=str, default="finegrid",
                        help="Model architecture param for Face Weights")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Data and detect available branches
    print(f"[INFO] Loading data from {args.input}...")
    with uproot.open(args.input) as f:
        tree = f["val_tree"]
        available_branches = set(tree.keys())
        print(f"[INFO] Available branches: {sorted(available_branches)}")

        # Load all available branches
        data = tree.arrays(list(available_branches), library="np")

    # 2. Angle plots (if available)
    if "pred_theta" in available_branches and "true_theta" in available_branches:
        print("[INFO] Generating angle plots...")

        pred = np.stack([data["pred_theta"], data["pred_phi"]], axis=1)
        true = np.stack([data["true_theta"], data["true_phi"]], axis=1)

        # Scatter Plot
        plot_pred_truth_scatter(pred, true, outfile=os.path.join(args.output_dir, "scatter_angle.pdf"))

        # Opening Angle Cosine Residual
        if "opening_angle" in available_branches:
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
            print(f"[INFO] Saved cos_residual.pdf")

        # Resolution Profiles
        plot_resolution_profile(pred, true, outfile=os.path.join(args.output_dir, "resolution_profile.pdf"))

        # Bias Profiles
        plot_profile(data["pred_theta"], data["true_theta"], label="Theta",
                     outfile=os.path.join(args.output_dir, "profile_theta.pdf"))
        plot_profile(data["pred_phi"], data["true_phi"], label="Phi",
                     outfile=os.path.join(args.output_dir, "profile_phi.pdf"))

    # 3. Energy plots (if available)
    if "pred_energy" in available_branches and "true_energy" in available_branches:
        print("[INFO] Generating energy plots...")
        plot_1d_comparison(
            data["pred_energy"], data["true_energy"],
            label="Energy", unit="[MeV]",
            outfile=os.path.join(args.output_dir, "energy_comparison.pdf")
        )

    # 4. Timing plots (if available)
    if "pred_timing" in available_branches and "true_timing" in available_branches:
        print("[INFO] Generating timing plots...")
        plot_1d_comparison(
            data["pred_timing"], data["true_timing"],
            label="Timing", unit="[ns]",
            outfile=os.path.join(args.output_dir, "timing_comparison.pdf")
        )

    # 5. Position (uvwFI) plots (if available)
    if "pred_u" in available_branches and "true_u" in available_branches:
        print("[INFO] Generating position plots...")
        for coord in ["u", "v", "w"]:
            if f"pred_{coord}" in available_branches:
                plot_1d_comparison(
                    data[f"pred_{coord}"], data[f"true_{coord}"],
                    label=f"Position {coord.upper()}", unit="[cm]",
                    outfile=os.path.join(args.output_dir, f"position_{coord}_comparison.pdf")
                )

    # 6. Load Model Checkpoint for Face Weights
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            print(f"[INFO] Loading model from {args.checkpoint} for Face Weights...")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load checkpoint to check model type
            ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

            # Determine if multi-task model
            active_tasks = ckpt.get("active_tasks", None)
            is_multi_task = active_tasks is not None and (
                len(active_tasks) > 1 or (len(active_tasks) == 1 and active_tasks[0] != "angle")
            )

            # Create appropriate model
            if is_multi_task:
                print(f"   Detected multi-task model with tasks: {active_tasks}")
                backbone = XECEncoder(outer_mode=args.outer_mode)
                model = XECMultiHeadModel(backbone=backbone, active_tasks=active_tasks).to(device)
            else:
                model = XECEncoder(outer_mode=args.outer_mode).to(device)

            # Extract state_dict (EMA or standard)
            state_dict = None
            if "ema_state_dict" in ckpt and ckpt["ema_state_dict"] is not None:
                print("   Loading EMA weights...")
                state_dict = ckpt["ema_state_dict"]
            elif "model_state_dict" in ckpt:
                print("   Loading standard weights...")
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = ckpt

            # Clean state_dict (remove 'module.' prefix)
            clean_state = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    clean_state[k[7:]] = v
                else:
                    clean_state[k] = v

            try:
                model.load_state_dict(clean_state, strict=False)

                # For multi-task model, extract backbone for face weights
                if is_multi_task:
                    backbone_model = model.backbone
                else:
                    backbone_model = model

                plot_face_weights(backbone_model, outfile=os.path.join(args.output_dir, "face_weights.pdf"))
            except Exception as e:
                print(f"[WARN] Failed to load model for face weights: {e}")
        else:
            print(f"[WARN] Checkpoint file not found: {args.checkpoint}")
    else:
        print("[INFO] No checkpoint provided. Skipping Face Weights plot.")

    print(f"[DONE] Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
