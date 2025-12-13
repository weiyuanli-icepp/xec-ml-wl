import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binned_statistic

from .angle_utils import angles_deg_to_unit_vec
from .angle_metrics import get_opening_angle_deg

def plot_resolution_profile(pred, true, bins=20, outfile=None):
    """
    Plots resolution profiles for analysis:
    Rows: [Theta Analysis, Phi Analysis]
    Cols: [Component Resolution, Opening Angle Resolution, Mean Opening Angle]
    """
    # 1. Opening Angle
    psi_deg = get_opening_angle_deg(pred, true)
    
    # 2. Component Residuals (Absolute errors for resolution)
    d_theta = np.abs(pred[:, 0] - true[:, 0])
    
    # Handle Phi wrapping for residual: result in [-180, 180] then abs
    d_phi_raw = pred[:, 1] - true[:, 1]
    d_phi = np.abs((d_phi_raw + 180) % 360 - 180)

    # Truth inputs for x-axes
    theta_true = true[:, 0]
    phi_true = true[:, 1]
    
    # Helper for binning and calculating statistics
    def get_binned_stat(x, y, stat_func, nbins):
        if len(x) == 0: return np.array([]), np.array([])
        bin_edges = np.linspace(x.min(), x.max(), nbins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_idx = np.digitize(x, bin_edges) - 1
        
        y_vals = []
        for i in range(nbins):
            mask = bin_idx == i
            if np.any(mask):
                y_vals.append(stat_func(y[mask]))
            else:
                y_vals.append(np.nan)
        return bin_centers, np.array(y_vals)

    percentile_68 = lambda x: np.percentile(x, 68)
    mean_func = np.mean

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Resolution Profiles", fontsize=16)

    # --- Row 1: Theta Dependence ---
    # 1. Theta Res vs Theta
    x, y = get_binned_stat(theta_true, d_theta, percentile_68, bins)
    axs[0, 0].plot(x, y, 'o-', color='tab:blue')
    axs[0, 0].set_xlabel("True Theta [deg]")
    axs[0, 0].set_ylabel("68% |dTheta| [deg]")
    axs[0, 0].set_title("Theta Resolution vs Theta")
    axs[0, 0].grid(True, alpha=0.3)

    # 2. Opening Angle Res vs Theta
    x, y = get_binned_stat(theta_true, psi_deg, percentile_68, bins)
    axs[0, 1].plot(x, y, 's-', color='tab:orange')
    axs[0, 1].set_xlabel("True Theta [deg]")
    axs[0, 1].set_ylabel("68% Opening Angle [deg]")
    axs[0, 1].set_title("Opening Angle Res vs Theta")
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Mean Opening Angle vs Theta
    x, y = get_binned_stat(theta_true, psi_deg, mean_func, bins)
    axs[0, 2].plot(x, y, '^-', color='tab:green')
    axs[0, 2].set_xlabel("True Theta [deg]")
    axs[0, 2].set_ylabel("Mean Opening Angle [deg]")
    axs[0, 2].set_title("Mean Opening Angle vs Theta")
    axs[0, 2].grid(True, alpha=0.3)

    # --- Row 2: Phi Dependence ---
    # 4. Phi Res vs Phi
    x, y = get_binned_stat(phi_true, d_phi, percentile_68, bins)
    axs[1, 0].plot(x, y, 'o-', color='tab:blue')
    axs[1, 0].set_xlabel("True Phi [deg]")
    axs[1, 0].set_ylabel("68% |dPhi| [deg]")
    axs[1, 0].set_title("Phi Resolution vs Phi")
    axs[1, 0].grid(True, alpha=0.3)

    # 5. Opening Angle Res vs Phi
    x, y = get_binned_stat(phi_true, psi_deg, percentile_68, bins)
    axs[1, 1].plot(x, y, 's-', color='tab:orange')
    axs[1, 1].set_xlabel("True Phi [deg]")
    axs[1, 1].set_ylabel("68% Opening Angle [deg]")
    axs[1, 1].set_title("Opening Angle Res vs Phi")
    axs[1, 1].grid(True, alpha=0.3)

    # 6. Mean Opening Angle vs Phi
    x, y = get_binned_stat(phi_true, psi_deg, mean_func, bins)
    axs[1, 2].plot(x, y, '^-', color='tab:green')
    axs[1, 2].set_xlabel("True Phi [deg]")
    axs[1, 2].set_ylabel("Mean Opening Angle [deg]")
    axs[1, 2].set_title("Mean Opening Angle vs Phi")
    axs[1, 2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()

def plot_face_weights(model, outfile=None):
    try:
        norms = model.get_concatenated_weight_norms()
        names = list(norms.keys())
        values = list(norms.values())
        
        plt.figure(figsize=(8, 5))
        plt.bar(names, values)
        plt.ylabel("Mean Abs Weight")
        plt.title("Face Importance (First Linear Layer)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if outfile:
            plt.savefig(outfile, dpi=120)
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Could not plot face weights: {e}")

def plot_profile(pred, true, bins=20, label="Theta", outfile=None):
    res = pred - true
    bin_means, bin_edges, _ = binned_statistic(true, res, statistic='mean', bins=bins)
    bin_std, _, _ = binned_statistic(true, res, statistic='std', bins=bins)
    bin_count, _, _ = binned_statistic(true, res, statistic='count', bins=bins)
    bin_sem = bin_std / np.sqrt(bin_count)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(bin_centers, bin_means, yerr=bin_sem, fmt='o-', capsize=5)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel(f"True {label} [deg]")
    plt.ylabel(f"Residual ({label}) [deg]")
    plt.title(f"Bias Profile: {label}")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()

def plot_input_distributions(data_dict, outfile=None):
    npho = data_dict.get("npho")
    time = data_dict.get("time")
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    if npho is not None:
        plt.hist(npho, bins=200, range=(-0.1, 1), log=True)
        plt.title("Input Npho (Log-y, subset)")
    plt.subplot(1, 2, 2)
    if time is not None:
        plt.hist(time, bins=100, log=True)
        plt.title("Input Time (Log-y, subset)")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()

def plot_cos_residuals(pred, true, outfile=None):
    v_pred = angles_deg_to_unit_vec(torch.from_numpy(pred))
    v_true = angles_deg_to_unit_vec(torch.from_numpy(true))
    cos_sim = torch.sum(v_pred * v_true, dim=1).clamp(-1.0, 1.0).numpy()
    cos_res = 1.0 - cos_sim
    plt.figure(figsize=(6,4))
    plt.hist(cos_res, bins=100, range=(0, 0.1), log=True) 
    plt.title("Cosine Residuals")
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()

def plot_pred_truth_scatter(pred, true, outfile=None):
    from matplotlib.colors import LogNorm 
    plt.figure(figsize=(12,5))
    labels = ["Theta", "Phi"]
    for comp in range(2):
        plt.subplot(1,2,comp+1)
        vmin = min(true[:,comp].min(), pred[:,comp].min())
        vmax = max(true[:,comp].max(), pred[:,comp].max())
        plt.hist2d(true[:,comp], pred[:,comp], bins=100, range=[[vmin, vmax], [vmin, vmax]], 
                   cmap="inferno", norm=LogNorm()) 
        plt.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.0, alpha=0.8, label="y=x")
        plt.colorbar(label="Count")
        plt.title(f"{labels[comp]} Regression")
        plt.xlabel(f"True {labels[comp]} [deg]")
        plt.ylabel(f"Pred {labels[comp]} [deg]")
        plt.legend()
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()

def plot_saliency_profile(saliency_data, outfile=None):
    """
    Plots the Gradient Saliency (Sensitivity) for Theta vs Phi,
    separated by Npho and Time.
    """
    # Extract face names from one of the keys
    faces = list(saliency_data["theta"]["npho"].keys())
    x = np.arange(len(faces))
    width = 0.35
    
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    
    # --- Subplot 1: Npho Sensitivity ---
    theta_npho = [saliency_data["theta"]["npho"][f] for f in faces]
    phi_npho   = [saliency_data["phi"]["npho"][f]   for f in faces]
    
    axs[0].bar(x - width/2, theta_npho, width, label='Theta', color='tab:blue', alpha=0.8)
    axs[0].bar(x + width/2, phi_npho,   width, label='Phi',   color='tab:orange', alpha=0.8)
    axs[0].set_title('Sensitivity to PHOTON COUNTS')
    axs[0].set_ylabel('Mean Gradient Magnitude')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(faces, rotation=45)
    axs[0].legend()
    axs[0].grid(True, axis='y', alpha=0.3)
    
    # --- Subplot 2: Time Sensitivity ---
    theta_time = [saliency_data["theta"]["time"][f] for f in faces]
    phi_time   = [saliency_data["phi"]["time"][f]   for f in faces]
    
    axs[1].bar(x - width/2, theta_time, width, label='Theta', color='tab:green', alpha=0.8)
    axs[1].bar(x + width/2, phi_time,   width, label='Phi',   color='tab:red', alpha=0.8)
    axs[1].set_title('Sensitivity to TIMING')
    axs[1].set_ylabel('Mean Gradient Magnitude')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(faces, rotation=45)
    axs[1].legend()
    axs[1].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()