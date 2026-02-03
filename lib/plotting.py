import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binned_statistic

from .utils import angles_deg_to_unit_vec
from .metrics import get_opening_angle_deg


def plot_scalar_scatter(pred, true, label="Value", outfile=None):
    """
    Plots pred vs true scatter for a scalar quantity (energy, timing, etc.)
    """
    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(figsize=(7, 6))

    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())

    h = ax.hist2d(true, pred, bins=80, range=[[vmin, vmax], [vmin, vmax]],
                  cmap='viridis', norm=LogNorm())
    ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='y=x')
    plt.colorbar(h[3], ax=ax, label='Count')

    ax.set_xlabel(f"True {label}")
    ax.set_ylabel(f"Pred {label}")
    ax.set_title(f"{label}: Pred vs True")
    ax.legend()
    ax.set_aspect('equal')

    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()

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
    axs[0, 0].plot(x, y, 'o', color='tab:blue', markersize=5)
    axs[0, 0].set_xlabel("True Theta [deg]")
    axs[0, 0].set_ylabel("68% |dTheta| [deg]")
    axs[0, 0].set_title("Theta Resolution vs Theta")

    # 2. Opening Angle Res vs Theta
    x, y = get_binned_stat(theta_true, psi_deg, percentile_68, bins)
    axs[0, 1].plot(x, y, 's', color='tab:orange', markersize=5)
    axs[0, 1].set_xlabel("True Theta [deg]")
    axs[0, 1].set_ylabel("68% Opening Angle [deg]")
    axs[0, 1].set_title("Opening Angle Res vs Theta")

    # 3. Mean Opening Angle vs Theta
    x, y = get_binned_stat(theta_true, psi_deg, mean_func, bins)
    axs[0, 2].plot(x, y, '^', color='tab:green', markersize=5)
    axs[0, 2].set_xlabel("True Theta [deg]")
    axs[0, 2].set_ylabel("Mean Opening Angle [deg]")
    axs[0, 2].set_title("Mean Opening Angle vs Theta")

    # --- Row 2: Phi Dependence ---
    # 4. Phi Res vs Phi
    x, y = get_binned_stat(phi_true, d_phi, percentile_68, bins)
    axs[1, 0].plot(x, y, 'o', color='tab:blue', markersize=5)
    axs[1, 0].set_xlabel("True Phi [deg]")
    axs[1, 0].set_ylabel("68% |dPhi| [deg]")
    axs[1, 0].set_title("Phi Resolution vs Phi")

    # 5. Opening Angle Res vs Phi
    x, y = get_binned_stat(phi_true, psi_deg, percentile_68, bins)
    axs[1, 1].plot(x, y, 's', color='tab:orange', markersize=5)
    axs[1, 1].set_xlabel("True Phi [deg]")
    axs[1, 1].set_ylabel("68% Opening Angle [deg]")
    axs[1, 1].set_title("Opening Angle Res vs Phi")

    # 6. Mean Opening Angle vs Phi
    x, y = get_binned_stat(phi_true, psi_deg, mean_func, bins)
    axs[1, 2].plot(x, y, '^', color='tab:green', markersize=5)
    axs[1, 2].set_xlabel("True Phi [deg]")
    axs[1, 2].set_ylabel("Mean Opening Angle [deg]")
    axs[1, 2].set_title("Mean Opening Angle vs Phi")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()

def plot_face_weights(model, outfile=None):
    """Plot face importance weights. Silently skips if model doesn't support it."""
    if not hasattr(model, 'get_concatenated_weight_norms'):
        return  # Silently skip - model doesn't support this visualization
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
        
def plot_energy_resolution_profile(pred, true, root_data=None, bins=20, outfile=None):
    """
    Plots energy resolution profile:
    - Row 1: Residual histogram, Resolution vs energy, Normalized resolution (sigma/E) vs energy
    - Row 2: Pred vs True scatter, Resolution vs U, V, W (first interaction point)

    Args:
        pred: Predicted energy values
        true: True energy values
        root_data: Dict with 'true_u', 'true_v', 'true_w' for position-profiled plots
        bins: Number of bins for profiling
        outfile: Output file path
    """
    from matplotlib.colors import LogNorm
    from scipy.optimize import curve_fit

    residual = pred - true
    abs_residual = np.abs(residual)

    # Helper for binning
    def get_binned_stat(x, y, stat_func, nbins):
        if len(x) == 0:
            return np.array([]), np.array([])
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

    # Calorimeter resolution model: sigma/E = sqrt((a/sqrt(E))^2 + b^2)
    # a = stochastic term, b = constant term
    def resolution_model(E, a, b):
        return np.sqrt((a / np.sqrt(E))**2 + b**2)

    percentile_68 = lambda x: np.percentile(x, 68)

    # Check if we have uvwFI data for position-profiled plots
    has_uvw = (root_data is not None and
               'true_u' in root_data and len(root_data.get('true_u', [])) > 0)

    # 2 rows x 4 cols if we have uvwFI data, otherwise 1 row x 4 cols
    if has_uvw:
        fig, axs = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle("Energy Resolution Profile", fontsize=14)
    else:
        fig, axs = plt.subplots(1, 4, figsize=(18, 4))
        fig.suptitle("Energy Resolution Profile", fontsize=14)
        axs = axs.reshape(1, -1)  # Make it 2D for consistent indexing

    # Row 1, Col 1: Residual histogram
    axs[0, 0].hist(residual, bins=100, alpha=0.7, color='tab:blue')
    axs[0, 0].axvline(0, color='red', linestyle='--', linewidth=1)
    axs[0, 0].set_xlabel("Residual (Pred - True)")
    axs[0, 0].set_ylabel("Count")
    axs[0, 0].set_title(f"Residual Distribution\nBias={np.mean(residual):.4f}, 68%={np.percentile(abs_residual, 68):.4f}")

    # Row 1, Col 2: Resolution vs True Energy
    x, y = get_binned_stat(true, abs_residual, percentile_68, bins)
    axs[0, 1].plot(x, y, 'o', color='tab:orange', markersize=5)
    axs[0, 1].set_xlabel("True Energy [GeV]")
    axs[0, 1].set_ylabel("68% |Residual| [GeV]")
    axs[0, 1].set_title("Resolution vs True Energy")

    # Row 1, Col 3: Normalized Resolution (sigma/E) vs True Energy with fit
    # Compute relative resolution: |residual| / true_energy
    # Use small epsilon to avoid division by zero
    safe_true = np.where(np.abs(true) > 1e-6, true, 1e-6)
    rel_residual = abs_residual / np.abs(safe_true)
    x, y = get_binned_stat(true, rel_residual, percentile_68, bins)
    axs[0, 2].plot(x, y, 'o', color='tab:green', markersize=5, label='Data')

    # Fit the resolution model
    fit_label = ""
    try:
        # Filter out NaN values for fitting
        valid = ~np.isnan(y) & ~np.isnan(x) & (x > 0)
        if np.sum(valid) >= 2:
            popt, pcov = curve_fit(resolution_model, x[valid], y[valid],
                                   p0=[0.02, 0.01], bounds=([0, 0], [1, 1]))
            a_fit, b_fit = popt
            # Plot fit curve
            x_fit = np.linspace(x[valid].min(), x[valid].max(), 100)
            y_fit = resolution_model(x_fit, a_fit, b_fit)
            axs[0, 2].plot(x_fit, y_fit, '-', color='tab:red', linewidth=1.5, label='Fit')
            fit_label = f"\na={a_fit*100:.2f}%/√E, b={b_fit*100:.2f}%"
    except Exception:
        pass  # Skip fit if it fails

    axs[0, 2].set_xlabel("True Energy [GeV]")
    axs[0, 2].set_ylabel("68% |Residual|/E")
    axs[0, 2].set_title(f"Relative Resolution vs Energy{fit_label}")
    axs[0, 2].legend(loc='upper right')

    # Row 1, Col 4: Pred vs True scatter
    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())
    h = axs[0, 3].hist2d(true, pred, bins=50, range=[[vmin, vmax], [vmin, vmax]],
                  cmap='viridis', norm=LogNorm())
    axs[0, 3].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1, label='y=x')
    axs[0, 3].set_xlabel("True Energy [GeV]")
    axs[0, 3].set_ylabel("Pred Energy [GeV]")
    axs[0, 3].set_title("Pred vs True")
    axs[0, 3].legend()
    plt.colorbar(h[3], ax=axs[0, 3], label='Count')

    if has_uvw:
        # Row 2: Resolution vs U, V, W (first interaction point)
        true_u = root_data['true_u']
        true_v = root_data['true_v']
        true_w = root_data['true_w']
        uvw_labels = ['U', 'V', 'W']
        uvw_data = [true_u, true_v, true_w]
        uvw_colors = ['tab:blue', 'tab:orange', 'tab:green']

        for i, (uvw_val, label, color) in enumerate(zip(uvw_data, uvw_labels, uvw_colors)):
            x, y = get_binned_stat(uvw_val, abs_residual, percentile_68, bins)
            axs[1, i].plot(x, y, 'o', color=color, markersize=5)
            axs[1, i].set_xlabel(f"True {label} [cm]")
            axs[1, i].set_ylabel("68% |Residual| [GeV]")
            axs[1, i].set_title(f"Resolution vs {label}")

        # Row 2, Col 4: Relative resolution vs U (or leave empty)
        # Let's add relative resolution vs U as an additional insight
        x, y = get_binned_stat(true_u, rel_residual, percentile_68, bins)
        axs[1, 3].plot(x, y, 'o', color='tab:purple', markersize=5)
        axs[1, 3].set_xlabel("True U [cm]")
        axs[1, 3].set_ylabel("68% |Residual|/E")
        axs[1, 3].set_title("Relative Resolution vs U")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()


def plot_timing_resolution_profile(pred, true, bins=20, outfile=None):
    """
    Plots timing resolution profile:
    - Residual distribution histogram
    - Resolution (68% |residual|) vs true timing
    - Pred vs True scatter plot
    """
    residual = pred - true
    abs_residual = np.abs(residual)

    # Helper for binning
    def get_binned_stat(x, y, stat_func, nbins):
        if len(x) == 0:
            return np.array([]), np.array([])
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

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Timing Resolution Profile", fontsize=14)

    # 1. Residual histogram
    axs[0].hist(residual, bins=100, alpha=0.7, color='tab:blue')
    axs[0].axvline(0, color='red', linestyle='--', linewidth=1)
    axs[0].set_xlabel("Residual (Pred - True)")
    axs[0].set_ylabel("Count")
    axs[0].set_title(f"Residual Distribution\nBias={np.mean(residual):.4f}, 68%={np.percentile(abs_residual, 68):.4f}")

    # 2. Resolution vs True Timing
    x, y = get_binned_stat(true, abs_residual, percentile_68, bins)
    axs[1].plot(x, y, 'o', color='tab:orange', markersize=5)
    axs[1].set_xlabel("True Timing")
    axs[1].set_ylabel("68% |Residual|")
    axs[1].set_title("Resolution vs True Timing")

    # 3. Pred vs True scatter
    from matplotlib.colors import LogNorm
    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())
    axs[2].hist2d(true, pred, bins=50, range=[[vmin, vmax], [vmin, vmax]],
                  cmap='viridis', norm=LogNorm())
    axs[2].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1, label='y=x')
    axs[2].set_xlabel("True Timing")
    axs[2].set_ylabel("Pred Timing")
    axs[2].set_title("Pred vs True")
    axs[2].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()


def plot_position_resolution_profile(pred_uvw, true_uvw, bins=20, outfile=None):
    """
    Plots position (uvwFI) resolution profile:
    - Row 1: U, V, W residual histograms
    - Row 2: U, V, W resolution vs true value
    """
    labels = ['U', 'V', 'W']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    # Helper for binning
    def get_binned_stat(x, y, stat_func, nbins):
        if len(x) == 0:
            return np.array([]), np.array([])
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

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Position (uvwFI) Resolution Profile", fontsize=14)

    for i in range(3):
        residual = pred_uvw[:, i] - true_uvw[:, i]
        abs_residual = np.abs(residual)
        true_val = true_uvw[:, i]

        # Row 1: Residual histograms
        axs[0, i].hist(residual, bins=100, alpha=0.7, color=colors[i])
        axs[0, i].axvline(0, color='red', linestyle='--', linewidth=1)
        axs[0, i].set_xlabel(f"{labels[i]} Residual")
        axs[0, i].set_ylabel("Count")
        axs[0, i].set_title(f"{labels[i]}: Bias={np.mean(residual):.3f}, 68%={np.percentile(abs_residual, 68):.3f}")

        # Row 2: Resolution vs True
        x, y = get_binned_stat(true_val, abs_residual, percentile_68, bins)
        axs[1, i].plot(x, y, 'o', color=colors[i], markersize=5)
        axs[1, i].set_xlabel(f"True {labels[i]}")
        axs[1, i].set_ylabel("68% |Residual|")
        axs[1, i].set_title(f"{labels[i]} Resolution vs True")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()


def plot_mae_reconstruction(truth, masked_input, recon, title="MAE Reconstruction", savepath=None):
    """
    Plots Truth vs Masked Input vs Reconstruction for Npho distribution.
    truth, masked_input, recon: 2D arrays for a specific face (e.g., Inner Face)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Truth
    im0 = axes[0].imshow(truth, cmap='viridis', aspect='auto')
    axes[0].set_title("Truth")
    plt.colorbar(im0, ax=axes[0])
    
    # Masked Input (What encoder saw)
    im1 = axes[1].imshow(masked_input, cmap='viridis', aspect='auto')
    axes[1].set_title("Masked Input")
    plt.colorbar(im1, ax=axes[1])
    
    # Reconstruction
    im2 = axes[2].imshow(recon, cmap='viridis', aspect='auto')
    axes[2].set_title("Reconstruction")
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(title)
    if savepath:
        plt.savefig(savepath)
    plt.close()