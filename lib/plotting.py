import contextlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binned_statistic

from .utils import angles_deg_to_unit_vec
from .metrics import get_opening_angle_deg


def _get_binned_stat(x, y, stat_func, nbins):
    """Bin y-values by x and apply stat_func per bin.

    Returns (bin_centers, values, errors) where errors is the standard
    error of the mean for each bin (usable as yerr in errorbar plots).
    """
    if len(x) == 0:
        return np.array([]), np.array([]), np.array([])
    bin_edges = np.linspace(x.min(), x.max(), nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(x, bin_edges) - 1
    y_vals = []
    y_errs = []
    for i in range(nbins):
        mask = bin_idx == i
        if np.any(mask):
            y_vals.append(stat_func(y[mask]))
            y_errs.append(np.std(y[mask]) / np.sqrt(mask.sum()))
        else:
            y_vals.append(np.nan)
            y_errs.append(np.nan)
    return bin_centers, np.array(y_vals), np.array(y_errs)


def _gaussian(x, A, mu, sigma):
    """Un-normalised Gaussian for histogram fitting."""
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _get_binned_gaussian(x, y, nbins):
    """Bin y by x, fit a Gaussian in each bin.

    Returns (bin_centers, sigmas, sigma_errors, bin_info) where bin_info
    is a list of (values, popt_or_None, bin_lo, bin_hi) for histogram
    diagnostic plotting.
    """
    from scipy.optimize import curve_fit as _curve_fit

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), []
    bin_edges = np.linspace(x.min(), x.max(), nbins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_idx = np.digitize(x, bin_edges) - 1

    sigmas, sigma_errs, bin_info = [], [], []
    for i in range(nbins):
        mask = bin_idx == i
        vals = y[mask]
        lo, hi = bin_edges[i], bin_edges[i + 1]

        if len(vals) < 10:
            sigmas.append(np.nan)
            sigma_errs.append(np.nan)
            bin_info.append((vals, None, lo, hi))
            continue

        try:
            counts, edges = np.histogram(vals, bins='auto')
            centers = (edges[:-1] + edges[1:]) / 2
            mu0 = np.mean(vals)
            sig0 = np.std(vals)
            dx = edges[1] - edges[0]
            A0 = len(vals) * dx / (sig0 * np.sqrt(2 * np.pi))

            popt, pcov = _curve_fit(
                _gaussian, centers, counts.astype(float),
                p0=[A0, mu0, sig0],
                bounds=([0, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
            )
            sigmas.append(abs(popt[2]))
            sigma_errs.append(np.sqrt(pcov[2, 2]))
            bin_info.append((vals, popt, lo, hi))
        except Exception:
            # Fallback: std with analytic error
            sigmas.append(np.std(vals))
            sigma_errs.append(np.std(vals) / np.sqrt(2 * max(len(vals) - 1, 1)))
            bin_info.append((vals, None, lo, hi))

    return bin_centers, np.array(sigmas), np.array(sigma_errs), bin_info


def _plot_bin_histograms(bin_info, xlabel, title):
    """Create a grid figure with per-bin histograms + Gaussian overlays.

    Args:
        bin_info: list of (values, popt_or_None, bin_lo, bin_hi)
        xlabel: x-axis label for histograms
        title: suptitle

    Returns:
        matplotlib Figure
    """
    n = len(bin_info)
    ncols = min(5, n)
    nrows = max(1, (n + ncols - 1) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3 * nrows))
    fig.suptitle(title, fontsize=13)
    axes = np.atleast_1d(axes).flatten()

    for i, (vals, popt, lo, hi) in enumerate(bin_info):
        ax = axes[i]
        if len(vals) == 0:
            ax.set_title(f"[{lo:.1f}, {hi:.1f}]\nEmpty", fontsize=8)
            ax.axis('off')
            continue
        counts, edges, _ = ax.hist(vals, bins='auto', alpha=0.7,
                                   color='tab:blue', density=False)
        if popt is not None:
            x_fit = np.linspace(edges[0], edges[-1], 200)
            ax.plot(x_fit, _gaussian(x_fit, *popt), 'r-', linewidth=1.5)
            ax.set_title(f"[{lo:.1f}, {hi:.1f}]\n"
                         f"μ={popt[1]:.3f} σ={abs(popt[2]):.3f}",
                         fontsize=8)
        else:
            ax.set_title(f"[{lo:.1f}, {hi:.1f}]\nno fit (N={len(vals)})",
                         fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlabel(xlabel, fontsize=7)

    for j in range(n, len(axes)):
        axes[j].axis('off')
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


@contextlib.contextmanager
def _dummy_pdf():
    """No-op context manager for interactive (non-file) mode."""
    yield None


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

def plot_resolution_profile(pred, true, root_data=None, bins=20, outfile=None):
    """
    Plots angle resolution profiles:
    Page 1 (2x3): Resolution vs own truth axes (theta, phi)
    Page 2 (2x3): Bias (mean residual) vs theta, phi, energy
    Page 3 (2x3): Resolution vs energy, U, V, W  (if root_data available)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    # 1. Opening Angle
    psi_deg = get_opening_angle_deg(pred, true)

    # 2. Component Residuals
    d_theta = np.abs(pred[:, 0] - true[:, 0])
    r_theta = pred[:, 0] - true[:, 0]  # signed residual for bias

    d_phi_raw = pred[:, 1] - true[:, 1]
    r_phi = (d_phi_raw + 180) % 360 - 180  # signed, wrapped
    d_phi = np.abs(r_phi)

    theta_true = true[:, 0]
    phi_true = true[:, 1]

    # Check cross-variable availability
    has_energy = (root_data is not None and
                  'true_energy' in root_data and len(root_data.get('true_energy', [])) > 0)
    has_uvw = (root_data is not None and
               'true_u' in root_data and len(root_data.get('true_u', [])) > 0)

    percentile_68 = lambda x: np.percentile(x, 68)
    mean_func = np.mean

    _eb = dict(fmt='none', capsize=3, elinewidth=0.8)  # shared errorbar style

    with PdfPages(outfile) if outfile else _dummy_pdf() as pdf:
        # --- Page 1: Resolution vs own truth axes ---
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Angle Resolution vs Own Truth Axes", fontsize=16)

        x, y, ye = _get_binned_stat(theta_true, d_theta, percentile_68, bins)
        axs[0, 0].errorbar(x, y, yerr=ye, marker='o', color='tab:blue', ms=5, **_eb)
        axs[0, 0].set_xlabel("True Theta [deg]"); axs[0, 0].set_ylabel("68% |dTheta| [deg]")
        axs[0, 0].set_title("Theta Resolution vs Theta")

        x, y, ye = _get_binned_stat(theta_true, psi_deg, percentile_68, bins)
        axs[0, 1].errorbar(x, y, yerr=ye, marker='s', color='tab:orange', ms=5, **_eb)
        axs[0, 1].set_xlabel("True Theta [deg]"); axs[0, 1].set_ylabel("68% Opening Angle [deg]")
        axs[0, 1].set_title("Opening Angle Res vs Theta")

        x, y, ye = _get_binned_stat(theta_true, psi_deg, mean_func, bins)
        axs[0, 2].errorbar(x, y, yerr=ye, marker='^', color='tab:green', ms=5, **_eb)
        axs[0, 2].set_xlabel("True Theta [deg]"); axs[0, 2].set_ylabel("Mean Opening Angle [deg]")
        axs[0, 2].set_title("Mean Opening Angle vs Theta")

        x, y, ye = _get_binned_stat(phi_true, d_phi, percentile_68, bins)
        axs[1, 0].errorbar(x, y, yerr=ye, marker='o', color='tab:blue', ms=5, **_eb)
        axs[1, 0].set_xlabel("True Phi [deg]"); axs[1, 0].set_ylabel("68% |dPhi| [deg]")
        axs[1, 0].set_title("Phi Resolution vs Phi")

        x, y, ye = _get_binned_stat(phi_true, psi_deg, percentile_68, bins)
        axs[1, 1].errorbar(x, y, yerr=ye, marker='s', color='tab:orange', ms=5, **_eb)
        axs[1, 1].set_xlabel("True Phi [deg]"); axs[1, 1].set_ylabel("68% Opening Angle [deg]")
        axs[1, 1].set_title("Opening Angle Res vs Phi")

        x, y, ye = _get_binned_stat(phi_true, psi_deg, mean_func, bins)
        axs[1, 2].errorbar(x, y, yerr=ye, marker='^', color='tab:green', ms=5, **_eb)
        axs[1, 2].set_xlabel("True Phi [deg]"); axs[1, 2].set_ylabel("Mean Opening Angle [deg]")
        axs[1, 2].set_title("Mean Opening Angle vs Phi")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if outfile: pdf.savefig(fig, dpi=120)
        else: plt.show()
        plt.close(fig)

        # --- Page 2: Bias vs own truth axes + energy ---
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Angle Bias (Mean Residual) Profiles", fontsize=16)

        x, y, ye = _get_binned_stat(theta_true, r_theta, mean_func, bins)
        axs[0, 0].errorbar(x, y, yerr=ye, marker='o', color='tab:blue', ms=5, **_eb)
        axs[0, 0].axhline(0, color='gray', ls='--', lw=1)
        axs[0, 0].set_xlabel("True Theta [deg]"); axs[0, 0].set_ylabel("Mean dTheta [deg]")
        axs[0, 0].set_title("Theta Bias vs Theta")

        x, y, ye = _get_binned_stat(phi_true, r_theta, mean_func, bins)
        axs[0, 1].errorbar(x, y, yerr=ye, marker='s', color='tab:orange', ms=5, **_eb)
        axs[0, 1].axhline(0, color='gray', ls='--', lw=1)
        axs[0, 1].set_xlabel("True Phi [deg]"); axs[0, 1].set_ylabel("Mean dTheta [deg]")
        axs[0, 1].set_title("Theta Bias vs Phi")

        x, y, ye = _get_binned_stat(theta_true, r_phi, mean_func, bins)
        axs[1, 0].errorbar(x, y, yerr=ye, marker='o', color='tab:blue', ms=5, **_eb)
        axs[1, 0].axhline(0, color='gray', ls='--', lw=1)
        axs[1, 0].set_xlabel("True Theta [deg]"); axs[1, 0].set_ylabel("Mean dPhi [deg]")
        axs[1, 0].set_title("Phi Bias vs Theta")

        x, y, ye = _get_binned_stat(phi_true, r_phi, mean_func, bins)
        axs[1, 1].errorbar(x, y, yerr=ye, marker='s', color='tab:orange', ms=5, **_eb)
        axs[1, 1].axhline(0, color='gray', ls='--', lw=1)
        axs[1, 1].set_xlabel("True Phi [deg]"); axs[1, 1].set_ylabel("Mean dPhi [deg]")
        axs[1, 1].set_title("Phi Bias vs Phi")

        if has_energy:
            true_energy = root_data['true_energy'] * 1000.0  # GeV -> MeV
            x, y, ye = _get_binned_stat(true_energy, r_theta, mean_func, bins)
            axs[0, 2].errorbar(x, y, yerr=ye, marker='D', color='tab:green', ms=5, **_eb)
            axs[0, 2].axhline(0, color='gray', ls='--', lw=1)
            axs[0, 2].set_xlabel("True Energy [MeV]"); axs[0, 2].set_ylabel("Mean dTheta [deg]")
            axs[0, 2].set_title("Theta Bias vs Energy")

            x, y, ye = _get_binned_stat(true_energy, r_phi, mean_func, bins)
            axs[1, 2].errorbar(x, y, yerr=ye, marker='D', color='tab:green', ms=5, **_eb)
            axs[1, 2].axhline(0, color='gray', ls='--', lw=1)
            axs[1, 2].set_xlabel("True Energy [MeV]"); axs[1, 2].set_ylabel("Mean dPhi [deg]")
            axs[1, 2].set_title("Phi Bias vs Energy")
        else:
            axs[0, 2].axis('off')
            axs[1, 2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if outfile: pdf.savefig(fig, dpi=120)
        else: plt.show()
        plt.close(fig)

        # --- Page 3: Resolution vs cross-variables (energy, U, V, W) ---
        if has_energy or has_uvw:
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle("Angle Resolution vs Cross-Variables", fontsize=16)

            if has_energy:
                true_energy = root_data['true_energy'] * 1000.0  # GeV -> MeV
                x, y, ye = _get_binned_stat(true_energy, d_theta, percentile_68, bins)
                axs[0, 0].errorbar(x, y, yerr=ye, marker='o', color='tab:red', ms=5, **_eb)
                axs[0, 0].set_xlabel("True Energy [MeV]"); axs[0, 0].set_ylabel("68% |dTheta| [deg]")
                axs[0, 0].set_title("Theta Resolution vs Energy")

                x, y, ye = _get_binned_stat(true_energy, psi_deg, percentile_68, bins)
                axs[1, 0].errorbar(x, y, yerr=ye, marker='s', color='tab:red', ms=5, **_eb)
                axs[1, 0].set_xlabel("True Energy [MeV]"); axs[1, 0].set_ylabel("68% Opening Angle [deg]")
                axs[1, 0].set_title("Opening Angle Res vs Energy")
            else:
                axs[0, 0].axis('off')
                axs[1, 0].axis('off')

            if has_uvw:
                _uvw_markers = ['o', 's', 'D']
                for i, (key, label, color, mk) in enumerate([
                    ('true_u', 'U', 'tab:blue', 'o'), ('true_v', 'V', 'tab:orange', 's'), ('true_w', 'W', 'tab:green', 'D')
                ]):
                    val = root_data[key]
                    x, y, ye = _get_binned_stat(val, d_theta, percentile_68, bins)
                    axs[0, i].errorbar(x, y, yerr=ye, marker=mk, color=color, ms=5, **_eb)
                    axs[0, i].set_xlabel(f"True {label} [cm]"); axs[0, i].set_ylabel("68% |dTheta| [deg]")
                    axs[0, i].set_title(f"Theta Res vs {label}")

                    x, y, ye = _get_binned_stat(val, psi_deg, percentile_68, bins)
                    axs[1, i].errorbar(x, y, yerr=ye, marker=mk, color=color, ms=5, **_eb)
                    axs[1, i].set_xlabel(f"True {label} [cm]"); axs[1, i].set_ylabel("68% Opening Angle [deg]")
                    axs[1, i].set_title(f"Opening Angle Res vs {label}")
            elif has_energy:
                for i in range(1, 3):
                    axs[0, i].axis('off')
                    axs[1, i].axis('off')

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if outfile: pdf.savefig(fig, dpi=120)
            else: plt.show()
            plt.close(fig)

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
    plt.errorbar(bin_centers, bin_means, yerr=bin_sem, fmt='o', capsize=3)
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
    
    plt.tight_layout()
    if outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()
        
def plot_energy_resolution_profile(pred, true, root_data=None, bins=20,
                                   outfile=None, gaussian_fit=False):
    """
    Plots energy resolution profile:
    - Row 1: Residual histogram, Resolution vs energy, Normalized resolution (sigma/E) vs energy
    - Row 2: Pred vs True scatter, Resolution vs U, V, W (first interaction point)
    - Row 3: Relative resolution vs U, V, W for signal region (50-55 MeV)

    When gaussian_fit=True, resolution and error bars come from per-bin
    Gaussian fits (sigma ± fit error) instead of 68th-percentile / SEM.
    Additional PDF pages show the per-bin histograms with fit overlays.

    Args:
        pred: Predicted energy values
        true: True energy values
        root_data: Dict with 'true_u', 'true_v', 'true_w' for position-profiled plots
        bins: Number of bins for profiling
        outfile: Output file path
        gaussian_fit: If True, use Gaussian fits for resolution & error bars
    """
    from matplotlib.colors import LogNorm
    from scipy.optimize import curve_fit

    # Convert from internal GeV to MeV for display
    pred = pred * 1000.0
    true = true * 1000.0

    residual = pred - true
    abs_residual = np.abs(residual)

    # Calorimeter resolution model: sigma/E = sqrt((a/sqrt(E))^2 + b^2 + (c/E)^2)
    # a = stochastic term, b = constant term, c = noise term
    def resolution_model(E, a, b, c):
        return np.sqrt((a / np.sqrt(E))**2 + b**2 + (c / E)**2)

    percentile_68 = lambda x: np.percentile(x, 68)

    # Collect extra histogram-diagnostic figures (only when gaussian_fit)
    hist_figs = []

    # Check if we have uvwFI data for position-profiled plots
    has_uvw = (root_data is not None and
               'true_u' in root_data and len(root_data.get('true_u', [])) > 0)

    # 3x4 if we have uvwFI data, otherwise 2x2
    if has_uvw:
        fig, axs = plt.subplots(3, 4, figsize=(18, 13))
        fig.suptitle("Energy Resolution Profile", fontsize=14)
        # Indices for the 4 main plots (first row)
        idx_hist = (0, 0)
        idx_res = (0, 1)
        idx_rel = (0, 2)
        idx_scatter = (0, 3)
    else:
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Energy Resolution Profile", fontsize=14)
        # Indices for 2x2 layout
        idx_hist = (0, 0)
        idx_res = (0, 1)
        idx_rel = (1, 0)
        idx_scatter = (1, 1)

    # Residual histogram
    axs[idx_hist].hist(residual, bins=100, alpha=0.7, color='tab:blue')
    axs[idx_hist].axvline(0, color='red', linestyle='--', linewidth=1)
    axs[idx_hist].set_xlabel("Residual (Pred - True)")
    axs[idx_hist].set_ylabel("Count")
    axs[idx_hist].set_title(f"Residual Distribution\nBias={np.mean(residual):.4f}, 68%={np.percentile(abs_residual, 68):.4f}")

    # Resolution vs True Energy
    _eb = dict(fmt='none', capsize=3, elinewidth=0.8)
    if gaussian_fit:
        x, y, ye, binfo = _get_binned_gaussian(true, residual, bins)
        hist_figs.append(_plot_bin_histograms(
            binfo, "Residual [MeV]", "Resolution vs True Energy – Bin Histograms"))
    else:
        x, y, ye = _get_binned_stat(true, abs_residual, percentile_68, bins)
    axs[idx_res].errorbar(x, y, yerr=ye, marker='o', color='tab:orange', ms=5, **_eb)
    axs[idx_res].set_xlabel("True Energy [MeV]")
    axs[idx_res].set_ylabel("σ [MeV] (Gauss fit)" if gaussian_fit else "68% |Residual| [MeV]")
    axs[idx_res].set_title("Resolution vs True Energy")

    # Normalized Resolution (sigma/E) vs True Energy with fit
    if gaussian_fit:
        # sigma / E_bin_center for relative resolution
        safe_x = np.where(x > 1e-6, x, 1e-6)
        x_rel, y_rel, ye_rel = x, y / safe_x, ye / safe_x
    else:
        safe_true = np.where(np.abs(true) > 1e-6, true, 1e-6)
        rel_residual = abs_residual / np.abs(safe_true)
        x_rel, y_rel, ye_rel = _get_binned_stat(true, rel_residual, percentile_68, bins)
    axs[idx_rel].errorbar(x_rel, y_rel, yerr=ye_rel, marker='o', color='tab:green',
                          ms=5, label='Data', **_eb)

    # Fit the resolution model
    fit_label = ""
    try:
        valid = ~np.isnan(y_rel) & ~np.isnan(x_rel) & (x_rel > 0)
        if np.sum(valid) >= 3:
            popt, pcov = curve_fit(resolution_model, x_rel[valid], y_rel[valid],
                                   p0=[0.02, 0.01, 0.001],
                                   bounds=([0, 0, 0], [1, 1, 1]))
            a_fit, b_fit, c_fit = popt
            x_fit = np.linspace(x_rel[valid].min(), x_rel[valid].max(), 100)
            y_fit = resolution_model(x_fit, a_fit, b_fit, c_fit)
            axs[idx_rel].plot(x_fit, y_fit, '-', color='tab:red', linewidth=1.5, label='Fit')
            fit_label = f"\na={a_fit*100:.2f}%/√E, b={b_fit*100:.2f}%, c={c_fit*100:.2f}%/E"
    except Exception:
        pass

    axs[idx_rel].set_xlabel("True Energy [MeV]")
    axs[idx_rel].set_ylabel("σ/E (Gauss fit)" if gaussian_fit else "68% |Residual|/E")
    axs[idx_rel].set_title(f"Relative Resolution vs Energy{fit_label}")
    axs[idx_rel].legend(loc='upper right')

    # Pred vs True scatter
    vmin = min(true.min(), pred.min())
    vmax = max(true.max(), pred.max())
    h = axs[idx_scatter].hist2d(true, pred, bins=50, range=[[vmin, vmax], [vmin, vmax]],
                  cmap='viridis', norm=LogNorm())
    axs[idx_scatter].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1, label='y=x')
    axs[idx_scatter].set_xlabel("True Energy [MeV]")
    axs[idx_scatter].set_ylabel("Pred Energy [MeV]")
    axs[idx_scatter].set_title("Pred vs True")
    axs[idx_scatter].legend()
    plt.colorbar(h[3], ax=axs[idx_scatter], label='Count')

    if has_uvw:
        # Row 2: Resolution vs U, V, W (first interaction point)
        true_u = root_data['true_u']
        true_v = root_data['true_v']
        true_w = root_data['true_w']
        uvw_labels = ['U', 'V', 'W']
        uvw_data = [true_u, true_v, true_w]
        uvw_colors = ['tab:blue', 'tab:orange', 'tab:green']

        uvw_markers = ['o', 's', 'D']
        for i, (uvw_val, label, color, mk) in enumerate(zip(uvw_data, uvw_labels, uvw_colors, uvw_markers)):
            if gaussian_fit:
                x, y, ye, binfo = _get_binned_gaussian(uvw_val, residual, bins)
                hist_figs.append(_plot_bin_histograms(
                    binfo, "Residual [MeV]", f"Resolution vs {label} – Bin Histograms"))
            else:
                x, y, ye = _get_binned_stat(uvw_val, abs_residual, percentile_68, bins)
            axs[1, i].errorbar(x, y, yerr=ye, marker=mk, color=color, ms=5, **_eb)
            axs[1, i].set_xlabel(f"True {label} [cm]")
            axs[1, i].set_ylabel("σ [MeV] (Gauss fit)" if gaussian_fit else "68% |Residual| [MeV]")
            axs[1, i].set_title(f"Resolution vs {label}")

        # Row 2, Col 4: Hide unused subplot
        axs[1, 3].axis('off')

        # Row 3: Relative resolution vs U, V, W for signal region (50-55 MeV)
        sig_mask = (true >= 50.0) & (true <= 55.0)
        n_sig = np.sum(sig_mask)
        sig_residual = residual[sig_mask]
        sig_true = true[sig_mask]
        sig_u = true_u[sig_mask]
        sig_v = true_v[sig_mask]
        sig_w = true_w[sig_mask]
        sig_uvw_data = [sig_u, sig_v, sig_w]
        mean_sig_e = np.mean(sig_true) if n_sig > 0 else 1.0

        if not gaussian_fit:
            safe_true_all = np.where(np.abs(true) > 1e-6, true, 1e-6)
            rel_residual = np.abs(residual) / np.abs(safe_true_all)
            sig_rel_residual = rel_residual[sig_mask]

        for i, (uvw_val, label, color, mk) in enumerate(zip(sig_uvw_data, uvw_labels, uvw_colors, uvw_markers)):
            if n_sig > 0:
                if gaussian_fit:
                    x, y, ye, binfo = _get_binned_gaussian(uvw_val, sig_residual, bins)
                    hist_figs.append(_plot_bin_histograms(
                        binfo, "Residual [MeV]",
                        f"Rel. Res. vs {label} (50–55 MeV) – Bin Histograms"))
                    # Convert absolute sigma to relative: sigma / <E>
                    y = y / mean_sig_e
                    ye = ye / mean_sig_e
                else:
                    x, y, ye = _get_binned_stat(uvw_val, sig_rel_residual, percentile_68, bins)
                axs[2, i].errorbar(x, y, yerr=ye, marker=mk, color=color, ms=5, **_eb)
            axs[2, i].set_xlabel(f"True {label} [cm]")
            axs[2, i].set_ylabel("σ/E (Gauss fit)" if gaussian_fit else "68% |Residual|/E")
            axs[2, i].set_title(f"Rel. Resolution vs {label}\n(50–55 MeV, N={n_sig})")

        axs[2, 3].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save — use PdfPages when we have histogram diagnostic pages
    if outfile and hist_figs:
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(outfile) as pdf:
            pdf.savefig(fig, dpi=120)
            for hf in hist_figs:
                pdf.savefig(hf, dpi=120)
        plt.close(fig)
        for hf in hist_figs:
            plt.close(hf)
    elif outfile:
        plt.savefig(outfile, dpi=120)
        plt.close()
    else:
        plt.show()
        for hf in hist_figs:
            plt.show()


def plot_timing_resolution_profile(pred, true, root_data=None, bins=20, outfile=None):
    """
    Plots timing resolution profiles (multi-page PDF):
    Page 1: Residual histogram, resolution vs true timing, pred vs true scatter
    Page 2: Bias vs true timing, resolution & bias vs energy (if available)
    Page 3: Resolution & bias vs U, V, W (if available)
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm

    residual = pred - true
    abs_residual = np.abs(residual)

    has_energy = (root_data is not None and
                  'true_energy' in root_data and len(root_data.get('true_energy', [])) > 0)
    has_uvw = (root_data is not None and
               'true_u' in root_data and len(root_data.get('true_u', [])) > 0)

    percentile_68 = lambda x: np.percentile(x, 68)
    mean_func = np.mean

    _eb = dict(fmt='none', capsize=3, elinewidth=0.8)

    with PdfPages(outfile) if outfile else _dummy_pdf() as pdf:
        # --- Page 1: Core plots ---
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Timing Resolution Profile", fontsize=14)

        axs[0].hist(residual, bins=100, alpha=0.7, color='tab:blue')
        axs[0].axvline(0, color='red', linestyle='--', linewidth=1)
        axs[0].set_xlabel("Residual (Pred - True)"); axs[0].set_ylabel("Count")
        axs[0].set_title(f"Residual Distribution\nBias={np.mean(residual):.4f}, 68%={np.percentile(abs_residual, 68):.4f}")

        x, y, ye = _get_binned_stat(true, abs_residual, percentile_68, bins)
        axs[1].errorbar(x, y, yerr=ye, marker='o', color='tab:orange', ms=5, **_eb)
        axs[1].set_xlabel("True Timing"); axs[1].set_ylabel("68% |Residual|")
        axs[1].set_title("Resolution vs True Timing")

        vmin = min(true.min(), pred.min()); vmax = max(true.max(), pred.max())
        axs[2].hist2d(true, pred, bins=50, range=[[vmin, vmax], [vmin, vmax]],
                      cmap='viridis', norm=LogNorm())
        axs[2].plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1, label='y=x')
        axs[2].set_xlabel("True Timing"); axs[2].set_ylabel("Pred Timing")
        axs[2].set_title("Pred vs True"); axs[2].legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if outfile: pdf.savefig(fig, dpi=120)
        else: plt.show()
        plt.close(fig)

        # --- Page 2: Bias vs timing + resolution/bias vs energy ---
        if has_energy:
            true_energy = root_data['true_energy'] * 1000.0  # GeV -> MeV
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle("Timing: Bias & Cross-Variable (Energy)", fontsize=14)

            x, y, ye = _get_binned_stat(true, residual, mean_func, bins)
            axs[0].errorbar(x, y, yerr=ye, marker='o', color='tab:blue', ms=5, **_eb)
            axs[0].axhline(0, color='gray', ls='--', lw=1)
            axs[0].set_xlabel("True Timing"); axs[0].set_ylabel("Mean Residual")
            axs[0].set_title("Bias vs True Timing")

            x, y, ye = _get_binned_stat(true_energy, abs_residual, percentile_68, bins)
            axs[1].errorbar(x, y, yerr=ye, marker='s', color='tab:red', ms=5, **_eb)
            axs[1].set_xlabel("True Energy [MeV]"); axs[1].set_ylabel("68% |Residual|")
            axs[1].set_title("Resolution vs Energy")

            x, y, ye = _get_binned_stat(true_energy, residual, mean_func, bins)
            axs[2].errorbar(x, y, yerr=ye, marker='D', color='tab:red', ms=5, **_eb)
            axs[2].axhline(0, color='gray', ls='--', lw=1)
            axs[2].set_xlabel("True Energy [MeV]"); axs[2].set_ylabel("Mean Residual")
            axs[2].set_title("Bias vs Energy")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if outfile: pdf.savefig(fig, dpi=120)
            else: plt.show()
            plt.close(fig)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(6, 4))
            fig.suptitle("Timing Bias", fontsize=14)
            x, y, ye = _get_binned_stat(true, residual, mean_func, bins)
            ax.errorbar(x, y, yerr=ye, marker='o', color='tab:blue', ms=5, **_eb)
            ax.axhline(0, color='gray', ls='--', lw=1)
            ax.set_xlabel("True Timing"); ax.set_ylabel("Mean Residual")
            ax.set_title("Bias vs True Timing")
            plt.tight_layout()
            if outfile: pdf.savefig(fig, dpi=120)
            else: plt.show()
            plt.close(fig)

        # --- Page 3: Resolution & bias vs U, V, W ---
        if has_uvw:
            fig, axs = plt.subplots(2, 3, figsize=(18, 8))
            fig.suptitle("Timing Resolution & Bias vs Position", fontsize=14)
            for i, (key, label, color, mk) in enumerate([
                ('true_u', 'U', 'tab:blue', 'o'), ('true_v', 'V', 'tab:orange', 's'), ('true_w', 'W', 'tab:green', 'D')
            ]):
                val = root_data[key]
                x, y, ye = _get_binned_stat(val, abs_residual, percentile_68, bins)
                axs[0, i].errorbar(x, y, yerr=ye, marker=mk, color=color, ms=5, **_eb)
                axs[0, i].set_xlabel(f"True {label} [cm]"); axs[0, i].set_ylabel("68% |Residual|")
                axs[0, i].set_title(f"Resolution vs {label}")

                x, y, ye = _get_binned_stat(val, residual, mean_func, bins)
                axs[1, i].errorbar(x, y, yerr=ye, marker=mk, color=color, ms=5, **_eb)
                axs[1, i].axhline(0, color='gray', ls='--', lw=1)
                axs[1, i].set_xlabel(f"True {label} [cm]"); axs[1, i].set_ylabel("Mean Residual")
                axs[1, i].set_title(f"Bias vs {label}")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if outfile: pdf.savefig(fig, dpi=120)
            else: plt.show()
            plt.close(fig)


def plot_position_resolution_profile(pred_uvw, true_uvw, root_data=None, bins=20, outfile=None):
    """
    Plots position (uvwFI) resolution profiles (multi-page PDF):
    Page 1: U, V, W residual histograms + resolution vs own true value
    Page 2: U, V, W bias vs own true value + resolution vs energy (if available)
    Page 3: Bias vs energy + 3D distance resolution vs energy, U, V, W (if available)
    """
    from matplotlib.backends.backend_pdf import PdfPages

    labels = ['U', 'V', 'W']
    colors = ['tab:blue', 'tab:orange', 'tab:green']

    has_energy = (root_data is not None and
                  'true_energy' in root_data and len(root_data.get('true_energy', [])) > 0)

    # Per-component residuals
    residuals = [pred_uvw[:, i] - true_uvw[:, i] for i in range(3)]
    abs_residuals = [np.abs(r) for r in residuals]
    # 3D distance error
    dist_3d = np.sqrt(sum(r**2 for r in residuals))

    percentile_68 = lambda x: np.percentile(x, 68)
    mean_func = np.mean

    markers = ['o', 's', 'D']
    _eb = dict(fmt='none', capsize=3, elinewidth=0.8)

    with PdfPages(outfile) if outfile else _dummy_pdf() as pdf:
        # --- Page 1: Residual histograms + resolution vs own true ---
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Position (uvwFI) Resolution Profile", fontsize=14)

        for i in range(3):
            axs[0, i].hist(residuals[i], bins=100, alpha=0.7, color=colors[i])
            axs[0, i].axvline(0, color='red', linestyle='--', linewidth=1)
            axs[0, i].set_xlabel(f"{labels[i]} Residual"); axs[0, i].set_ylabel("Count")
            axs[0, i].set_title(f"{labels[i]}: Bias={np.mean(residuals[i]):.3f}, "
                                f"68%={np.percentile(abs_residuals[i], 68):.3f}")

            x, y, ye = _get_binned_stat(true_uvw[:, i], abs_residuals[i], percentile_68, bins)
            axs[1, i].errorbar(x, y, yerr=ye, marker=markers[i], color=colors[i], ms=5, **_eb)
            axs[1, i].set_xlabel(f"True {labels[i]}"); axs[1, i].set_ylabel("68% |Residual|")
            axs[1, i].set_title(f"{labels[i]} Resolution vs True")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if outfile: pdf.savefig(fig, dpi=120)
        else: plt.show()
        plt.close(fig)

        # --- Page 2: Bias vs own true + resolution vs energy ---
        fig, axs = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle("Position Bias vs Own Truth & Energy Cross-Variable", fontsize=14)

        for i in range(3):
            x, y, ye = _get_binned_stat(true_uvw[:, i], residuals[i], mean_func, bins)
            axs[0, i].errorbar(x, y, yerr=ye, marker=markers[i], color=colors[i], ms=5, **_eb)
            axs[0, i].axhline(0, color='gray', ls='--', lw=1)
            axs[0, i].set_xlabel(f"True {labels[i]}"); axs[0, i].set_ylabel("Mean Residual")
            axs[0, i].set_title(f"{labels[i]} Bias vs True {labels[i]}")

        if has_energy:
            true_energy = root_data['true_energy'] * 1000.0  # GeV -> MeV
            for i in range(3):
                x, y, ye = _get_binned_stat(true_energy, abs_residuals[i], percentile_68, bins)
                axs[1, i].errorbar(x, y, yerr=ye, marker=markers[i], color=colors[i], ms=5, **_eb)
                axs[1, i].set_xlabel("True Energy [MeV]"); axs[1, i].set_ylabel("68% |Residual|")
                axs[1, i].set_title(f"{labels[i]} Resolution vs Energy")
        else:
            for i in range(3):
                axs[1, i].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if outfile: pdf.savefig(fig, dpi=120)
        else: plt.show()
        plt.close(fig)

        # --- Page 3: Bias vs energy + 3D distance error profiles ---
        if has_energy:
            true_energy = root_data['true_energy'] * 1000.0  # GeV -> MeV
            fig, axs = plt.subplots(2, 3, figsize=(15, 8))
            fig.suptitle("Position Bias vs Energy & 3D Distance Profiles", fontsize=14)

            for i in range(3):
                x, y, ye = _get_binned_stat(true_energy, residuals[i], mean_func, bins)
                axs[0, i].errorbar(x, y, yerr=ye, marker=markers[i], color=colors[i], ms=5, **_eb)
                axs[0, i].axhline(0, color='gray', ls='--', lw=1)
                axs[0, i].set_xlabel("True Energy [MeV]"); axs[0, i].set_ylabel("Mean Residual")
                axs[0, i].set_title(f"{labels[i]} Bias vs Energy")

            x, y, ye = _get_binned_stat(true_energy, dist_3d, percentile_68, bins)
            axs[1, 0].errorbar(x, y, yerr=ye, marker='o', color='tab:red', ms=5, **_eb)
            axs[1, 0].set_xlabel("True Energy [MeV]"); axs[1, 0].set_ylabel("68% 3D Distance")
            axs[1, 0].set_title("3D Distance Res vs Energy")

            x, y, ye = _get_binned_stat(true_uvw[:, 0], dist_3d, percentile_68, bins)
            axs[1, 1].errorbar(x, y, yerr=ye, marker='s', color='tab:blue', ms=5, **_eb)
            axs[1, 1].set_xlabel("True U"); axs[1, 1].set_ylabel("68% 3D Distance")
            axs[1, 1].set_title("3D Distance Res vs U")

            x, y, ye = _get_binned_stat(true_uvw[:, 2], dist_3d, percentile_68, bins)
            axs[1, 2].errorbar(x, y, yerr=ye, marker='D', color='tab:green', ms=5, **_eb)
            axs[1, 2].set_xlabel("True W"); axs[1, 2].set_ylabel("68% 3D Distance")
            axs[1, 2].set_title("3D Distance Res vs W")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            if outfile: pdf.savefig(fig, dpi=120)
            else: plt.show()
            plt.close(fig)


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