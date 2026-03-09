#!/usr/bin/env python3
"""
Combine per-patch CEX validation results into a single ROOT file and
produce resolution plots with Gaussian fits.

Standard mode — reads prediction CSVs from val_data/cex/patch{1..24}/:
    python macro/combine_cex_results.py
    python macro/combine_cex_results.py --input-base val_data/cex
    python macro/combine_cex_results.py --patches 13 12 21

Dead-channel mode — reads regressor_*.root files with recovery strategies:
    python macro/combine_cex_results.py --dead-channel
    python macro/combine_cex_results.py --dead-channel --patches 13 12 21
"""

import argparse
import os
import sys
import glob

import numpy as np

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


ALL_PATCHES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
               13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


def _gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (_gaussian(x, A1, mu1, sigma1)
            + _gaussian(x, A2, mu2, sigma2))


def fit_gaussian(values, nbins='auto'):
    """Fit a Gaussian to a 1D array. Returns (popt, pcov) or (None, None)."""
    from scipy.optimize import curve_fit

    if len(values) < 10:
        return None, None
    try:
        counts, edges = np.histogram(values, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2
        mu0 = np.mean(values)
        sig0 = np.std(values)
        dx = edges[1] - edges[0]
        A0 = len(values) * dx / (sig0 * np.sqrt(2 * np.pi))

        popt, pcov = curve_fit(
            _gaussian, centers, counts.astype(float),
            p0=[A0, mu0, sig0],
            bounds=([0, -np.inf, 1e-8], [np.inf, np.inf, np.inf]),
        )
        return popt, pcov
    except Exception:
        return None, None


def fit_double_gaussian(values, nbins='auto'):
    """Fit a double Gaussian to a 1D array.

    Returns (popt, pcov) or (None, None).
    popt = [A_core, mu_core, sigma_core, A_tail, mu_tail, sigma_tail]
    where the core is the component with the smaller |sigma|.
    """
    from scipy.optimize import curve_fit

    if len(values) < 30:
        return None, None
    try:
        counts, edges = np.histogram(values, bins=nbins)
        centers = (edges[:-1] + edges[1:]) / 2
        mu0 = np.median(values)
        sig0 = np.std(values)
        dx = edges[1] - edges[0]
        A0 = len(values) * dx / (sig0 * np.sqrt(2 * np.pi))

        # Initial guess: core (70% of events, narrow) + tail (30%, wider)
        p0 = [0.7 * A0, mu0, 0.6 * sig0,
               0.3 * A0, mu0, 2.0 * sig0]
        bounds_lo = [0, -np.inf, 1e-8, 0, -np.inf, 1e-8]
        bounds_hi = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]

        popt, pcov = curve_fit(
            _double_gaussian, centers, counts.astype(float),
            p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=10000,
        )

        # Ensure component 1 is the core (narrower sigma)
        if abs(popt[5]) < abs(popt[2]):
            popt = np.array([popt[3], popt[4], popt[5],
                             popt[0], popt[1], popt[2]])
            # Swap covariance blocks: reorder indices [3,4,5,0,1,2]
            idx = [3, 4, 5, 0, 1, 2]
            pcov = pcov[np.ix_(idx, idx)]

        return popt, pcov
    except Exception:
        return None, None


STRATEGIES = ["raw", "neighavg", "inpainted"]
STRATEGY_LABELS = {"raw": "Raw", "neighavg": "Neighbor Avg", "inpainted": "Inpainted"}
STRATEGY_COLORS = {"raw": "tab:red", "neighavg": "tab:orange", "inpainted": "tab:blue"}
STRATEGY_MARKERS = {"raw": "o", "neighavg": "s", "inpainted": "D"}


def find_csv(patch_dir):
    """Find the predictions CSV in a patch directory."""
    candidates = sorted(glob.glob(os.path.join(patch_dir, "predictions_energy_*.csv")))
    if candidates:
        return candidates[-1]
    return None


def find_dc_root(patch_dir):
    """Find dead-channel recovery ROOT files in a patch directory."""
    candidates = sorted(glob.glob(os.path.join(patch_dir, "regressor_*.root")))
    return candidates or None


def make_plots(patch_data, combined_residual, output_dir,
               comb_dg_popt=None, comb_dg_pcov=None):
    """Generate multi-page PDF with double-Gaussian-fit resolution plots.

    Args:
        patch_data: list of (patch_id, n_events, residual_mev,
                             sg_popt, sg_pcov, dg_popt, dg_pcov)
        combined_residual: 1D array of all residuals in MeV
        output_dir: directory for output PDF
        comb_dg_popt: double Gaussian popt for combined residual
        comb_dg_pcov: double Gaussian pcov for combined residual
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    pdf_path = os.path.join(output_dir, "CEX23_resolution.pdf")

    with PdfPages(pdf_path) as pdf:
        # ============================================================
        # Page 1: Resolution vs patch (summary) — core σ from double Gaussian
        # ============================================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("CEX23 Energy Resolution by Patch", fontsize=14)

        patch_ids = []
        core_sigmas = []
        core_sigma_errs = []
        sg_sigmas = []
        sg_sigma_errs = []
        core_mus = []
        core_mu_errs = []
        sg_mus = []
        sg_mu_errs = []
        n_events_list = []

        for item in patch_data:
            pid, n_ev, res = item[0], item[1], item[2]
            dg_popt = item[5] if len(item) > 5 else None
            dg_pcov = item[6] if len(item) > 6 else None
            sg_popt = item[3]
            sg_pcov = item[4]
            if dg_popt is None and sg_popt is None:
                continue
            patch_ids.append(pid)
            n_events_list.append(n_ev)
            if dg_popt is not None:
                core_sigmas.append(abs(dg_popt[2]))
                core_sigma_errs.append(np.sqrt(dg_pcov[2, 2]))
                core_mus.append(dg_popt[1])
                core_mu_errs.append(np.sqrt(dg_pcov[1, 1]))
            else:
                core_sigmas.append(np.nan)
                core_sigma_errs.append(0)
                core_mus.append(np.nan)
                core_mu_errs.append(0)
            if sg_popt is not None:
                sg_sigmas.append(abs(sg_popt[2]))
                sg_sigma_errs.append(np.sqrt(sg_pcov[2, 2]))
                sg_mus.append(sg_popt[1])
                sg_mu_errs.append(np.sqrt(sg_pcov[1, 1]))
            else:
                sg_sigmas.append(np.nan)
                sg_sigma_errs.append(0)
                sg_mus.append(np.nan)
                sg_mu_errs.append(0)

        if patch_ids:
            x = np.arange(len(patch_ids))
            labels = [str(p) for p in patch_ids]

            # Top: sigma (resolution) per patch — both core and single-G
            ax1.errorbar(x - 0.1, core_sigmas, yerr=core_sigma_errs, fmt='o',
                         color='tab:blue', capsize=4, markersize=6,
                         label='Double-G core')
            ax1.errorbar(x + 0.1, sg_sigmas, yerr=sg_sigma_errs, fmt='s',
                         color='tab:red', capsize=4, markersize=5,
                         label='Single-G')
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=8)
            ax1.set_xlabel("Patch")
            ax1.set_ylabel("σ [MeV]")
            ax1.set_title("Resolution (σ)")
            ax1.legend(fontsize=9)

            # Bottom: bias (mu) per patch — both core and single-G
            ax2.errorbar(x - 0.1, core_mus, yerr=core_mu_errs, fmt='o',
                         color='tab:blue', capsize=4, markersize=6,
                         label='Double-G core')
            ax2.errorbar(x + 0.1, sg_mus, yerr=sg_mu_errs, fmt='s',
                         color='tab:red', capsize=4, markersize=5,
                         label='Single-G')
            ax2.axhline(0, color='black', ls='-', lw=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_xlabel("Patch")
            ax2.set_ylabel("μ [MeV]")
            ax2.set_title("Bias (μ)")
            ax2.legend(fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # Page 2: Combined residual histogram with double Gaussian fit
        # ============================================================
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        plot_range = (-40, 40)
        nbins = 100
        counts, edges, _ = ax.hist(combined_residual, bins=nbins,
                                   range=plot_range, alpha=0.7,
                                   color='tab:blue',
                                   label=f"N = {len(combined_residual)}")
        # Refit on the plotted histogram so amplitudes match
        centers = (edges[:-1] + edges[1:]) / 2
        from scipy.optimize import curve_fit
        mu0 = np.median(combined_residual)
        sig0 = np.std(combined_residual)
        dx = edges[1] - edges[0]
        A0 = len(combined_residual) * dx / (sig0 * np.sqrt(2 * np.pi))

        # Single Gaussian refit
        sg_refit = None
        try:
            sg_refit, _ = curve_fit(
                _gaussian, centers, counts.astype(float),
                p0=[A0, mu0, sig0],
                bounds=([0, -np.inf, 1e-8], [np.inf, np.inf, np.inf]))
        except Exception:
            pass

        # Double Gaussian refit
        dg_refit = None
        try:
            p0 = [0.7 * A0, mu0, 0.6 * sig0, 0.3 * A0, mu0, 2.0 * sig0]
            bounds_lo = [0, -np.inf, 1e-8, 0, -np.inf, 1e-8]
            bounds_hi = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
            dg_refit, _ = curve_fit(
                _double_gaussian, centers, counts.astype(float),
                p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=10000)
            if abs(dg_refit[5]) < abs(dg_refit[2]):
                dg_refit = np.array([dg_refit[3], dg_refit[4], dg_refit[5],
                                     dg_refit[0], dg_refit[1], dg_refit[2]])
        except Exception:
            pass

        x_fit = np.linspace(plot_range[0], plot_range[1], 300)
        if sg_refit is not None:
            ax.plot(x_fit, _gaussian(x_fit, *sg_refit), 'g-', lw=2,
                    label=(f"Single Gaussian\n"
                           f"  μ={sg_refit[1]:.2f}, "
                           f"σ={abs(sg_refit[2]):.2f} MeV"))
        if dg_refit is not None:
            ax.plot(x_fit, _double_gaussian(x_fit, *dg_refit), 'r-', lw=2,
                    label=(f"Double Gaussian\n"
                           f"  core: μ={dg_refit[1]:.2f}, "
                           f"σ={abs(dg_refit[2]):.2f} MeV\n"
                           f"  tail: μ={dg_refit[4]:.2f}, "
                           f"σ={abs(dg_refit[5]):.2f} MeV"))
            ax.plot(x_fit, _gaussian(x_fit, *dg_refit[:3]),
                    'r--', lw=1, alpha=0.6, label="Core")
            ax.plot(x_fit, _gaussian(x_fit, *dg_refit[3:]),
                    'r:', lw=1, alpha=0.6, label="Tail")
        ax.set_xlabel("Pred − True Energy [MeV]")
        ax.set_ylabel("Events")
        ax.set_title("CEX23 Combined Energy Residual (All Patches)")
        ax.legend(fontsize=9)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # Pages 3+: Per-patch histograms (grid, 6 per page)
        # ============================================================
        patches_with_data = [item for item in patch_data if len(item[2]) > 0]
        per_page = 6
        for page_start in range(0, len(patches_with_data), per_page):
            page_items = patches_with_data[page_start:page_start + per_page]
            nrows = (len(page_items) + 2) // 3
            ncols = min(3, len(page_items))
            fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
            fig.suptitle("Per-Patch Residual Histograms", fontsize=13)
            axes = np.atleast_1d(axes).flatten()

            for i, item in enumerate(page_items):
                pid, n_ev, res = item[0], item[1], item[2]
                dg_popt = item[5] if len(item) > 5 else None
                dg_pcov = item[6] if len(item) > 6 else None

                ax = axes[i]
                counts, edges, _ = ax.hist(res, bins='auto', range=plot_range,
                                           alpha=0.7, color='tab:blue')
                # Refit on plotted histogram bins
                centers_p = (edges[:-1] + edges[1:]) / 2
                mu0_p = np.median(res)
                sig0_p = np.std(res)
                dx_p = edges[1] - edges[0]
                A0_p = len(res) * dx_p / (sig0_p * np.sqrt(2 * np.pi))

                sg_refit_p = None
                try:
                    sg_refit_p, _ = curve_fit(
                        _gaussian, centers_p, counts.astype(float),
                        p0=[A0_p, mu0_p, sig0_p],
                        bounds=([0, -np.inf, 1e-8], [np.inf, np.inf, np.inf]))
                except Exception:
                    pass

                dg_refit_p = None
                try:
                    p0_p = [0.7*A0_p, mu0_p, 0.6*sig0_p,
                            0.3*A0_p, mu0_p, 2.0*sig0_p]
                    dg_refit_p, dg_pcov_p = curve_fit(
                        _double_gaussian, centers_p, counts.astype(float),
                        p0=p0_p,
                        bounds=([0,-np.inf,1e-8,0,-np.inf,1e-8],
                                [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]),
                        maxfev=10000)
                    if abs(dg_refit_p[5]) < abs(dg_refit_p[2]):
                        dg_refit_p = np.array([dg_refit_p[3], dg_refit_p[4],
                                               dg_refit_p[5], dg_refit_p[0],
                                               dg_refit_p[1], dg_refit_p[2]])
                        idx = [3, 4, 5, 0, 1, 2]
                        dg_pcov_p = dg_pcov_p[np.ix_(idx, idx)]
                except Exception:
                    pass

                x_fit = np.linspace(plot_range[0], plot_range[1], 200)
                if sg_refit_p is not None:
                    ax.plot(x_fit, _gaussian(x_fit, *sg_refit_p),
                            'g-', lw=1.5, alpha=0.8)
                if dg_refit_p is not None:
                    ax.plot(x_fit, _double_gaussian(x_fit, *dg_refit_p),
                            'r-', lw=1.5)
                    ax.plot(x_fit, _gaussian(x_fit, *dg_refit_p[:3]),
                            'r--', lw=1, alpha=0.5)
                    ax.plot(x_fit, _gaussian(x_fit, *dg_refit_p[3:]),
                            'r:', lw=1, alpha=0.5)

                # Build title with available fit info
                title_parts = [f"Patch {pid} (N={n_ev})"]
                if dg_refit_p is not None:
                    core_sig_err = np.sqrt(dg_pcov_p[2, 2])
                    title_parts.append(
                        f"core σ={abs(dg_refit_p[2]):.2f}±{core_sig_err:.2f}")
                if sg_refit_p is not None:
                    title_parts.append(
                        f"σ={abs(sg_refit_p[2]):.2f} MeV")
                else:
                    title_parts[-1] += " MeV"
                ax.set_title("\n".join(title_parts), fontsize=10)
                ax.set_xlabel("Pred − True [MeV]", fontsize=9)
                ax.set_ylabel("Events", fontsize=9)

            for j in range(len(page_items), len(axes)):
                axes[j].axis('off')

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPlots: {pdf_path}")


def make_plots_dead_channel(patch_data_dc, combined_residuals_dc,
                            active_strategies, output_dir):
    """Generate multi-page PDF comparing dead-channel recovery strategies.

    Args:
        patch_data_dc: list of (patch_id, n_events,
                        {strategy: (residual, dg_popt, dg_pcov)})
        combined_residuals_dc: {strategy: residual_mev_array}
        active_strategies: list of strategy names with valid data
        output_dir: directory for output PDF
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from scipy.optimize import curve_fit

    pdf_path = os.path.join(output_dir, "CEX23_dead_channel_resolution.pdf")
    n_strat = len(active_strategies)
    offsets = np.linspace(-0.15 * (n_strat - 1), 0.15 * (n_strat - 1), n_strat)

    with PdfPages(pdf_path) as pdf:
        # ==============================================================
        # Page 1: Resolution (core sigma) vs patch for each strategy
        # ==============================================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("CEX23 Dead-Channel Recovery: Energy Resolution", fontsize=14)

        patch_ids = []
        strat_sigmas = {s: [] for s in active_strategies}
        strat_sigma_errs = {s: [] for s in active_strategies}
        strat_mus = {s: [] for s in active_strategies}
        strat_mu_errs = {s: [] for s in active_strategies}

        for pid, n_ev, strat_dict in patch_data_dc:
            has_any = False
            for s in active_strategies:
                if s in strat_dict and strat_dict[s][1] is not None:
                    has_any = True
                    break
            if not has_any:
                continue
            patch_ids.append(pid)
            for s in active_strategies:
                if s in strat_dict and strat_dict[s][1] is not None:
                    dg = strat_dict[s][1]
                    dg_cov = strat_dict[s][2]
                    strat_sigmas[s].append(abs(dg[2]))
                    strat_sigma_errs[s].append(np.sqrt(dg_cov[2, 2]))
                    strat_mus[s].append(dg[1])
                    strat_mu_errs[s].append(np.sqrt(dg_cov[1, 1]))
                else:
                    strat_sigmas[s].append(np.nan)
                    strat_sigma_errs[s].append(0)
                    strat_mus[s].append(np.nan)
                    strat_mu_errs[s].append(0)

        if patch_ids:
            x = np.arange(len(patch_ids))
            labels = [str(p) for p in patch_ids]

            for i, s in enumerate(active_strategies):
                ax1.errorbar(x + offsets[i], strat_sigmas[s],
                             yerr=strat_sigma_errs[s],
                             fmt=STRATEGY_MARKERS[s], color=STRATEGY_COLORS[s],
                             capsize=4, markersize=6, label=STRATEGY_LABELS[s])
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=8)
            ax1.set_xlabel("Patch")
            ax1.set_ylabel("Core $\\sigma$ [MeV]")
            ax1.set_title("Resolution (Double-Gaussian Core $\\sigma$)")
            ax1.legend(fontsize=9)

            for i, s in enumerate(active_strategies):
                ax2.errorbar(x + offsets[i], strat_mus[s],
                             yerr=strat_mu_errs[s],
                             fmt=STRATEGY_MARKERS[s], color=STRATEGY_COLORS[s],
                             capsize=4, markersize=6, label=STRATEGY_LABELS[s])
            ax2.axhline(0, color='black', ls='-', lw=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_xlabel("Patch")
            ax2.set_ylabel("Core $\\mu$ [MeV]")
            ax2.set_title("Bias (Double-Gaussian Core $\\mu$)")
            ax2.legend(fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ==============================================================
        # Page 2: Combined residual histograms (one per strategy)
        # ==============================================================
        fig, axes = plt.subplots(1, n_strat, figsize=(6 * n_strat, 5))
        if n_strat == 1:
            axes = [axes]
        fig.suptitle("CEX23 Combined Residual by Strategy", fontsize=14)
        plot_range = (-40, 40)
        nbins = 100

        for ax, s in zip(axes, active_strategies):
            res = combined_residuals_dc.get(s, np.array([]))
            if res.size == 0:
                ax.set_title(STRATEGY_LABELS[s])
                continue
            counts, edges, _ = ax.hist(res, bins=nbins, range=plot_range,
                                       alpha=0.7, color=STRATEGY_COLORS[s],
                                       label=f"N = {len(res)}")
            # Double Gaussian fit
            centers = (edges[:-1] + edges[1:]) / 2
            mu0 = np.median(res)
            sig0 = np.std(res)
            dx = edges[1] - edges[0]
            A0 = len(res) * dx / (sig0 * np.sqrt(2 * np.pi))
            try:
                p0 = [0.7*A0, mu0, 0.6*sig0, 0.3*A0, mu0, 2.0*sig0]
                dg, _ = curve_fit(
                    _double_gaussian, centers, counts.astype(float),
                    p0=p0,
                    bounds=([0,-np.inf,1e-8,0,-np.inf,1e-8],
                            [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf]),
                    maxfev=10000)
                if abs(dg[5]) < abs(dg[2]):
                    dg = np.array([dg[3], dg[4], dg[5], dg[0], dg[1], dg[2]])
                x_fit = np.linspace(plot_range[0], plot_range[1], 300)
                ax.plot(x_fit, _double_gaussian(x_fit, *dg), 'k-', lw=2)
                ax.plot(x_fit, _gaussian(x_fit, *dg[:3]), 'k--', lw=1, alpha=0.5)
                ax.plot(x_fit, _gaussian(x_fit, *dg[3:]), 'k:', lw=1, alpha=0.5)
                label = (f"core: $\\sigma$={abs(dg[2]):.2f}, "
                         f"$\\mu$={dg[1]:.2f} MeV")
                ax.text(0.97, 0.95, label, transform=ax.transAxes,
                        fontsize=9, va='top', ha='right')
            except Exception:
                pass
            ax.set_xlabel("Pred - True [MeV]")
            ax.set_ylabel("Events")
            ax.set_title(STRATEGY_LABELS[s])
            ax.legend(fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ==============================================================
        # Pages 3+: Per-patch overlaid histograms (6 per page)
        # ==============================================================
        patches_with_data = [item for item in patch_data_dc
                             if any(s in item[2] and len(item[2][s][0]) > 0
                                    for s in active_strategies)]
        per_page = 6
        for page_start in range(0, len(patches_with_data), per_page):
            page_items = patches_with_data[page_start:page_start + per_page]
            nrows = (len(page_items) + 2) // 3
            ncols = min(3, len(page_items))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(5 * ncols, 4 * nrows))
            fig.suptitle("Per-Patch Residual by Strategy", fontsize=13)
            axes = np.atleast_1d(axes).flatten()

            for i, (pid, n_ev, strat_dict) in enumerate(page_items):
                ax = axes[i]
                title_parts = [f"Patch {pid} (N={n_ev})"]
                for s in active_strategies:
                    if s not in strat_dict:
                        continue
                    res = strat_dict[s][0]
                    if len(res) == 0:
                        continue
                    ax.hist(res, bins='auto', range=plot_range, alpha=0.4,
                            color=STRATEGY_COLORS[s], label=STRATEGY_LABELS[s])
                    dg = strat_dict[s][1]
                    if dg is not None:
                        title_parts.append(
                            f"{STRATEGY_LABELS[s]}: "
                            f"$\\sigma$={abs(dg[2]):.2f} MeV")
                ax.set_title("\n".join(title_parts), fontsize=9)
                ax.set_xlabel("Pred - True [MeV]", fontsize=9)
                ax.set_ylabel("Events", fontsize=9)
                ax.legend(fontsize=7)

            for j in range(len(page_items), len(axes)):
                axes[j].axis('off')

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPlots: {pdf_path}")


def _run_dead_channel_mode(args, patches, input_base):
    """Dead-channel mode: load regressor_*.root files and compare strategies."""
    if not HAS_UPROOT:
        print("[ERROR] uproot is required for dead-channel mode")
        sys.exit(1)

    patch_data_dc = []  # (patch_id, n_events, {strat: (residual, dg_popt, dg_pcov)})
    combined_per_strat = {s: [] for s in STRATEGIES}
    found = 0
    missing = []

    for patch in patches:
        patch_dir = os.path.join(input_base, f"patch{patch}")
        root_files = find_dc_root(patch_dir)

        if root_files is None:
            missing.append(patch)
            continue

        # Read and concatenate all ROOT files for this patch
        all_arrays = {}
        for rf in root_files:
            with uproot.open(rf) as f:
                tree = f["tree"]
                arrays = tree.arrays(library="np")
                for k, v in arrays.items():
                    all_arrays.setdefault(k, []).append(v)
        arrays = {k: np.concatenate(v) for k, v in all_arrays.items()}

        n = len(arrays.get("run", []))
        found += 1

        # Get truth
        truth = arrays.get("energyTruth", None)
        if truth is None:
            print(f"  Patch {patch:>2d}: {n:>6d} events | no energyTruth branch")
            patch_data_dc.append((patch, n, {}))
            continue

        # Detect available strategies
        strat_dict = {}
        parts = [f"  Patch {patch:>2d}: {n:>6d} events"]

        for s in STRATEGIES:
            branch = f"energy_{s}"
            pred = arrays.get(branch, None)
            if pred is None:
                continue
            # Skip inpainted if all sentinel (no inpainter was provided)
            if s == "inpainted" and np.all(pred > 1e9):
                continue

            valid = (truth < 1e9) & (pred < 1e9)
            if valid.sum() == 0:
                continue

            residual = (pred[valid] - truth[valid]) * 1e3  # MeV
            dg_popt, dg_pcov = fit_double_gaussian(residual)
            strat_dict[s] = (residual, dg_popt, dg_pcov)
            combined_per_strat[s].append(residual)

            if dg_popt is not None:
                core_sig = abs(dg_popt[2])
                core_sig_err = np.sqrt(dg_pcov[2, 2])
                parts.append(f"{STRATEGY_LABELS[s]}: "
                             f"σ={core_sig:.2f}\u00b1{core_sig_err:.2f}")
            else:
                res68 = np.percentile(np.abs(residual), 68)
                parts.append(f"{STRATEGY_LABELS[s]}: res68={res68:.2f}")

        print(" | ".join(parts))
        patch_data_dc.append((patch, n, strat_dict))

    if found == 0:
        print("\n[ERROR] No dead-channel results found. "
              "Check that validation jobs have completed.")
        sys.exit(1)

    if missing:
        print(f"\n[WARN] Missing patches: {missing}")

    # Determine active strategies (those with data)
    active_strategies = [s for s in STRATEGIES if combined_per_strat[s]]

    # Combined stats
    print(f"\n{'='*60}")
    print(f"Combined: {found} patches")
    combined_residuals_dc = {}
    for s in active_strategies:
        res = np.concatenate(combined_per_strat[s])
        combined_residuals_dc[s] = res
        dg_popt, dg_pcov = fit_double_gaussian(res)
        parts = [f"  {STRATEGY_LABELS[s]:>14s}: N={len(res):>7d}"]
        if dg_popt is not None:
            core_sig = abs(dg_popt[2])
            core_sig_err = np.sqrt(dg_pcov[2, 2])
            parts.append(f"core σ={core_sig:.2f}\u00b1{core_sig_err:.2f} MeV")
            parts.append(f"μ={dg_popt[1]:+.2f} MeV")
        parts.append(f"res68={np.percentile(np.abs(res), 68):.2f} MeV")
        print(" | ".join(parts))

    # Save combined ROOT file
    if not args.no_plots and patch_data_dc:
        make_plots_dead_channel(patch_data_dc, combined_residuals_dc,
                                active_strategies, input_base)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Combine per-patch CEX validation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input-base", default="val_data/cex",
                        help="Base directory with patch*/ subdirectories")
    parser.add_argument("--output", default=None,
                        help="Output file path (default: <input-base>/CEX23_combined.root or .csv)")
    parser.add_argument("--patches", type=int, nargs="*", default=None,
                        help="Specific patches to combine (default: all 1-24)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip plot generation")
    parser.add_argument("--dead-channel", action="store_true",
                        help="Dead-channel recovery mode: read regressor_*.root files")
    args = parser.parse_args()

    patches = args.patches or ALL_PATCHES
    input_base = args.input_base

    print(f"Input base: {input_base}")
    print(f"Patches:    {patches}")
    print()

    if args.dead_channel:
        _run_dead_channel_mode(args, patches, input_base)
        return

    if not HAS_PANDAS:
        print("[ERROR] pandas is required. Install with: pip install pandas")
        sys.exit(1)

    all_dfs = []
    patch_data = []   # (patch_id, n_events, residual_mev, popt, pcov)
    found = 0
    missing = []

    for patch in patches:
        patch_dir = os.path.join(input_base, f"patch{patch}")
        csv_path = find_csv(patch_dir)

        if csv_path is None:
            missing.append(patch)
            continue

        df = pd.read_csv(csv_path)
        df["patch"] = patch
        n = len(df)
        all_dfs.append(df)
        found += 1

        # Compute residual and fit double Gaussian
        if "pred_energy" in df.columns and "true_energy" in df.columns:
            valid = df["true_energy"] < 1e9
            if valid.sum() > 0:
                residual = (df.loc[valid, "pred_energy"].values
                            - df.loc[valid, "true_energy"].values) * 1e3
                dg_popt, dg_pcov = fit_double_gaussian(residual)
                sg_popt, sg_pcov = fit_gaussian(residual)
                patch_data.append((patch, n, residual, sg_popt, sg_pcov,
                                   dg_popt, dg_pcov))

                parts = [f"  Patch {patch:>2d}: {n:>6d} events"]
                if dg_popt is not None:
                    core_sig = abs(dg_popt[2])
                    core_sig_err = np.sqrt(dg_pcov[2, 2])
                    f_core = abs(dg_popt[0] * dg_popt[2]) / (
                        abs(dg_popt[0] * dg_popt[2]) + abs(dg_popt[3] * dg_popt[5]))
                    parts.append(f"core σ={core_sig:.2f}±{core_sig_err:.2f} MeV "
                                 f"(f={f_core:.0%})")
                if sg_popt is not None:
                    sg_sig_err = np.sqrt(sg_pcov[2, 2])
                    parts.append(f"σ={abs(sg_popt[2]):.2f}±{sg_sig_err:.2f} MeV")
                if dg_popt is None and sg_popt is None:
                    parts.append(f"fit failed, "
                                 f"res68={np.percentile(np.abs(residual), 68):.2f} MeV")
                print(" | ".join(parts))
            else:
                pred_mev = df["pred_energy"].values * 1e3
                patch_data.append((patch, n, pred_mev, None, None, None, None))
                print(f"  Patch {patch:>2d}: {n:>6d} events | "
                      f"pred mean={np.mean(pred_mev):.2f} MeV (no truth)")
        else:
            patch_data.append((patch, n, np.array([]), None, None, None, None))
            print(f"  Patch {patch:>2d}: {n:>6d} events")

    if not all_dfs:
        print("\n[ERROR] No patch results found. Check that validation jobs have completed.")
        sys.exit(1)

    if missing:
        print(f"\n[WARN] Missing patches: {missing}")

    combined = pd.concat(all_dfs, ignore_index=True)
    n_total = len(combined)
    n_patches = found

    # Overall stats with Gaussian fit
    print(f"\n{'='*60}")
    print(f"Combined: {n_total} events from {n_patches} patches")

    combined_residual = np.array([])
    comb_dg_popt = None
    comb_dg_pcov = None
    if "pred_energy" in combined.columns and "true_energy" in combined.columns:
        valid = combined["true_energy"] < 1e9
        if valid.sum() > 0:
            combined_residual = (combined.loc[valid, "pred_energy"].values
                                 - combined.loc[valid, "true_energy"].values) * 1e3
            popt, pcov = fit_gaussian(combined_residual)
            comb_dg_popt, comb_dg_pcov = fit_double_gaussian(combined_residual)
            if comb_dg_popt is not None:
                core_sig = abs(comb_dg_popt[2])
                core_sig_err = np.sqrt(comb_dg_pcov[2, 2])
                tail_sig = abs(comb_dg_popt[5])
                tail_sig_err = np.sqrt(comb_dg_pcov[5, 5])
                f_core = abs(comb_dg_popt[0] * comb_dg_popt[2]) / (
                    abs(comb_dg_popt[0] * comb_dg_popt[2])
                    + abs(comb_dg_popt[3] * comb_dg_popt[5]))
                print(f"  Core μ:      {comb_dg_popt[1]:+.2f} ± {np.sqrt(comb_dg_pcov[1,1]):.2f} MeV")
                print(f"  Core σ:      {core_sig:.2f} ± {core_sig_err:.2f} MeV")
                print(f"  Tail σ:      {tail_sig:.2f} ± {tail_sig_err:.2f} MeV")
                print(f"  Core frac:   {f_core:.1%}")
            if popt is not None:
                sig_err = np.sqrt(pcov[2, 2])
                print(f"  Single-G σ:  {abs(popt[2]):.2f} ± {sig_err:.2f} MeV")
            print(f"  MAE:         {np.mean(np.abs(combined_residual)):.2f} MeV")
            print(f"  RMSE:        {np.sqrt(np.mean(combined_residual**2)):.2f} MeV")
            print(f"  Res68:       {np.percentile(np.abs(combined_residual), 68):.2f} MeV")
        else:
            pred_mev = combined["pred_energy"].values * 1e3
            print(f"  Pred mean: {np.mean(pred_mev):.2f} MeV")
            print(f"  Pred std:  {np.std(pred_mev):.2f} MeV")
            print(f"  (No truth available for real data)")

    # Save combined data
    if args.output:
        outpath = args.output
    elif HAS_UPROOT:
        outpath = os.path.join(input_base, "CEX23_combined.root")
    else:
        outpath = os.path.join(input_base, "CEX23_combined.csv")

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)

    if outpath.endswith(".root") and not HAS_UPROOT:
        print("[WARN] uproot not available, falling back to CSV")
        outpath = outpath.replace(".root", ".csv")

    if outpath.endswith(".root"):
        branches = {}
        for col in combined.columns:
            arr = combined[col].values
            if arr.dtype == np.float64:
                arr = arr.astype(np.float32)
            branches[col] = arr
        with uproot.recreate(outpath) as f:
            f.mktree("tree", branches)
        print(f"\nData:  {outpath} ({n_total} events)")
    else:
        combined.to_csv(outpath, index=False)
        print(f"\nData:  {outpath} ({n_total} events)")

    # Generate plots
    if not args.no_plots and patch_data:
        make_plots(patch_data, combined_residual, input_base,
                   comb_dg_popt=comb_dg_popt, comb_dg_pcov=comb_dg_pcov)

    print("Done!")


if __name__ == "__main__":
    main()
