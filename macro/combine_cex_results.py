#!/usr/bin/env python3
"""
Combine per-patch CEX validation results into a single ROOT file and
produce resolution plots with Gaussian fits.

Reads prediction CSVs from val_data/cex/patch{1..24}/ and:
  1. Prints per-patch and overall statistics
  2. Writes a combined ROOT file (or CSV fallback) with patch ID
  3. Produces a multi-page PDF with:
     - Page 1: Summary — resolution vs patch (Gaussian σ ± fit error)
     - Page 2: Combined residual histogram with Gaussian fit
     - Pages 3+: Per-patch residual histograms with Gaussian overlays

Usage:
    python macro/combine_cex_results.py
    python macro/combine_cex_results.py --input-base val_data/cex
    python macro/combine_cex_results.py --patches 13 12 21
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


def find_csv(patch_dir):
    """Find the predictions CSV in a patch directory."""
    candidates = sorted(glob.glob(os.path.join(patch_dir, "predictions_energy_*.csv")))
    if candidates:
        return candidates[-1]
    return None


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
        fig.suptitle("CEX23 Energy Resolution by Patch (Double Gaussian Core)",
                     fontsize=14)

        patch_ids = []
        core_sigmas = []
        core_sigma_errs = []
        core_mus = []
        core_mu_errs = []
        n_events_list = []

        for item in patch_data:
            pid, n_ev, res = item[0], item[1], item[2]
            dg_popt = item[5] if len(item) > 5 else None
            dg_pcov = item[6] if len(item) > 6 else None
            sg_popt = item[3]
            sg_pcov = item[4]
            # Prefer double Gaussian core, fall back to single Gaussian
            if dg_popt is not None:
                patch_ids.append(pid)
                core_sigmas.append(abs(dg_popt[2]))
                core_sigma_errs.append(np.sqrt(dg_pcov[2, 2]))
                core_mus.append(dg_popt[1])
                core_mu_errs.append(np.sqrt(dg_pcov[1, 1]))
                n_events_list.append(n_ev)
            elif sg_popt is not None:
                patch_ids.append(pid)
                core_sigmas.append(abs(sg_popt[2]))
                core_sigma_errs.append(np.sqrt(sg_pcov[2, 2]))
                core_mus.append(sg_popt[1])
                core_mu_errs.append(np.sqrt(sg_pcov[1, 1]))
                n_events_list.append(n_ev)

        if patch_ids:
            x = np.arange(len(patch_ids))
            labels = [str(p) for p in patch_ids]

            # Top: core sigma (resolution) per patch
            ax1.errorbar(x, core_sigmas, yerr=core_sigma_errs, fmt='o',
                         color='tab:blue', capsize=4, markersize=6)
            # Overall combined fit
            if comb_dg_popt is not None:
                ax1.axhline(abs(comb_dg_popt[2]), color='tab:red', ls='--', lw=1.5,
                            label=f"Combined core σ = {abs(comb_dg_popt[2]):.2f} MeV")
                ax1.legend()
            ax1.set_xticks(x)
            ax1.set_xticklabels(labels, fontsize=8)
            ax1.set_xlabel("Patch")
            ax1.set_ylabel("Core σ [MeV]")
            ax1.set_title("Resolution (Double Gaussian Core σ)")

            # Bottom: bias (core mu) per patch
            ax2.errorbar(x, core_mus, yerr=core_mu_errs, fmt='s',
                         color='tab:orange', capsize=4, markersize=6)
            if comb_dg_popt is not None:
                ax2.axhline(comb_dg_popt[1], color='tab:red', ls='--', lw=1.5,
                            label=f"Combined core μ = {comb_dg_popt[1]:+.2f} MeV")
                ax2.legend()
            ax2.axhline(0, color='black', ls='-', lw=0.5)
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels, fontsize=8)
            ax2.set_xlabel("Patch")
            ax2.set_ylabel("Core μ [MeV]")
            ax2.set_title("Bias (Double Gaussian Core μ)")

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # ============================================================
        # Page 2: Combined residual histogram with double Gaussian fit
        # ============================================================
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        counts, edges, _ = ax.hist(combined_residual, bins=100, alpha=0.7,
                                   color='tab:blue',
                                   label=f"N = {len(combined_residual)}")
        x_fit = np.linspace(edges[0], edges[-1], 300)
        if comb_dg_popt is not None:
            ax.plot(x_fit, _double_gaussian(x_fit, *comb_dg_popt), 'r-', lw=2,
                    label=(f"Double Gaussian\n"
                           f"  core: μ={comb_dg_popt[1]:.2f}, "
                           f"σ={abs(comb_dg_popt[2]):.2f} MeV\n"
                           f"  tail: μ={comb_dg_popt[4]:.2f}, "
                           f"σ={abs(comb_dg_popt[5]):.2f} MeV"))
            # Show individual components
            ax.plot(x_fit, _gaussian(x_fit, *comb_dg_popt[:3]),
                    'r--', lw=1, alpha=0.6, label="Core")
            ax.plot(x_fit, _gaussian(x_fit, *comb_dg_popt[3:]),
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
                counts, edges, _ = ax.hist(res, bins='auto', alpha=0.7,
                                           color='tab:blue')
                x_fit = np.linspace(edges[0], edges[-1], 200)
                if dg_popt is not None:
                    ax.plot(x_fit, _double_gaussian(x_fit, *dg_popt),
                            'r-', lw=1.5)
                    ax.plot(x_fit, _gaussian(x_fit, *dg_popt[:3]),
                            'r--', lw=1, alpha=0.5)
                    ax.plot(x_fit, _gaussian(x_fit, *dg_popt[3:]),
                            'r:', lw=1, alpha=0.5)
                    core_sig_err = np.sqrt(dg_pcov[2, 2])
                    ax.set_title(
                        f"Patch {pid} (N={n_ev})\n"
                        f"core: μ={dg_popt[1]:.2f}, "
                        f"σ={abs(dg_popt[2]):.2f}±{core_sig_err:.2f} MeV",
                        fontsize=10)
                else:
                    ax.set_title(f"Patch {pid} (N={n_ev})\n"
                                 f"Double-G fit failed", fontsize=10)
                ax.set_xlabel("Pred − True [MeV]", fontsize=9)
                ax.set_ylabel("Events", fontsize=9)

            for j in range(len(page_items), len(axes)):
                axes[j].axis('off')

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

    print(f"\nPlots: {pdf_path}")


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
    args = parser.parse_args()

    if not HAS_PANDAS:
        print("[ERROR] pandas is required. Install with: pip install pandas")
        sys.exit(1)

    patches = args.patches or ALL_PATCHES
    input_base = args.input_base

    print(f"Input base: {input_base}")
    print(f"Patches:    {patches}")
    print()

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

                if dg_popt is not None:
                    core_sig = abs(dg_popt[2])
                    core_sig_err = np.sqrt(dg_pcov[2, 2])
                    core_mu = dg_popt[1]
                    # Core fraction from amplitudes
                    f_core = abs(dg_popt[0] * dg_popt[2]) / (
                        abs(dg_popt[0] * dg_popt[2]) + abs(dg_popt[3] * dg_popt[5]))
                    print(f"  Patch {patch:>2d}: {n:>6d} events | "
                          f"core μ={core_mu:+.2f} MeV | "
                          f"core σ={core_sig:.2f}±{core_sig_err:.2f} MeV | "
                          f"f_core={f_core:.0%}")
                elif sg_popt is not None:
                    sig_err = np.sqrt(sg_pcov[2, 2])
                    print(f"  Patch {patch:>2d}: {n:>6d} events | "
                          f"μ={sg_popt[1]:+.2f} MeV | "
                          f"σ={abs(sg_popt[2]):.2f}±{sig_err:.2f} MeV (single)")
                else:
                    print(f"  Patch {patch:>2d}: {n:>6d} events | "
                          f"fit failed, "
                          f"res68={np.percentile(np.abs(residual), 68):.2f} MeV")
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
            f["tree"] = branches
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
