#!/usr/bin/env python3
"""
Compare ML energy regressor CEX results with the conventional method.

Reads:
  1. Conventional results from peakpos text file (run-based peak/resolution)
  2. ML results from combine_cex_results.py output (per-patch CSVs)

Produces a comparison PDF with resolution and bias vs patch for both methods.

Usage:
    python macro/compare_cex_conventional.py
    python macro/compare_cex_conventional.py --conv /path/to/peakpos_nsum2_2023CEX.txt
    python macro/compare_cex_conventional.py --ml-base val_data/cex --output comparison.pdf
"""

import argparse
import os
import sys
import glob

import numpy as np

try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas is required. Install with: pip install pandas")
    sys.exit(1)


# Conversion factor from ADC counts to MeV
ADC_TO_MEV = 0.0000015634

# CEX23 patch configs: (srun, nfiles, patch)
# From others/submit_cex.py
CEX23_CONFIGS = [
    (557545, 100, 13),
    (558304, 100, 12),
    (558394, 100, 21),
    (559081, 100, 20),
    (559171, 100,  5),
    (559862, 100,  4),
    (558991, 100, 22),
    (558214, 100, 14),
    (559772, 100,  6),
    (558900, 100, 19),
    (558124, 100, 11),
    (559682, 100,  3),
    (559261, 100,  1),
    (559498, 100,  2),
    (559592, 100,  7),
    (559408, 100,  8),
    (557628, 100,  9),
    (557809, 100, 10),
    (558034, 100, 15),
    (557718, 100, 16),
    (558484, 100, 17),
    (558717, 100, 18),
    (558807, 100, 23),
    (558575, 100, 24),
]


def run_to_patch(run_number):
    """Map a run number to its CEX patch."""
    for srun, nfiles, patch in CEX23_CONFIGS:
        if srun <= run_number < srun + nfiles:
            return patch
    return None


def parse_conventional(filepath):
    """Parse the conventional peakpos text file.

    Format:
        run 558114
        peak 35025938.57 36871.49947
        reso 2.964110395 0.07639236497

    Returns list of dicts with keys:
        run, patch, peak_mev, peak_err_mev, reso_pct, reso_err_pct
    """
    results = []
    current = {}

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'run':
                if current:
                    results.append(current)
                current = {'run': int(parts[1])}
            elif parts[0] == 'peak':
                current['peak_adc'] = float(parts[1])
                current['peak_err_adc'] = float(parts[2])
                current['peak_mev'] = float(parts[1]) * ADC_TO_MEV
                current['peak_err_mev'] = float(parts[2]) * ADC_TO_MEV
            elif parts[0] == 'reso':
                # reso = core_sigma / core_mean in percent
                current['reso_pct'] = float(parts[1])
                current['reso_err_pct'] = float(parts[2])

    if current:
        results.append(current)

    # Map runs to patches
    for r in results:
        r['patch'] = run_to_patch(r['run'])

    return results


def _gaussian(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def _double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    return (_gaussian(x, A1, mu1, sigma1)
            + _gaussian(x, A2, mu2, sigma2))


def fit_double_gaussian(values, nbins='auto'):
    """Fit a double Gaussian. Returns (popt, pcov) or (None, None).

    popt = [A_core, mu_core, sigma_core, A_tail, mu_tail, sigma_tail]
    where core is the component with the smaller |sigma|.
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
        p0 = [0.7 * A0, mu0, 0.6 * sig0,
              0.3 * A0, mu0, 2.0 * sig0]
        bounds_lo = [0, -np.inf, 1e-8, 0, -np.inf, 1e-8]
        bounds_hi = [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(
            _double_gaussian, centers, counts.astype(float),
            p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=10000,
        )
        # Ensure component 1 is core (narrower sigma)
        if abs(popt[5]) < abs(popt[2]):
            popt = np.array([popt[3], popt[4], popt[5],
                             popt[0], popt[1], popt[2]])
            idx = [3, 4, 5, 0, 1, 2]
            pcov = pcov[np.ix_(idx, idx)]
        return popt, pcov
    except Exception:
        return None, None


def load_ml_results(input_base, patches):
    """Load ML results from per-patch prediction CSVs.

    Returns list of dicts with keys:
        patch, n_events, sigma_mev, sigma_err_mev, mu_mev, mu_err_mev, res68_mev
    """
    results = []
    for patch in patches:
        patch_dir = os.path.join(input_base, f"patch{patch}")
        candidates = sorted(glob.glob(os.path.join(patch_dir, "predictions_energy_*.csv")))
        if not candidates:
            continue
        csv_path = candidates[-1]
        df = pd.read_csv(csv_path)
        if "pred_energy" not in df.columns or "true_energy" not in df.columns:
            continue
        valid = df["true_energy"] < 1e9
        if valid.sum() < 10:
            continue

        residual_mev = (df.loc[valid, "pred_energy"].values
                        - df.loc[valid, "true_energy"].values) * 1e3

        dg_popt, dg_pcov = fit_double_gaussian(residual_mev)
        res68 = np.percentile(np.abs(residual_mev), 68)

        entry = {
            'patch': patch,
            'n_events': int(valid.sum()),
            'res68_mev': res68,
        }
        # Use double Gaussian core sigma/mu
        if dg_popt is not None:
            entry['sigma_mev'] = abs(dg_popt[2])
            entry['sigma_err_mev'] = np.sqrt(dg_pcov[2, 2])
            entry['mu_mev'] = dg_popt[1]
            entry['mu_err_mev'] = np.sqrt(dg_pcov[1, 1])
            # Core fraction
            f_core = abs(dg_popt[0] * dg_popt[2]) / (
                abs(dg_popt[0] * dg_popt[2]) + abs(dg_popt[3] * dg_popt[5]))
            entry['core_frac'] = f_core
            entry['tail_sigma_mev'] = abs(dg_popt[5])
        results.append(entry)

    return results


def print_comparison(conv_by_patch, ml_by_patch, patches):
    """Print a formatted comparison table."""
    print("\n" + "=" * 90)
    print("  CEX23 ENERGY RESOLUTION: ML vs CONVENTIONAL")
    print("=" * 90)

    # Conventional: resolution is in %, peak is in MeV
    # ML: resolution is Gaussian sigma in MeV

    print(f"\n{'Patch':>5s}  {'Conv σ [MeV]':>14s}  {'Conv [%]':>10s}  "
          f"{'ML σ [MeV]':>14s}  {'ML [%]':>10s}  "
          f"{'ML μ [MeV]':>14s}  {'ML/Conv':>8s}")
    print("-" * 90)

    conv_sigmas = []
    conv_pcts = []
    ml_sigmas = []
    ml_pcts = []
    ratios = []

    # Approximate true energy for percentage calculation
    E_TRUE_APPROX = 54.9  # MeV

    for patch in sorted(patches):
        conv = conv_by_patch.get(patch)
        ml = ml_by_patch.get(patch)

        # Conventional: sigma = reso_pct / 100 * peak_mev
        if conv is not None:
            conv_sigma = conv['reso_pct'] / 100.0 * conv['peak_mev']
            conv_sigma_err = conv['reso_err_pct'] / 100.0 * conv['peak_mev']
            conv_str = f"{conv_sigma:.2f} ± {conv_sigma_err:.2f}"
            conv_pct_str = f"{conv['reso_pct']:.2f}"
            conv_sigmas.append(conv_sigma)
            conv_pcts.append(conv['reso_pct'])
        else:
            conv_str = "---"
            conv_pct_str = "---"
            conv_sigma = None

        if ml is not None and 'sigma_mev' in ml:
            ml_str = f"{ml['sigma_mev']:.2f} ± {ml['sigma_err_mev']:.2f}"
            ml_pct = ml['sigma_mev'] / E_TRUE_APPROX * 100.0
            ml_pct_str = f"{ml_pct:.2f}"
            ml_sigmas.append(ml['sigma_mev'])
            ml_pcts.append(ml_pct)
            ml_mu_str = f"{ml['mu_mev']:.2f} ± {ml['mu_err_mev']:.2f}"
        else:
            ml_str = "---"
            ml_pct_str = "---"
            ml_mu_str = "---"

        # Ratio
        if conv_sigma is not None and ml is not None and 'sigma_mev' in ml:
            ratio = ml['sigma_mev'] / conv_sigma
            ratio_str = f"{ratio:.3f}"
            ratios.append(ratio)
        else:
            ratio_str = "---"

        print(f"{patch:>5d}  {conv_str:>14s}  {conv_pct_str:>10s}  "
              f"{ml_str:>14s}  {ml_pct_str:>10s}  "
              f"{ml_mu_str:>14s}  {ratio_str:>8s}")

    print("-" * 90)

    # Averages
    parts = ["  Avg"]
    if conv_sigmas:
        parts.append(f"  Conv: {np.mean(conv_sigmas):.2f} MeV ({np.mean(conv_pcts):.2f}%)")
    if ml_sigmas:
        parts.append(f"  ML: {np.mean(ml_sigmas):.2f} MeV ({np.mean(ml_pcts):.2f}%)")
    if ratios:
        parts.append(f"  Ratio: {np.mean(ratios):.3f}")
    print("  ".join(parts))

    print("=" * 90)

    # Note about conventional resolution interpretation
    print("\nNote: Conv σ = reso_pct/100 × peak_mev (reso = fitted core σ / fitted core mean, in %)")
    print("      ML σ  = Double-Gaussian core σ of (pred - true) residual in MeV")
    print(f"      ADC→MeV conversion factor: {ADC_TO_MEV}")


def make_comparison_plot(conv_by_patch, ml_by_patch, patches, output_path):
    """Generate comparison PDF."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    sorted_patches = sorted(patches)

    with PdfPages(output_path) as pdf:
        # --- Page 1: Resolution comparison ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        fig.suptitle("CEX23 Energy Resolution: ML vs Conventional", fontsize=14)

        x_conv, y_conv, yerr_conv = [], [], []
        x_ml, y_ml, yerr_ml = [], [], []
        x_ml68, y_ml68 = [], []

        for i, patch in enumerate(sorted_patches):
            conv = conv_by_patch.get(patch)
            ml = ml_by_patch.get(patch)

            if conv is not None:
                sigma = conv['reso_pct'] / 100.0 * conv['peak_mev']
                sigma_err = conv['reso_err_pct'] / 100.0 * conv['peak_mev']
                x_conv.append(i)
                y_conv.append(sigma)
                yerr_conv.append(sigma_err)

            if ml is not None and 'sigma_mev' in ml:
                x_ml.append(i)
                y_ml.append(ml['sigma_mev'])
                yerr_ml.append(ml['sigma_err_mev'])
            if ml is not None:
                x_ml68.append(i)
                y_ml68.append(ml['res68_mev'])

        # Resolution (sigma)
        if x_conv:
            ax1.errorbar(np.array(x_conv) - 0.15, y_conv, yerr=yerr_conv,
                         fmt='s', color='tab:red', capsize=4, markersize=6,
                         label='Conventional')
        if x_ml:
            ax1.errorbar(np.array(x_ml) + 0.15, y_ml, yerr=yerr_ml,
                         fmt='o', color='tab:blue', capsize=4, markersize=6,
                         label='ML (Gaussian σ)')
        if x_ml68:
            ax1.plot(np.array(x_ml68) + 0.15, y_ml68,
                     '^', color='tab:green', markersize=5,
                     label='ML (68th pct)')
        ax1.set_xticks(range(len(sorted_patches)))
        ax1.set_xticklabels([str(p) for p in sorted_patches], fontsize=8)
        ax1.set_xlabel("Patch")
        ax1.set_ylabel("Resolution σ [MeV]")
        ax1.set_title("Resolution Comparison")
        ax1.legend(fontsize=9)

        # Bias (mu)
        x_conv_mu, y_conv_mu, yerr_conv_mu = [], [], []
        x_ml_mu, y_ml_mu, yerr_ml_mu = [], [], []

        # For conventional, bias = peak - 54.9 MeV (expected energy)
        E_EXPECTED = 54.9  # MeV — approximate expected CEX gamma energy
        for i, patch in enumerate(sorted_patches):
            conv = conv_by_patch.get(patch)
            ml = ml_by_patch.get(patch)

            if conv is not None:
                x_conv_mu.append(i)
                y_conv_mu.append(conv['peak_mev'] - E_EXPECTED)
                yerr_conv_mu.append(conv['peak_err_mev'])

            if ml is not None and 'mu_mev' in ml:
                x_ml_mu.append(i)
                y_ml_mu.append(ml['mu_mev'])
                yerr_ml_mu.append(ml['mu_err_mev'])

        if x_conv_mu:
            ax2.errorbar(np.array(x_conv_mu) - 0.15, y_conv_mu, yerr=yerr_conv_mu,
                         fmt='s', color='tab:red', capsize=4, markersize=6,
                         label='Conventional (peak − 54.9)')
        if x_ml_mu:
            ax2.errorbar(np.array(x_ml_mu) + 0.15, y_ml_mu, yerr=yerr_ml_mu,
                         fmt='o', color='tab:blue', capsize=4, markersize=6,
                         label='ML (residual μ)')
        ax2.axhline(0, color='black', ls='-', lw=0.5)
        ax2.set_xticks(range(len(sorted_patches)))
        ax2.set_xticklabels([str(p) for p in sorted_patches], fontsize=8)
        ax2.set_xlabel("Patch")
        ax2.set_ylabel("Bias [MeV]")
        ax2.set_title("Bias Comparison")
        ax2.legend(fontsize=9)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Resolution ratio (ML / Conv) ---
        fig, ax = plt.subplots(figsize=(12, 5))

        x_ratio, y_ratio, yerr_ratio = [], [], []
        for i, patch in enumerate(sorted_patches):
            conv = conv_by_patch.get(patch)
            ml = ml_by_patch.get(patch)
            if conv is not None and ml is not None and 'sigma_mev' in ml:
                conv_sigma = conv['reso_pct'] / 100.0 * conv['peak_mev']
                ratio = ml['sigma_mev'] / conv_sigma
                # Error propagation
                conv_sigma_err = conv['reso_err_pct'] / 100.0 * conv['peak_mev']
                ratio_err = ratio * np.sqrt(
                    (ml['sigma_err_mev'] / ml['sigma_mev']) ** 2
                    + (conv_sigma_err / conv_sigma) ** 2
                )
                x_ratio.append(i)
                y_ratio.append(ratio)
                yerr_ratio.append(ratio_err)

        if x_ratio:
            ax.errorbar(x_ratio, y_ratio, yerr=yerr_ratio,
                        fmt='o', color='tab:purple', capsize=4, markersize=6)
            ax.axhline(1.0, color='black', ls='--', lw=1, label='Equal')
            avg_ratio = np.mean(y_ratio)
            ax.axhline(avg_ratio, color='tab:purple', ls=':', lw=1,
                        label=f'Average = {avg_ratio:.2f}')
        ax.set_xticks(range(len(sorted_patches)))
        ax.set_xticklabels([str(p) for p in sorted_patches], fontsize=8)
        ax.set_xlabel("Patch")
        ax.set_ylabel("ML σ / Conv σ")
        ax.set_title("Resolution Ratio (ML / Conventional)")
        ax.legend(fontsize=10)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\nPlots: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare ML vs conventional CEX energy resolution",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--conv",
                        default="/data/project/meg/shared/subprojects/xec/shared/"
                                "peakpos_nsum2_2023CEX.txt",
                        help="Conventional results text file")
    parser.add_argument("--ml-base", default="val_data/cex",
                        help="ML results base directory (with patch*/ subdirs)")
    parser.add_argument("--output", default="val_data/cex/CEX23_ml_vs_conventional.pdf",
                        help="Output PDF path")
    parser.add_argument("--patches", type=int, nargs="*", default=None,
                        help="Specific patches (default: all available)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Print table only, skip PDF")
    args = parser.parse_args()

    # --- Load conventional results ---
    if not os.path.exists(args.conv):
        print(f"[ERROR] Conventional results file not found: {args.conv}")
        sys.exit(1)
    conv_data = parse_conventional(args.conv)
    print(f"Loaded {len(conv_data)} runs from conventional results")

    # Aggregate per patch (average peak and resolution across runs)
    conv_by_patch = {}
    patch_runs = {}
    for r in conv_data:
        patch = r.get('patch')
        if patch is None:
            continue
        if patch not in patch_runs:
            patch_runs[patch] = []
        patch_runs[patch].append(r)

    for patch, runs in patch_runs.items():
        peaks = np.array([r['peak_mev'] for r in runs])
        peak_errs = np.array([r['peak_err_mev'] for r in runs])
        resos = np.array([r['reso_pct'] for r in runs])
        reso_errs = np.array([r['reso_err_pct'] for r in runs])

        # Weighted average (weight = 1/err^2)
        if len(peaks) > 0:
            w_peak = 1.0 / peak_errs**2
            avg_peak = np.average(peaks, weights=w_peak)
            avg_peak_err = 1.0 / np.sqrt(np.sum(w_peak))

            w_reso = 1.0 / reso_errs**2
            avg_reso = np.average(resos, weights=w_reso)
            avg_reso_err = 1.0 / np.sqrt(np.sum(w_reso))

            conv_by_patch[patch] = {
                'peak_mev': avg_peak,
                'peak_err_mev': avg_peak_err,
                'reso_pct': avg_reso,
                'reso_err_pct': avg_reso_err,
                'n_runs': len(runs),
            }

    n_mapped = sum(1 for r in conv_data if r.get('patch') is not None)
    n_unmapped = len(conv_data) - n_mapped
    print(f"  Mapped to patches: {n_mapped} runs → {len(conv_by_patch)} patches")
    if n_unmapped > 0:
        print(f"  Unmapped runs: {n_unmapped} (run numbers outside CEX23 configs)")

    # --- Load ML results ---
    all_patches = list(range(1, 25))
    patches = args.patches or all_patches
    ml_data = load_ml_results(args.ml_base, patches)
    ml_by_patch = {r['patch']: r for r in ml_data}
    print(f"Loaded ML results for {len(ml_data)} patches")

    # --- Determine which patches to compare ---
    available_patches = sorted(
        set(conv_by_patch.keys()) | set(ml_by_patch.keys())
    )
    if args.patches:
        available_patches = [p for p in args.patches if p in available_patches]

    # --- Print comparison ---
    print_comparison(conv_by_patch, ml_by_patch, available_patches)

    # --- Plot ---
    if not args.no_plot:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        make_comparison_plot(conv_by_patch, ml_by_patch, available_patches,
                             args.output)


if __name__ == "__main__":
    main()
