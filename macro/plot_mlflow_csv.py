#!/usr/bin/env python3
"""
Plot training curves from MLflow CSV export.

Usage:
    python macro/plot_mlflow_csv.py ~/Downloads/train_loss_npho-val_loss_npho.csv
    python macro/plot_mlflow_csv.py file1.csv file2.csv  # overlay multiple files
    python macro/plot_mlflow_csv.py data.csv -o output.pdf
    python macro/plot_mlflow_csv.py data.csv --ylabelstyle decade   # 1e-2, 1e-3 ticks
    python macro/plot_mlflow_csv.py data.csv --ylabelstyle scaled   # x10^-3 with 1, 2, 3
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def main():
    parser = argparse.ArgumentParser(description="Plot MLflow CSV exports")
    parser.add_argument("inputs", nargs="+", help="CSV file(s) exported from MLflow")
    parser.add_argument("-o", "--output", default=None, help="Output PDF path (default: auto)")
    parser.add_argument("--xlim", nargs=2, type=float, default=None, metavar=("MIN", "MAX"),
                        help="X-axis limits (e.g. --xlim 5 50)")
    parser.add_argument("--ylabelstyle", default="decade", choices=["decade", "scaled"],
                        help="Y-axis label style: 'decade' (1e-2, 1e-3) or 'scaled' (x10^-N with plain numbers)")
    parser.add_argument("--title", default=None, help="Plot title")
    parser.add_argument("--chain", action="store_true",
                        help="Chain runs end-to-end: offset each run's epochs "
                             "so they appear as a continuous training curve")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Custom labels for each input file (default: Run column)")
    args = parser.parse_args()

    # Load and concatenate all CSV files
    dfs = []
    epoch_offset = 0
    for i, path in enumerate(args.inputs):
        chunk = pd.read_csv(path)
        # Apply custom labels if provided
        if args.labels and i < len(args.labels):
            chunk["Run"] = args.labels[i]
        # Chain mode: offset epochs so runs appear sequential
        if args.chain and i > 0:
            epoch_offset += dfs[-1]["step"].max() + 1
            chunk["step"] = chunk["step"] + epoch_offset
        dfs.append(chunk)
    df = pd.concat(dfs, ignore_index=True)

    # Get unique runs and metrics
    runs = df["Run"].unique()
    metrics = df["metric"].unique()

    # Group metrics by base name (strip train/val prefix) for subplot layout
    base_metrics = {}
    for m in metrics:
        base = m.replace("train/", "").replace("val/", "")
        base_metrics.setdefault(base, []).append(m)

    n_bases = len(base_metrics)
    fig, axes = plt.subplots(1, n_bases, figsize=(7 * n_bases, 5), squeeze=False)
    axes = axes[0]

    # Color cycle per run
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for ax_idx, (base, metric_list) in enumerate(sorted(base_metrics.items())):
        ax = axes[ax_idx]

        # Determine scale factor for "scaled" mode
        if args.ylabelstyle == "scaled":
            # Find the order of magnitude from visible data
            if args.xlim:
                xlo, xhi = args.xlim
                vis = df[(df["step"] >= xlo) & (df["step"] <= xhi)]
            else:
                vis = df
            median_val = vis["value"].median()
            exponent = int(np.floor(np.log10(median_val)))
            scale = 10 ** (-exponent)
        else:
            scale = 1

        for run_idx, run in enumerate(runs):
            color = colors[run_idx % len(colors)]

            for metric in sorted(metric_list):
                mask = (df["Run"] == run) & (df["metric"] == metric)
                sub = df[mask].sort_values("step")
                if sub.empty:
                    continue

                is_val = metric.startswith("val/")
                marker = "s" if is_val else "o"
                prefix = "val" if is_val else "train"
                # Use per-run colors when multiple runs; fixed colors for single run
                if len(runs) > 1:
                    line_color = color
                    label = f"{run} ({prefix})"
                else:
                    line_color = "#e74c3c" if is_val else "#2c3e50"
                    label = prefix

                ax.plot(sub["step"], sub["value"] * scale, marker=marker,
                        color=line_color, linestyle="none", markersize=8,
                        alpha=0.4, label=label)

                # Smoothed trend line (exponential moving average in log space)
                if len(sub) > 3:
                    from scipy.ndimage import uniform_filter1d
                    win = min(15, len(sub) // 2)
                    log_vals = np.log(sub["value"].values * scale)
                    smoothed = np.exp(uniform_filter1d(log_vals, size=win))
                    skip = win // 2  # skip boundary-affected points
                    ax.plot(sub["step"].values[skip:], smoothed[skip:], linestyle="-",
                            color=line_color, linewidth=2, alpha=0.9)

        ax.set_xlabel("Epoch", fontsize=15)
        ax.set_ylabel("Loss", fontsize=15)
        ax.set_title(args.title or "Training Loss", fontsize=15)
        ax.legend(fontsize=15, frameon=False)
        ax.tick_params(axis="both", labelsize=15)
        ax.grid(False)
        ax.set_yscale("log")

        # Set axis limits first
        if args.xlim:
            ax.set_xlim(args.xlim)
            xlo, xhi = args.xlim
            visible = df[(df["step"] >= xlo) & (df["step"] <= xhi)]
            if not visible.empty:
                ymin = visible["value"].min() * scale
                ymax = visible["value"].max() * scale
                margin = 0.1
                ax.set_ylim(ymin * (1 - margin), ymax * (1 + margin))

        # Then configure tick formatting
        if args.ylabelstyle == "decade":
            def _decade_fmt(v, _):
                if v <= 0:
                    return ""
                exp = int(np.round(np.log10(v)))
                return r"$10^{%d}$" % exp
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(_decade_fmt))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())
            ax.tick_params(axis="y", labelsize=11)
        else:
            # Scaled mode: annotate exponent, use plain number ticks
            ax.annotate(r"$\times 10^{%d}$" % exponent, xy=(0, 1.01),
                        xycoords="axes fraction", fontsize=15,
                        ha="left", va="bottom")
            # Pick nice ticks based on visible range
            ylo, yhi = ax.get_ylim()
            ratio = yhi / ylo if ylo > 0 else 10
            if ratio < 4:
                step = 0.5
            elif ratio < 8:
                step = 1.0
            else:
                step = 2.0
            major_ticks = []
            v = np.ceil(ylo / step) * step
            while v <= yhi:
                major_ticks.append(v)
                v += step
            while len(major_ticks) > 8:
                major_ticks = major_ticks[::2]
            ax.yaxis.set_major_locator(ticker.FixedLocator(major_ticks))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(
                lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    plt.tight_layout()
    output = args.output or args.inputs[0].rsplit(".", 1)[0] + ".pdf"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
