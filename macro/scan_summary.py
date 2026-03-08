#!/usr/bin/env python3
"""
Summarize hyperparameter scan results from MLflow.

Usage:
    python macro/scan_summary.py --experiment gamma_angle
    python macro/scan_summary.py --experiment gamma_angle --prefix ang_scan
    python macro/scan_summary.py --experiment gamma_timing --prefix tim_scan
    python macro/scan_summary.py --experiment gamma_position --prefix pos_scan
    python macro/scan_summary.py --experiment gamma_angle --output scan_angle_summary.pdf
    python macro/scan_summary.py --experiment my_exp --task position
"""

import argparse
import os
import sys

import mlflow
import pandas as pd
import numpy as np


# Metrics to extract (at the final epoch of each run)
ANGLE_METRICS = [
    "val/loss", "train/loss",
    "val/l1", "val/cos",
    "theta_bias", "theta_rms", "theta_skew",
    "phi_bias", "phi_rms", "phi_skew",
    "angle_resolution_68pct",
    "train/grad_norm_max",
    "system/lr",
]

TIMING_METRICS = [
    "val/loss", "train/loss",
    "val/l1", "val/smooth_l1", "val/mse",
    "train/grad_norm_max",
    "system/lr",
]

ENERGY_METRICS = [
    "val/loss", "train/loss",
    "val/l1", "val/smooth_l1", "val/mse",
    "train/grad_norm_max",
    "system/lr",
]

POSITION_METRICS = [
    "val/loss", "train/loss",
    "val/l1", "val/smooth_l1", "val/mse", "val/cos_pos",
    "uvw_u_res_68pct", "uvw_v_res_68pct", "uvw_w_res_68pct", "uvw_dist_68pct",
    "uvw_u_bias", "uvw_v_bias", "uvw_w_bias", "uvw_dist_bias",
    "train/grad_norm_max",
    "system/lr",
]

# Key params to show
KEY_PARAMS = [
    "lr", "weight_decay", "grad_clip", "batch_size", "grad_accum_steps",
    "epochs", "loss_fn", "loss_beta",
    "encoder_dim", "num_fusion_layers", "drop_path_rate",
    "channel_dropout_rate", "lr_scheduler",
    "train_path",
]


def get_final_metrics(client, run_id, metric_keys):
    """Get the last logged value for each metric."""
    result = {}
    for key in metric_keys:
        try:
            history = client.get_metric_history(run_id, key)
            if history:
                # Get the value at the max step
                last = max(history, key=lambda m: m.step)
                result[key] = last.value
                result[f"{key}__step"] = last.step
        except Exception:
            pass
    return result


def get_best_val_loss(client, run_id):
    """Get the minimum val/loss across all epochs."""
    try:
        history = client.get_metric_history(run_id, "val/loss")
        if history:
            best = min(history, key=lambda m: m.value)
            return best.value, best.step
    except Exception:
        pass
    return None, None


def get_metric_history_df(client, run_id, metric_key):
    """Get full history of a metric as a list of (step, value) tuples."""
    try:
        history = client.get_metric_history(run_id, metric_key)
        return [(m.step, m.value) for m in sorted(history, key=lambda m: m.step)]
    except Exception:
        return []


def detect_task(experiment_name):
    """Detect task type from experiment name."""
    name = experiment_name.lower()
    if "angle" in name:
        return "angle"
    elif "timing" in name:
        return "timing"
    elif "position" in name or "uvw" in name:
        return "position"
    elif "energy" in name:
        return "energy"
    return "angle"  # default


def build_summary(experiment_name, prefix=None, tracking_uri=None):
    """Build a summary DataFrame from MLflow runs."""
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = mlflow.tracking.MlflowClient()

    # Find experiment
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"[ERROR] Experiment '{experiment_name}' not found.")
        print("Available experiments:")
        for exp in client.search_experiments():
            print(f"  - {exp.name}")
        sys.exit(1)

    # Get all runs
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
    )

    if prefix:
        runs = [r for r in runs if r.info.run_name and r.info.run_name.startswith(prefix)]

    if not runs:
        print(f"[ERROR] No runs found in experiment '{experiment_name}'"
              + (f" with prefix '{prefix}'" if prefix else ""))
        sys.exit(1)

    task = detect_task(experiment_name)
    if task == "angle":
        metric_keys = ANGLE_METRICS
    elif task == "timing":
        metric_keys = TIMING_METRICS
    elif task == "position":
        metric_keys = POSITION_METRICS
    else:
        metric_keys = ENERGY_METRICS

    rows = []
    for run in runs:
        run_name = run.info.run_name or run.info.run_id[:8]
        params = run.data.params

        # Final metrics
        final = get_final_metrics(client, run.info.run_id, metric_keys)

        # Best val/loss
        best_val, best_epoch = get_best_val_loss(client, run.info.run_id)

        # Final epoch
        final_epoch = final.get("val/loss__step", None)

        row = {"run_name": run_name, "run_id": run.info.run_id[:8]}

        # Key params
        for p in KEY_PARAMS:
            # MLflow stores params with various prefixes
            val = (params.get(p) or params.get(f"task/angle_{p}")
                   or params.get(f"task/energy_{p}") or params.get(f"task/timing_{p}"))
            if val is not None:
                row[p] = val

        row["final_epoch"] = int(final_epoch) if final_epoch else None
        row["best_val_loss"] = best_val
        row["best_epoch"] = int(best_epoch) if best_epoch else None

        # Train/val gap
        tr_loss = final.get("train/loss")
        val_loss = final.get("val/loss")
        if tr_loss is not None and val_loss is not None:
            row["train_loss"] = tr_loss
            row["val_loss"] = val_loss
            row["overfit_gap"] = val_loss - tr_loss
            row["overfit_ratio"] = val_loss / tr_loss if tr_loss > 0 else None

        # Task-specific metrics
        if task == "angle":
            for key in ["val/l1", "val/cos", "angle_resolution_68pct",
                        "theta_bias", "theta_rms", "theta_skew",
                        "phi_bias", "phi_rms", "phi_skew"]:
                row[key.replace("/", "_")] = final.get(key)
        elif task == "position":
            for key in ["val/l1", "val/cos_pos",
                        "uvw_u_res_68pct", "uvw_v_res_68pct", "uvw_w_res_68pct",
                        "uvw_dist_68pct",
                        "uvw_u_bias", "uvw_v_bias", "uvw_w_bias", "uvw_dist_bias"]:
                row[key.replace("/", "_")] = final.get(key)
        else:
            for key in ["val/l1", "val/smooth_l1", "val/mse"]:
                row[key.replace("/", "_")] = final.get(key)

        row["grad_norm_max"] = final.get("train/grad_norm_max")
        row["final_lr"] = final.get("system/lr")

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by run_name for consistent ordering
    df = df.sort_values("run_name").reset_index(drop=True)

    return df, task, client, runs


def print_summary(df, task):
    """Print a formatted summary table."""
    print("\n" + "=" * 100)
    print(f"  SCAN SUMMARY — {task.upper()} REGRESSOR")
    print("=" * 100)

    # --- Config differences ---
    print("\n--- Configuration ---")
    config_cols = [c for c in KEY_PARAMS if c in df.columns]
    # Only show columns that vary across runs
    varying = [c for c in config_cols if df[c].nunique() > 1]
    if varying:
        print(df[["run_name"] + varying].to_string(index=False))
    else:
        print("  (all runs have the same configuration)")

    # --- Performance ---
    print("\n--- Performance (sorted by best_val_loss) ---")
    perf_cols = ["run_name", "final_epoch", "best_val_loss", "best_epoch",
                 "train_loss", "val_loss", "overfit_gap"]
    if task == "angle":
        perf_cols += ["val_cos", "angle_resolution_68pct"]
    elif task == "position":
        perf_cols += ["uvw_dist_68pct"]
    available = [c for c in perf_cols if c in df.columns]
    perf_df = df[available].sort_values("best_val_loss")
    print(perf_df.to_string(index=False, float_format=lambda x: f"{x:.4e}" if abs(x) < 0.01 else f"{x:.4f}"))

    # --- Bias & Resolution (angle / position) ---
    if task == "angle":
        print("\n--- Bias & Skew (sorted by best_val_loss) ---")
        bias_cols = ["run_name", "theta_bias", "theta_rms", "theta_skew",
                     "phi_bias", "phi_rms", "phi_skew"]
        available = [c for c in bias_cols if c in df.columns]
        bias_df = df[available].sort_values("run_name")
        # Merge sort order from performance
        order = perf_df["run_name"].tolist()
        bias_df = bias_df.set_index("run_name").loc[order].reset_index()
        print(bias_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if task == "position":
        print("\n--- Resolution 68th pct (sorted by best_val_loss) ---")
        res_cols = ["run_name", "uvw_u_res_68pct", "uvw_v_res_68pct",
                    "uvw_w_res_68pct", "uvw_dist_68pct"]
        available = [c for c in res_cols if c in df.columns]
        res_df = df[available].sort_values("run_name")
        order = perf_df["run_name"].tolist()
        res_df = res_df.set_index("run_name").loc[order].reset_index()
        print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        print("\n--- Bias (sorted by best_val_loss) ---")
        bias_cols = ["run_name", "uvw_u_bias", "uvw_v_bias",
                     "uvw_w_bias", "uvw_dist_bias"]
        available = [c for c in bias_cols if c in df.columns]
        bias_df = df[available].sort_values("run_name")
        bias_df = bias_df.set_index("run_name").loc[order].reset_index()
        print(bias_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Gradient norms ---
    print("\n--- Training Diagnostics ---")
    diag_cols = ["run_name", "grad_norm_max", "final_lr", "overfit_ratio"]
    available = [c for c in diag_cols if c in df.columns]
    diag_df = df[available].sort_values("run_name")
    print(diag_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # --- Ranking ---
    print("\n--- Rankings ---")
    if "best_val_loss" in df.columns:
        ranked = df.sort_values("best_val_loss")[["run_name", "best_val_loss"]].reset_index(drop=True)
        ranked.index += 1
        ranked.index.name = "rank"
        print(ranked.to_string(float_format=lambda x: f"{x:.4e}" if abs(x) < 0.01 else f"{x:.4f}"))

    print("\n" + "=" * 100)


def make_plots(df, task, client, runs, output_path, prefix=None):
    """Generate comparison plots as a PDF."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError:
        print("[WARN] matplotlib not available, skipping plots.")
        return

    # Collect loss histories for all runs
    run_histories = {}
    for run in runs:
        name = run.info.run_name or run.info.run_id[:8]
        if prefix and not name.startswith(prefix):
            continue
        mc = mlflow.tracking.MlflowClient()
        val_hist = get_metric_history_df(mc, run.info.run_id, "val/loss")
        tr_hist = get_metric_history_df(mc, run.info.run_id, "train/loss")
        run_histories[name] = {"val": val_hist, "train": tr_hist}

        if task == "angle":
            for key in ["val/cos", "angle_resolution_68pct",
                        "theta_bias", "theta_rms", "phi_bias", "phi_rms"]:
                hist = get_metric_history_df(mc, run.info.run_id, key)
                run_histories[name][key] = hist

        if task == "position":
            for key in ["uvw_dist_68pct", "uvw_u_res_68pct", "uvw_v_res_68pct",
                        "uvw_w_res_68pct", "uvw_u_bias", "uvw_v_bias", "uvw_w_bias"]:
                hist = get_metric_history_df(mc, run.info.run_id, key)
                run_histories[name][key] = hist

    sorted_names = sorted(run_histories.keys())

    with PdfPages(output_path) as pdf:
        # --- Page 1: Val loss curves ---
        fig, ax = plt.subplots(figsize=(12, 7))
        for name in sorted_names:
            hist = run_histories[name]["val"]
            if hist:
                steps, vals = zip(*hist)
                ax.plot(steps, vals, label=name, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation Loss")
        ax.set_title("Validation Loss Curves")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # --- Page 2: Train vs Val (overfit check) ---
        fig, ax = plt.subplots(figsize=(12, 7))
        for name in sorted_names:
            tr = run_histories[name]["train"]
            vl = run_histories[name]["val"]
            if tr and vl:
                tr_steps, tr_vals = zip(*tr)
                vl_steps, vl_vals = zip(*vl)
                ax.plot(tr_steps, tr_vals, linestyle="--", alpha=0.6, linewidth=1)
                ax.plot(vl_steps, vl_vals, label=name, linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Train (dashed) vs Val (solid) Loss")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        if task == "angle":
            # --- Page 3: Angular resolution ---
            fig, ax = plt.subplots(figsize=(12, 7))
            for name in sorted_names:
                hist = run_histories[name].get("angle_resolution_68pct", [])
                if hist:
                    steps, vals = zip(*hist)
                    ax.plot(steps, vals, label=name, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Angular Resolution (68th pct) [deg]")
            ax.set_title("Angular Resolution vs Epoch")
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --- Page 4: Cosine similarity ---
            fig, ax = plt.subplots(figsize=(12, 7))
            for name in sorted_names:
                hist = run_histories[name].get("val/cos", [])
                if hist:
                    steps, vals = zip(*hist)
                    ax.plot(steps, vals, label=name, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Cosine Loss (1 - cos_sim)")
            ax.set_title("Cosine Loss vs Epoch")
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --- Page 5: Theta bias & RMS ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for name in sorted_names:
                hist = run_histories[name].get("theta_bias", [])
                if hist:
                    steps, vals = zip(*hist)
                    axes[0].plot(steps, vals, label=name, linewidth=1.5)
            axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Theta Bias [deg]")
            axes[0].set_title("Theta Bias vs Epoch")
            axes[0].legend(fontsize=6, loc="best")

            for name in sorted_names:
                hist = run_histories[name].get("theta_rms", [])
                if hist:
                    steps, vals = zip(*hist)
                    axes[1].plot(steps, vals, label=name, linewidth=1.5)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Theta RMS [deg]")
            axes[1].set_title("Theta RMS vs Epoch")
            axes[1].legend(fontsize=6, loc="best")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --- Page 6: Phi bias & RMS ---
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            for name in sorted_names:
                hist = run_histories[name].get("phi_bias", [])
                if hist:
                    steps, vals = zip(*hist)
                    axes[0].plot(steps, vals, label=name, linewidth=1.5)
            axes[0].axhline(0, color="gray", linestyle=":", linewidth=0.8)
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Phi Bias [deg]")
            axes[0].set_title("Phi Bias vs Epoch")
            axes[0].legend(fontsize=6, loc="best")

            for name in sorted_names:
                hist = run_histories[name].get("phi_rms", [])
                if hist:
                    steps, vals = zip(*hist)
                    axes[1].plot(steps, vals, label=name, linewidth=1.5)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Phi RMS [deg]")
            axes[1].set_title("Phi RMS vs Epoch")
            axes[1].legend(fontsize=6, loc="best")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        if task == "position":
            # --- Position: Distance resolution ---
            fig, ax = plt.subplots(figsize=(12, 7))
            for name in sorted_names:
                hist = run_histories[name].get("uvw_dist_68pct", [])
                if hist:
                    steps, vals = zip(*hist)
                    ax.plot(steps, vals, label=name, linewidth=1.5)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("3D Distance Resolution (68th pct) [cm]")
            ax.set_title("Position Resolution vs Epoch")
            ax.legend(fontsize=7, loc="upper right")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --- Position: Per-axis resolution ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for i, coord in enumerate(["u", "v", "w"]):
                for name in sorted_names:
                    hist = run_histories[name].get(f"uvw_{coord}_res_68pct", [])
                    if hist:
                        steps, vals = zip(*hist)
                        axes[i].plot(steps, vals, label=name, linewidth=1.5)
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(f"{coord.upper()} Resolution (68th pct) [cm]")
                axes[i].set_title(f"{coord.upper()} Resolution vs Epoch")
                axes[i].legend(fontsize=6, loc="best")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # --- Position: Per-axis bias ---
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            for i, coord in enumerate(["u", "v", "w"]):
                for name in sorted_names:
                    hist = run_histories[name].get(f"uvw_{coord}_bias", [])
                    if hist:
                        steps, vals = zip(*hist)
                        axes[i].plot(steps, vals, label=name, linewidth=1.5)
                axes[i].axhline(0, color="gray", linestyle=":", linewidth=0.8)
                axes[i].set_xlabel("Epoch")
                axes[i].set_ylabel(f"{coord.upper()} Bias [cm]")
                axes[i].set_title(f"{coord.upper()} Bias vs Epoch")
                axes[i].legend(fontsize=6, loc="best")
            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # --- Final page: Bar chart of best val loss ---
        fig, ax = plt.subplots(figsize=(12, 7))
        sorted_df = df.dropna(subset=["best_val_loss"]).sort_values("best_val_loss")
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
        bars = ax.barh(range(len(sorted_df)), sorted_df["best_val_loss"], color=colors)
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df["run_name"], fontsize=8)
        ax.set_xlabel("Best Validation Loss")
        ax.set_title("Run Ranking by Best Validation Loss")
        ax.invert_yaxis()
        # Add value labels
        for bar, val in zip(bars, sorted_df["best_val_loss"]):
            ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                    f" {val:.4f}", va="center", fontsize=8)
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n[INFO] Plots saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Summarize MLflow scan results")
    parser.add_argument("--experiment", required=True,
                        help="MLflow experiment name (e.g., gamma_angle)")
    parser.add_argument("--prefix", default=None,
                        help="Filter runs by name prefix (e.g., ang_scan)")
    parser.add_argument("--tracking-uri", default=None,
                        help="MLflow tracking URI (default: sqlite:///mlruns.db)")
    parser.add_argument("--output", default=None,
                        help="Output PDF path (default: scan_<experiment>_summary.pdf)")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip PDF generation, print table only")
    parser.add_argument("--task", default=None,
                        choices=["angle", "timing", "energy", "position"],
                        help="Override task type (default: auto-detect from experiment name)")
    args = parser.parse_args()

    uri = args.tracking_uri or f"sqlite:///{os.path.join(os.getcwd(), 'mlruns.db')}"
    mlflow.set_tracking_uri(uri)

    df, task, client, runs = build_summary(
        args.experiment, prefix=args.prefix, tracking_uri=uri
    )

    if args.task:
        task = args.task

    print_summary(df, task)

    if not args.no_plot:
        output = args.output or f"scan_{args.experiment.replace('/', '_')}_summary.pdf"
        make_plots(df, task, client, runs, output, prefix=args.prefix)


if __name__ == "__main__":
    main()
