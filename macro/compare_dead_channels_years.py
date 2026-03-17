#!/usr/bin/env python3
"""
Compare dead channels across years using representative runs.

Usage:
    python macro/compare_dead_channels_years.py
    python macro/compare_dead_channels_years.py --plot  # also save bar chart PDF
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.db_utils import get_dead_channel_info
from lib.geom_defs import FACE_SENSOR_IDS

# Representative runs for each year/period
RUNS = [
    (391066, "2021"),
    (427120, "2022"),
    (520134, "2023"),
    (580904, "2024"),
    (600681, "2025 start"),
    (676170, "2025 end"),
]

FACES = ["inner", "outer", "us", "ds", "top", "bot"]
FACE_TOTALS = {face: len(ids) for face, ids in FACE_SENSOR_IDS.items()}
TOTAL_SENSORS = 4760


def main():
    parser = argparse.ArgumentParser(description="Compare dead channels across years")
    parser.add_argument("--plot", action="store_true", help="Save bar chart PDF")
    parser.add_argument("--output", "-o", default="dead_channel_comparison.pdf",
                        help="Output PDF path (default: dead_channel_comparison.pdf)")
    args = parser.parse_args()

    # Fetch dead channel info for all runs
    results = []
    for run, label in RUNS:
        print(f"Querying run {run} ({label})...")
        info = get_dead_channel_info(run)
        results.append((run, label, info))

    # --- Summary table ---
    print("\n" + "=" * 90)
    print("Dead Channel Comparison Across Years")
    print("=" * 90)

    # Header
    header = f"{'Period':<14s} {'Run':>7s} {'Total':>14s}"
    for face in FACES:
        header += f" | {face:>13s}"
    print(header)
    print("-" * len(header))

    for run, label, info in results:
        dead_set = set(info['dead_channels'])
        total_str = f"{info['n_dead']}/{TOTAL_SENSORS} ({info['dead_fraction']*100:.1f}%)"
        row = f"{label:<14s} {run:>7d} {total_str:>14s}"
        for face in FACES:
            face_ids = FACE_SENSOR_IDS[face]
            count = sum(1 for idx in face_ids if int(idx) in dead_set)
            total = FACE_TOTALS[face]
            pct = 100.0 * count / total
            row += f" | {count:>4d}/{total:>4d} {pct:>4.1f}%"
        print(row)

    print("=" * len(header))

    # --- Overlap analysis ---
    print("\n" + "=" * 60)
    print("Dead Channel Overlap (Jaccard similarity)")
    print("=" * 60)

    dead_sets = [(label, set(info['dead_channels'])) for _, label, info in results]
    header2 = f"{'':>14s}"
    for label, _ in dead_sets:
        header2 += f" {label:>11s}"
    print(header2)

    for i, (label_i, set_i) in enumerate(dead_sets):
        row = f"{label_i:>14s}"
        for j, (label_j, set_j) in enumerate(dead_sets):
            if i == j:
                row += f" {'---':>11s}"
            else:
                union = len(set_i | set_j)
                inter = len(set_i & set_j)
                jaccard = inter / union if union > 0 else 0
                row += f" {jaccard:>11.3f}"
        print(row)

    # --- New dead channels between consecutive periods ---
    print("\n" + "=" * 60)
    print("New / Recovered Dead Channels Between Periods")
    print("=" * 60)
    for i in range(1, len(results)):
        _, prev_label, prev_info = results[i - 1]
        _, curr_label, curr_info = results[i]
        prev_set = set(prev_info['dead_channels'])
        curr_set = set(curr_info['dead_channels'])
        new = curr_set - prev_set
        recovered = prev_set - curr_set
        print(f"  {prev_label} -> {curr_label}: +{len(new)} new, -{len(recovered)} recovered "
              f"(net {len(new) - len(recovered):+d})")

    # --- Plot ---
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[WARNING] matplotlib not available, skipping plot")
            return

        labels = [label for _, label, _ in results]
        n_dead = [info['n_dead'] for _, _, info in results]

        # Per-face dead counts
        face_dead = {face: [] for face in FACES}
        for _, _, info in results:
            dead_set = set(info['dead_channels'])
            for face in FACES:
                face_ids = FACE_SENSOR_IDS[face]
                count = sum(1 for idx in face_ids if int(idx) in dead_set)
                face_dead[face].append(count)

        fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

        # Top: total dead channels
        ax = axes[0]
        x = np.arange(len(labels))
        bars = ax.bar(x, n_dead, color='steelblue', edgecolor='black', linewidth=0.5)
        for bar, n in zip(bars, n_dead):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                    f"{n}\n({n/TOTAL_SENSORS*100:.1f}%)", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Dead Channels")
        ax.set_title("Total Dead Channels by Year")
        ax.set_ylim(0, max(n_dead) * 1.25)

        # Bottom: stacked per-face
        ax = axes[1]
        bottom = np.zeros(len(labels))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for face, color in zip(FACES, colors):
            counts = np.array(face_dead[face])
            ax.bar(x, counts, bottom=bottom, label=face, color=color,
                   edgecolor='black', linewidth=0.3)
            bottom += counts
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Dead Channels")
        ax.set_title("Dead Channels by Face")
        ax.legend(loc='upper left', ncol=3, fontsize=8)
        ax.set_ylim(0, max(n_dead) * 1.25)

        plt.tight_layout()
        plt.savefig(args.output, dpi=150)
        plt.close()
        print(f"\n[INFO] Saved plot to {args.output}")


if __name__ == "__main__":
    main()
