#!/usr/bin/env python3
"""Check solid-angle ratio distribution for sensorfront validation.

Verifies whether the omega_m/omega_n clamp at 5.0 in
SolidAngleWeightedBaseline is affecting results.

Usage:
    python macro/check_sa_ratios.py
    python macro/check_sa_ratios.py --shared-dir artifacts/sensorfront_shared
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.inpainter_baselines import build_neighbor_map


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-dir", default="artifacts/sensorfront_shared",
                        help="Path to sensorfront shared directory")
    parser.add_argument("--max-events", type=int, default=500,
                        help="Max events to sample for ratio computation")
    args = parser.parse_args()

    data_path = Path(args.shared_dir) / "_sensorfront_data.npz"
    manifest_path = Path(args.shared_dir) / "_sensorfront_manifest.npz"

    print(f"[INFO] Loading {data_path}")
    d = np.load(data_path)
    print(f"  Keys: {list(d.keys())}")

    sa = d["solid_angles"]
    print(f"  solid_angles shape: {sa.shape}")

    m = np.load(manifest_path)
    sid = m["matched_sid"]
    print(f"  matched_sid shape: {sid.shape}")

    # Solid angle of masked (sensorfront) sensors
    omega_m = sa[np.arange(len(sid)), sid]
    print(f"\nomega_m (sensorfront sensors):")
    print(f"  median = {np.median(omega_m):.6f}")
    print(f"  min    = {np.min(omega_m):.6f}")
    print(f"  max    = {np.max(omega_m):.6f}")
    print(f"  mean   = {np.mean(omega_m):.6f}")

    # Compute omega_m / omega_n ratios for nearest neighbors
    print(f"\n[INFO] Building neighbor map (k=1)...")
    nbrs = build_neighbor_map(k=1)

    n_sample = min(args.max_events, len(sid))
    ratios = []
    for i in range(n_sample):
        s = sid[i]
        om = omega_m[i]
        for n in nbrs.get(s, []):
            on = sa[i, n]
            if on > 0:
                ratios.append(om / on)
    ratios = np.array(ratios)

    print(f"\nomega_m / omega_n ratios ({len(ratios)} neighbor pairs, "
          f"{n_sample} events):")
    print(f"  median = {np.median(ratios):.3f}")
    print(f"  mean   = {np.mean(ratios):.3f}")
    print(f"  p90    = {np.percentile(ratios, 90):.3f}")
    print(f"  p95    = {np.percentile(ratios, 95):.3f}")
    print(f"  p99    = {np.percentile(ratios, 99):.3f}")
    print(f"  max    = {np.max(ratios):.3f}")
    print(f"  fraction > 2.0:  {(ratios > 2).mean():.3f}")
    print(f"  fraction > 5.0:  {(ratios > 5).mean():.3f}")
    print(f"  fraction > 10.0: {(ratios > 10).mean():.3f}")


if __name__ == "__main__":
    main()
