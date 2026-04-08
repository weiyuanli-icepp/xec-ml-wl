#!/usr/bin/env python3
"""
Compare lib/sensor_directions.txt (used by our Python SolidAngleComputer)
against the XECPMRunHeader geometry dumped from a rec file.

Reads:
  - Meganalyzer dump (from macro/dump_xec_geometry.C):
        # id is_sipm face x y z dir_x dir_y dir_z
  - Our Python geometry (lib/sensor_directions.txt):
        # sensor_id  dir_x  dir_y  dir_z  pos_x  pos_y  pos_z  face

Checks, per sensor:
  - Position (x, y, z) difference
  - Direction (dir_x, dir_y, dir_z) difference
  - Face assignment match
  - SiPM/PMT boundary consistency with our SIPM_PMT_BOUNDARY=4596

Reports counts of mismatches and the worst-N entries.

Usage:
    python macro/compare_xec_geometry.py \\
        --meg xec_geom_559261.txt \\
        [--python lib/sensor_directions.txt] \\
        [--pos-tol 1e-3] [--dir-tol 1e-4]
"""

import argparse
import sys
from pathlib import Path

import numpy as np


def load_meganalyzer_dump(path):
    """Load meganalyzer XECPMRunHeader dump.

    Returns dict with keys: id, is_sipm, face, xyz (N, 3), direction (N, 3).
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 9:
        raise RuntimeError(
            f"Unexpected shape for meganalyzer dump {path}: {data.shape}; "
            f"expected (N, 9)"
        )
    return {
        "id": data[:, 0].astype(np.int32),
        "is_sipm": data[:, 1].astype(np.int32),
        "face": data[:, 2].astype(np.int32),
        "xyz": data[:, 3:6],
        "direction": data[:, 6:9],
    }


def load_python_geometry(path):
    """Load lib/sensor_directions.txt used by our SolidAngleComputer.

    Format: sensor_id dir_x dir_y dir_z pos_x pos_y pos_z face

    Returns dict with keys: id, face, xyz (N, 3), direction (N, 3).
    """
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] != 8:
        raise RuntimeError(
            f"Unexpected shape for Python geometry {path}: {data.shape}; "
            f"expected (N, 8)"
        )
    return {
        "id": data[:, 0].astype(np.int32),
        "direction": data[:, 1:4],
        "xyz": data[:, 4:7],
        "face": data[:, 7].astype(np.int32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare meganalyzer XECPMRunHeader dump vs "
                    "lib/sensor_directions.txt"
    )
    parser.add_argument("--meg", required=True,
                        help="Meganalyzer dump file (from dump_xec_geometry.C)")
    parser.add_argument(
        "--python", default=None,
        help="Our Python geometry file (default: lib/sensor_directions.txt)",
    )
    parser.add_argument("--pos-tol", type=float, default=1e-3,
                        help="Position tolerance in cm (default: 1e-3)")
    parser.add_argument("--dir-tol", type=float, default=1e-4,
                        help="Direction unit-vector tolerance (default: 1e-4)")
    parser.add_argument("--max-print", type=int, default=20,
                        help="Number of worst mismatches to print per category")
    parser.add_argument("--sipm-pmt-boundary", type=int, default=4596,
                        help="Python's SIPM_PMT_BOUNDARY constant "
                             "(sensors >= this are PMTs)")
    args = parser.parse_args()

    if args.python is None:
        args.python = str(
            Path(__file__).resolve().parent.parent / "lib" / "sensor_directions.txt"
        )

    print(f"[INFO] Meganalyzer dump: {args.meg}")
    print(f"[INFO] Python geometry : {args.python}")
    print(f"[INFO] Position tol    : {args.pos_tol} cm")
    print(f"[INFO] Direction tol   : {args.dir_tol}")
    print(f"[INFO] Python SIPM_PMT_BOUNDARY = {args.sipm_pmt_boundary}")

    meg = load_meganalyzer_dump(args.meg)
    pyg = load_python_geometry(args.python)

    # --- Align by sensor id ---
    n_meg = len(meg["id"])
    n_pyg = len(pyg["id"])
    print(f"\n[INFO] Meganalyzer: {n_meg} sensors")
    print(f"[INFO] Python     : {n_pyg} sensors")

    if n_meg != n_pyg:
        print(f"[WARN] Count mismatch: {n_meg} vs {n_pyg}")

    # Build id -> index maps
    meg_idx = {int(sid): i for i, sid in enumerate(meg["id"])}
    pyg_idx = {int(sid): i for i, sid in enumerate(pyg["id"])}

    common_ids = sorted(set(meg_idx.keys()) & set(pyg_idx.keys()))
    only_meg = sorted(set(meg_idx.keys()) - set(pyg_idx.keys()))
    only_pyg = sorted(set(pyg_idx.keys()) - set(meg_idx.keys()))

    print(f"[INFO] Common sensors : {len(common_ids)}")
    if only_meg:
        print(f"[WARN] Only in meganalyzer: {len(only_meg)} "
              f"(first 10: {only_meg[:10]})")
    if only_pyg:
        print(f"[WARN] Only in Python    : {len(only_pyg)} "
              f"(first 10: {only_pyg[:10]})")

    if not common_ids:
        print("[ERROR] No common sensors — cannot compare")
        sys.exit(1)

    # Build aligned arrays
    mi = np.array([meg_idx[i] for i in common_ids])
    pi = np.array([pyg_idx[i] for i in common_ids])
    ids = np.asarray(common_ids, dtype=np.int32)

    meg_xyz = meg["xyz"][mi]
    pyg_xyz = pyg["xyz"][pi]
    meg_dir = meg["direction"][mi]
    pyg_dir = pyg["direction"][pi]
    meg_face = meg["face"][mi]
    pyg_face = pyg["face"][pi]
    meg_issipm = meg["is_sipm"][mi]

    # --- Position comparison ---
    pos_diff = np.linalg.norm(meg_xyz - pyg_xyz, axis=1)
    n_pos_bad = int((pos_diff > args.pos_tol).sum())

    print("\n" + "=" * 60)
    print("Position comparison")
    print("=" * 60)
    print(f"  ||Δxyz|| median      : {np.median(pos_diff):.6g} cm")
    print(f"  ||Δxyz|| max         : {pos_diff.max():.6g} cm")
    print(f"  N |Δxyz| > {args.pos_tol:g} cm : {n_pos_bad:,}")
    if n_pos_bad > 0:
        worst = np.argsort(pos_diff)[::-1][:args.max_print]
        print(f"\n  Worst {min(args.max_print, n_pos_bad)}:")
        print(f"  {'id':>6s} {'meg_xyz':>30s} {'pyg_xyz':>30s} "
              f"{'||Δ||':>12s}")
        for k in worst:
            if pos_diff[k] <= args.pos_tol:
                break
            mxyz = meg_xyz[k]
            pxyz = pyg_xyz[k]
            print(f"  {ids[k]:>6d} "
                  f"({mxyz[0]:>8.4f},{mxyz[1]:>8.4f},{mxyz[2]:>8.4f}) "
                  f"({pxyz[0]:>8.4f},{pxyz[1]:>8.4f},{pxyz[2]:>8.4f}) "
                  f"{pos_diff[k]:>12.6g}")

    # --- Direction comparison ---
    dir_diff = np.linalg.norm(meg_dir - pyg_dir, axis=1)
    # Also check 1 - |cos θ| for robustness (handles normalization differences)
    dots = np.sum(meg_dir * pyg_dir, axis=1)
    meg_mag = np.linalg.norm(meg_dir, axis=1)
    pyg_mag = np.linalg.norm(pyg_dir, axis=1)
    # Guard zero-magnitude
    denom = np.where((meg_mag > 0) & (pyg_mag > 0), meg_mag * pyg_mag, 1.0)
    cos_theta = np.clip(dots / denom, -1.0, 1.0)
    ang_diff = 1.0 - cos_theta  # 0 = parallel; 2 = antiparallel

    n_dir_bad = int((dir_diff > args.dir_tol).sum())

    print("\n" + "=" * 60)
    print("Direction comparison")
    print("=" * 60)
    print(f"  ||Δdir|| median      : {np.median(dir_diff):.6g}")
    print(f"  ||Δdir|| max         : {dir_diff.max():.6g}")
    print(f"  (1 - cos θ) max      : {ang_diff.max():.6g}")
    print(f"  N |Δdir| > {args.dir_tol:g}   : {n_dir_bad:,}")
    if n_dir_bad > 0:
        worst = np.argsort(dir_diff)[::-1][:args.max_print]
        print(f"\n  Worst {min(args.max_print, n_dir_bad)}:")
        print(f"  {'id':>6s} {'meg_dir':>30s} {'pyg_dir':>30s} "
              f"{'||Δ||':>12s}")
        for k in worst:
            if dir_diff[k] <= args.dir_tol:
                break
            md = meg_dir[k]
            pd = pyg_dir[k]
            print(f"  {ids[k]:>6d} "
                  f"({md[0]:>+8.5f},{md[1]:>+8.5f},{md[2]:>+8.5f}) "
                  f"({pd[0]:>+8.5f},{pd[1]:>+8.5f},{pd[2]:>+8.5f}) "
                  f"{dir_diff[k]:>12.6g}")

    # --- Face comparison ---
    face_mismatch = (meg_face != pyg_face)
    n_face_bad = int(face_mismatch.sum())

    print("\n" + "=" * 60)
    print("Face assignment comparison")
    print("=" * 60)
    print(f"  N face mismatches    : {n_face_bad:,}")
    if n_face_bad > 0:
        # Confusion table
        face_names = {
            0: "inner", 1: "outer", 2: "us", 3: "ds", 4: "top", 5: "bot"
        }
        print(f"\n  Confusion (meg_face -> pyg_face) counts:")
        bad = np.where(face_mismatch)[0]
        pairs = {}
        for k in bad:
            pair = (int(meg_face[k]), int(pyg_face[k]))
            pairs[pair] = pairs.get(pair, 0) + 1
        for (mf, pf), cnt in sorted(pairs.items(), key=lambda x: -x[1]):
            mn = face_names.get(mf, str(mf))
            pn = face_names.get(pf, str(pf))
            print(f"    {mn:>6s}({mf}) -> {pn:>6s}({pf}): {cnt:>6,}")

        # Show first N mismatched sensor ids
        k_show = min(args.max_print, n_face_bad)
        print(f"\n  First {k_show} mismatched sensor ids:")
        print(f"  {'id':>6s} {'meg_face':>10s} {'pyg_face':>10s}")
        for k in bad[:k_show]:
            mn = face_names.get(int(meg_face[k]), "?")
            pn = face_names.get(int(pyg_face[k]), "?")
            print(f"  {ids[k]:>6d} {mn:>8s}({meg_face[k]}) "
                  f"{pn:>8s}({pyg_face[k]})")

    # --- SiPM/PMT boundary check ---
    print("\n" + "=" * 60)
    print("SiPM/PMT classification check")
    print("=" * 60)
    # Per meganalyzer
    meg_pmt = (meg_issipm == 0)
    meg_sipm = (meg_issipm == 1)
    n_meg_sipm = int(meg_sipm.sum())
    n_meg_pmt = int(meg_pmt.sum())
    print(f"  Meganalyzer: {n_meg_sipm:,} SiPMs, {n_meg_pmt:,} PMTs")

    # Per our Python convention (sensor_id >= boundary -> PMT)
    pyg_pmt_pred = (ids >= args.sipm_pmt_boundary)
    pyg_sipm_pred = ~pyg_pmt_pred
    n_pyg_sipm = int(pyg_sipm_pred.sum())
    n_pyg_pmt = int(pyg_pmt_pred.sum())
    print(f"  Python (id >= {args.sipm_pmt_boundary}): "
          f"{n_pyg_sipm:,} SiPMs, {n_pyg_pmt:,} PMTs")

    # Disagreement
    sipm_mismatch = (meg_sipm & pyg_pmt_pred) | (meg_pmt & pyg_sipm_pred)
    n_sipm_mismatch = int(sipm_mismatch.sum())
    print(f"  Classification mismatches: {n_sipm_mismatch:,}")
    if n_sipm_mismatch > 0:
        # Show which sensor ids are misclassified
        bad = np.where(sipm_mismatch)[0]
        k_show = min(args.max_print, n_sipm_mismatch)
        print(f"\n  First {k_show}:")
        print(f"  {'id':>6s} {'meg is_sipm':>12s} {'pyg expects':>12s}")
        for k in bad[:k_show]:
            exp = "SiPM" if pyg_sipm_pred[k] else "PMT"
            got = "SiPM" if meg_sipm[k] else "PMT"
            print(f"  {ids[k]:>6d} {got:>12s} {exp:>12s}")

        # Report the actual split point
        sipm_ids = ids[meg_sipm]
        pmt_ids = ids[meg_pmt]
        if len(sipm_ids) > 0 and len(pmt_ids) > 0:
            print(f"\n  Meganalyzer SiPM id range : "
                  f"[{sipm_ids.min()}, {sipm_ids.max()}]")
            print(f"  Meganalyzer PMT  id range : "
                  f"[{pmt_ids.min()}, {pmt_ids.max()}]")

    # --- Overall verdict ---
    print("\n" + "=" * 60)
    print("Overall")
    print("=" * 60)
    all_ok = (n_pos_bad == 0 and n_dir_bad == 0
              and n_face_bad == 0 and n_sipm_mismatch == 0)
    if all_ok:
        print("  ✓ Geometry matches within tolerances.")
    else:
        print("  ✗ Geometry mismatches found:")
        if n_pos_bad:
            print(f"    - {n_pos_bad} position mismatches")
        if n_dir_bad:
            print(f"    - {n_dir_bad} direction mismatches")
        if n_face_bad:
            print(f"    - {n_face_bad} face mismatches")
        if n_sipm_mismatch:
            print(f"    - {n_sipm_mismatch} SiPM/PMT classification mismatches")


if __name__ == "__main__":
    main()
