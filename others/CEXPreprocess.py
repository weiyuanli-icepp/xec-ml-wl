#!/usr/bin/env python3
"""
CEXPreprocess.py — Real-data CEX preprocessing for energy regressor validation.

Python replacement for CEXPreprocess.C. Uses uproot to read rec files
(no MEG framework or ROOT dictionaries needed).

Usage:
    python others/CEXPreprocess.py --srun 557545 --nfiles 1 --patch 13
    python others/CEXPreprocess.py --srun 557545 --nfiles 100 --patch 13 \
        --output-dir /my/output --dead-file data/dead_channels_run430000.txt
"""
import os
import sys
import argparse
import numpy as np
import uproot
import time

# Constants
N_CHANNELS = 4760

# π⁰ kinematics (GeV)
Epi0 = 0.1378    # π⁰ kinetic energy + mass at rest in MEG target
mpi0 = 0.13497   # π⁰ mass (GeV/c²)

# CEX23 energy window for 55 MeV peak (GeV)
E_MIN = 0.054
E_MAX = 0.057

REC_DIR = "/data/project/meg/offline/run"


def load_dead_channels(path):
    """Load dead channel indices from text file (one per line, # comments)."""
    channels = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                channels.append(int(line))
    return np.array(channels, dtype=np.int32)


def discover_branches(tree):
    """Print available branch keys for debugging."""
    print("[DEBUG] Available branches:")
    for key in sorted(tree.keys()):
        print(f"  {key}")


def process_run(iRun, dead_mask, out_arrays):
    """Process one rec file, appending selected events to out_arrays.

    Returns number of events selected, or -1 if file/tree not found.
    """
    filename = f"{REC_DIR}/{iRun // 1000:>3d}xxx/rec{iRun:05d}.root"

    if not os.path.exists(filename):
        return -1

    try:
        f = uproot.open(filename)
    except Exception as e:
        print(f"  Warning: cannot open {filename}: {e}")
        return -1

    if "rec" not in f:
        print(f"  Warning: 'rec' tree not found in {filename}")
        return -1

    rec = f["rec"]
    n_entries = rec.num_entries
    print(f"Run {iRun} ({filename})")

    # --- Read branches ---
    # uproot reads non-split object/TClonesArray members via StreamerInfo.
    # Discover what's available on first run if needed.
    available = set(rec.keys())

    # Find the right key names (uproot may use different naming conventions)
    # Try common patterns for the branches we need
    def find_key(patterns):
        for p in patterns:
            if p in available:
                return p
        return None

    key_mask = find_key(["eventheader./fmask", "eventheader.fmask",
                         "eventheader./eventheader.fmask"])
    key_egamma = find_key(["reco./fEGamma", "reco.fEGamma",
                           "reco./reco.fEGamma"])
    key_evstat = find_key(["reco./fEvstatGamma", "reco.fEvstatGamma",
                           "reco./reco.fEvstatGamma"])
    key_ugamma = find_key(["reco./fUGamma", "reco.fUGamma",
                           "reco./reco.fUGamma"])
    key_vgamma = find_key(["reco./fVGamma", "reco.fVGamma",
                           "reco./reco.fVGamma"])
    key_wgamma = find_key(["reco./fWGamma", "reco.fWGamma",
                           "reco./reco.fWGamma"])
    key_npho = find_key(["xeccl/xeccl.fnpho", "xeccl.fnpho",
                         "xeccl/fnpho"])
    key_nphe = find_key(["xeccl/xeccl.fnphe", "xeccl.fnphe",
                         "xeccl/fnphe"])
    key_tpm = find_key(["xeccl/xeccl.ftpm", "xeccl.ftpm",
                        "xeccl/ftpm"])
    key_openangle = find_key(["bgocexresult/bgocexresult.fopeningAngle",
                              "bgocexresult.fopeningAngle",
                              "bgocexresult/fopeningAngle"])
    key_bgoenergy = find_key(["bgocexresult/bgocexresult.fbgoEnergy",
                              "bgocexresult.fbgoEnergy",
                              "bgocexresult/fbgoEnergy"])

    # Check critical branches
    missing = []
    for name, key in [("trigger mask", key_mask), ("EGamma", key_egamma),
                      ("npho", key_npho), ("tpm", key_tpm),
                      ("openingAngle", key_openangle)]:
        if key is None:
            missing.append(name)

    if missing:
        print(f"  ERROR: Cannot find branches: {missing}")
        discover_branches(rec)
        return -1

    # Read all data at once
    read_keys = [k for k in [key_mask, key_egamma, key_evstat,
                             key_ugamma, key_vgamma, key_wgamma,
                             key_npho, key_nphe, key_tpm,
                             key_openangle, key_bgoenergy] if k is not None]

    arrays = rec.arrays(read_keys, library="np")

    mask_arr = arrays[key_mask]
    egamma_arr = arrays[key_egamma]
    npho_arr = arrays[key_npho]
    tpm_arr = arrays[key_tpm]
    openangle_arr = arrays[key_openangle]

    # Optional branches
    evstat_arr = arrays.get(key_evstat) if key_evstat else None
    ugamma_arr = arrays.get(key_ugamma) if key_ugamma else None
    vgamma_arr = arrays.get(key_vgamma) if key_vgamma else None
    wgamma_arr = arrays.get(key_wgamma) if key_wgamma else None
    nphe_arr = arrays.get(key_nphe) if key_nphe else None
    bgoenergy_arr = arrays.get(key_bgoenergy) if key_bgoenergy else None

    # --- Handle array shapes ---
    # For multi-dimensional members (e.g. fEGamma[nGamma]), take index 0
    def get_col0(arr):
        """Get first column if 2D, else return as-is."""
        if arr is not None and arr.ndim > 1:
            return arr[:, 0]
        return arr

    mask_vals = get_col0(mask_arr)
    egamma_vals = get_col0(egamma_arr)
    openangle_vals = get_col0(openangle_arr)
    bgoenergy_vals = get_col0(bgoenergy_arr) if bgoenergy_arr is not None else None

    evstat_vals = get_col0(evstat_arr) if evstat_arr is not None else None
    ugamma_vals = get_col0(ugamma_arr) if ugamma_arr is not None else None
    vgamma_vals = get_col0(vgamma_arr) if vgamma_arr is not None else None
    wgamma_vals = get_col0(wgamma_arr) if wgamma_arr is not None else None

    # npho/nphe/tpm: shape might be (N, 4760) or (N, 4760, nFit)
    if npho_arr.ndim == 3:
        npho_vals = npho_arr[:, :, 0]
    else:
        npho_vals = npho_arr

    if tpm_arr.ndim == 3:
        tpm_vals = tpm_arr[:, :, 0]
    else:
        tpm_vals = tpm_arr

    if nphe_arr is not None:
        if nphe_arr.ndim == 3:
            nphe_vals = nphe_arr[:, :, 0]
        else:
            nphe_vals = nphe_arr
    else:
        nphe_vals = np.full_like(npho_vals, 1e10)

    # --- Event selection (vectorized) ---
    # 1. Physics triggers only (50, 51)
    trig_ok = (mask_vals == 50) | (mask_vals == 51)

    # 2. Energy window
    energy_ok = (egamma_vals >= E_MIN) & (egamma_vals <= E_MAX)

    # 3. Opening angle → Etrue, must be physical (sqrtarg >= 0)
    cos_oa = np.cos(np.deg2rad(openangle_vals))
    sqrtarg = 0.25 * Epi0**2 - mpi0**2 / (2.0 * (1.0 - cos_oa))
    phys_ok = sqrtarg >= 0

    sel = trig_ok & energy_ok & phys_ok

    # 4. Require valid npho_max and time_min
    npho_sel = npho_vals[sel]
    nphe_sel = nphe_vals[sel]
    tpm_sel = tpm_vals[sel]

    valid_npho = np.isfinite(npho_sel) & (npho_sel >= 0) & (npho_sel < 1e9)
    npho_max = np.where(valid_npho, npho_sel, -np.inf).max(axis=1)
    ch_npho_max = np.where(valid_npho, npho_sel, -np.inf).argmax(axis=1)

    valid_time = np.isfinite(tpm_sel) & (tpm_sel < 1e9) & (nphe_sel > 50)
    tpm_for_min = np.where(valid_time, tpm_sel, np.inf)
    time_min = tpm_for_min.min(axis=1)
    ch_time_min = tpm_for_min.argmin(axis=1)

    has_npho = npho_max < 1e9
    has_time = time_min < 1e9
    valid_evt = has_npho & has_time

    # Apply final cut
    sel_idx = np.where(sel)[0][valid_evt]
    n_selected = len(sel_idx)

    if n_selected == 0:
        print(f"  -> 0 events selected")
        return 0

    # --- Fill output arrays ---
    etrue = Epi0 / 2.0 - np.sqrt(sqrtarg[sel_idx])

    # npho / nphe / time for selected events
    npho_out = npho_vals[sel_idx].astype(np.float32)
    nphe_out = nphe_vals[sel_idx].astype(np.float32)
    tpm_out = tpm_vals[sel_idx].astype(np.float32)

    # Set invalid values to sentinel
    npho_out[~np.isfinite(npho_out)] = 1e10
    nphe_out[~np.isfinite(nphe_out)] = 1e10
    tpm_out[~np.isfinite(tpm_out)] = 1e10

    # relative_time
    valid_time_sel = valid_time[valid_evt]
    time_min_sel = time_min[valid_evt]
    relative_time = np.full_like(tpm_out, 1e10)
    for i in range(n_selected):
        valid_ch = np.isfinite(tpm_out[i]) & (tpm_out[i] < 1e9) & (time_min_sel[i] < 1e9)
        relative_time[i, valid_ch] = tpm_out[i, valid_ch] - time_min_sel[i]

    # Metadata
    npho_max_final = npho_max[valid_evt].astype(np.float32)
    time_min_final = time_min[valid_evt].astype(np.float32)
    npho_max_final[npho_max_final >= 1e9] = 1e10
    time_min_final[time_min_final >= 1e9] = 1e10
    ch_npho_max_final = ch_npho_max[valid_evt].astype(np.int16)
    ch_time_min_final = ch_time_min[valid_evt].astype(np.int16)

    # Append to output
    out = out_arrays
    out["run"].append(np.full(n_selected, iRun, dtype=np.int32))
    out["event"].append(sel_idx.astype(np.int32))
    out["energyTruth"].append(etrue.astype(np.float32))
    out["energyReco"].append(egamma_vals[sel_idx].astype(np.float32))
    out["Angle"].append(openangle_vals[sel_idx].astype(np.float32))
    out["gstatus"].append(
        evstat_vals[sel_idx].astype(np.int32) if evstat_vals is not None
        else np.zeros(n_selected, dtype=np.int32))
    out["Ebgo"].append(
        bgoenergy_vals[sel_idx].astype(np.float32) if bgoenergy_vals is not None
        else np.full(n_selected, 1e10, dtype=np.float32))

    out["uvwRecoFI"].append(np.column_stack([
        ugamma_vals[sel_idx] if ugamma_vals is not None else np.full(n_selected, 1e10),
        vgamma_vals[sel_idx] if vgamma_vals is not None else np.full(n_selected, 1e10),
        wgamma_vals[sel_idx] if wgamma_vals is not None else np.full(n_selected, 1e10),
    ]).astype(np.float32))

    out["npho"].append(npho_out)
    out["nphe"].append(nphe_out)
    out["time"].append(tpm_out)
    out["relative_time"].append(relative_time)

    out["ch_npho_max"].append(ch_npho_max_final)
    out["ch_time_min"].append(ch_time_min_final)
    out["npho_max_used"].append(npho_max_final)
    out["time_min_used"].append(time_min_final)

    # Dead channel mask
    out["dead"].append(np.tile(dead_mask, (n_selected, 1)))
    out["nDead"].append(np.full(n_selected, int(dead_mask.sum()), dtype=np.int32))

    # Truth placeholders (not available for real data)
    out["timeTruth"].append(np.full(n_selected, 1e10, dtype=np.float32))
    out["uvwTruth"].append(np.full((n_selected, 3), 1e10, dtype=np.float32))
    out["xyzTruth"].append(np.full((n_selected, 3), 1e10, dtype=np.float32))
    out["emiAng"].append(np.full((n_selected, 2), 1e10, dtype=np.float32))
    out["emiVec"].append(np.full((n_selected, 3), 1e10, dtype=np.float32))
    out["xyzVTX"].append(np.full((n_selected, 3), 1e10, dtype=np.float32))

    print(f"  -> {n_selected} events selected")
    return n_selected


def main():
    parser = argparse.ArgumentParser(
        description="CEX real-data preprocessing for regressor validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--srun", type=int, required=True, help="Starting run number")
    parser.add_argument("--nfiles", type=int, required=True, help="Number of runs to try")
    parser.add_argument("--patch", type=int, required=True, help="Patch number (for output filename)")
    parser.add_argument("--output-dir", default=".", help="Output directory")
    parser.add_argument("--dead-file", default=None,
                        help="Dead channel file (one index per line)")
    args = parser.parse_args()

    # Load dead channels
    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    if args.dead_file:
        dead_indices = load_dead_channels(args.dead_file)
        dead_mask[dead_indices] = True
        print(f"Loaded {len(dead_indices)} dead channels from {args.dead_file}")
    else:
        print("No dead channel file provided — dead branch will be all False")

    # Output accumulators
    out_arrays = {
        "run": [], "event": [],
        "energyTruth": [], "energyReco": [], "timeTruth": [],
        "uvwRecoFI": [], "uvwTruth": [], "xyzTruth": [],
        "emiAng": [], "emiVec": [], "xyzVTX": [],
        "npho": [], "nphe": [], "time": [], "relative_time": [],
        "ch_npho_max": [], "ch_time_min": [],
        "npho_max_used": [], "time_min_used": [],
        "dead": [], "nDead": [],
        "Ebgo": [], "Angle": [], "gstatus": [],
    }

    t0 = time.time()
    total_events = 0
    runs_processed = 0

    for iRun in range(args.srun, args.srun + args.nfiles):
        n = process_run(iRun, dead_mask, out_arrays)
        if n >= 0:
            runs_processed += 1
            total_events += n

    if total_events == 0:
        print("\nNo events selected — not writing output.")
        return

    # Concatenate
    final = {}
    for key, chunks in out_arrays.items():
        final[key] = np.concatenate(chunks)

    # Write output
    outpath = os.path.join(
        args.output_dir,
        f"CEX23_patch{args.patch}_r{args.srun}_n{args.nfiles}.root"
    )

    print(f"\nWriting {outpath} ...")

    with uproot.recreate(outpath) as f_out:
        branch_types = {}
        for key, arr in final.items():
            if arr.dtype == np.int32:
                branch_types[key] = np.int32
            elif arr.dtype == np.int16:
                branch_types[key] = np.int16
            elif arr.dtype == np.bool_:
                branch_types[key] = np.bool_
            else:
                branch_types[key] = np.float32

        f_out.mktree("tree", branch_types)
        f_out["tree"].extend(final)

    elapsed = time.time() - t0

    print(f"\n=== Summary ===")
    print(f"Runs processed: {runs_processed}")
    print(f"Events selected: {total_events}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {outpath}")


if __name__ == "__main__":
    main()
