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
import awkward as ak
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

# Tolerance for EGamma + bgoEnergy ≈ Epi0 cut (GeV)
ETOTAL_TOL = 0.02

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


def load_dead_mask_for_run(iRun, dead_dir):
    """Load dead channel mask from dead_dir/dead_run{RUN}.txt if it exists."""
    dead_mask = np.zeros(N_CHANNELS, dtype=bool)
    path = os.path.join(dead_dir, f"dead_run{iRun}.txt")
    if os.path.exists(path):
        indices = load_dead_channels(path)
        dead_mask[indices] = True
    return dead_mask


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
    # uproot key names (from rec tree StreamerInfo):
    #   Single-object branches: "branch./branch.member"
    #   TClonesArray branches:  "branch/branch.member"
    key_mask      = "eventheader./eventheader.mask"
    key_egamma    = "reco./reco.EGamma"
    key_evstat    = "reco./reco.EvstatGamma"
    key_ugamma    = "reco./reco.UGamma"
    key_vgamma    = "reco./reco.VGamma"
    key_wgamma    = "reco./reco.WGamma"
    key_npho      = "xeccl/xeccl.npho"
    key_nphe      = "xeccl/xeccl.nphe"
    key_tpm       = "xeccl/xeccl.tpm"
    key_openangle = "bgocexresult/bgocexresult.openingAngle"
    key_bgoenergy = "bgocexresult/bgocexresult.bgoEnergy"

    # Check critical branches exist
    available = set(rec.keys())
    critical = {"trigger mask": key_mask, "EGamma": key_egamma,
                "npho": key_npho, "tpm": key_tpm,
                "openingAngle": key_openangle}
    missing = [name for name, key in critical.items() if key not in available]

    if missing:
        print(f"  ERROR: Cannot find branches: {missing}")
        discover_branches(rec)
        return -1

    # Read all data at once using awkward arrays (handles jagged/nested data)
    read_keys = [key_mask, key_egamma, key_evstat,
                 key_ugamma, key_vgamma, key_wgamma,
                 key_npho, key_nphe, key_tpm,
                 key_openangle, key_bgoenergy]
    read_keys = [k for k in read_keys if k in available]

    arrays = rec.arrays(read_keys, library="ak")

    def get_field(key):
        """Get field from awkward record, or None if not read."""
        if key in read_keys:
            return arrays[key]
        return None

    def ak_to_flat(arr, dtype=np.float32):
        """Convert awkward array to flat 1-D numpy, taking index 0 for nested dims."""
        if arr is None:
            return None
        while arr.ndim > 1:
            arr = arr[:, 0]
        return ak.to_numpy(arr).astype(dtype)

    def ak_to_2d(arr, fallback_shape=None, dtype=np.float32):
        """Convert awkward array to (N, 4760) numpy, taking index 0 of innermost dim."""
        if arr is None:
            return np.full(fallback_shape, 1e10, dtype=dtype) if fallback_shape else None
        while arr.ndim > 2:
            arr = arr[:, :, 0]
        return ak.to_numpy(arr).astype(dtype)

    mask_vals = ak_to_flat(get_field(key_mask), dtype=np.int32)
    egamma_vals = ak_to_flat(get_field(key_egamma))
    openangle_vals = ak_to_flat(get_field(key_openangle))
    bgoenergy_vals = ak_to_flat(get_field(key_bgoenergy))

    evstat_vals = ak_to_flat(get_field(key_evstat), dtype=np.int32)
    ugamma_vals = ak_to_flat(get_field(key_ugamma))
    vgamma_vals = ak_to_flat(get_field(key_vgamma))
    wgamma_vals = ak_to_flat(get_field(key_wgamma))

    npho_vals = ak_to_2d(get_field(key_npho))
    tpm_vals = ak_to_2d(get_field(key_tpm))
    nphe_vals = ak_to_2d(get_field(key_nphe), fallback_shape=npho_vals.shape)

    # --- Event selection (vectorized) ---
    # 1. Trigger
    trig_ok = (mask_vals == 51)

    # 2. Energy window
    energy_ok = (egamma_vals >= E_MIN) & (egamma_vals <= E_MAX)

    # 3. EGamma + bgoEnergy ≈ Epi0 (wide tolerance)
    if bgoenergy_vals is not None:
        etotal = egamma_vals + bgoenergy_vals
        etotal_ok = np.abs(etotal - Epi0) <= ETOTAL_TOL
    else:
        etotal_ok = np.ones(len(egamma_vals), dtype=bool)

    # 4. Opening angle → Etrue, must be physical (sqrtarg >= 0)
    cos_oa = np.cos(np.deg2rad(openangle_vals))
    sqrtarg = 0.25 * Epi0**2 - mpi0**2 / (2.0 * (1.0 - cos_oa))
    phys_ok = sqrtarg >= 0

    sel = trig_ok & energy_ok & etotal_ok & phys_ok

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
    dead_group = parser.add_mutually_exclusive_group()
    dead_group.add_argument("--dead-file", default=None,
                            help="Dead channel file (one index per line, same for all runs)")
    dead_group.add_argument("--dead-dir", default=None,
                            help="Directory with per-run dead channel files "
                                 "(dead_run{RUN}.txt, auto-generated by submit_cex.py)")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Stop after this many selected events (for testing)")
    args = parser.parse_args()

    # Load dead channels (fixed file for all runs, or per-run from directory)
    fixed_dead_mask = None
    if args.dead_file:
        dead_indices = load_dead_channels(args.dead_file)
        fixed_dead_mask = np.zeros(N_CHANNELS, dtype=bool)
        fixed_dead_mask[dead_indices] = True
        print(f"Loaded {len(dead_indices)} dead channels from {args.dead_file}")
    elif args.dead_dir:
        print(f"Using per-run dead channel files from {args.dead_dir}")
    else:
        print("No dead channel source — dead branch will be all False")
        fixed_dead_mask = np.zeros(N_CHANNELS, dtype=bool)

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
        if fixed_dead_mask is not None:
            dead_mask = fixed_dead_mask
        else:
            dead_mask = load_dead_mask_for_run(iRun, args.dead_dir)
            print(f"  Run {iRun}: {int(dead_mask.sum())} dead channels from {args.dead_dir}")
        n = process_run(iRun, dead_mask, out_arrays)
        if n >= 0:
            runs_processed += 1
            total_events += n
        if args.max_events and total_events >= args.max_events:
            print(f"  Reached --max-events {args.max_events}, stopping.")
            break

    if total_events == 0:
        print("\nNo events selected — not writing output.")
        return

    # Concatenate and truncate if --max-events
    final = {}
    for key, chunks in out_arrays.items():
        final[key] = np.concatenate(chunks)
    if args.max_events and len(final["run"]) > args.max_events:
        for key in final:
            final[key] = final[key][:args.max_events]
        total_events = args.max_events

    # Write output
    outpath = os.path.join(
        args.output_dir,
        f"CEX23_patch{args.patch}_r{args.srun}_n{args.nfiles}.root"
    )

    print(f"\nWriting {outpath} ...")

    with uproot.recreate(outpath) as f_out:
        branch_types = {}
        for key, arr in final.items():
            base = arr.dtype
            if base == np.float64:
                base = np.float32
                final[key] = arr.astype(np.float32)
            # For multi-dim arrays, encode inner shape into the dtype
            if arr.ndim == 1:
                branch_types[key] = base
            else:
                branch_types[key] = np.dtype((base, arr.shape[1:]))

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
