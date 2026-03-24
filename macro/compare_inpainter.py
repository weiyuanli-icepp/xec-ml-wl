#!/usr/bin/env python3
"""
Cross-configuration inpainter comparison.

Overlays per-face MAE vs truth npho (denormalized, raw photons) for multiple
inpainter runs on a single 2x3 grid.  Use --mode to switch between validation
types.

Usage:
    python macro/compare_inpainter.py --mode mc            # MC validation (default)
    python macro/compare_inpainter.py --mode sensorfront   # sensor-front validation
    python macro/compare_inpainter.py --mode data          # real data validation
    python macro/compare_inpainter.py --localfit path/to/localfit.root  # overlay local fit
    python macro/compare_inpainter.py -o out.pdf           # custom output path
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import uproot

# Allow imports from the project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.normalization import NphoTransform

# ---------------------------------------------------------------------------
# Configuration — edit these lists to add/remove runs
# ---------------------------------------------------------------------------
_SCAN_STEPS = [
    {"label": "S1: baseline",    "color": "blue",      "dir": "inp_scan_s1_baseline"},
    {"label": "S2: +flat mask",  "color": "orange",    "dir": "inp_scan_s2_flatmask"},
    {"label": "S3: +npho wt",   "color": "green",     "dir": "inp_scan_s3_nphowt"},
    {"label": "S4: flat+npho wt","color": "red",       "dir": "inp_scan_s4_flat_nphowt"},
    {"label": "S5: sqrt",       "color": "purple",    "dir": "inp_scan_s5_sqrt"},
    {"label": "S6: mask 0.15",  "color": "brown",     "dir": "inp_scan_s6_mask015"},
    {"label": "S7: sqrt+wt+m15","color": "deeppink",  "dir": "inp_scan_s7_sqrt_nphowt_mask015"},
    {"label": "S8: mask 0.20",  "color": "teal",      "dir": "inp_scan_s8_mask020"},
]

ENTRIES_MC = [
    {**s, "path": f"artifacts/{s['dir']}/validation_mc/predictions_mc_run430000.root"}
    for s in _SCAN_STEPS
]

ENTRIES_SENSORFRONT = [
    {**s, "path": f"artifacts/{s['dir']}/validation_sensorfront/predictions_sensorfront.root"}
    for s in _SCAN_STEPS
]

ENTRIES_DATA = [
    {**s, "path": f"artifacts/{s['dir']}/validation_data/predictions_real_runcustom.root"}
    for s in _SCAN_STEPS
]

ENTRIES_BY_MODE = {
    "mc": ENTRIES_MC,
    "sensorfront": ENTRIES_SENSORFRONT,
    "data": ENTRIES_DATA,
}

# Labels and colors for cross-mode ("all") comparison
MODE_DISPLAY = {
    "mc":          {"label": "MC",                "color": "blue"},
    "sensorfront": {"label": "MC light map peak", "color": "green"},
    "data":        {"label": "Data",              "color": "red"},
}

FACE_INT_TO_NAME = {0: 'inner', 1: 'us', 2: 'ds', 3: 'outer', 4: 'top', 5: 'bot'}

BASELINE_DEFS = {
    'avg': {'label': 'Neighbor Avg', 'color': 'orange'},
    'sa':  {'label': 'Solid-Angle Wt', 'color': 'green'},
    'lf':  {'label': 'Local Fit', 'color': 'darkred'},
}

MIN_BIN_COUNT = 10
TRUTH_RAW_MIN = 100.0   # photon cut for log-space binning
N_BINS = 30


def _plot_valid(ax, x, y, *args, **kwargs):
    """Plot x vs y, skipping NaN entries but connecting valid points."""
    valid = np.isfinite(y)
    if valid.any():
        ax.plot(x[valid], y[valid], *args, **kwargs)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_entry(entry):
    """Load one prediction file and return denormalized arrays + baselines."""
    path = entry['path']
    print(f"[INFO] Loading {path}")

    with uproot.open(path) as f:
        # --- metadata ---
        meta = {'npho_scheme': 'log1p', 'npho_scale': 1000.0, 'npho_scale2': 4.08}
        has_npho_scheme = False
        if 'metadata' in f:
            mt = f['metadata']
            mk = mt.keys()
            if 'npho_scheme' in mk:
                has_npho_scheme = True
                v = mt['npho_scheme'].array(library='np')[0]
                meta['npho_scheme'] = v.decode() if isinstance(v, bytes) else v
            for k in ('npho_scale', 'npho_scale2'):
                if k in mk:
                    v = mt[k].array(library='np')[0]
                    if not np.isnan(v):
                        meta[k] = float(v)
        # Real data validation stores raw photons with no npho_scheme in metadata.
        # All MC/sensorfront validation scripts always write npho_scheme, so
        # absence reliably indicates a real-data output file.
        if not has_npho_scheme and 'metadata' in f:
            meta['npho_scheme'] = 'raw'
            print(f"[INFO]   No npho_scheme in metadata → treating as raw photons")

        # --- predictions tree ---
        for tname in ('predictions', 'tree', 'Tree'):
            if tname in f:
                tree = f[tname]
                break
        else:
            tree = f[f.keys()[0]]

        keys = tree.keys()
        data = {k: tree[k].array(library='np') for k in keys}

    # Remap baseline_localfit_* -> baseline_lf_*
    for suffix in ('npho', 'error_npho'):
        src = f'baseline_localfit_{suffix}'
        dst = f'baseline_lf_{suffix}'
        if src in data and dst not in data:
            data[dst] = data.pop(src)

    # Valid mask
    has_mask_type = 'mask_type' in data
    valid = data['error_npho'] > -998
    if has_mask_type:
        valid = valid & (data['mask_type'] == 0)

    # Denormalize (or skip if values are already raw photons)
    is_raw = (meta['npho_scheme'] == 'raw')
    if is_raw:
        truth_raw = data['truth_npho']
        pred_raw = data['pred_npho']
    else:
        xf = NphoTransform(
            scheme=meta['npho_scheme'],
            npho_scale=meta['npho_scale'],
            npho_scale2=meta['npho_scale2'],
        )
        truth_raw = xf.inverse(data['truth_npho'])
        pred_raw = xf.inverse(data['pred_npho'])
    error_raw = pred_raw - truth_raw

    # Baselines (denormalized)
    baselines = {}
    for bname in BASELINE_DEFS:
        err_key = f'baseline_{bname}_error_npho'
        pred_key = f'baseline_{bname}_npho'
        if err_key not in data:
            continue
        if is_raw:
            bl_pred_raw = data[pred_key]
        else:
            bl_pred_raw = xf.inverse(data[pred_key])
        bl_error_raw = bl_pred_raw - truth_raw
        # Validity: finite (NaN = no prediction, -999 = legacy sentinel)
        bl_valid = valid & np.isfinite(data[err_key]) & (data[err_key] > -998)
        bl_result = {
            'truth_raw': truth_raw[bl_valid],
            'pred_raw': bl_pred_raw[bl_valid],
            'error_raw': bl_error_raw[bl_valid],
            'face': data['face'][bl_valid],
        }
        if 'run_number' in data:
            bl_result['run_number'] = data['run_number'][bl_valid]
        if 'event_number' in data:
            bl_result['event_number'] = data['event_number'][bl_valid]
        baselines[bname] = bl_result

    result = {
        'truth_raw': truth_raw[valid],
        'pred_raw': pred_raw[valid],
        'error_raw': error_raw[valid],
        'face': data['face'][valid],
        'baselines': baselines,
        'meta': meta,
    }
    # Carry run/event numbers for energy matching
    if 'run_number' in data:
        result['run_number'] = data['run_number'][valid]
    if 'event_number' in data:
        result['event_number'] = data['event_number'][valid]
    return result


def _load_localfit(path):
    """Load standalone LocalFitBaseline output (raw photon values, no denorm)."""
    print(f"[INFO] Loading localfit: {path}")
    with uproot.open(path) as f:
        for tname in ('predictions', 'tree', 'Tree'):
            if tname in f:
                tree = f[tname]
                break
        else:
            tree = f[f.keys()[0]]

        data = {k: tree[k].array(library='np') for k in tree.keys()}

    valid = data['error_npho'] > -998
    if 'mask_type' in data:
        valid = valid & (data['mask_type'] == 0)

    return {
        'truth_raw': data['truth_npho'][valid],
        'pred_raw': data['pred_npho'][valid],
        'error_raw': data['error_npho'][valid],
        'face': data['face'][valid],
        'baselines': {},
        'meta': {'npho_scheme': 'raw', 'npho_scale': 1.0, 'npho_scale2': 1.0},
    }


def _load_baselines(path):
    """Load standalone baseline ROOT file (from compute_inpainter_baselines.py).

    Returns a dict mapping baseline name to {truth_raw, pred_raw, error_raw, face, ...}.
    Same structure as the 'baselines' sub-dict in _load_entry().
    """
    print(f"[INFO] Loading standalone baselines: {path}")
    with uproot.open(path) as f:
        # metadata
        meta = {'npho_scheme': 'log1p', 'npho_scale': 1000.0, 'npho_scale2': 4.08}
        if 'metadata' in f:
            mt = f['metadata']
            mk = mt.keys()
            if 'npho_scheme' in mk:
                v = mt['npho_scheme'].array(library='np')[0]
                meta['npho_scheme'] = v.decode() if isinstance(v, bytes) else str(v)
            for k in ('npho_scale', 'npho_scale2'):
                if k in mk:
                    v = mt[k].array(library='np')[0]
                    if not np.isnan(v):
                        meta[k] = float(v)

        # predictions tree
        for tname in ('predictions', 'tree'):
            if tname in f:
                tree = f[tname]
                break
        else:
            tree = f[f.keys()[0]]
        data = {k: tree[k].array(library='np') for k in tree.keys()}

    # Validity
    has_mask_type = 'mask_type' in data
    # For baselines, use mask_type == 0 (artificial, has truth)
    valid_base = np.ones(len(data['face']), dtype=bool)
    if has_mask_type:
        valid_base = data['mask_type'] == 0

    # Denormalize truth (skip if already in raw photon space)
    is_raw = (meta['npho_scheme'] == 'raw')
    if is_raw:
        truth_raw = data['truth_npho']
    else:
        xf = NphoTransform(
            scheme=meta['npho_scheme'],
            npho_scale=meta['npho_scale'],
            npho_scale2=meta['npho_scale2'],
        )
        truth_raw = xf.inverse(data['truth_npho'])

    baselines = {}
    for bname in BASELINE_DEFS:
        pred_key = f'baseline_{bname}_npho'
        err_key = f'baseline_{bname}_error_npho'
        if pred_key not in data:
            continue
        bl_pred_raw = data[pred_key] if is_raw else xf.inverse(data[pred_key])
        bl_error_raw = bl_pred_raw - truth_raw
        bl_valid = valid_base & np.isfinite(data[err_key])
        bl_result = {
            'truth_raw': truth_raw[bl_valid],
            'pred_raw': bl_pred_raw[bl_valid],
            'error_raw': bl_error_raw[bl_valid],
            'face': data['face'][bl_valid],
        }
        if 'run_number' in data:
            bl_result['run_number'] = data['run_number'][bl_valid]
        if 'event_number' in data:
            bl_result['event_number'] = data['event_number'][bl_valid]
        baselines[bname] = bl_result

    print(f"[INFO]   Found baselines: {list(baselines.keys())}")
    return baselines


def _compute_slice_metrics(truth, error, bin_edges):
    """Binned MAE, bias, RMS of error.  Returns bin_centers, mae, bias, rms, counts."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices = np.clip(np.digitize(truth, bin_edges) - 1, 0, len(bin_centers) - 1)

    mae = np.full(len(bin_centers), np.nan)
    bias = np.full(len(bin_centers), np.nan)
    rms = np.full(len(bin_centers), np.nan)
    counts = np.zeros(len(bin_centers), dtype=int)

    for i in range(len(bin_centers)):
        m = indices == i
        counts[i] = m.sum()
        if counts[i] < MIN_BIN_COUNT:
            continue
        err = error[m]
        mae[i] = np.mean(np.abs(err))
        bias[i] = np.mean(err)
        rms[i] = np.sqrt(np.mean(err ** 2))

    return bin_centers, mae, bias, rms, counts


def _load_mc_energy(mc_path, tree_name='tree'):
    """Load (run, event) → energyTruth [MeV] mapping from MC ROOT file(s).

    Args:
        mc_path: Path to MC ROOT file, directory, or glob pattern.
        tree_name: Tree name in the ROOT files.

    Returns:
        dict mapping (run, event) → energy in MeV.
    """
    from pathlib import Path as _P

    mc_path = str(mc_path)
    if os.path.isdir(mc_path):
        files = sorted(str(p) for p in _P(mc_path).glob('*.root'))
    elif '*' in mc_path or '?' in mc_path:
        import glob
        files = sorted(glob.glob(mc_path))
    else:
        files = [mc_path]

    if not files:
        print(f"[WARN] No MC files found at {mc_path}")
        return {}

    energy_map = {}
    for fpath in files:
        with uproot.open(fpath) as f:
            tree = f[tree_name]
            keys = tree.keys()
            if 'energyTruth' not in keys:
                print(f"[WARN] No energyTruth branch in {fpath}, skipping")
                continue
            run = tree['run'].array(library='np') if 'run' in keys else None
            event = tree['event'].array(library='np') if 'event' in keys else None
            energy = tree['energyTruth'].array(library='np')
            if run is None or event is None:
                continue
            for r, e, en in zip(run, event, energy):
                energy_map[(int(r), int(e))] = float(en) * 1e3  # GeV → MeV

    print(f"[INFO] Loaded MC energy for {len(energy_map)} events "
          f"from {len(files)} file(s)")
    return energy_map


def _attach_energy(d, energy_map):
    """Attach per-sensor energy array to a loaded entry dict using run/event matching.

    Returns energy array (same length as d['truth_raw']), NaN where no match.
    """
    run = d.get('run_number')
    evt = d.get('event_number')
    if run is None or evt is None:
        return None
    energy = np.full(len(run), np.nan, dtype=np.float64)
    for i in range(len(run)):
        key = (int(run[i]), int(evt[i]))
        if key in energy_map:
            energy[i] = energy_map[key]
    n_matched = np.isfinite(energy).sum()
    return energy if n_matched > 0 else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare inpainter runs (per-face MAE in raw photons)",
    )
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output PDF path (default: compare_inpainter_{mode}.pdf)')
    parser.add_argument('--mode', type=str, default='mc',
                        choices=list(ENTRIES_BY_MODE.keys()) + ['all'],
                        help='Validation type to compare (default: mc). '
                             '"all" overlays mc/sensorfront/data for selected steps.')
    parser.add_argument('--baselines', type=str, default=None,
                        help='Path to standalone baseline ROOT file '
                             '(from compute_inpainter_baselines.py)')
    parser.add_argument('--localfit', type=str, default=None,
                        help='Path to LocalFitBaseline output ROOT file (raw photons)')
    parser.add_argument('--mc-data', type=str, default=None,
                        help='Path to MC ROOT file(s) for energy matching '
                             '(default: auto-detect from mode)')
    parser.add_argument('--energy-range', type=float, nargs=2, default=None,
                        metavar=('E_MIN', 'E_MAX'),
                        help='Energy range in MeV for additional filtered pages '
                             '(e.g. --energy-range 45 50)')
    parser.add_argument('--steps', type=int, nargs='+', default=None,
                        help='Only include these scan steps (e.g. --steps 3)')
    parser.add_argument('--ylim-mae', type=float, default=1.5,
                        help='Y-axis upper limit for relative MAE/RMS plots (default: 1.5)')
    parser.add_argument('--show-counts', action='store_true',
                        help='Show number of events per bin on secondary y-axis (log scale, right)')
    parser.add_argument('--only-baselines', type=str, nargs='+', default=None,
                        choices=list(BASELINE_DEFS.keys()),
                        help='Only show these baselines (e.g. --only-baselines sa)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'compare_inpainter_{args.mode}.pdf'

    cross_mode = (args.mode == 'all')
    if cross_mode:
        # Cross-mode comparison: load selected step(s) from each mode,
        # and promote per-mode baselines into separate entries so each
        # mode's SA baseline is plotted independently.
        entries = []
        _cross_mode_baseline_entries = []  # (entry_dict, mode_name) to load later
        for mode_name, mode_entries in ENTRIES_BY_MODE.items():
            disp = MODE_DISPLAY[mode_name]
            filtered = mode_entries
            if args.steps is not None:
                step_set = set(args.steps)
                filtered = [e for i, e in enumerate(filtered) if (i + 1) in step_set]
            for e in filtered:
                entries.append({**e, 'label': f"{disp['label']} (inpainter)",
                                'color': disp['color'], '_mode': mode_name})
    else:
        entries = ENTRIES_BY_MODE[args.mode]
        # Filter entries by step number if --steps is given
        if args.steps is not None:
            step_set = set(args.steps)
            entries = [e for i, e in enumerate(entries) if (i + 1) in step_set]

    # Filter baselines if --only-baselines is given
    if args.only_baselines is not None:
        keep = set(args.only_baselines)
        for bname in list(BASELINE_DEFS.keys()):
            if bname not in keep:
                del BASELINE_DEFS[bname]
    print(f"[INFO] Mode: {args.mode} ({len(entries)} entries)")

    # Load all entries (skip missing files)
    loaded = []
    for entry in entries:
        if not Path(entry['path']).exists():
            print(f"[WARN] Skipping {entry['label']}: {entry['path']} not found")
            continue
        loaded.append((entry, _load_entry(entry)))

    # In cross-mode ("all"), promote each mode's baselines into separate
    # loaded entries so that e.g. SA-wt is compared fairly per mode.
    if cross_mode and BASELINE_DEFS:
        # Load standalone baselines (--baselines) for modes missing embedded ones
        standalone_bl = None
        if args.baselines and Path(args.baselines).exists():
            standalone_bl = _load_baselines(args.baselines)

        extra = []
        # Lighter/dashed versions of mode colors for baselines
        _BL_COLOR = {
            "mc":          "cornflowerblue",
            "sensorfront": "mediumseagreen",
            "data":        "salmon",
        }
        for entry, d in loaded:
            mode_name = entry.get('_mode', 'mc')
            for bname in list(BASELINE_DEFS.keys()):
                # Try embedded baselines first, fall back to standalone
                if bname in d['baselines']:
                    bl = d['baselines'][bname]
                elif standalone_bl is not None and bname in standalone_bl:
                    bl = standalone_bl[bname]
                else:
                    continue
                bl_label = f"{MODE_DISPLAY[mode_name]['label']} (SA-wt)"
                bl_entry = {
                    'label': bl_label,
                    'color': _BL_COLOR.get(mode_name, 'gray'),
                    'path': entry['path'],
                    'is_baseline': True,
                }
                bl_data = {
                    'truth_raw': bl['truth_raw'],
                    'pred_raw': bl['pred_raw'],
                    'error_raw': bl['error_raw'],
                    'face': bl['face'],
                    'baselines': {},
                    'meta': d['meta'],
                }
                if 'run_number' in bl:
                    bl_data['run_number'] = bl['run_number']
                if 'event_number' in bl:
                    bl_data['event_number'] = bl['event_number']
                extra.append((bl_entry, bl_data))
            # Clear baselines from the ML entry so they don't get plotted again
            d['baselines'] = {}
        loaded.extend(extra)
        # Clear BASELINE_DEFS so the old single-baseline code path is skipped
        BASELINE_DEFS.clear()

    # Optionally add standalone LocalFitBaseline result
    if args.localfit:
        lf_entry = {'label': 'Local Fit', 'color': 'darkred', 'path': args.localfit,
                     'is_baseline': True}
        loaded.append((lf_entry, _load_localfit(args.localfit)))

    if not loaded:
        print("[ERROR] No entries loaded. Check that prediction files exist.")
        sys.exit(1)

    # --- Load MC energy for energy-filtered pages ---
    energy_map = {}
    if args.energy_range is not None:
        mc_path = args.mc_data
        if mc_path is None and args.mode == 'mc':
            mc_path = 'data/E15to60_AngUni_PosSQ/val2/'
        if mc_path is not None:
            energy_map = _load_mc_energy(mc_path)
        else:
            print("[WARN] --energy-range requires --mc-data or mc mode")

    # Attach energy to each loaded entry and baselines
    if energy_map:
        for _, d in loaded:
            d['energy'] = _attach_energy(d, energy_map)
            for bname, bl in d['baselines'].items():
                bl['energy'] = _attach_energy(bl, energy_map)

    # Shared bin edges (log-spaced in raw photons, full range)
    max_vals = []
    for _, d in loaded:
        tr = d['truth_raw']
        above = tr[tr >= TRUTH_RAW_MIN]
        if len(above) > 0:
            max_vals.append(np.max(above))
    global_max = max(max_vals) if max_vals else 1e5
    bin_edges = np.logspace(np.log10(TRUTH_RAW_MIN), np.log10(global_max), N_BINS + 1)

    # Decide baseline source: standalone file (--baselines) or embedded in entries
    standalone_baselines = None
    if args.baselines:
        if not Path(args.baselines).exists():
            print(f"[ERROR] Baseline file not found: {args.baselines}")
            sys.exit(1)
        standalone_baselines = _load_baselines(args.baselines)

    baseline_entry_idx = None
    if standalone_baselines is None:
        for i, (_, d) in enumerate(loaded):
            if d['baselines']:
                baseline_entry_idx = i
                break

    # Unified baseline dict for all downstream code
    bl_dict = (standalone_baselines if standalone_baselines is not None
               else (loaded[baseline_entry_idx][1]['baselines']
                     if baseline_entry_idx is not None else {}))

    # --- Precompute per-face metrics for all entries & baselines ---
    # metrics_cache[(entry_idx_or_bname, face_int)] = (centers, mae, bias, rms, counts)
    metrics_cache = {}
    for ei, (entry, d) in enumerate(loaded):
        for face_int in FACE_INT_TO_NAME:
            fm = d['face'] == face_int
            tr = d['truth_raw'][fm]
            er = d['error_raw'][fm]
            cut = tr >= TRUTH_RAW_MIN
            tr, er = tr[cut], er[cut]
            if len(tr) < MIN_BIN_COUNT:
                continue
            metrics_cache[(ei, face_int)] = _compute_slice_metrics(tr, er, bin_edges)

    if bl_dict:
        for bname in BASELINE_DEFS:
            if bname not in bl_dict:
                continue
            bl = bl_dict[bname]
            for face_int in FACE_INT_TO_NAME:
                fm = bl['face'] == face_int
                tr = bl['truth_raw'][fm]
                er = bl['error_raw'][fm]
                cut = tr >= TRUTH_RAW_MIN
                tr, er = tr[cut], er[cut]
                if len(tr) < MIN_BIN_COUNT:
                    continue
                metrics_cache[(bname, face_int)] = _compute_slice_metrics(tr, er, bin_edges)

    # --- Pred-based bin edges (same range logic, using pred_raw) ---
    pred_max_vals = []
    for _, d in loaded:
        pr = d['pred_raw']
        above = pr[pr >= TRUTH_RAW_MIN]
        if len(above) > 0:
            pred_max_vals.append(np.max(above))
    pred_global_max = max(pred_max_vals) if pred_max_vals else 1e5
    pred_bin_edges = np.logspace(np.log10(TRUTH_RAW_MIN),
                                 np.log10(pred_global_max), N_BINS + 1)

    # --- Precompute pred-based per-face metrics ---
    # pred_metrics_cache[(entry_idx_or_bname, face_int)] = (centers, mae, bias, rms, counts)
    pred_metrics_cache = {}
    for ei, (entry, d) in enumerate(loaded):
        for face_int in FACE_INT_TO_NAME:
            fm = d['face'] == face_int
            pr = d['pred_raw'][fm]
            er = d['error_raw'][fm]
            cut = pr >= TRUTH_RAW_MIN
            pr, er = pr[cut], er[cut]
            if len(pr) < MIN_BIN_COUNT:
                continue
            pred_metrics_cache[(ei, face_int)] = _compute_slice_metrics(
                pr, er, pred_bin_edges)

    if bl_dict:
        for bname in BASELINE_DEFS:
            if bname not in bl_dict:
                continue
            bl = bl_dict[bname]
            for face_int in FACE_INT_TO_NAME:
                fm = bl['face'] == face_int
                pr = bl['pred_raw'][fm]
                er = bl['error_raw'][fm]
                cut = pr >= TRUTH_RAW_MIN
                pr, er = pr[cut], er[cut]
                if len(pr) < MIN_BIN_COUNT:
                    continue
                pred_metrics_cache[(bname, face_int)] = _compute_slice_metrics(
                    pr, er, pred_bin_edges)

    # --- Collect all methods for summary & stdout ---
    # (label, color, truth_raw, error_raw, face, is_baseline)
    methods = []
    for entry, d in loaded:
        cut = d['truth_raw'] >= TRUTH_RAW_MIN
        is_bl = entry.get('is_baseline', False)
        methods.append((entry['label'], entry['color'],
                         d['truth_raw'][cut], d['error_raw'][cut],
                         d['face'][cut], is_bl))
    if bl_dict:
        for bname, bdef in BASELINE_DEFS.items():
            if bname not in bl_dict:
                continue
            bl = bl_dict[bname]
            cut = bl['truth_raw'] >= TRUTH_RAW_MIN
            methods.append((bdef['label'], bdef['color'],
                             bl['truth_raw'][cut], bl['error_raw'][cut],
                             bl['face'][cut], True))

    # Precompute global & per-face metrics for each method
    method_metrics = []
    for label, color, tr, er, fc, is_bl in methods:
        global_rel_mae = np.mean(np.abs(er) / tr)
        global_mae = np.mean(np.abs(er))
        global_rms = np.sqrt(np.mean(er ** 2))
        global_bias = np.mean(er)
        global_rel_rms = np.sqrt(np.mean((er / tr) ** 2))
        face_rel_mae = {}
        for fi, fn in FACE_INT_TO_NAME.items():
            m = fc == fi
            if m.sum() >= MIN_BIN_COUNT:
                face_rel_mae[fn] = np.mean(np.abs(er[m]) / tr[m])
        method_metrics.append({
            'label': label, 'color': color, 'is_bl': is_bl, 'n': len(tr),
            'rel_mae': global_rel_mae, 'mae': global_mae,
            'rms': global_rms, 'bias': global_bias, 'rel_rms': global_rel_rms,
            'face_rel_mae': face_rel_mae,
        })

    # --- Determine which faces have data (for compact plots) ---
    active_faces = []
    for face_int, face_name in FACE_INT_TO_NAME.items():
        has_data = any(
            (ei, face_int) in metrics_cache for ei in range(len(loaded))
        )
        if has_data:
            active_faces.append((face_int, face_name))

    # --- Multi-page PDF ---
    page_defs_truth = [
        ('mae',  'Relative MAE vs Truth Npho (per face)',  'Relative MAE'),
        ('rms',  'Relative RMS vs Truth Npho (per face)',  'Relative RMS'),
        ('bias', 'Relative Bias vs Truth Npho (per face)', 'Relative Bias'),
    ]
    page_defs_pred = [
        ('mae',  'Relative MAE vs Pred Npho (per face)',  'Relative MAE'),
        ('rms',  'Relative RMS vs Pred Npho (per face)',  'Relative RMS'),
        ('bias', 'Relative Bias vs Pred Npho (per face)', 'Relative Bias'),
    ]

    with PdfPages(args.output) as pdf:
        # ===== Page 1: Summary =====
        show_per_face = len(active_faces) > 1
        if show_per_face:
            fig, (ax_bar, ax_face) = plt.subplots(1, 2, figsize=(18, 8),
                                                   gridspec_kw={'width_ratios': [1, 1.5]})
        else:
            fig, ax_bar = plt.subplots(1, 1, figsize=(10, 8))

        # -- Left (or only) panel: global relative MAE horizontal bar chart --
        sorted_mm = sorted(method_metrics, key=lambda m: m['rel_mae'])
        labels_bar = [m['label'] for m in sorted_mm]
        vals_bar = [m['rel_mae'] for m in sorted_mm]
        colors_bar = [m['color'] for m in sorted_mm]
        hatches_bar = ['//' if m['is_bl'] else '' for m in sorted_mm]
        y_pos = np.arange(len(labels_bar))

        # Cap x-axis: use 2x the largest non-outlier value
        # (outlier = > 3x median) so extreme baselines don't squash the rest
        median_val = np.median(vals_bar)
        non_outlier = [v for v in vals_bar if v <= 3 * median_val]
        x_max = max(non_outlier) * 1.5 if non_outlier else max(vals_bar) * 1.25
        vals_bar_clipped = [min(v, x_max) for v in vals_bar]

        bars = ax_bar.barh(y_pos, vals_bar_clipped, color=colors_bar,
                           edgecolor='black', linewidth=0.5)
        for bar, hatch in zip(bars, hatches_bar):
            bar.set_hatch(hatch)
        for i, (v, v_clip) in enumerate(zip(vals_bar, vals_bar_clipped)):
            if v > x_max:
                ax_bar.text(v_clip - 0.01, i, f'{v:.1f}', va='center',
                            ha='right', fontsize=9, color='white', fontweight='bold')
            else:
                ax_bar.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(labels_bar, fontsize=9)
        ax_bar.set_xlabel('Global Relative MAE')
        ax_bar.set_title('Global Relative MAE  (truth >= 100 photons)')
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, x_max * 1.1)

        # -- Right panel: per-face grouped bar chart (skip if only 1 face) --
        if show_per_face:
            face_names = [fn for _, fn in active_faces]
            n_faces = len(face_names)

            # Sort methods: group by color (ML + SA-wt pair), ordered
            # MC, Data, MC light map peak. ML entry first, then its SA-wt.
            if cross_mode:
                _MODE_ORDER = ['mc', 'data', 'sensorfront']
                _MODE_LABEL = {v['label']: k for k, v in MODE_DISPLAY.items()}

                def _sort_key(mm):
                    label = mm['label']
                    # Strip " (SA-wt)" / " (inpainter)" to get base mode label
                    base = label.replace(' (SA-wt)', '').replace(' (inpainter)', '')
                    mode = _MODE_LABEL.get(base, 'zzz')
                    order = _MODE_ORDER.index(mode) if mode in _MODE_ORDER else 99
                    # SA-wt first (0), then inpainter (1)
                    is_inpainter = 0 if '(SA-wt)' in label else 1
                    return (is_inpainter, order)

                method_metrics_sorted = sorted(method_metrics, key=_sort_key)
            else:
                method_metrics_sorted = method_metrics

            n_methods = len(method_metrics_sorted)
            bar_width = 0.8 / max(n_methods, 1)
            x_faces = np.arange(n_faces)

            for mi, mm in enumerate(method_metrics_sorted):
                face_vals = [mm['face_rel_mae'].get(fn, 0.0) for fn in face_names]
                face_mask = [fn in mm['face_rel_mae'] for fn in face_names]
                x_offset = (mi - n_methods / 2 + 0.5) * bar_width
                bar_objs = ax_face.bar(x_faces + x_offset, face_vals, bar_width,
                                       color=mm['color'], edgecolor='black',
                                       linewidth=0.3, label=mm['label'],
                                       hatch='//' if mm['is_bl'] else '')
                for bi, present in enumerate(face_mask):
                    if not present:
                        bar_objs[bi].set_alpha(0.0)

            ax_face.set_xticks(x_faces)
            ax_face.set_xticklabels(face_names, fontsize=10)
            ax_face.set_ylabel('Relative MAE', fontsize=16)
            ax_face.set_title('Per-Face Relative MAE  (truth >= 100 photons)', fontsize=16)
            ax_face.set_ylim(0, args.ylim_mae)
            ax_face.legend(fontsize=15, loc='upper right', ncol=1)

        fig.suptitle(f'Inpainter Comparison Summary — {args.mode}', fontsize=16)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # ===== Pages 2–4: per-face binned metrics vs Truth Npho =====
        n_active = len(active_faces)
        ncols = min(n_active, 3)
        nrows = (n_active + ncols - 1) // ncols
        for metric_key, suptitle, ylabel in page_defs_truth:
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 5 * nrows),
                                     squeeze=False)
            axes_flat = axes.flatten()

            for idx, (face_int, face_name) in enumerate(active_faces):
                ax = axes_flat[idx]

                for ei, (entry, _) in enumerate(loaded):
                    key = (ei, face_int)
                    if key not in metrics_cache:
                        continue
                    centers, mae, bias, rms, cnts = metrics_cache[key]
                    if metric_key == 'mae':
                        vals = mae / centers
                    elif metric_key == 'rms':
                        vals = rms / centers
                    else:
                        vals = bias / centers
                    is_bl = entry.get('is_baseline', False)
                    marker = 's' if is_bl else 'o'
                    alpha = 0.8 if is_bl else 1.0
                    _plot_valid(ax, centers, vals, marker, color=entry['color'],
                               markersize=5, alpha=alpha, linestyle='none',
                               label=entry['label'])

                if bl_dict:
                    for bname, bdef in BASELINE_DEFS.items():
                        key = (bname, face_int)
                        if key not in metrics_cache:
                            continue
                        centers, mae, bias, rms, cnts = metrics_cache[key]
                        if metric_key == 'mae':
                            vals = mae / centers
                        elif metric_key == 'rms':
                            vals = rms / centers
                        else:
                            vals = bias / centers
                        _plot_valid(ax, centers, vals, 's', color=bdef['color'],
                                   markersize=5, alpha=0.8, linestyle='none',
                                   label=bdef['label'])

                ax.set_xscale('log')
                ax.set_xlabel('Truth Npho [photons]')
                ax.set_ylabel(ylabel)
                ax.set_title(face_name)
                if metric_key in ('mae', 'rms'):
                    ax.set_ylim(0, args.ylim_mae)
                else:
                    ax.set_ylim(-args.ylim_mae, args.ylim_mae)
                ax.legend(fontsize=7)

                # Overlay bin counts on secondary y-axis
                if args.show_counts:
                    # Use counts from first available entry
                    for ei, (entry, _) in enumerate(loaded):
                        key = (ei, face_int)
                        if key in metrics_cache:
                            _, _, _, _, first_cnts = metrics_cache[key]
                            ax2 = ax.twinx()
                            valid_c = first_cnts > 0
                            if valid_c.any():
                                c_centers = bin_edges[:-1] * 0.5 + bin_edges[1:] * 0.5
                                ax2.bar(c_centers[valid_c], first_cnts[valid_c],
                                        width=np.diff(bin_edges)[valid_c] * 0.8,
                                        color='gray', alpha=0.15, zorder=0)
                            ax2.set_yscale('log')
                            ax2.set_ylabel('Events / bin', fontsize=7, color='gray')
                            ax2.tick_params(axis='y', labelsize=6, colors='gray')
                            break

            # Hide unused axes
            for idx in range(n_active, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(suptitle, fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ===== Pages 5–7: per-face binned metrics vs Pred Npho =====
        for metric_key, suptitle, ylabel in page_defs_pred:
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 5 * nrows),
                                     squeeze=False)
            axes_flat = axes.flatten()

            for idx, (face_int, face_name) in enumerate(active_faces):
                ax = axes_flat[idx]

                for ei, (entry, _) in enumerate(loaded):
                    key = (ei, face_int)
                    if key not in pred_metrics_cache:
                        continue
                    centers, mae, bias, rms, cnts = pred_metrics_cache[key]
                    if metric_key == 'mae':
                        vals = mae / centers
                    elif metric_key == 'rms':
                        vals = rms / centers
                    else:
                        vals = bias / centers
                    _plot_valid(ax, centers, vals, 'o', color=entry['color'],
                               markersize=5, linestyle='none', label=entry['label'])

                if bl_dict:
                    for bname, bdef in BASELINE_DEFS.items():
                        key = (bname, face_int)
                        if key not in pred_metrics_cache:
                            continue
                        centers, mae, bias, rms, cnts = pred_metrics_cache[key]
                        if metric_key == 'mae':
                            vals = mae / centers
                        elif metric_key == 'rms':
                            vals = rms / centers
                        else:
                            vals = bias / centers
                        _plot_valid(ax, centers, vals, 's', color=bdef['color'],
                                   markersize=5, alpha=0.8, linestyle='none',
                                   label=bdef['label'])

                ax.set_xscale('log')
                ax.set_xlabel('Pred Npho [photons]')
                ax.set_ylabel(ylabel)
                ax.set_title(face_name)
                if metric_key in ('mae', 'rms'):
                    ax.set_ylim(0, args.ylim_mae)
                else:
                    ax.set_ylim(-args.ylim_mae, args.ylim_mae)
                ax.legend(fontsize=7)

                if args.show_counts:
                    for ei, (entry, _) in enumerate(loaded):
                        key = (ei, face_int)
                        if key in pred_metrics_cache:
                            _, _, _, _, first_cnts = pred_metrics_cache[key]
                            ax2 = ax.twinx()
                            valid_c = first_cnts > 0
                            if valid_c.any():
                                c_centers = pred_bin_edges[:-1] * 0.5 + pred_bin_edges[1:] * 0.5
                                ax2.bar(c_centers[valid_c], first_cnts[valid_c],
                                        width=np.diff(pred_bin_edges)[valid_c] * 0.8,
                                        color='gray', alpha=0.15, zorder=0)
                            ax2.set_yscale('log')
                            ax2.set_ylabel('Events / bin', fontsize=7, color='gray')
                            ax2.tick_params(axis='y', labelsize=6, colors='gray')
                            break

            for idx in range(n_active, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(suptitle, fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # ===== Energy-filtered pages (if --energy-range given) =====
        n_energy_pages = 0
        if args.energy_range and energy_map:
            e_lo, e_hi = args.energy_range
            e_tag = f"E ∈ [{e_lo:.0f}, {e_hi:.0f}] MeV"

            # Build energy-filtered metrics cache
            ecut_metrics = {}
            for ei, (entry, d) in enumerate(loaded):
                en = d.get('energy')
                if en is None:
                    continue
                for face_int in FACE_INT_TO_NAME:
                    fm = d['face'] == face_int
                    em = np.isfinite(en) & (en >= e_lo) & (en <= e_hi)
                    sel = fm & em
                    tr = d['truth_raw'][sel]
                    er = d['error_raw'][sel]
                    cut = tr >= TRUTH_RAW_MIN
                    tr, er = tr[cut], er[cut]
                    if len(tr) < MIN_BIN_COUNT:
                        continue
                    ecut_metrics[(ei, face_int)] = _compute_slice_metrics(
                        tr, er, bin_edges)

            if bl_dict:
                for bname in BASELINE_DEFS:
                    if bname not in bl_dict:
                        continue
                    bl = bl_dict[bname]
                    bl_en = bl.get('energy')
                    if bl_en is None:
                        continue
                    for face_int in FACE_INT_TO_NAME:
                        fm = bl['face'] == face_int
                        em = np.isfinite(bl_en) & (bl_en >= e_lo) & (bl_en <= e_hi)
                        sel = fm & em
                        tr = bl['truth_raw'][sel]
                        er = bl['error_raw'][sel]
                        cut = tr >= TRUTH_RAW_MIN
                        tr, er = tr[cut], er[cut]
                        if len(tr) < MIN_BIN_COUNT:
                            continue
                        ecut_metrics[(bname, face_int)] = _compute_slice_metrics(
                            tr, er, bin_edges)

            # Determine active faces for energy-filtered data
            ecut_active = []
            for face_int, face_name in FACE_INT_TO_NAME.items():
                if any((ei, face_int) in ecut_metrics
                       for ei in range(len(loaded))):
                    ecut_active.append((face_int, face_name))

            if ecut_active:
                ecut_n_active = len(ecut_active)
                ecut_ncols = min(ecut_n_active, 3)
                ecut_nrows = (ecut_n_active + ecut_ncols - 1) // ecut_ncols

                ecut_page_defs = [
                    ('mae',  f'Relative MAE vs Truth Npho — {e_tag}',
                     'Relative MAE'),
                    ('rms',  f'Relative RMS vs Truth Npho — {e_tag}',
                     'Relative RMS'),
                    ('bias', f'Relative Bias vs Truth Npho — {e_tag}',
                     'Relative Bias'),
                ]

                for metric_key, suptitle, ylabel in ecut_page_defs:
                    fig, axes = plt.subplots(
                        ecut_nrows, ecut_ncols,
                        figsize=(6 * ecut_ncols, 5 * ecut_nrows),
                        squeeze=False)
                    axes_flat = axes.flatten()

                    for idx, (face_int, face_name) in enumerate(ecut_active):
                        ax = axes_flat[idx]
                        for ei, (entry, _) in enumerate(loaded):
                            key = (ei, face_int)
                            if key not in ecut_metrics:
                                continue
                            centers, mae, bias, rms, cnts = ecut_metrics[key]
                            if metric_key == 'mae':
                                vals = mae / centers
                            elif metric_key == 'rms':
                                vals = rms / centers
                            else:
                                vals = bias / centers
                            _plot_valid(ax, centers, vals, 'o', color=entry['color'],
                                       markersize=5, linestyle='none',
                                       label=entry['label'])

                        if bl_dict:
                            for bname, bdef in BASELINE_DEFS.items():
                                key = (bname, face_int)
                                if key not in ecut_metrics:
                                    continue
                                centers, mae, bias, rms, cnts = ecut_metrics[key]
                                if metric_key == 'mae':
                                    vals = mae / centers
                                elif metric_key == 'rms':
                                    vals = rms / centers
                                else:
                                    vals = bias / centers
                                _plot_valid(ax, centers, vals, 's',
                                            color=bdef['color'],
                                            markersize=5, alpha=0.8,
                                            linestyle='none',
                                            label=bdef['label'])

                        ax.set_xscale('log')
                        ax.set_xlabel('Truth Npho [photons]')
                        ax.set_ylabel(ylabel)
                        ax.set_title(face_name)
                        if metric_key in ('mae', 'rms'):
                            ax.set_ylim(0, 1.5)
                        else:
                            ax.set_ylim(-1.5, 1.5)
                        ax.legend(fontsize=7)

                    for idx in range(ecut_n_active, len(axes_flat)):
                        axes_flat[idx].set_visible(False)

                    fig.suptitle(suptitle, fontsize=14)
                    plt.tight_layout()
                    pdf.savefig(fig)
                    plt.close(fig)
                    n_energy_pages += 1

    n_pages = 1 + 3 + 3 + n_energy_pages
    print(f"[INFO] Saved {args.output} ({n_pages} pages, {n_active} active face(s))")

    # --- Stdout summary ---

    # --- Per-entry config ---
    print()
    print("=" * 70)
    print("ENTRIES")
    print("=" * 70)
    for entry, d in loaded:
        m = d['meta']
        cut = d['truth_raw'] >= TRUTH_RAW_MIN
        print(f"  {entry['label']}")
        print(f"    path   : {entry['path']}")
        print(f"    scheme : {m['npho_scheme']}  "
              f"(scale={m['npho_scale']:.1f}, scale2={m['npho_scale2']:.2f})")
        print(f"    sensors: {cut.sum():,} (truth >= {TRUTH_RAW_MIN:.0f} photons)")

    # --- Global metrics table ---
    print()
    print("=" * 70)
    print("GLOBAL METRICS  (truth >= 100 photons)")
    print("=" * 70)
    hdr = f"{'Method':<28s} {'MAE':>10s} {'RMS':>10s} {'Bias':>10s} {'Rel MAE':>10s} {'Rel RMS':>10s} {'N':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for mm in method_metrics:
        tag = " *" if mm['is_bl'] else ""
        print(f"{mm['label']:<28s} {mm['mae']:>10.1f} {mm['rms']:>10.1f} {mm['bias']:>+10.1f} "
              f"{mm['rel_mae']:>10.3f} {mm['rel_rms']:>10.3f} {mm['n']:>10,}{tag}")

    # --- Per-face breakdown ---
    print()
    face_names = list(FACE_INT_TO_NAME.values())
    print("=" * 70)
    print("PER-FACE MAE  (truth >= 100 photons)")
    print("=" * 70)
    hdr_face = f"{'Method':<28s} " + " ".join(f"{fn:>10s}" for fn in face_names)
    print(hdr_face)
    print("-" * len(hdr_face))
    for label, color, tr, er, fc, is_bl in methods:
        vals = []
        for fi in FACE_INT_TO_NAME:
            m = fc == fi
            if m.sum() < MIN_BIN_COUNT:
                vals.append(f"{'---':>10s}")
            else:
                vals.append(f"{np.mean(np.abs(er[m])):>10.1f}")
        tag = " *" if is_bl else ""
        print(f"{label:<28s} " + " ".join(vals) + tag)

    # --- Per-face relative MAE ---
    print()
    print("=" * 70)
    print("PER-FACE RELATIVE MAE  (truth >= 100 photons)")
    print("=" * 70)
    print(hdr_face)
    print("-" * len(hdr_face))
    for label, color, tr, er, fc, is_bl in methods:
        vals = []
        for fi in FACE_INT_TO_NAME:
            m = fc == fi
            if m.sum() < MIN_BIN_COUNT:
                vals.append(f"{'---':>10s}")
            else:
                vals.append(f"{np.mean(np.abs(er[m]) / tr[m]):>10.3f}")
        tag = " *" if is_bl else ""
        print(f"{label:<28s} " + " ".join(vals) + tag)

    print()
    print("  (* = baseline)")
    print(f"  Output: {args.output}")


if __name__ == '__main__':
    main()
