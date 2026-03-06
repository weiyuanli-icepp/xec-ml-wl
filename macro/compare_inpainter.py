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
    {**s, "path": f"artifacts/{s['dir']}/validation_data/real_data_predictions.root"}
    for s in _SCAN_STEPS
]

ENTRIES_BY_MODE = {
    "mc": ENTRIES_MC,
    "sensorfront": ENTRIES_SENSORFRONT,
    "data": ENTRIES_DATA,
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
    valid = data['error_npho'] > -999
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
        # Validity: finite and not sentinel
        bl_valid = valid & ~np.isnan(data[err_key]) & (data[err_key] > -999)
        baselines[bname] = {
            'truth_raw': truth_raw[bl_valid],
            'pred_raw': bl_pred_raw[bl_valid],
            'error_raw': bl_error_raw[bl_valid],
            'face': data['face'][bl_valid],
        }

    return {
        'truth_raw': truth_raw[valid],
        'pred_raw': pred_raw[valid],
        'error_raw': error_raw[valid],
        'face': data['face'][valid],
        'baselines': baselines,
        'meta': meta,
    }


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

    valid = data['error_npho'] > -999
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


def _compute_slice_metrics(truth, error, bin_edges):
    """Binned MAE, bias, RMS of error.  Returns bin_centers and all three."""
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices = np.clip(np.digitize(truth, bin_edges) - 1, 0, len(bin_centers) - 1)

    mae = np.full(len(bin_centers), np.nan)
    bias = np.full(len(bin_centers), np.nan)
    rms = np.full(len(bin_centers), np.nan)

    for i in range(len(bin_centers)):
        m = indices == i
        if m.sum() < MIN_BIN_COUNT:
            continue
        err = error[m]
        mae[i] = np.mean(np.abs(err))
        bias[i] = np.mean(err)
        rms[i] = np.sqrt(np.mean(err ** 2))

    return bin_centers, mae, bias, rms


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
                        choices=list(ENTRIES_BY_MODE.keys()),
                        help='Validation type to compare (default: mc)')
    parser.add_argument('--localfit', type=str, default=None,
                        help='Path to LocalFitBaseline output ROOT file (raw photons)')
    args = parser.parse_args()

    if args.output is None:
        args.output = f'compare_inpainter_{args.mode}.pdf'

    entries = ENTRIES_BY_MODE[args.mode]
    print(f"[INFO] Mode: {args.mode} ({len(entries)} entries)")

    # Load all entries (skip missing files)
    loaded = []
    for entry in entries:
        if not Path(entry['path']).exists():
            print(f"[WARN] Skipping {entry['label']}: {entry['path']} not found")
            continue
        loaded.append((entry, _load_entry(entry)))

    # Optionally add standalone LocalFitBaseline result
    if args.localfit:
        lf_entry = {'label': 'Local Fit', 'color': 'darkred', 'path': args.localfit}
        loaded.append((lf_entry, _load_localfit(args.localfit)))

    if not loaded:
        print("[ERROR] No entries loaded. Check that prediction files exist.")
        sys.exit(1)

    # Shared bin edges (log-spaced in raw photons, full range)
    max_vals = []
    for _, d in loaded:
        tr = d['truth_raw']
        above = tr[tr >= TRUTH_RAW_MIN]
        if len(above) > 0:
            max_vals.append(np.max(above))
    global_max = max(max_vals) if max_vals else 1e5
    bin_edges = np.logspace(np.log10(TRUTH_RAW_MIN), np.log10(global_max), N_BINS + 1)

    # Decide which entry supplies baselines (first one that has any)
    baseline_entry_idx = None
    for i, (_, d) in enumerate(loaded):
        if d['baselines']:
            baseline_entry_idx = i
            break

    # --- Precompute per-face metrics for all entries & baselines ---
    # metrics_cache[(entry_idx_or_bname, face_int)] = (centers, mae, bias, rms)
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

    if baseline_entry_idx is not None:
        bl_dict = loaded[baseline_entry_idx][1]['baselines']
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
    # pred_metrics_cache[(entry_idx_or_bname, face_int)] = (centers, mae, bias, rms)
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

    if baseline_entry_idx is not None:
        bl_dict_pred = loaded[baseline_entry_idx][1]['baselines']
        for bname in BASELINE_DEFS:
            if bname not in bl_dict_pred:
                continue
            bl = bl_dict_pred[bname]
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
        methods.append((entry['label'], entry['color'],
                         d['truth_raw'][cut], d['error_raw'][cut],
                         d['face'][cut], False))
    if baseline_entry_idx is not None:
        bl_dict = loaded[baseline_entry_idx][1]['baselines']
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
        fig, (ax_bar, ax_face) = plt.subplots(1, 2, figsize=(18, 8),
                                               gridspec_kw={'width_ratios': [1, 1.5]})

        # -- Left panel: global relative MAE horizontal bar chart (sorted) --
        sorted_mm = sorted(method_metrics, key=lambda m: m['rel_mae'])
        labels_bar = [m['label'] for m in sorted_mm]
        vals_bar = [m['rel_mae'] for m in sorted_mm]
        colors_bar = [m['color'] for m in sorted_mm]
        hatches_bar = ['//' if m['is_bl'] else '' for m in sorted_mm]
        y_pos = np.arange(len(labels_bar))

        bars = ax_bar.barh(y_pos, vals_bar, color=colors_bar, edgecolor='black',
                           linewidth=0.5)
        for bar, hatch in zip(bars, hatches_bar):
            bar.set_hatch(hatch)
        for i, v in enumerate(vals_bar):
            ax_bar.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(labels_bar, fontsize=9)
        ax_bar.set_xlabel('Global Relative MAE')
        ax_bar.set_title('Global Relative MAE  (truth >= 100 photons)')
        ax_bar.invert_yaxis()
        ax_bar.set_xlim(0, max(vals_bar) * 1.25)

        # -- Right panel: per-face relative MAE grouped bar chart --
        face_names = [fn for _, fn in active_faces]
        n_methods = len(method_metrics)
        n_faces = len(face_names)
        bar_width = 0.8 / n_methods
        x_faces = np.arange(n_faces)

        for mi, mm in enumerate(method_metrics):
            face_vals = [mm['face_rel_mae'].get(fn, 0.0) for fn in face_names]
            face_mask = [fn in mm['face_rel_mae'] for fn in face_names]
            x_offset = (mi - n_methods / 2 + 0.5) * bar_width
            bar_objs = ax_face.bar(x_faces + x_offset, face_vals, bar_width,
                                   color=mm['color'], edgecolor='black',
                                   linewidth=0.3, label=mm['label'],
                                   hatch='//' if mm['is_bl'] else '')
            # Gray out missing faces
            for bi, present in enumerate(face_mask):
                if not present:
                    bar_objs[bi].set_alpha(0.0)

        ax_face.set_xticks(x_faces)
        ax_face.set_xticklabels(face_names, fontsize=10)
        ax_face.set_ylabel('Relative MAE')
        ax_face.set_title('Per-Face Relative MAE  (truth >= 100 photons)')
        ax_face.legend(fontsize=7, loc='upper right', ncol=2)

        fig.suptitle(f'Inpainter Comparison Summary — {args.mode}', fontsize=14)
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
                    centers, mae, bias, rms = metrics_cache[key]
                    if metric_key == 'mae':
                        vals = mae / centers
                    elif metric_key == 'rms':
                        vals = rms / centers
                    else:
                        vals = bias / centers
                    ax.plot(centers, vals, 'o-', color=entry['color'],
                            markersize=3, label=entry['label'])

                if baseline_entry_idx is not None:
                    for bname, bdef in BASELINE_DEFS.items():
                        key = (bname, face_int)
                        if key not in metrics_cache:
                            continue
                        centers, mae, bias, rms = metrics_cache[key]
                        if metric_key == 'mae':
                            vals = mae / centers
                        elif metric_key == 'rms':
                            vals = rms / centers
                        else:
                            vals = bias / centers
                        ax.plot(centers, vals, 's--', color=bdef['color'],
                                markersize=3, alpha=0.8, label=bdef['label'])

                ax.set_xscale('log')
                ax.set_xlabel('Truth Npho [photons]')
                ax.set_ylabel(ylabel)
                ax.set_title(face_name)
                if metric_key in ('mae', 'rms'):
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(-1, 1)
                ax.legend(fontsize=7)

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
                    centers, mae, bias, rms = pred_metrics_cache[key]
                    if metric_key == 'mae':
                        vals = mae / centers
                    elif metric_key == 'rms':
                        vals = rms / centers
                    else:
                        vals = bias / centers
                    ax.plot(centers, vals, 'o-', color=entry['color'],
                            markersize=3, label=entry['label'])

                if baseline_entry_idx is not None:
                    for bname, bdef in BASELINE_DEFS.items():
                        key = (bname, face_int)
                        if key not in pred_metrics_cache:
                            continue
                        centers, mae, bias, rms = pred_metrics_cache[key]
                        if metric_key == 'mae':
                            vals = mae / centers
                        elif metric_key == 'rms':
                            vals = rms / centers
                        else:
                            vals = bias / centers
                        ax.plot(centers, vals, 's--', color=bdef['color'],
                                markersize=3, alpha=0.8, label=bdef['label'])

                ax.set_xscale('log')
                ax.set_xlabel('Pred Npho [photons]')
                ax.set_ylabel(ylabel)
                ax.set_title(face_name)
                if metric_key in ('mae', 'rms'):
                    ax.set_ylim(0, 1)
                else:
                    ax.set_ylim(-1, 1)
                ax.legend(fontsize=7)

            for idx in range(n_active, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(suptitle, fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    n_pages = 1 + 3 + 3  # summary + truth pages + pred pages
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
