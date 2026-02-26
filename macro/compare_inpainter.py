#!/usr/bin/env python3
"""
Cross-configuration inpainter comparison.

Overlays per-face MAE vs truth npho (denormalized, raw photons) for multiple
inpainter runs on a single 2x3 grid.  Edit the ENTRIES list below to add or
remove models.

Usage:
    python macro/compare_inpainter.py                # default output
    python macro/compare_inpainter.py -o out.pdf     # custom output path
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
# Configuration — edit this list to add/remove runs
# ---------------------------------------------------------------------------
ENTRIES = [
    {
        "path": "artifacts/mask0.10_nopre_ep20_cr-at/validation_mc/predictions_mc_runcustom.root",
        "label": "log1p (no pretrain)",
        "color": "blue",
    },
    {
        "path": "artifacts/mask0.10_nopre_ep30_cr-at_sqrt/validation_mc/predictions_mc_run430000.root",
        "label": "sqrt + flat mask + npho wt",
        "color": "red",
    },
]

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
        if 'metadata' in f:
            mt = f['metadata']
            mk = mt.keys()
            if 'npho_scheme' in mk:
                v = mt['npho_scheme'].array(library='np')[0]
                meta['npho_scheme'] = v.decode() if isinstance(v, bytes) else v
            for k in ('npho_scale', 'npho_scale2'):
                if k in mk:
                    v = mt[k].array(library='np')[0]
                    if not np.isnan(v):
                        meta[k] = float(v)

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

    # Denormalize
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
        bl_pred_raw = xf.inverse(data[pred_key])
        bl_error_raw = bl_pred_raw - truth_raw
        # Validity: finite and not sentinel
        bl_valid = valid & ~np.isnan(data[err_key]) & (data[err_key] > -999)
        baselines[bname] = {
            'truth_raw': truth_raw[bl_valid],
            'error_raw': bl_error_raw[bl_valid],
            'face': data['face'][bl_valid],
        }

    return {
        'truth_raw': truth_raw[valid],
        'error_raw': error_raw[valid],
        'face': data['face'][valid],
        'baselines': baselines,
        'meta': meta,
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
    parser.add_argument('-o', '--output', type=str, default='compare_inpainter.pdf',
                        help='Output PDF path')
    args = parser.parse_args()

    # Load all entries
    loaded = []
    for entry in ENTRIES:
        loaded.append((entry, _load_entry(entry)))

    # Shared bin edges (log-spaced in raw photons)
    pct95_vals = []
    for _, d in loaded:
        tr = d['truth_raw']
        above = tr[tr >= TRUTH_RAW_MIN]
        if len(above) > 0:
            pct95_vals.append(np.percentile(above, 95))
    global_max = max(pct95_vals) if pct95_vals else 1e5
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

    # --- Multi-page PDF: relative MAE, relative RMS, relative bias ---
    page_defs = [
        ('mae',  'Relative MAE vs Truth Npho (per face)',  'Relative MAE'),
        ('rms',  'Relative RMS vs Truth Npho (per face)',  'Relative RMS'),
        ('bias', 'Relative Bias vs Truth Npho (per face)', 'Relative Bias'),
    ]

    with PdfPages(args.output) as pdf:
        for metric_key, suptitle, ylabel in page_defs:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes_flat = axes.flatten()

            for idx, (face_int, face_name) in enumerate(FACE_INT_TO_NAME.items()):
                ax = axes_flat[idx]

                # ML entries
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

                # Baselines
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
                ax.legend(fontsize=7)

            fig.suptitle(suptitle, fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[INFO] Saved {args.output} (3 pages: relative MAE, RMS, bias)")

    # --- Stdout summary ---
    print()
    print(f"{'Method':<30s}  {'Global MAE [photons]':>20s}")
    print('-' * 53)
    for entry, d in loaded:
        cut = d['truth_raw'] >= TRUTH_RAW_MIN
        mae_val = np.mean(np.abs(d['error_raw'][cut]))
        print(f"{entry['label']:<30s}  {mae_val:>20.1f}")

    if baseline_entry_idx is not None:
        bl_dict = loaded[baseline_entry_idx][1]['baselines']
        for bname, bdef in BASELINE_DEFS.items():
            if bname not in bl_dict:
                continue
            bl = bl_dict[bname]
            cut = bl['truth_raw'] >= TRUTH_RAW_MIN
            mae_val = np.mean(np.abs(bl['error_raw'][cut]))
            print(f"{bdef['label']:<30s}  {mae_val:>20.1f}")


if __name__ == '__main__':
    main()
