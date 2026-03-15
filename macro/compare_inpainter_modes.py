#!/usr/bin/env python3
"""
Cross-mode inpainter comparison: MC vs Sensorfront vs Real Data.

Compares a single inpainter (default: S3) and baselines across
validation modes on the same plots.

Step 1: Export per-mode metrics to npz files (one-time):
    python macro/compare_inpainter_modes.py --export mc \\
        --predictions artifacts/inp_scan_s3_nphowt/validation_mc/predictions_mc_run430000.root \\
        --baselines data/inp_baselines_val2_dcp430000.root \\
        --localfit localfit_mc_all.root \\
        --output data/comparison_cache/mc.npz

    python macro/compare_inpainter_modes.py --export sensorfront \\
        --predictions artifacts/inp_scan_s3_nphowt/validation_sensorfront/predictions_sensorfront.root \\
        --output data/comparison_cache/sensorfront.npz

    python macro/compare_inpainter_modes.py --export data \\
        --predictions artifacts/inp_scan_s3_nphowt/validation_data/predictions_real_runcustom.root \\
        --baselines data/inp_baselines_realdata_dcp430000.root \\
        --localfit localfit_realdata.root \\
        --output data/comparison_cache/data.npz

Step 2: Plot comparison from cached files:
    python macro/compare_inpainter_modes.py --plot \\
        --mc data/comparison_cache/mc.npz \\
        --sensorfront data/comparison_cache/sensorfront.npz \\
        --data data/comparison_cache/data.npz \\
        -o cross_mode_comparison.pdf
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.normalization import NphoTransform

FACE_INT_TO_NAME = {0: 'inner', 1: 'us', 2: 'ds', 3: 'outer', 4: 'top', 5: 'bot'}
MIN_BIN_COUNT = 10
TRUTH_RAW_MIN = 100.0
N_BINS = 30

BASELINE_DEFS = {
    'avg': {'label': 'Neighbor Avg'},
    'sa':  {'label': 'Solid-Angle Wt'},
    'lf':  {'label': 'Local Fit'},
}

# Methods to include: ML inpainter + baselines
METHOD_STYLES = {
    'ml':  {'marker': 'o', 'ls': '-',  'lw': 1.5},
    'avg': {'marker': 's', 'ls': '--', 'lw': 1.2},
    'sa':  {'marker': '^', 'ls': '--', 'lw': 1.2},
    'lf':  {'marker': 'D', 'ls': '--', 'lw': 1.2},
}

MODE_COLORS = {
    'mc':           {'ml': 'C0', 'avg': 'C0', 'sa': 'C0', 'lf': 'C0'},
    'sensorfront':  {'ml': 'C1', 'avg': 'C1', 'sa': 'C1', 'lf': 'C1'},
    'data':         {'ml': 'C3', 'avg': 'C3', 'sa': 'C3', 'lf': 'C3'},
}

MODE_LABELS = {
    'mc': 'MC',
    'sensorfront': 'Sensorfront',
    'data': 'Real Data',
}


def _load_entry(path):
    """Load prediction file, return (truth_raw, error_raw, face) arrays."""
    import uproot
    print(f"[INFO] Loading {path}")
    with uproot.open(path) as f:
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
            if 'npho_scheme' not in mk:
                meta['npho_scheme'] = 'raw'

        for tname in ('predictions', 'tree'):
            if tname in f:
                tree = f[tname]
                break
        else:
            tree = f[f.keys()[0]]
        data = {k: tree[k].array(library='np') for k in tree.keys()}

    # Remap baseline_localfit_* -> baseline_lf_*
    for suffix in ('npho', 'error_npho'):
        src = f'baseline_localfit_{suffix}'
        dst = f'baseline_lf_{suffix}'
        if src in data and dst not in data:
            data[dst] = data.pop(src)

    has_mask_type = 'mask_type' in data
    valid = data['error_npho'] > -998
    if has_mask_type:
        valid = valid & (data['mask_type'] == 0)

    is_raw = (meta['npho_scheme'] == 'raw')
    if is_raw:
        truth_raw = data['truth_npho']
        pred_raw = data['pred_npho']
    else:
        xf = NphoTransform(scheme=meta['npho_scheme'],
                           npho_scale=meta['npho_scale'],
                           npho_scale2=meta['npho_scale2'])
        truth_raw = xf.inverse(data['truth_npho'])
        pred_raw = xf.inverse(data['pred_npho'])
    error_raw = pred_raw - truth_raw

    result = {
        'ml': {
            'truth_raw': truth_raw[valid],
            'error_raw': error_raw[valid],
            'face': data['face'][valid],
        }
    }

    # Embedded baselines
    for bname in BASELINE_DEFS:
        pred_key = f'baseline_{bname}_npho'
        err_key = f'baseline_{bname}_error_npho'
        if pred_key not in data:
            continue
        bl_pred = data[pred_key] if is_raw else xf.inverse(data[pred_key])
        bl_error = bl_pred - truth_raw
        bl_valid = valid & np.isfinite(data[err_key]) & (data[err_key] > -998)
        result[bname] = {
            'truth_raw': truth_raw[bl_valid],
            'error_raw': bl_error[bl_valid],
            'face': data['face'][bl_valid],
        }

    return result


def _load_baselines(path):
    """Load standalone baseline file."""
    import uproot
    print(f"[INFO] Loading baselines: {path}")
    with uproot.open(path) as f:
        meta = {'npho_scheme': 'raw', 'npho_scale': 1000.0, 'npho_scale2': 4.08}
        if 'metadata' in f:
            mt = f['metadata']
            if 'npho_scheme' in mt.keys():
                v = mt['npho_scheme'].array(library='np')[0]
                meta['npho_scheme'] = v.decode() if isinstance(v, bytes) else str(v)
            for k in ('npho_scale', 'npho_scale2'):
                if k in mt.keys():
                    v = mt[k].array(library='np')[0]
                    if not np.isnan(v):
                        meta[k] = float(v)

        for tname in ('predictions', 'tree'):
            if tname in f:
                tree = f[tname]
                break
        else:
            tree = f[f.keys()[0]]
        data = {k: tree[k].array(library='np') for k in tree.keys()}

    has_mask_type = 'mask_type' in data
    valid_base = np.ones(len(data['face']), dtype=bool)
    if has_mask_type:
        valid_base = data['mask_type'] == 0

    is_raw = (meta['npho_scheme'] == 'raw')
    if is_raw:
        truth_raw = data['truth_npho']
    else:
        xf = NphoTransform(scheme=meta['npho_scheme'],
                           npho_scale=meta['npho_scale'],
                           npho_scale2=meta['npho_scale2'])
        truth_raw = xf.inverse(data['truth_npho'])

    result = {}
    for bname in BASELINE_DEFS:
        pred_key = f'baseline_{bname}_npho'
        err_key = f'baseline_{bname}_error_npho'
        if pred_key not in data:
            continue
        bl_pred = data[pred_key] if is_raw else xf.inverse(data[pred_key])
        bl_error = bl_pred - truth_raw
        bl_valid = valid_base & np.isfinite(data[err_key])
        result[bname] = {
            'truth_raw': truth_raw[bl_valid],
            'error_raw': bl_error[bl_valid],
            'face': data['face'][bl_valid],
        }
    return result


def _load_localfit(path):
    """Load LocalFit file."""
    import uproot
    print(f"[INFO] Loading localfit: {path}")
    with uproot.open(path) as f:
        for tname in ('predictions', 'tree'):
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
        'lf': {
            'truth_raw': data['truth_npho'][valid],
            'error_raw': data['error_npho'][valid],
            'face': data['face'][valid],
        }
    }


def _compute_slice_metrics(truth, error, bin_edges):
    """Binned MAE, bias, RMS."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    indices = np.clip(np.digitize(truth, bin_edges) - 1, 0, len(centers) - 1)
    mae = np.full(len(centers), np.nan)
    bias = np.full(len(centers), np.nan)
    rms = np.full(len(centers), np.nan)
    for i in range(len(centers)):
        m = indices == i
        if m.sum() < MIN_BIN_COUNT:
            continue
        err = error[m]
        mae[i] = np.mean(np.abs(err))
        bias[i] = np.mean(err)
        rms[i] = np.sqrt(np.mean(err**2))
    return centers, mae, bias, rms


def export_mode(args):
    """Export metrics for one mode to npz file."""
    methods = _load_entry(args.predictions)

    # Override baselines from standalone file
    if args.baselines:
        standalone = _load_baselines(args.baselines)
        for bname, bdata in standalone.items():
            methods[bname] = bdata

    # Override localfit
    if args.localfit:
        lf = _load_localfit(args.localfit)
        methods.update(lf)

    # Compute bin edges from ML truth
    ml = methods['ml']
    above = ml['truth_raw'][ml['truth_raw'] >= TRUTH_RAW_MIN]
    if len(above) == 0:
        print("[ERROR] No entries above threshold")
        sys.exit(1)
    bin_edges = np.logspace(np.log10(TRUTH_RAW_MIN),
                            np.log10(np.max(above)), N_BINS + 1)

    # Compute per-method, per-face metrics
    save_data = {'bin_edges': bin_edges, 'mode': args.export}
    for mname, mdata in methods.items():
        tr = mdata['truth_raw']
        er = mdata['error_raw']
        fc = mdata['face']
        cut = tr >= TRUTH_RAW_MIN
        tr, er, fc = tr[cut], er[cut], fc[cut]

        # Global metrics
        if len(tr) > 0:
            save_data[f'{mname}_global_rel_mae'] = np.mean(np.abs(er) / tr)
            save_data[f'{mname}_global_mae'] = np.mean(np.abs(er))
            save_data[f'{mname}_global_n'] = len(tr)

        # Per-face global relative MAE
        for fi, fn in FACE_INT_TO_NAME.items():
            fm = fc == fi
            if fm.sum() >= MIN_BIN_COUNT:
                save_data[f'{mname}_{fn}_rel_mae'] = np.mean(np.abs(er[fm]) / tr[fm])

        # Per-face binned metrics
        for fi, fn in FACE_INT_TO_NAME.items():
            fm = fc == fi
            if fm.sum() < MIN_BIN_COUNT:
                continue
            centers, mae, bias, rms = _compute_slice_metrics(
                tr[fm], er[fm], bin_edges)
            save_data[f'{mname}_{fn}_centers'] = centers
            save_data[f'{mname}_{fn}_mae'] = mae
            save_data[f'{mname}_{fn}_bias'] = bias
            save_data[f'{mname}_{fn}_rms'] = rms

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **save_data)
    n_methods = sum(1 for k in save_data if k.endswith('_global_n'))
    print(f"[INFO] Saved {args.export} metrics ({n_methods} methods) to {args.output}")


def plot_comparison(args):
    """Plot cross-mode comparison from cached npz files."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    # Load all modes
    modes = {}
    for mode_name, path in [('mc', args.mc), ('sensorfront', args.sensorfront),
                             ('data', args.data)]:
        if path and os.path.exists(path):
            modes[mode_name] = dict(np.load(path, allow_pickle=True))
            print(f"[INFO] Loaded {mode_name}: {path}")

    if not modes:
        print("[ERROR] No mode data loaded")
        sys.exit(1)

    # Determine which methods exist across modes
    all_methods = set()
    for mode_data in modes.values():
        for key in mode_data:
            if key.endswith('_global_n'):
                all_methods.add(key.replace('_global_n', ''))
    method_order = [m for m in ['ml', 'avg', 'sa', 'lf'] if m in all_methods]

    # Determine active faces
    active_faces = []
    for fi, fn in FACE_INT_TO_NAME.items():
        has_data = any(
            f'ml_{fn}_centers' in mode_data
            for mode_data in modes.values()
        )
        if has_data:
            active_faces.append((fi, fn))

    n_active = len(active_faces)
    ncols = min(n_active, 3)
    nrows = (n_active + ncols - 1) // ncols

    method_labels = {'ml': 'ML Inpainter', **{k: v['label'] for k, v in BASELINE_DEFS.items()}}

    page_defs = [
        ('mae',  'Relative MAE vs Truth Npho', 'Relative MAE'),
        ('rms',  'Relative RMS vs Truth Npho', 'Relative RMS'),
        ('bias', 'Relative Bias vs Truth Npho', 'Relative Bias'),
    ]

    def _plot_valid(ax, x, y, *plot_args, **kwargs):
        valid = np.isfinite(y)
        if valid.any():
            ax.plot(x[valid], y[valid], *plot_args, **kwargs)

    # Methods for per-face chart (skip localfit)
    perface_methods = [m for m in method_order if m != 'lf']

    with PdfPages(args.o) as pdf:
        # Page 1: global + per-face summary
        fig, (ax_global, ax_face) = plt.subplots(
            2, 1, figsize=(14, 12),
            gridspec_kw={'height_ratios': [1, 1.2]})

        # -- Top: global bar chart (grouped by method) --
        bar_data = []
        bar_colors = []
        bar_hatches = []
        for mname in method_order:
            for mode_name in ['mc', 'data', 'sensorfront']:
                if mode_name not in modes:
                    continue
                mode_data = modes[mode_name]
                key = f'{mname}_global_rel_mae'
                if key not in mode_data:
                    continue
                val = float(mode_data[key])
                label = f'{method_labels[mname]} — {MODE_LABELS[mode_name]}'
                bar_data.append((label, val))
                bar_colors.append(MODE_COLORS[mode_name][mname])
                bar_hatches.append('//' if mname != 'ml' else '')

        if bar_data:
            labels = [d[0] for d in bar_data]
            vals = [d[1] for d in bar_data]
            y_pos = np.arange(len(labels))

            median_val = np.median(vals)
            non_outlier = [v for v in vals if v <= 3 * median_val]
            x_max = max(non_outlier) * 1.5 if non_outlier else max(vals) * 1.25
            vals_clipped = [min(v, x_max) for v in vals]

            bars = ax_global.barh(y_pos, vals_clipped, color=bar_colors,
                                  edgecolor='black', linewidth=0.5)
            for bar, hatch in zip(bars, bar_hatches):
                bar.set_hatch(hatch)
            for i, (v, vc) in enumerate(zip(vals, vals_clipped)):
                if v > x_max:
                    ax_global.text(vc - 0.01, i, f'{v:.1f}', va='center',
                                   ha='right', fontsize=8, color='white',
                                   fontweight='bold')
                else:
                    ax_global.text(v + 0.002, i, f'{v:.3f}', va='center',
                                   fontsize=8)
            ax_global.set_yticks(y_pos)
            ax_global.set_yticklabels(labels, fontsize=9)
            ax_global.set_xlabel('Global Relative MAE')
            ax_global.set_title('Global Relative MAE  (truth >= 100 photons)')
            ax_global.invert_yaxis()
            ax_global.set_xlim(0, x_max * 1.1)

        # -- Bottom: per-face grouped bar chart --
        # Group: for each face, show (method × mode) bars
        face_names = [fn for _, fn in active_faces]
        n_faces = len(face_names)
        mode_list = [m for m in ['mc', 'data', 'sensorfront'] if m in modes]
        n_groups = len(perface_methods) * len(mode_list)
        bar_width = 0.8 / max(n_groups, 1)
        x_faces = np.arange(n_faces)

        legend_handles = []
        legend_labels = []
        gi = 0
        for mname in perface_methods:
            for mode_name in mode_list:
                mode_data = modes[mode_name]
                face_vals = []
                face_present = []
                for fn in face_names:
                    key = f'{mname}_{fn}_rel_mae'
                    if key in mode_data:
                        face_vals.append(float(mode_data[key]))
                        face_present.append(True)
                    else:
                        face_vals.append(0.0)
                        face_present.append(False)

                x_offset = (gi - n_groups / 2 + 0.5) * bar_width
                color = MODE_COLORS[mode_name][mname]
                hatch = '//' if mname != 'ml' else ''
                bar_objs = ax_face.bar(
                    x_faces + x_offset, face_vals, bar_width,
                    color=color, edgecolor='black', linewidth=0.3,
                    hatch=hatch)
                for bi, present in enumerate(face_present):
                    if not present:
                        bar_objs[bi].set_alpha(0.0)

                label = f'{method_labels[mname]} — {MODE_LABELS[mode_name]}'
                legend_handles.append(bar_objs[0])
                legend_labels.append(label)
                gi += 1

        ax_face.set_ylim(0, 1.5)
        ax_face.set_xticks(x_faces)
        ax_face.set_xticklabels(face_names, fontsize=11)
        ax_face.set_ylabel('Relative MAE')
        ax_face.set_title('Per-Face Relative MAE  (truth >= 100 photons)')
        ax_face.legend(legend_handles, legend_labels,
                       fontsize=8, loc='upper right', ncol=2)

        fig.suptitle('Cross-Mode Inpainter Comparison', fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # Pages 2-4: per-face binned metrics
        for metric_key, suptitle, ylabel in page_defs:
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(6 * ncols, 5 * nrows),
                                     squeeze=False)
            axes_flat = axes.flatten()

            for idx, (face_int, face_name) in enumerate(active_faces):
                ax = axes_flat[idx]

                for mode_name in ['mc', 'data', 'sensorfront']:
                    if mode_name not in modes:
                        continue
                    mode_data = modes[mode_name]

                    for mname in method_order:
                        key_c = f'{mname}_{face_name}_centers'
                        key_m = f'{mname}_{face_name}_{metric_key}'
                        if key_c not in mode_data:
                            continue
                        centers = mode_data[key_c]
                        metric_vals = mode_data[key_m]

                        if metric_key in ('mae', 'rms'):
                            vals = metric_vals / centers
                        else:
                            vals = metric_vals / centers

                        style = METHOD_STYLES[mname]
                        color = MODE_COLORS[mode_name][mname]
                        label = f'{MODE_LABELS[mode_name]} {method_labels[mname]}'

                        _plot_valid(ax, centers, vals,
                                    style['marker'] + style['ls'],
                                    color=color, markersize=3,
                                    linewidth=style['lw'],
                                    alpha=0.8 if mname != 'ml' else 1.0,
                                    label=label)

                ax.set_xscale('log')
                ax.set_xlabel('Truth Npho [photons]')
                ax.set_ylabel(ylabel)
                ax.set_title(face_name)
                if metric_key in ('mae', 'rms'):
                    ax.set_ylim(0, 1.5)
                else:
                    ax.set_ylim(-1.5, 1.5)
                ax.legend(fontsize=6, loc='upper right', ncol=1)

            for idx in range(n_active, len(axes_flat)):
                axes_flat[idx].set_visible(False)

            fig.suptitle(suptitle, fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"[INFO] Saved {args.o}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-mode inpainter comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest='command')

    # Export subcommand
    exp = sub.add_parser('export', help='Export one mode to npz cache')
    exp.add_argument('mode', choices=['mc', 'data', 'sensorfront'],
                     help='Validation mode')
    exp.add_argument('--predictions', '-p', required=True,
                     help='ML prediction ROOT file (single inpainter step)')
    exp.add_argument('--baselines', '-b', default=None,
                     help='Standalone baselines ROOT file')
    exp.add_argument('--localfit', '-l', default=None,
                     help='LocalFit ROOT file')
    exp.add_argument('--output', '-o', required=True,
                     help='Output npz file')

    # Plot subcommand
    plt_cmd = sub.add_parser('plot', help='Plot cross-mode comparison')
    plt_cmd.add_argument('--mc', default=None, help='MC npz cache file')
    plt_cmd.add_argument('--sensorfront', default=None,
                         help='Sensorfront npz cache file')
    plt_cmd.add_argument('--data', default=None,
                         help='Real data npz cache file')
    plt_cmd.add_argument('-o', default='cross_mode_comparison.pdf',
                         help='Output PDF (default: cross_mode_comparison.pdf)')

    args = parser.parse_args()

    if args.command == 'export':
        # Remap for export_mode
        args.export = args.mode
        export_mode(args)
    elif args.command == 'plot':
        plot_comparison(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
