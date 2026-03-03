#!/usr/bin/env python3
"""
LocalFitBaseline Event Diagnostic Visualization.

Shows per-event fit quality diagnostics for the 2-stage local fit
(Stage 1: U/V projection fits, Stage 2: solid-angle MINUIT fit).
Useful for diagnosing fit failures caused by pileup gammas.

Usage:
    # 1. Run LocalFitBaseline macro
    root -l -b -q 'others/LocalFitBaseline.C("input.root", "", "localfit_output.root", "perevent_dead.txt")'

    # 2. Visualize selected events
    python macro/show_localfit_event.py \
        --localfit-root localfit_output.root \
        --input-root input.root \
        --events 0 5 10 42 \
        --output localfit_diagnostics.pdf

    # 3. Visualize all events (capped)
    python macro/show_localfit_event.py \
        --localfit-root localfit_output.root \
        --input-root input.root \
        --all --max-events 50 \
        --output localfit_diagnostics.pdf
"""

from __future__ import annotations

import argparse
import os
import sys
import numpy as np
import uproot
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.sensor_geometry import load_sensor_positions

# =========================================================================
#  Constants (from LocalFitBaseline.C)
# =========================================================================

N_INNER = 4092
N_COLS = 44
N_ROWS = 93
SIPM_SIZE = 1.2          # cm
XERIN = 64.84            # cm
MPPC_HEIGHT = 0.13       # cm
REFF = XERIN + MPPC_HEIGHT  # 64.97 cm
XE_PHI_RAD = 125.52 * np.pi / 180.0
ATTEN = 1e6              # cm (effectively infinite)
QE_SIPM = 0.12

# PM intervals
PM_INTERVAL_U = 1.5097727  # cm
PM_INTERVAL_V = XE_PHI_RAD * REFF / N_ROWS  # cm

# Stage 1 fixed parameters (in PM units)
HALF_SIZE_U = SIPM_SIZE / (2.0 * PM_INTERVAL_U)
HALF_PHI_V = np.arcsin(SIPM_SIZE / XERIN) / 2.0
ATTEN_U = ATTEN / PM_INTERVAL_U
ATTEN_V = ATTEN / PM_INTERVAL_V
RADIUS_V = XERIN / PM_INTERVAL_V

# Solid angle chip layout
CHIP_DISTANCE = 0.05     # cm (0.5 mm gap)
CHIP_SIZE = 0.59          # cm (5.9 mm chip)


# =========================================================================
#  Stage 1 fit functions (matching FitFunc1PointU/V in LocalFitBaseline.C)
# =========================================================================

def fit_func_1point_u(x, scale, u_pm, w_pm,
                      half_size=HALF_SIZE_U, atten=ATTEN_U):
    """Stage 1 U projection: scale * arctan_profile * exp_attenuation."""
    w_pm = np.maximum(w_pm, 1e-4)
    fitval = (np.arctan2((x - u_pm) - half_size, w_pm)
              - np.arctan2((x - u_pm) + half_size, w_pm))
    norm = np.arctan2(-half_size, w_pm) - np.arctan2(half_size, w_pm)
    with np.errstate(invalid='ignore', divide='ignore'):
        fitval = np.where(norm != 0, fitval / norm, 0.0)
    dist = np.sqrt((x - u_pm)**2 + w_pm**2)
    return scale * fitval * np.exp(-dist / atten)


def fit_func_1point_v(x, scale, v_pm, w_pm,
                      half_phi=HALF_PHI_V, radius=RADIUS_V,
                      atten=ATTEN_V):
    """Stage 1 V projection: scale * arctan_profile * exp_attenuation."""
    w_pm = np.maximum(w_pm, 1e-4)
    half_phi = np.maximum(half_phi, 1e-4)
    phi_plus = (x - v_pm) / radius + half_phi
    phi_minus = (x - v_pm) / radius - half_phi
    fitval = (np.arctan2(radius * np.sin(phi_plus),
                         w_pm + radius - radius * np.cos(phi_plus))
              - np.arctan2(radius * np.sin(phi_minus),
                           w_pm + radius - radius * np.cos(phi_minus)))
    norm = (np.arctan2(radius * np.sin(half_phi),
                       w_pm + radius - radius * np.cos(half_phi))
            - np.arctan2(radius * np.sin(-half_phi),
                         w_pm + radius - radius * np.cos(-half_phi)))
    with np.errstate(invalid='ignore', divide='ignore'):
        fitval = np.where(norm != 0, fitval / norm, 0.0)
    dist = np.sqrt((x - v_pm)**2 + w_pm**2)
    return scale * fitval * np.exp(-dist / atten)


# =========================================================================
#  Coordinate transforms (matching LocalFitBaseline.C)
# =========================================================================

def uvw_to_xyz(uvw: np.ndarray) -> np.ndarray:
    """Convert UVW to XYZ. Vectorized. uvw: (..., 3)."""
    u, v, w = uvw[..., 0], uvw[..., 1], uvw[..., 2]
    x = -1.0 * (w + REFF) * np.cos(v / REFF)
    y = (w + REFF) * np.sin(v / REFF)
    z = u
    return np.stack([x, y, z], axis=-1)


def xyz_to_uvw(xyz: np.ndarray) -> np.ndarray:
    """Convert XYZ to UVW. Vectorized. xyz: (..., 3)."""
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    r = np.sqrt(x**2 + y**2)
    phi = np.where(
        (x == 0.0) & (y == 0.0),
        0.0,
        np.pi + np.arctan2(-y, -x),
    )
    u = z
    v = -(phi - np.pi) * REFF
    w = r - REFF
    return np.stack([u, v, w], axis=-1)


# =========================================================================
#  Analytical PM geometry (matching ComputeAnalyticalPMGeometry)
# =========================================================================

def compute_inner_pm_geometry():
    """Compute positions, normals, and UV coords for all inner SiPMs.

    Returns:
        pm_pos: (4092, 3) XYZ positions
        pm_dir: (4092, 3) normals (outward radial)
        pm_u:   (4092,) U coordinate in cm
        pm_v:   (4092,) V coordinate in cm
    """
    ch = np.arange(N_INNER)
    col = ch % N_COLS
    row = ch // N_COLS

    u = (col - (N_COLS - 1) / 2.0) * PM_INTERVAL_U
    v = -((row - (N_ROWS - 1) / 2.0) * PM_INTERVAL_V)
    w = np.zeros(N_INNER)

    uvw = np.stack([u, v, w], axis=-1)
    xyz = uvw_to_xyz(uvw)

    r = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
    normals = np.stack([xyz[:, 0] / r, xyz[:, 1] / r, np.zeros(N_INNER)], axis=-1)

    return xyz, normals, u, v


# =========================================================================
#  Solid angle computation (port of PMSolidAngleMPPC)
# =========================================================================

def _chip_solid_angle(view, corner1, corner2, unit):
    """Solid angle for one chip given two opposite corners."""
    v1 = corner1 - view
    v2 = corner2 - view

    d1_u = np.dot(v1, unit[0])
    d1_v = np.dot(v1, unit[1])
    d1_w = np.dot(v1, unit[2])
    d2_u = np.dot(v2, unit[0])
    d2_v = np.dot(v2, unit[1])
    d2_w = np.dot(v2, unit[2])

    sin_a1 = d1_u / np.sqrt(d1_u**2 + d1_w**2)
    sin_a2 = d2_u / np.sqrt(d2_u**2 + d2_w**2)
    sin_b1 = d1_v / np.sqrt(d1_v**2 + d1_w**2)
    sin_b2 = d2_v / np.sqrt(d2_v**2 + d2_w**2)

    omega = np.abs(
        np.arcsin(sin_a1 * sin_b1)
        + np.arcsin(sin_a2 * sin_b2)
        - np.arcsin(sin_a1 * sin_b2)
        - np.arcsin(sin_b1 * sin_a2)
    ) / (4.0 * np.pi)

    return omega


def pm_solid_angle_mppc(view, center, normal):
    """Compute solid angle of a 4-chip MPPC sensor from viewpoint.

    Args:
        view:   (3,) viewpoint XYZ
        center: (3,) sensor center XYZ
        normal: (3,) sensor normal (outward)

    Returns:
        Fractional solid angle (omega / 4pi already applied).
    """
    center_view = center - view
    if np.dot(center_view, normal) > 0:
        return 0.0

    # Local coordinate system
    unit_u = np.array([0.0, 0.0, 1.0])
    unit_v = np.cross(unit_u, normal)
    norm_v = np.linalg.norm(unit_v)
    if norm_v < 1e-12:
        return 0.0
    unit_v /= norm_v
    unit_w = normal / np.linalg.norm(normal)
    unit_u = np.cross(unit_v, unit_w)
    unit_u /= np.linalg.norm(unit_u)

    unit = np.array([unit_u, unit_v, unit_w])
    hd = CHIP_DISTANCE / 2.0

    total = 0.0
    for su, sv in [(+1, +1), (-1, +1), (+1, -1), (-1, -1)]:
        c1 = center + su * hd * unit_u + sv * hd * unit_v
        c2 = c1 + su * CHIP_SIZE * unit_u + sv * CHIP_SIZE * unit_v
        total += _chip_solid_angle(view, c1, c2, unit)

    return total


def compute_predicted_npho(view_xyz, fit_scale, pm_pos, pm_dir):
    """Compute predicted npho for all inner sensors.

    Args:
        view_xyz: (3,) fitted position in XYZ
        fit_scale: scale factor from MINUIT fit
        pm_pos: (N, 3) sensor positions
        pm_dir: (N, 3) sensor normals

    Returns:
        pred: (N,) predicted npho
    """
    n = len(pm_pos)
    pred = np.zeros(n)
    for ch in range(n):
        omega = pm_solid_angle_mppc(view_xyz, pm_pos[ch], pm_dir[ch])
        pred[ch] = fit_scale * omega
    return pred


# =========================================================================
#  Data loading
# =========================================================================

def load_localfit_data(localfit_root: str):
    """Load position and prediction trees from LocalFitBaseline output.

    Returns:
        pos: dict with arrays from position tree
        pred: dict with arrays from predictions tree
    """
    with uproot.open(localfit_root) as f:
        pos_tree = f["position"]
        pos = {
            "event_idx":    pos_tree["event_idx"].array(library="np"),
            "run_number":   pos_tree["run_number"].array(library="np"),
            "event_number": pos_tree["event_number"].array(library="np"),
            "uvwTruth":     pos_tree["uvwTruth"].array(library="np"),
            "uvwRecoFI":    pos_tree["uvwRecoFI"].array(library="np"),
            "uvwFitNoDead": pos_tree["uvwFitNoDead"].array(library="np"),
            "uvwStage1":    pos_tree["uvwStage1"].array(library="np"),
            "fitScale":     pos_tree["fitScale"].array(library="np"),
            "fitChisq":     pos_tree["fitChisq"].array(library="np"),
            "nPMUsed":      pos_tree["nPMUsed"].array(library="np"),
            "energyTruth":  pos_tree["energyTruth"].array(library="np"),
        }

        pred_tree = f["predictions"]
        pred = {
            "event_idx":  pred_tree["event_idx"].array(library="np"),
            "sensor_id":  pred_tree["sensor_id"].array(library="np"),
            "truth_npho": pred_tree["truth_npho"].array(library="np"),
            "pred_npho":  pred_tree["pred_npho"].array(library="np"),
            "error_npho": pred_tree["error_npho"].array(library="np"),
        }

    return pos, pred


def load_input_npho(input_root: str, event_indices: np.ndarray,
                    pos: dict = None):
    """Load npho[4760] for selected events from input ROOT file.

    When ``pos`` is provided (with run_number/event_number arrays), events
    are matched by (run, event) instead of by entry index.  This is
    essential when the localfit was run on a filtered file (e.g. sensorfront
    validation) where event_idx refers to filtered-file entry indices,
    not the original file.

    Returns:
        npho_dict: dict mapping event_idx -> npho array (4760,)
        uvw_recofi_dict: dict mapping event_idx -> uvwRecoFI (3,)
    """
    npho_dict = {}
    uvw_dict = {}
    with uproot.open(input_root) as f:
        tree = f["tree"]
        npho_all = tree["npho"].array(library="np")
        uvw_all = tree["uvwRecoFI"].array(library="np")

        # Match by (run, event) if available
        if (pos is not None and "run_number" in pos
                and "event_number" in pos):
            run_all = tree["run"].array(library="np") if "run" in tree.keys() else None
            event_all = tree["event"].array(library="np") if "event" in tree.keys() else None

            if run_all is not None and event_all is not None:
                # Build lookup: (run, event) -> entry index in original file
                orig_lookup = {}
                for j in range(len(run_all)):
                    orig_lookup[(int(run_all[j]), int(event_all[j]))] = j

                for ev_idx in event_indices:
                    ev_idx = int(ev_idx)
                    pi = np.where(pos["event_idx"] == ev_idx)[0]
                    if len(pi) == 0:
                        continue
                    pi = pi[0]
                    key = (int(pos["run_number"][pi]),
                           int(pos["event_number"][pi]))
                    orig_j = orig_lookup.get(key)
                    if orig_j is not None:
                        npho_dict[ev_idx] = npho_all[orig_j]
                        uvw_dict[ev_idx] = uvw_all[orig_j]
                    else:
                        print(f"[WARN] Event {ev_idx} (run={key[0]}, "
                              f"event={key[1]}) not found in input file")
                return npho_dict, uvw_dict

        # Fallback: use event_idx as direct entry index
        for idx in event_indices:
            if idx < len(npho_all):
                npho_dict[int(idx)] = npho_all[idx]
                uvw_dict[int(idx)] = uvw_all[idx]
    return npho_dict, uvw_dict


# =========================================================================
#  Projection building
# =========================================================================

def build_projections(npho_inner, dead_channels=None):
    """Build U and V projections from inner face npho.

    Args:
        npho_inner: (4092,) npho values for inner sensors
        dead_channels: set of dead channel IDs to exclude

    Returns:
        proj_u: (44,) summed along V (columns)
        proj_v: (93,) summed along U (rows)
    """
    if dead_channels is None:
        dead_channels = set()

    proj_u = np.zeros(N_COLS)
    proj_v = np.zeros(N_ROWS)

    for ch in range(N_INNER):
        if ch in dead_channels:
            continue
        n = npho_inner[ch]
        if not np.isfinite(n) or n <= 0:
            continue
        col = ch % N_COLS
        row = ch // N_COLS
        proj_u[col] += n
        proj_v[row] += n

    return proj_u, proj_v


# =========================================================================
#  Plotting
# =========================================================================

def plot_event_page(fig, event_info, pm_pos, pm_dir, pm_u, pm_v):
    """Plot one event's diagnostic page.

    Args:
        fig: matplotlib Figure
        event_info: dict with all event data
        pm_pos, pm_dir, pm_u, pm_v: inner PM geometry
    """
    ei = event_info

    # Layout: 3 rows (text + 2x3 grid)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.15, 1, 1],
                          hspace=0.35, wspace=0.35,
                          left=0.06, right=0.94, top=0.95, bottom=0.05)

    # --- Text panel ---
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis("off")

    dead_str = ""
    if ei["dead_sensors"]:
        for ds in ei["dead_sensors"]:
            dead_str += (f"  Dead sensor {ds['sid']}: "
                         f"truth={ds['truth']:.0f}, "
                         f"pred={ds['pred']:.0f}, "
                         f"error={ds['error']:.0f}\n")

    text = (
        f"Event {ei['event_idx']}  |  "
        f"Energy = {ei['energy'] * 1000:.1f} MeV\n"
        f"Truth:     U={ei['uvw_truth'][0]:7.2f}  "
        f"V={ei['uvw_truth'][1]:7.2f}  "
        f"W={ei['uvw_truth'][2]:7.2f} cm\n"
        f"RecoFI:    U={ei['uvw_recofi'][0]:7.2f}  "
        f"V={ei['uvw_recofi'][1]:7.2f}  "
        f"W={ei['uvw_recofi'][2]:7.2f} cm\n"
        f"Stage1:    U={ei['uvw_stage1'][0]:7.2f}  "
        f"V={ei['uvw_stage1'][1]:7.2f}  "
        f"W={ei['uvw_stage1'][2]:7.2f} cm\n"
        f"FitNoDead: U={ei['uvw_fit'][0]:7.2f}  "
        f"V={ei['uvw_fit'][1]:7.2f}  "
        f"W={ei['uvw_fit'][2]:7.2f} cm\n"
        f"chi2/ndf = {ei['chi2ndf']:.2f}  |  "
        f"nPMUsed = {ei['n_pm_used']}  |  "
        f"fitScale = {ei['fit_scale']:.3e}\n"
    )
    if dead_str:
        text += dead_str

    ax_text.text(0.02, 0.95, text, transform=ax_text.transAxes,
                 fontsize=7, fontfamily="monospace",
                 verticalalignment="top")

    # Pre-compute Stage 2 predicted npho for all inner sensors (shared by panels)
    fit_xyz = uvw_to_xyz(ei["uvw_fit"])
    pred_all = compute_predicted_npho(fit_xyz, ei["fit_scale"], pm_pos, pm_dir)
    ei["_pred_inner"] = pred_all


    # --- Panel (A): U Projection + Stage 2 prediction ---
    ax_u = fig.add_subplot(gs[1, 0])
    _plot_u_projection(ax_u, ei)

    # --- Panel (B): V Projection + Stage 2 prediction ---
    ax_v = fig.add_subplot(gs[1, 1])
    _plot_v_projection(ax_v, ei)

    # --- Panel (C): 2D Truth Npho ---
    ax_truth = fig.add_subplot(gs[1, 2])
    vmin, vmax = _plot_2d_truth(ax_truth, ei)

    # --- Panel (D): 2D Predicted Npho ---
    ax_pred = fig.add_subplot(gs[2, 0])
    _plot_2d_pred(ax_pred, ei, vmin, vmax)

    # --- Panel (E): 2D Residual ---
    ax_resid = fig.add_subplot(gs[2, 1])
    _plot_2d_residual(ax_resid, ei)

    # --- Panel (F): Truth vs Predicted Scatter ---
    ax_scatter = fig.add_subplot(gs[2, 2])
    _plot_scatter(ax_scatter, ei)


def _uvw_to_grid(u_cm, v_cm):
    """Convert UVW cm coords to grid (col, row) float indices."""
    col = u_cm / PM_INTERVAL_U + (N_COLS - 1) / 2.0
    row = -v_cm / PM_INTERVAL_V + (N_ROWS - 1) / 2.0
    return col, row


def _plot_u_projection(ax, ei):
    """Panel A: U projection of data + Stage 1 fit curve + Stage 2 projection."""
    x_bins = np.arange(N_COLS) - N_COLS / 2.0 + 0.5
    ax.bar(x_bins, ei["proj_u"], width=0.9, color="steelblue", alpha=0.7,
           label="Data")

    # Stage 1 analytical fit curve (arctan + attenuation)
    u_pm = ei["uvw_stage1"][0] / PM_INTERVAL_U
    w_pm = ei["uvw_stage1"][2] / PM_INTERVAL_U
    # Estimate scale: match peak of fit function to data in a ±5 PM region
    region = 5.0
    mask = (x_bins >= u_pm - region) & (x_bins <= u_pm + region)
    if mask.any():
        shape_at_bins = fit_func_1point_u(x_bins[mask], 1.0, u_pm, w_pm)
        peak_shape = shape_at_bins.max()
        peak_data = ei["proj_u"][mask].max()
        scale_est = peak_data / peak_shape if peak_shape > 0 else 0.0
    else:
        scale_est = 0.0
    x_fine = np.linspace(x_bins[0] - 0.5, x_bins[-1] + 0.5, 300)
    y_s1 = fit_func_1point_u(x_fine, scale_est, u_pm, w_pm)
    ax.plot(x_fine, y_s1, "orange", linewidth=1.5, label="Stage1 fit")

    # Stage 2 predicted projection (solid-angle model)
    s2 = ei["_pred_inner"].copy()
    s2[~np.isfinite(s2) | (s2 < 0)] = 0.0
    s2_grid = s2[:N_INNER].reshape(N_ROWS, N_COLS)
    s2_proj_u = s2_grid.sum(axis=0)  # (44,)
    ax.bar(x_bins, s2_proj_u, width=0.9, color="red", alpha=0.3,
           label="Stage2")

    ax.set_xlabel("U [PM units]", fontsize=8)
    ax.set_ylabel("Sum npho", fontsize=8)
    ax.set_title("(A) U Projection", fontsize=9)
    ax.legend(fontsize=5, loc="upper right")
    ax.tick_params(labelsize=7)


def _plot_v_projection(ax, ei):
    """Panel B: V projection of data + Stage 1 fit curve + Stage 2 projection."""
    x_bins = np.arange(N_ROWS) - N_ROWS / 2.0 + 0.5
    ax.bar(x_bins, ei["proj_v"], width=0.9, color="steelblue", alpha=0.7,
           label="Data")

    # Stage 1 analytical fit curve (arctan + attenuation)
    v_pm = -ei["uvw_stage1"][1] / PM_INTERVAL_V  # negative sign convention
    w_pm = ei["uvw_stage1"][2] / PM_INTERVAL_V
    # Estimate scale: match peak of fit function to data in a ±5 PM region
    region = 5.0
    mask = (x_bins >= v_pm - region) & (x_bins <= v_pm + region)
    if mask.any():
        shape_at_bins = fit_func_1point_v(x_bins[mask], 1.0, v_pm, w_pm)
        peak_shape = shape_at_bins.max()
        peak_data = ei["proj_v"][mask].max()
        scale_est = peak_data / peak_shape if peak_shape > 0 else 0.0
    else:
        scale_est = 0.0
    x_fine = np.linspace(x_bins[0] - 0.5, x_bins[-1] + 0.5, 500)
    y_s1 = fit_func_1point_v(x_fine, scale_est, v_pm, w_pm)
    ax.plot(x_fine, y_s1, "orange", linewidth=1.5, label="Stage1 fit")

    # Stage 2 predicted projection (solid-angle model)
    s2 = ei["_pred_inner"].copy()
    s2[~np.isfinite(s2) | (s2 < 0)] = 0.0
    s2_grid = s2[:N_INNER].reshape(N_ROWS, N_COLS)
    s2_proj_v = s2_grid.sum(axis=1)  # (93,)
    ax.bar(x_bins, s2_proj_v, width=0.9, color="red", alpha=0.3,
           label="Stage2")

    ax.set_xlabel("V [PM units]", fontsize=8)
    ax.set_ylabel("Sum npho", fontsize=8)
    ax.set_title("(B) V Projection", fontsize=9)
    ax.legend(fontsize=5, loc="upper right")
    ax.tick_params(labelsize=7)


def _inner_to_grid(npho_inner):
    """Reshape inner face npho (4092,) to (93, 44) grid."""
    grid = npho_inner[:N_INNER].reshape(N_ROWS, N_COLS)
    return grid


def _plot_2d_truth(ax, ei):
    """Panel C: 2D truth npho map."""
    npho = ei["npho_inner"].copy()
    npho[npho <= 0] = np.nan
    grid = _inner_to_grid(npho)

    valid = grid[np.isfinite(grid) & (grid > 0)]
    if len(valid) > 0:
        vmin = max(1, np.nanpercentile(valid, 1))
        vmax = np.nanpercentile(valid, 99.5)
    else:
        vmin, vmax = 1, 1000

    im = ax.imshow(grid, aspect="auto", origin="upper",
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Position markers (in grid coords: col=x, row=y)
    _add_position_markers(ax, ei)

    # Highlight dead sensors
    for ds in ei["dead_sensors"]:
        sid = ds["sid"]
        if sid < N_INNER:
            row, col = sid // N_COLS, sid % N_COLS
            rect = plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                 linewidth=2, edgecolor="magenta",
                                 facecolor="none")
            ax.add_patch(rect)

    ax.set_xlabel("U (col)", fontsize=8)
    ax.set_ylabel("V (row)", fontsize=8)
    ax.set_title("(C) Truth Npho", fontsize=9)
    ax.tick_params(labelsize=7)
    return vmin, vmax


def _plot_2d_pred(ax, ei, vmin, vmax):
    """Panel D: 2D predicted npho map (from Stage 2 solid-angle fit)."""
    pred = ei["_pred_inner"].copy()
    pred[pred <= 0] = np.nan

    grid = _inner_to_grid(pred)

    im = ax.imshow(grid, aspect="auto", origin="upper",
                   norm=LogNorm(vmin=vmin, vmax=vmax),
                   cmap="viridis", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _add_position_markers(ax, ei)

    ax.set_xlabel("U (col)", fontsize=8)
    ax.set_ylabel("V (row)", fontsize=8)
    ax.set_title("(D) Predicted Npho", fontsize=9)
    ax.tick_params(labelsize=7)


def _plot_2d_residual(ax, ei):
    """Panel E: 2D residual (truth - predicted) map."""
    truth = ei["npho_inner"][:N_INNER].copy().astype(float)
    pred = ei["_pred_inner"].copy()
    truth[truth <= 0] = 0.0
    pred[np.isnan(pred)] = 0.0
    resid = truth - pred
    grid = _inner_to_grid(resid)

    vmax = max(np.nanpercentile(np.abs(resid), 97), 1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(grid, aspect="auto", origin="upper",
                   norm=norm, cmap="RdBu_r", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _add_position_markers(ax, ei)

    ax.set_xlabel("U (col)", fontsize=8)
    ax.set_ylabel("V (row)", fontsize=8)
    ax.set_title("(E) Residual (T-P)", fontsize=9)
    ax.tick_params(labelsize=7)


def _plot_scatter(ax, ei):
    """Panel F: Truth vs Predicted scatter for sensors in fit region."""
    truth = ei["npho_inner"][:N_INNER].copy().astype(float)
    pred = ei["_pred_inner"].copy()

    # Select sensors in fit region (same as Stage 2 selection)
    stage1_u = ei["uvw_stage1"][0]
    stage1_v = ei["uvw_stage1"][1]
    region = 10.0
    region_cm_u = region * 1.05 * PM_INTERVAL_U
    region_cm_v = region * 1.05 * PM_INTERVAL_V

    ch = np.arange(N_INNER)
    pm_u_arr = (ch % N_COLS - (N_COLS - 1) / 2.0) * PM_INTERVAL_U
    pm_v_arr = -((ch // N_COLS - (N_ROWS - 1) / 2.0) * PM_INTERVAL_V)

    in_region = ((np.abs(stage1_u - pm_u_arr) <= region_cm_u) &
                 (np.abs(stage1_v - pm_v_arr) <= region_cm_v) &
                 (truth > 0) & np.isfinite(pred) & (pred > 0))

    dead_set = {ds["sid"] for ds in ei["dead_sensors"]}
    is_dead = np.array([ch_i in dead_set for ch_i in range(N_INNER)])

    mask_regular = in_region & ~is_dead
    mask_dead = in_region & is_dead

    if mask_regular.any():
        ax.scatter(truth[mask_regular], pred[mask_regular],
                   s=4, alpha=0.4, c="steelblue", edgecolors="none",
                   label="Used")
    if mask_dead.any():
        ax.scatter(truth[mask_dead], pred[mask_dead],
                   s=20, alpha=0.8, c="red", marker="x",
                   label="Dead")

    # Identity line
    all_vals = np.concatenate([truth[in_region], pred[in_region]])
    if len(all_vals) > 0:
        lo = max(0, np.min(all_vals) * 0.8)
        hi = np.max(all_vals) * 1.2
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

    ax.set_xlabel("Truth npho", fontsize=8)
    ax.set_ylabel("Predicted npho", fontsize=8)
    ax.set_title("(F) Truth vs Pred", fontsize=9)

    # Text annotation
    ax.text(0.03, 0.95,
            f"chi2/ndf={ei['chi2ndf']:.2f}\n"
            f"fitScale={ei['fit_scale']:.2e}\n"
            f"nPMUsed={ei['n_pm_used']}",
            transform=ax.transAxes, fontsize=6, fontfamily="monospace",
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))

    ax.legend(fontsize=5, loc="lower right")
    ax.tick_params(labelsize=7)


def _add_position_markers(ax, ei):
    """Add truth/fit/recofi position markers to a 2D grid plot."""
    truth_col, truth_row = _uvw_to_grid(ei["uvw_truth"][0], ei["uvw_truth"][1])
    fit_col, fit_row = _uvw_to_grid(ei["uvw_fit"][0], ei["uvw_fit"][1])
    recofi_col, recofi_row = _uvw_to_grid(ei["uvw_recofi"][0],
                                            ei["uvw_recofi"][1])
    stage1_col, stage1_row = _uvw_to_grid(ei["uvw_stage1"][0],
                                            ei["uvw_stage1"][1])

    ax.plot(truth_col, truth_row, marker="*", color="lime",
            markersize=10, markeredgecolor="black", markeredgewidth=0.5,
            label="Truth", zorder=10)
    ax.plot(fit_col, fit_row, marker="o", color="red",
            markersize=7, markeredgecolor="black", markeredgewidth=0.5,
            label="Fit", zorder=10)
    ax.plot(recofi_col, recofi_row, marker="D", color="cyan",
            markersize=5, markeredgecolor="black", markeredgewidth=0.5,
            label="RecoFI", zorder=10)
    ax.plot(stage1_col, stage1_row, marker="s", color="yellow",
            markersize=5, markeredgecolor="black", markeredgewidth=0.5,
            label="Stage1", zorder=10)


# =========================================================================
#  Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LocalFitBaseline per-event diagnostic visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("--localfit-root", required=True,
                        help="LocalFitBaseline output ROOT file")
    parser.add_argument("--input-root", required=True,
                        help="Original input ROOT file (has npho[4760])")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--events", type=int, nargs="+",
                       help="Event indices to visualize")
    group.add_argument("--all", action="store_true",
                       help="Visualize all events")
    parser.add_argument("--max-events", type=int, default=50,
                        help="Cap when using --all (default: 50)")
    parser.add_argument("--output", "-o", type=str,
                        default="localfit_diagnostics.pdf",
                        help="Output PDF path")

    args = parser.parse_args()

    print(f"Loading LocalFit results from {args.localfit_root} ...")
    pos, pred = load_localfit_data(args.localfit_root)

    # Determine which events to plot
    available_events = pos["event_idx"]
    if args.all:
        event_indices = available_events[:args.max_events]
    else:
        event_indices = np.array(args.events, dtype=np.int32)
        # Filter to available events
        avail_set = set(available_events)
        missing = [e for e in event_indices if e not in avail_set]
        if missing:
            print(f"WARNING: Events {missing} not found in localfit output, "
                  f"skipping")
        event_indices = np.array([e for e in event_indices if e in avail_set],
                                 dtype=np.int32)

    if len(event_indices) == 0:
        print("ERROR: No events to visualize")
        sys.exit(1)

    print(f"Loading npho from {args.input_root} for "
          f"{len(event_indices)} events ...")
    npho_dict, uvw_recofi_dict = load_input_npho(
        args.input_root, event_indices, pos=pos)

    # Compute PM geometry once
    print("Computing inner PM geometry ...")
    pm_pos, pm_dir, pm_u, pm_v = compute_inner_pm_geometry()

    # Build per-event info and plot
    print(f"Generating {len(event_indices)} pages ...")
    with PdfPages(args.output) as pdf:
        for i, ev_idx in enumerate(event_indices):
            ev_idx = int(ev_idx)
            if ev_idx not in npho_dict:
                print(f"  Event {ev_idx}: npho not available, skipping")
                continue

            # Position data
            pos_mask = pos["event_idx"] == ev_idx
            if not pos_mask.any():
                continue
            pi = np.where(pos_mask)[0][0]

            # Prediction data for dead channels
            pred_mask = pred["event_idx"] == ev_idx
            dead_sensors = []
            if pred_mask.any():
                for j in np.where(pred_mask)[0]:
                    dead_sensors.append({
                        "sid": int(pred["sensor_id"][j]),
                        "truth": float(pred["truth_npho"][j]),
                        "pred": float(pred["pred_npho"][j]),
                        "error": float(pred["error_npho"][j]),
                    })

            npho_full = npho_dict[ev_idx]
            npho_inner = npho_full[:N_INNER].copy()

            dead_ch_set = {ds["sid"] for ds in dead_sensors}
            proj_u, proj_v = build_projections(npho_inner, dead_ch_set)

            event_info = {
                "event_idx":   ev_idx,
                "energy":      float(pos["energyTruth"][pi]),
                "uvw_truth":   pos["uvwTruth"][pi],
                "uvw_recofi":  pos["uvwRecoFI"][pi],
                "uvw_stage1":  pos["uvwStage1"][pi],
                "uvw_fit":     pos["uvwFitNoDead"][pi],
                "fit_scale":   float(pos["fitScale"][pi]),
                "chi2ndf":     float(pos["fitChisq"][pi]),
                "n_pm_used":   int(pos["nPMUsed"][pi]),
                "dead_sensors": dead_sensors,
                "npho_inner":  npho_inner,
                "proj_u":      proj_u,
                "proj_v":      proj_v,
            }

            fig = plt.figure(figsize=(14, 10))
            plot_event_page(fig, event_info, pm_pos, pm_dir, pm_u, pm_v)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)

            if (i + 1) % 10 == 0 or i == 0:
                print(f"  {i + 1}/{len(event_indices)} pages done")

    print(f"Saved {len(event_indices)} pages to {args.output}")


if __name__ == "__main__":
    main()
