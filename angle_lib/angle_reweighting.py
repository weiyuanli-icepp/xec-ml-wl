import numpy as np
import os
from .angle_utils import iterate_chunks

def scan_angle_hist_1d(root, tree="tree", comp=0, nbins=30, step_size=4000):
    root = os.path.expanduser(root)
    vmin, vmax = +np.inf, -np.inf
    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        vals = ang[:, comp]
        if vals.size == 0: continue
        vmin = min(vmin, vals.min())
        vmax = max(vmax, vals.max())

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError("scan_angle_hist_1d: invalid range")

    edges = np.linspace(vmin, vmax, nbins + 1)
    counts = np.zeros(nbins, dtype=np.int64)

    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        vals = ang[:, comp]
        if vals.size == 0: continue
        h, _ = np.histogram(vals, bins=edges)
        counts += h
        
    weights = np.zeros(nbins, dtype=np.float64)
    valid = counts > 0
    if valid.any():
        total_counts = counts.sum()
        k = total_counts / valid.sum()
        weights[valid] = k / counts[valid]

    return edges, weights

def scan_angle_hist_2d(root, tree="tree", nbins_theta=20, nbins_phi=20, step_size=4000):
    root = os.path.expanduser(root)
    th_min, th_max = +np.inf, -np.inf
    ph_min, ph_max = +np.inf, -np.inf

    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        theta = ang[:, 0]; phi = ang[:, 1]
        if theta.size == 0: continue
        th_min = min(th_min, theta.min()); th_max = max(th_max, theta.max())
        ph_min = min(ph_min, phi.min()); ph_max = max(ph_max, phi.max())

    if not (np.isfinite(th_min) and np.isfinite(th_max) and np.isfinite(ph_min) and np.isfinite(ph_max)):
        raise RuntimeError("scan_angle_hist_2d: invalid range")

    edges_theta = np.linspace(th_min, th_max, nbins_theta + 1)
    edges_phi   = np.linspace(ph_min, ph_max, nbins_phi + 1)
    counts = np.zeros((nbins_theta, nbins_phi), dtype=np.int64)

    for arr in iterate_chunks(root, tree, ["emiAng"], step_size):
        ang = arr["emiAng"].astype("float64")
        theta = ang[:, 0]; phi = ang[:, 1]
        if theta.size == 0: continue
        h, _, _ = np.histogram2d(theta, phi, bins=[edges_theta, edges_phi])
        counts += h.astype(np.int64)
        
    weights_2d = np.zeros_like(counts, dtype=np.float64)
    valid = counts > 0
    if valid.any():
        total_counts = counts.sum()
        k = total_counts / valid.sum()
        weights_2d[valid] = k / counts[valid]

    return edges_theta, edges_phi, weights_2d