import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
import torch

from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP
)

def plot_event_faces(npho_tensor, title="Event Display", savepath=None, outer_mode="split", outer_fine_pool=None):
    """
    Visualizes the photon counts on the geometric faces.
    npho_tensor: (4760,) torch tensor or numpy array of photon counts.
    """
    if isinstance(npho_tensor, torch.Tensor):
        npho_tensor = npho_tensor.cpu().numpy()
        
    def get_img(idx_map):
        flat_idx = idx_map.reshape(-1)
        valid = (flat_idx >= 0) & (flat_idx < len(npho_tensor))
        img_flat = np.zeros_like(flat_idx, dtype=float)
        img_flat[valid] = npho_tensor[flat_idx[valid]]
        img_flat[~valid] = np.nan
        return img_flat.reshape(idx_map.shape)

    img_inner = get_img(INNER_INDEX_MAP)
    img_us = get_img(US_INDEX_MAP)
    img_ds = get_img(DS_INDEX_MAP)
    img_outer_coarse = get_img(OUTER_COARSE_FULL_INDEX_MAP)
    img_outer_center = get_img(OUTER_CENTER_INDEX_MAP)

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(3, 4, height_ratios=[1, 2, 1])

    cmap = "viridis"
    vmax = np.nanmax(npho_tensor) if np.nanmax(npho_tensor) > 0 else 1.0
    norm = Normalize(vmin=0, vmax=vmax)

    ax_inner = fig.add_subplot(gs[1, 1:3])
    im = ax_inner.imshow(img_inner.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_inner.set_title("Inner Face")
    plt.colorbar(im, ax=ax_inner, label="Npho")

    ax_us = fig.add_subplot(gs[1, 0])
    ax_us.imshow(img_us.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_us.set_title("Upstream")

    ax_ds = fig.add_subplot(gs[1, 3])
    ax_ds.imshow(img_ds.T, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_ds.set_title("Downstream")

    ax_out_c = fig.add_subplot(gs[0, 1])
    ax_out_c.imshow(img_outer_coarse, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_out_c.set_title("Outer (Coarse)")

    ax_out_ctr = fig.add_subplot(gs[0, 2])
    ax_out_ctr.imshow(img_outer_center, origin='lower', cmap=cmap, norm=norm, aspect='auto')
    ax_out_ctr.set_title("Outer (Center)")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if savepath:
        plt.savefig(savepath, dpi=100)
        plt.close()
    else:
        plt.show()

def plot_event_time(*args, **kwargs):
    pass