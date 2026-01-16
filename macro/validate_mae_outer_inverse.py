# Usage:
# python macro/validate_mae_outer_inverse.py --file artifacts/<RUN_NAME>/mae_predictions_epoch_1.root --pool 3 3 --max_events 200
import sys
import os
import argparse
import numpy as np
import uproot
import torch
import torch.nn.functional as F

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from lib.geom_utils import build_outer_fine_grid_tensor, gather_face
    from lib.geom_defs import (
        OUTER_COARSE_FULL_INDEX_MAP,
        OUTER_CENTER_INDEX_MAP,
        OUTER_FINE_COARSE_SCALE,
        OUTER_FINE_CENTER_SCALE,
        OUTER_FINE_CENTER_START,
    )
except ImportError:
    print("Error: Could not import geometry helpers.")
    print("Run from repo root (xec-ml-wl/) or set PYTHONPATH.")
    sys.exit(1)


def reconstruct_outer_from_fine(fine_pred, pool_kernel):
    fine = fine_pred
    if pool_kernel:
        if isinstance(pool_kernel, int):
            scale = (pool_kernel, pool_kernel)
        else:
            scale = tuple(pool_kernel)
        fine = F.interpolate(fine, scale_factor=scale, mode="nearest")

    cr, cc = OUTER_FINE_COARSE_SCALE
    Hc, Wc = OUTER_COARSE_FULL_INDEX_MAP.shape
    npho = fine[:, 0:1].contiguous().view(-1, 1, Hc, cr, Wc, cc).sum(dim=(3, 5))
    time = fine[:, 1:2].contiguous().view(-1, 1, Hc, cr, Wc, cc).mean(dim=(3, 5))
    coarse_pred = torch.cat([npho, time], dim=1)

    sr, sc = OUTER_FINE_CENTER_SCALE
    Hc_center, Wc_center = OUTER_CENTER_INDEX_MAP.shape
    top = OUTER_FINE_CENTER_START[0] * cr
    left = OUTER_FINE_CENTER_START[1] * cc
    center_fine = fine[:, :, top:top + Hc_center * sr, left:left + Wc_center * sc]
    c_npho = center_fine[:, 0:1].contiguous().view(-1, 1, Hc_center, sr, Wc_center, sc).sum(dim=(3, 5))
    c_time = center_fine[:, 1:2].contiguous().view(-1, 1, Hc_center, sr, Wc_center, sc).mean(dim=(3, 5))
    center_pred = torch.cat([c_npho, c_time], dim=1)

    return coarse_pred, center_pred


def scatter_rect_face(full, face_pred, index_map):
    idx_flat = torch.tensor(index_map.reshape(-1), device=face_pred.device, dtype=torch.long)
    valid = idx_flat >= 0
    idx = idx_flat[valid]
    vals = face_pred.permute(0, 2, 3, 1).reshape(face_pred.size(0), -1, 2)[:, valid]
    full[:, idx, :] = vals


def compute_stats(pred, truth):
    diff = pred - truth
    mae = diff.abs().mean(dim=(0, 2, 3)).cpu().numpy()
    rmse = torch.sqrt((diff * diff).mean(dim=(0, 2, 3))).cpu().numpy()
    return mae, rmse


def compute_stats_flat(pred, truth):
    diff = pred - truth
    mae = diff.abs().mean(dim=(0, 1)).cpu().numpy()
    rmse = torch.sqrt((diff * diff).mean(dim=(0, 1))).cpu().numpy()
    return mae, rmse


def compute_stats_masked(pred, truth, mask):
    mask_t = torch.from_numpy(mask).to(pred.device).unsqueeze(0).unsqueeze(0).float()
    diff = (pred - truth) * mask_t
    denom = mask_t.sum(dim=(0, 2, 3)).clamp_min(1.0)
    mae = diff.abs().sum(dim=(0, 2, 3)) / denom
    rmse = torch.sqrt((diff * diff).sum(dim=(0, 2, 3)) / denom)
    return mae.cpu().numpy(), rmse.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Validate outer finegrid inverse mapping using truth arrays."
    )
    parser.add_argument("--file", type=str, required=True, help="MAE predictions ROOT file")
    parser.add_argument("--tree", type=str, default="tree", help="TTree name")
    parser.add_argument("--max_events", type=int, default=100, help="Max events to evaluate")
    parser.add_argument("--pool", type=int, nargs=2, default=None, help="Outer fine pool kernel, e.g. 3 3")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File not found at {args.file}")
        sys.exit(1)

    with uproot.open(args.file) as f:
        if args.tree not in f:
            print(f"Error: Tree '{args.tree}' not found. Available: {f.keys()}")
            sys.exit(1)

        tree = f[args.tree]
        n_entries = tree.num_entries
        n_read = min(args.max_events, n_entries)
        arrays = tree.arrays(["truth_npho", "truth_time"], library="np", entry_stop=n_read)

    truth_npho = arrays["truth_npho"].astype("float32")
    truth_time = arrays["truth_time"].astype("float32")
    x_truth = np.stack([truth_npho, truth_time], axis=-1)
    x_truth_t = torch.from_numpy(x_truth)

    fine = build_outer_fine_grid_tensor(x_truth_t, pool_kernel=args.pool)
    coarse_pred, center_pred = reconstruct_outer_from_fine(fine, args.pool)

    truth_coarse = gather_face(x_truth_t, OUTER_COARSE_FULL_INDEX_MAP)
    truth_center = gather_face(x_truth_t, OUTER_CENTER_INDEX_MAP)

    coarse_mae, coarse_rmse = compute_stats(coarse_pred, truth_coarse)
    center_mae, center_rmse = compute_stats(center_pred, truth_center)

    overlap_mask = np.isin(OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP)
    non_overlap_mask = ~overlap_mask
    coarse_no_overlap_mae, coarse_no_overlap_rmse = compute_stats_masked(
        coarse_pred, truth_coarse, non_overlap_mask
    )

    full_pred = torch.zeros_like(x_truth_t)
    scatter_rect_face(full_pred, coarse_pred, OUTER_COARSE_FULL_INDEX_MAP)
    scatter_rect_face(full_pred, center_pred, OUTER_CENTER_INDEX_MAP)

    coarse_idx = OUTER_COARSE_FULL_INDEX_MAP.reshape(-1)
    center_idx = OUTER_CENTER_INDEX_MAP.reshape(-1)
    idx_union = np.unique(np.concatenate([coarse_idx, center_idx]))
    idx_union = idx_union[idx_union >= 0]
    pred_union = full_pred[:, idx_union, :]
    truth_union = x_truth_t[:, idx_union, :]
    union_mae, union_rmse = compute_stats_flat(pred_union, truth_union)

    print(f"[INFO] Evaluated {n_read} events")
    print("Coarse face MAE (npho, time):", coarse_mae)
    print("Coarse face RMSE (npho, time):", coarse_rmse)
    print("Coarse no-overlap MAE (npho, time):", coarse_no_overlap_mae)
    print("Coarse no-overlap RMSE (npho, time):", coarse_no_overlap_rmse)
    print("Center face MAE (npho, time):", center_mae)
    print("Center face RMSE (npho, time):", center_rmse)
    print("Outer union MAE (npho, time):", union_mae)
    print("Outer union RMSE (npho, time):", union_rmse)


if __name__ == "__main__":
    main()
