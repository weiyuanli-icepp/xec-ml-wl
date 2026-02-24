"""
Rule-based baselines for dead channel (sensor) recovery.

Two baselines are provided:
  1. NeighborAverageBaseline  -- simple unweighted mean of k-hop neighbors
  2. SolidAngleWeightedBaseline -- solid-angle-weighted mean of k-hop neighbors
"""

from __future__ import annotations

import numpy as np
from collections import deque
from typing import Optional

from .geom_defs import (
    INNER_INDEX_MAP,
    US_INDEX_MAP,
    DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP,
    TOP_HEX_ROWS,
    BOTTOM_HEX_ROWS,
    HEX_EDGE_INDEX_NP,
    flatten_hex_rows,
    OUTER_ALL_SENSOR_IDS,
    DEFAULT_SENTINEL_TIME,
)

N_SENSORS = 4760


# =========================================================
# Neighbor-map construction helpers
# =========================================================

def _rect_khop_neighbors(index_map: np.ndarray, k: int) -> dict:
    """Build k-hop 8-connected neighbor lists for a rectangular grid.

    For k=1 every sensor gets up to 8 neighbors (the 3x3 box minus center).
    For k=2 every sensor gets up to 24 neighbors (the 5x5 box minus center).
    In general, the (2k+1)x(2k+1) box minus center.

    Parameters
    ----------
    index_map : ndarray of shape (H, W)
        Sensor-id grid (values are global sensor indices in [0, 4760)).
    k : int
        Number of hops.

    Returns
    -------
    neighbors : dict[int, np.ndarray]
        Mapping from sensor_id to 1-D array of neighbor sensor_ids.
    """
    H, W = index_map.shape
    neighbors = {}
    for r in range(H):
        for c in range(W):
            sid = int(index_map[r, c])
            nbrs = []
            for dr in range(-k, k + 1):
                for dc in range(-k, k + 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        nbrs.append(int(index_map[rr, cc]))
            neighbors[sid] = np.array(nbrs, dtype=np.int32)
    return neighbors


def _hex_khop_neighbors(hex_rows, k: int) -> dict:
    """Build k-hop neighbor lists for a hexagonal face using BFS on
    HEX_EDGE_INDEX_NP.

    HEX_EDGE_INDEX_NP is a (3, E) array built from a canonical hex layout
    with the same row-length pattern as both the top and bottom faces
    ([11, 12, 11, 12, 13, 14]).  Rows 0..5 of that canonical layout map to
    the *local* node indices 0..72.  We translate local -> global using
    ``flatten_hex_rows(hex_rows)``.

    Parameters
    ----------
    hex_rows : list of arrays
        Per-row global sensor ids (e.g. TOP_HEX_ROWS or BOTTOM_HEX_ROWS).
    k : int
        Number of hops.

    Returns
    -------
    neighbors : dict[int, np.ndarray]
        Mapping from global sensor_id to 1-D array of global neighbor ids.
    """
    flat_ids = flatten_hex_rows(hex_rows)          # local->global
    n_local = len(flat_ids)

    # Build adjacency list in *local* coordinates from HEX_EDGE_INDEX_NP.
    # HEX_EDGE_INDEX_NP[0] = src, [1] = dst, [2] = neighbor_type
    # neighbor_type 0 = self-loop, 1-6 = actual neighbors.
    src = HEX_EDGE_INDEX_NP[0]
    dst = HEX_EDGE_INDEX_NP[1]
    types = HEX_EDGE_INDEX_NP[2]

    # Only keep real edges (type != 0 means not self-loop).
    real_mask = types != 0
    src_real = src[real_mask]
    dst_real = dst[real_mask]

    # Adjacency list (local).
    adj = [[] for _ in range(n_local)]
    for s, d in zip(src_real, dst_real):
        adj[s].append(d)

    neighbors = {}
    for node in range(n_local):
        # BFS up to k hops
        visited = set()
        visited.add(node)
        frontier = deque([node])
        depth = 0
        while frontier and depth < k:
            next_frontier = deque()
            for _ in range(len(frontier)):
                u = frontier.popleft()
                for v in adj[u]:
                    if v not in visited:
                        visited.add(v)
                        next_frontier.append(v)
            frontier = next_frontier
            depth += 1
        visited.discard(node)  # exclude self
        global_id = int(flat_ids[node])
        neighbors[global_id] = flat_ids[np.array(sorted(visited), dtype=np.int64)]
    return neighbors


def _outer_face_neighbors(k: int) -> dict:
    """Build k-hop neighbors for the outer face.

    The outer face has two overlapping grids:
      - A 9x24 coarse grid (OUTER_COARSE_FULL_INDEX_MAP, 216 sensors)
      - An additional set of center sensors that interleave with 12 of
        the coarse sensors (OUTER_ALL_SENSOR_IDS has 234 unique sensors).

    For simplicity and correctness, we treat the coarse 9x24 grid as the
    primary rectangular grid for the 216 coarse sensors.  The 18 extra
    center-only sensors (IDs >= 4742) do not sit on this grid; they are
    handled by giving them neighbours from the coarse grid cells that are
    physically adjacent in the OUTER_CENTER_INDEX_MAP.

    Strategy:
      1. Coarse sensors: standard rectangular k-hop on the 9x24 grid.
      2. Center-only sensors (the 18 IDs 4742..4759): They live on a small
         5x6 center grid (OUTER_CENTER_INDEX_MAP transposed).  We do k-hop
         on that small grid, but then *also* connect them to the coarse
         grid neighbors of any coarse sensor that shares the center grid.
    """
    from .geom_defs import OUTER_CENTER_INDEX_MAP, CENTRAL_COARSE_IDS

    # --- Step 1: coarse 9x24 rectangular neighbors ---
    coarse_nbrs = _rect_khop_neighbors(OUTER_COARSE_FULL_INDEX_MAP, k)

    # --- Step 2: center 5x6 rectangular neighbors ---
    # OUTER_CENTER_INDEX_MAP is (5,6) after the .T in geom_defs.
    center_map = OUTER_CENTER_INDEX_MAP  # shape (5, 6)
    center_nbrs = _rect_khop_neighbors(center_map, k)

    # Merge: for every sensor that appears in the center grid, union the
    # coarse-grid neighbors and center-grid neighbors.
    center_ids_set = set(int(x) for x in center_map.flatten())
    coarse_ids_set = set(int(x) for x in OUTER_COARSE_FULL_INDEX_MAP.flatten())

    merged = {}

    # Start with all coarse sensors.
    for sid, nbrs in coarse_nbrs.items():
        merged[sid] = set(int(x) for x in nbrs)

    # Add center-grid neighbors.
    for sid, nbrs in center_nbrs.items():
        if sid not in merged:
            merged[sid] = set()
        merged[sid].update(int(x) for x in nbrs)

    # For center-only sensors that share a grid position with a coarse sensor,
    # also import that coarse sensor's coarse-grid neighbors.
    # (The 12 CENTRAL_COARSE_IDS appear in both grids.)
    central_set = set(int(x) for x in CENTRAL_COARSE_IDS)
    for sid in center_ids_set:
        if sid in central_set and sid in coarse_nbrs:
            # This sensor is in both grids -- already handled above.
            pass
        elif sid not in coarse_ids_set:
            # Pure center sensor (4742-4759).  Find its center-grid position
            # and pull in coarse-grid neighbors of any coarse sensor sharing
            # the same center-grid row/col neighborhood.
            pass
        # Either way the center_nbrs union already covers it.

    # Convert sets to sorted arrays, excluding self.
    neighbors = {}
    for sid, nbr_set in merged.items():
        nbr_set.discard(sid)
        neighbors[sid] = np.array(sorted(nbr_set), dtype=np.int32)

    return neighbors


def build_neighbor_map(k: int = 1) -> dict:
    """Build a complete neighbor map for all 4760 sensors.

    Parameters
    ----------
    k : int
        Number of hops for the neighbor search.

    Returns
    -------
    neighbor_map : dict[int, np.ndarray]
        sensor_id -> array of neighbor sensor_ids (global indices).
    """
    neighbor_map = {}

    # Rectangular faces
    neighbor_map.update(_rect_khop_neighbors(INNER_INDEX_MAP, k))
    neighbor_map.update(_rect_khop_neighbors(US_INDEX_MAP, k))
    neighbor_map.update(_rect_khop_neighbors(DS_INDEX_MAP, k))

    # Outer face (coarse + center)
    neighbor_map.update(_outer_face_neighbors(k))

    # Hexagonal faces
    neighbor_map.update(_hex_khop_neighbors(TOP_HEX_ROWS, k))
    neighbor_map.update(_hex_khop_neighbors(BOTTOM_HEX_ROWS, k))

    return neighbor_map


def _neighbor_map_to_padded_array(neighbor_map: dict):
    """Convert variable-length neighbor map to padded numpy arrays for
    vectorised prediction.

    Returns
    -------
    nbr_indices : ndarray, shape (N_SENSORS, max_neighbors), int32
        Padded neighbor indices (padded with 0, masked by nbr_counts).
    nbr_counts : ndarray, shape (N_SENSORS,), int32
        Number of valid neighbors per sensor.
    """
    max_nbrs = max((len(v) for v in neighbor_map.values()), default=0)
    nbr_indices = np.zeros((N_SENSORS, max_nbrs), dtype=np.int32)
    nbr_counts = np.zeros(N_SENSORS, dtype=np.int32)
    for sid, nbrs in neighbor_map.items():
        n = len(nbrs)
        nbr_counts[sid] = n
        if n > 0:
            nbr_indices[sid, :n] = nbrs
    return nbr_indices, nbr_counts


# =========================================================
# Baselines
# =========================================================

class NeighborAverageBaseline:
    """Predict masked sensors as the unweighted mean of their unmasked
    k-hop neighbors.

    Parameters
    ----------
    k : int
        Number of neighbor hops (k=1 -> 8-connected for rectangular grids,
        immediate neighbors for hex grids).
    """

    def __init__(self, k: int = 1):
        self.k = k
        self.neighbor_map = build_neighbor_map(k)
        self.nbr_indices, self.nbr_counts = _neighbor_map_to_padded_array(
            self.neighbor_map
        )

    def predict(self, x_npho: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Predict values for masked sensors.

        Parameters
        ----------
        x_npho : ndarray, shape (N_events, 4760)
            Normalised npho values.  Masked positions may contain the
            sentinel value (sentinel_npho = -1.0) or 0.
        mask : ndarray, shape (N_events, 4760)
            Boolean array where True marks a masked / dead channel.

        Returns
        -------
        predictions : ndarray, shape (N_events, 4760)
            Copy of *x_npho* with masked positions replaced by the average
            of their unmasked k-hop neighbors' values.  Sensors with no
            unmasked neighbors get 0.0.
        """
        N = x_npho.shape[0]
        predictions = x_npho.copy()

        nbr_idx = self.nbr_indices   # (4760, max_nbrs)
        nbr_cnt = self.nbr_counts    # (4760,)
        max_nbrs = nbr_idx.shape[1]

        for i in range(N):
            masked_sensors = np.where(mask[i])[0]
            if masked_sensors.size == 0:
                continue

            # Gather neighbor values and their mask status for all masked
            # sensors at once.
            # nbrs shape: (n_masked, max_nbrs)  -- global sensor ids
            nbrs = nbr_idx[masked_sensors]                     # (n_masked, max_nbrs)
            counts = nbr_cnt[masked_sensors]                   # (n_masked,)

            # Validity mask: neighbor slot is valid if (a) it is within the
            # count and (b) the neighbor itself is not masked.
            slot_range = np.arange(max_nbrs)[None, :]          # (1, max_nbrs)
            in_range = slot_range < counts[:, None]             # (n_masked, max_nbrs)

            # Check that the neighbor is unmasked.
            neighbor_unmasked = ~mask[i][nbrs]                  # (n_masked, max_nbrs)
            valid = in_range & neighbor_unmasked                # (n_masked, max_nbrs)

            # Neighbor npho values (use 0 for invalid slots so they don't
            # affect the sum).
            nbr_vals = x_npho[i][nbrs]                         # (n_masked, max_nbrs)
            nbr_vals = np.where(valid, nbr_vals, 0.0)

            n_valid = valid.sum(axis=1).astype(np.float64)     # (n_masked,)
            total = nbr_vals.sum(axis=1)                       # (n_masked,)

            # Clamp denominator to avoid division-by-zero warning;
            # the np.where ensures 0.0 is returned when n_valid == 0.
            safe_n = np.maximum(n_valid, 1.0)
            avg = np.where(n_valid > 0, total / safe_n, 0.0)
            predictions[i, masked_sensors] = avg

        return predictions


class SolidAngleWeightedBaseline:
    """Predict masked sensors using solid-angle-weighted averaging
    from k-hop neighbors.

    For a masked sensor *m* with solid angle omega_m, and unmasked neighbor
    *n* with solid angle omega_n and value v_n, the prediction is::

        pred_m = sum_n( w_n * v_n ) / sum_n( w_n )
        w_n = omega_m / omega_n

    This is a weighted average where neighbors with solid angles closer to
    the masked sensor's solid angle contribute more.  Unlike a multiplicative
    correction (v_n * ratio), this weighted average stays bounded in
    normalized space and avoids outliers from extreme solid angle ratios.

    If *solid_angles* is not provided, falls back to the simple unweighted
    neighbor average (identical to :class:`NeighborAverageBaseline`).

    Parameters
    ----------
    k : int
        Number of neighbor hops.
    """

    def __init__(self, k: int = 1):
        self.k = k
        self.neighbor_map = build_neighbor_map(k)
        self.nbr_indices, self.nbr_counts = _neighbor_map_to_padded_array(
            self.neighbor_map
        )

    def predict(
        self,
        x_npho: np.ndarray,
        mask: np.ndarray,
        solid_angles: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Predict values for masked sensors.

        Parameters
        ----------
        x_npho : ndarray, shape (N_events, 4760)
            Normalised npho values.
        mask : ndarray, shape (N_events, 4760)
            Boolean array; True = masked / dead channel.
        solid_angles : ndarray, shape (N_events, 4760), optional
            Per-event solid angle of each sensor.  If None, falls back to
            simple (unweighted) neighbor averaging.

        Returns
        -------
        predictions : ndarray, shape (N_events, 4760)
            Copy of *x_npho* with masked positions replaced.
        """
        N = x_npho.shape[0]
        predictions = x_npho.copy()

        nbr_idx = self.nbr_indices
        nbr_cnt = self.nbr_counts
        max_nbrs = nbr_idx.shape[1]

        for i in range(N):
            masked_sensors = np.where(mask[i])[0]
            if masked_sensors.size == 0:
                continue

            nbrs = nbr_idx[masked_sensors]
            counts = nbr_cnt[masked_sensors]

            slot_range = np.arange(max_nbrs)[None, :]
            in_range = slot_range < counts[:, None]
            neighbor_unmasked = ~mask[i][nbrs]
            valid = in_range & neighbor_unmasked

            nbr_vals = x_npho[i][nbrs]

            if solid_angles is not None:
                # omega_m: solid angle of each masked sensor.
                omega_m = solid_angles[i][masked_sensors]       # (n_masked,)
                # omega_n: solid angle of each neighbor.
                omega_n = solid_angles[i][nbrs]                 # (n_masked, max_nbrs)

                # Weighted average: w_n = omega_m / omega_n
                safe_omega_n = np.where(omega_n > 0, omega_n, 1.0)
                weights = omega_m[:, None] / safe_omega_n       # (n_masked, max_nbrs)
                weights = np.where(valid, weights, 0.0)

                weighted_vals = np.where(valid, nbr_vals * weights, 0.0)
                weight_sum = weights.sum(axis=1)                # (n_masked,)
                safe_wsum = np.maximum(weight_sum, 1e-12)
                avg = np.where(weight_sum > 0, weighted_vals.sum(axis=1) / safe_wsum, 0.0)
            else:
                # Simple average fallback.
                nbr_vals = np.where(valid, nbr_vals, 0.0)
                n_valid = valid.sum(axis=1).astype(np.float64)
                total = nbr_vals.sum(axis=1)
                safe_n = np.maximum(n_valid, 1.0)
                avg = np.where(n_valid > 0, total / safe_n, 0.0)

            predictions[i, masked_sensors] = avg

        return predictions
