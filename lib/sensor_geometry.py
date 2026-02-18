"""
Sensor geometry utilities for cross-attention inpainting.

Provides:
- 3D position loading from sensor_positions.txt
- Face ID mapping (sensor_id -> face_id 0-5)
- KNN graph construction by 3D Euclidean distance
"""

import numpy as np
from scipy.spatial import cKDTree

from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows,
)

NUM_SENSORS = 4760


def load_sensor_positions(filepath: str) -> np.ndarray:
    """
    Load (4760, 3) array of sensor 3D positions from file.

    Expected format (whitespace-separated, '#' comments allowed):
        sensor_id  x  y  z
        0 -30.4163 57.4104 -32.4601
        1 ...
        ...
        4759 ...

    Returns:
        positions: (4760, 3) float64 array, indexed by sensor_id
    """
    raw = np.loadtxt(filepath, comments="#")
    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(
            f"Expected columns [sensor_id, x, y, z], got shape {raw.shape}"
        )

    ids = raw[:, 0].astype(int)
    coords = raw[:, 1:4]

    if len(ids) != NUM_SENSORS:
        raise ValueError(
            f"Expected {NUM_SENSORS} sensors, got {len(ids)}"
        )

    # Sort by sensor_id and validate contiguous 0..4759
    order = np.argsort(ids)
    ids_sorted = ids[order]
    expected = np.arange(NUM_SENSORS)
    if not np.array_equal(ids_sorted, expected):
        raise ValueError("Sensor IDs must be contiguous 0..4759")

    positions = np.empty((NUM_SENSORS, 3), dtype=np.float64)
    positions[ids] = coords
    return positions


def build_sensor_face_ids() -> np.ndarray:
    """
    Map each sensor_id (0-4759) to a face_id (0-5).

    Face assignments:
        0 = Inner   (sensor IDs from INNER_INDEX_MAP:  0-4091)
        1 = US      (sensor IDs from US_INDEX_MAP:     4308-4451)
        2 = DS      (sensor IDs from DS_INDEX_MAP:     4452-4595)
        3 = Outer   (sensor IDs from OUTER_COARSE + OUTER_CENTER: 4092-4307, 4742-4759)
        4 = Top     (sensor IDs from TOP_HEX_ROWS:    4596-4668)
        5 = Bottom  (sensor IDs from BOTTOM_HEX_ROWS: 4669-4741)

    Returns:
        face_ids: (4760,) int32 array
    """
    face_ids = np.full(NUM_SENSORS, -1, dtype=np.int32)

    # Face 0: Inner
    face_ids[INNER_INDEX_MAP.flatten()] = 0

    # Face 1: US
    face_ids[US_INDEX_MAP.flatten()] = 1

    # Face 2: DS
    face_ids[DS_INDEX_MAP.flatten()] = 2

    # Face 3: Outer (coarse + center, some overlap with coarse central IDs)
    face_ids[OUTER_COARSE_FULL_INDEX_MAP.flatten()] = 3
    face_ids[OUTER_CENTER_INDEX_MAP.flatten()] = 3

    # Face 4: Top hex
    face_ids[flatten_hex_rows(TOP_HEX_ROWS)] = 4

    # Face 5: Bottom hex
    face_ids[flatten_hex_rows(BOTTOM_HEX_ROWS)] = 5

    assert (face_ids >= 0).all(), (
        f"Some sensors have no face assignment: "
        f"{np.where(face_ids < 0)[0][:10]}..."
    )
    return face_ids


def build_knn_graph(positions: np.ndarray, k: int) -> np.ndarray:
    """
    Compute k-nearest neighbors for each sensor by 3D Euclidean distance.

    Args:
        positions: (N, 3) array of 3D coordinates
        k: number of neighbors (excluding self)

    Returns:
        knn_indices: (N, k) int64 array of neighbor indices
    """
    tree = cKDTree(positions)
    # Query k+1 because the first match is self
    _, indices = tree.query(positions, k=k + 1)
    # Remove self (first column)
    knn_indices = indices[:, 1:].astype(np.int64)
    return knn_indices
