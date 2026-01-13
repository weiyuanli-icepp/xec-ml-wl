import numpy as np

# =========================================================
# GEOMETRY DEFINITIONS
# =========================================================

# --- Inner SiPM face (93x44) ---
INNER_INDEX_MAP = np.arange(0, 4092, dtype=np.int32).reshape(93, 44)

# --- US and DS faces (24x6) ---
US_INDEX_MAP = np.arange(4308, 4308 + 6*24, dtype=np.int32).reshape(24, 6)
DS_INDEX_MAP = np.arange(4452, 4452 + 6*24, dtype=np.int32).reshape(24, 6)

# --- Outer Face (Coarse + Center) ---
CENTRAL_COARSE_IDS = [
    4185, 4186, 4187, 4194, 4195, 4196,
    4203, 4204, 4205, 4212, 4213, 4214,
]
OUTER_COARSE_FULL_INDEX_MAP = np.arange(4092, 4308, dtype=np.int32).reshape(9, 24)
OUTER_CENTER_INDEX_MAP = np.array([
    [4185, 4742, 4186, 4743, 4187],
    [4744, 4745, 4746, 4747, 4748],
    [4194, 4749, 4195, 4750, 4196],
    [4203, 4751, 4204, 4752, 4205],
    [4753, 4754, 4755, 4756, 4757],
    [4212, 4758, 4213, 4759, 4214],
], dtype=np.int32).T

# --- Outer Fine Grid Scaling ---
OUTER_FINE_COARSE_SCALE = (5, 3)
OUTER_FINE_CENTER_SCALE = (3, 2)
OUTER_FINE_CENTER_START = (3, 10)
OUTER_FINE_H = OUTER_COARSE_FULL_INDEX_MAP.shape[0] * OUTER_FINE_COARSE_SCALE[0]
OUTER_FINE_W = OUTER_COARSE_FULL_INDEX_MAP.shape[1] * OUTER_FINE_COARSE_SCALE[1]

# --- TOP & BOTTOM HEX DEFINITIONS ---
TOP_ROWS_LIST = [
    np.arange(4596, 4607), np.arange(4607, 4619), np.arange(4619, 4630),
    np.arange(4630, 4642), np.arange(4642, 4655), np.arange(4655, 4669),
]
BOTTOM_ROWS_LIST = [
    np.arange(4669, 4680), np.arange(4680, 4692), np.arange(4692, 4703),
    np.arange(4703, 4715), np.arange(4715, 4728), np.arange(4728, 4742),
]
TOP_HEX_ROWS = TOP_ROWS_LIST
BOTTOM_HEX_ROWS = BOTTOM_ROWS_LIST

def flatten_hex_rows(rows) -> np.ndarray:
    return np.concatenate([np.asarray(r, dtype=np.int32) for r in rows])

def build_hex_edge_index(row_lengths):
    id_map = {}
    node = 0
    for r, L in enumerate(row_lengths):
        for c in range(L):
            id_map[(r, c)] = node
            node += 1
    edges = set()
    for (r, c), u in id_map.items():
        if r % 2 == 0:
            neigh = [(r, c-1), (r, c+1), (r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
        else:
            neigh = [(r, c-1), (r, c+1), (r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]
        for rr, cc in neigh:
            if (rr, cc) in id_map:
                edges.add((u, id_map[(rr, cc)]))
                edges.add((id_map[(rr, cc)], u))
    if edges:
        edge_index = np.array(list(edges), dtype=np.int64).T
    else:
        edge_index = np.empty((2, 0), dtype=np.int64)
    dst = edge_index[1] if edge_index.size else np.array([], dtype=np.int64)
    deg = np.bincount(dst, minlength=node) if dst.size else np.zeros(node, dtype=np.int64)
    return edge_index, deg

HEX_EDGE_INDEX_NP, HEX_DEG_NP = build_hex_edge_index([len(r) for r in TOP_ROWS_LIST])

# Default Normalization Factors
DEFAULT_NPHO_SCALE     =  0.58
DEFAULT_NPHO_SCALE2    =  1.0
DEFAULT_TIME_SCALE     =  6.5e8
DEFAULT_TIME_SHIFT     =  0.5
DEFAULT_SENTINEL_VALUE = -5.0