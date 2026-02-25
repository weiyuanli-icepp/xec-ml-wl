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
    # edges = set()
    edges = []
    for (r, c), u in id_map.items():
        # Self-connection
        edges.append([u, u, 0])
        
        # Neighbor connections
        if r % 2 == 0:
            neigh = [(r, c-1), (r, c+1), (r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]
        else:
            neigh = [(r, c-1), (r, c+1), (r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]

        for i, (rr, cc) in enumerate(neigh):
            if (rr, cc) in id_map:
                v = id_map[(rr, cc)]
                edges.append([u, v, i+1])
    if edges:
        edge_index = np.array(edges, dtype=np.int64).T
    else:
        edge_index = np.empty((3, 0), dtype=np.int64)
    
    dst = edge_index[1] if edge_index.size else np.array([], dtype=np.int64)
    deg = np.bincount(dst, minlength=node) if dst.size else np.zeros(node, dtype=np.int64)
    return edge_index, deg

HEX_EDGE_INDEX_NP, HEX_DEG_NP = build_hex_edge_index([len(r) for r in TOP_ROWS_LIST])

# --- Outer Sensor to Finegrid Mapping ---
# For sensor-level predictions, we need to map each outer sensor to its finegrid region

def build_outer_sensor_to_finegrid_map():
    """
    Build mapping from each outer sensor ID to its finegrid region bounds.

    Returns:
        dict: sensor_id -> (h_start, h_end, w_start, w_end) in finegrid coordinates

    Finegrid dimensions: 45 x 72 (9*5 x 24*3)
    - Coarse sensors (non-central): each maps to 5×3 region
    - Center sensors: each maps to 3×2 region (at finegrid position [15:30, 30:42])
    """
    mapping = {}

    cr, cc = OUTER_FINE_COARSE_SCALE  # (5, 3)
    sr, sc = OUTER_FINE_CENTER_SCALE  # (3, 2)
    c_start_r, c_start_c = OUTER_FINE_CENTER_START  # (3, 10) in coarse coords

    # Center region starts at finegrid position (15, 30)
    center_fine_top = c_start_r * cr  # 15
    center_fine_left = c_start_c * cc  # 30

    # Set of central coarse IDs (these use center mapping, not coarse mapping)
    central_set = set(CENTRAL_COARSE_IDS)

    # Non-central coarse sensors: use full 5×3 coarse region
    for h in range(9):
        for w in range(24):
            sensor_id = OUTER_COARSE_FULL_INDEX_MAP[h, w]
            if sensor_id not in central_set:
                mapping[int(sensor_id)] = (h * cr, (h + 1) * cr, w * cc, (w + 1) * cc)

    # Center sensors (including 12 overlapping coarse sensors): use 3×2 center region
    # OUTER_CENTER_INDEX_MAP has shape (5, 6) after transpose
    H_center, W_center = OUTER_CENTER_INDEX_MAP.shape  # 5, 6
    for h in range(H_center):
        for w in range(W_center):
            sensor_id = OUTER_CENTER_INDEX_MAP[h, w]
            h_start = center_fine_top + h * sr
            h_end = center_fine_top + (h + 1) * sr
            w_start = center_fine_left + w * sc
            w_end = center_fine_left + (w + 1) * sc
            mapping[int(sensor_id)] = (h_start, h_end, w_start, w_end)

    return mapping

# Build the mapping once at import time
OUTER_SENSOR_TO_FINEGRID = build_outer_sensor_to_finegrid_map()

# All unique outer sensor IDs in sorted order (234 total)
# 204 non-central coarse (from 4092-4307 minus 12 central) + 12 central coarse + 18 dense center
_outer_center_set = set(OUTER_CENTER_INDEX_MAP.flatten())
_outer_coarse_non_central = [
    int(sid) for sid in OUTER_COARSE_FULL_INDEX_MAP.flatten()
    if sid not in CENTRAL_COARSE_IDS
]
_outer_center_all = [int(sid) for sid in OUTER_CENTER_INDEX_MAP.flatten()]
OUTER_ALL_SENSOR_IDS = np.array(
    sorted(set(_outer_coarse_non_central + _outer_center_all)),
    dtype=np.int32
)

# Create reverse mapping: flat sensor ID -> index in OUTER_ALL_SENSOR_IDS (0-233)
OUTER_SENSOR_ID_TO_IDX = {int(sid): idx for idx, sid in enumerate(OUTER_ALL_SENSOR_IDS)}


# --- Face Sensor IDs ---
# Dict mapping face name to sorted numpy array of global sensor indices.
FACE_SENSOR_IDS = {
    'inner': np.arange(0, 4092, dtype=np.int32),
    'outer': OUTER_ALL_SENSOR_IDS,
    'us':    US_INDEX_MAP.flatten().astype(np.int32),
    'ds':    DS_INDEX_MAP.flatten().astype(np.int32),
    'top':   flatten_hex_rows(TOP_HEX_ROWS),
    'bot':   flatten_hex_rows(BOTTOM_HEX_ROWS),
}


# Default Normalization Factors
# npho_norm = log1p(raw_npho / NPHO_SCALE) / NPHO_SCALE2
# time_norm = (raw_time / TIME_SCALE) - TIME_SHIFT
DEFAULT_NPHO_SCALE     =  1000.0
DEFAULT_NPHO_SCALE2    =  4.08
DEFAULT_TIME_SCALE     =  1.14e-7
DEFAULT_TIME_SHIFT     = -0.46
DEFAULT_SENTINEL_TIME = -1.0
DEFAULT_SENTINEL_NPHO = -1.0

# Conditional Time Loss Threshold
# Time loss is only computed for sensors where npho > threshold (raw scale).
# Based on conventional timing reconstruction (MEGTXECTimeFit.cpp), sensors with
# nphe < threshold are rejected because timing uncertainty ~ 1/sqrt(nphe) diverges.
# Default of 100 is a conservative threshold; can be tuned via config.
# Sensors with npho < threshold will have valid npho but invalid (sentinel) time.
DEFAULT_NPHO_THRESHOLD = 100.0