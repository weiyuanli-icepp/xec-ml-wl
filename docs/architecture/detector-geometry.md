# Detector Geometry & Sensor Mapping

The MEG II LXe detector has **4760 sensors** (4092 SiPMs + 668 PMTs) arranged across 6 faces. This section documents the geometry definitions used throughout the codebase.

## A. Sensor Overview

| Face | Type | Sensors | Shape | Index Range | Description |
|------|------|---------|-------|-------------|-------------|
| **Inner** | SiPM | 4092 | 93×44 | 0–4091 | Cylindrical inner surface |
| **Outer Coarse** | SiPM | 216 | 9×24 | 4092–4307 | Outer cylindrical surface |
| **Outer Center** | SiPM | 18 | 5×6 | 4742–4759 | High-granularity center patch (replaces 12 coarse) |
| **US (Upstream)** | SiPM | 144 | 24×6 | 4308–4451 | Upstream endcap |
| **DS (Downstream)** | SiPM | 144 | 24×6 | 4452–4595 | Downstream endcap |
| **Top** | PMT | 73 | Hex | 4596–4668 | Top hexagonal PMT array |
| **Bottom** | PMT | 73 | Hex | 4669–4741 | Bottom hexagonal PMT array |

**Total: 4760 sensors** (input tensor shape: `(B, 4760, 2)` for npho and time)

## B. Index Maps (`lib/geom_defs.py`)

The geometry is defined using numpy index maps that translate 2D grid positions to flat sensor indices:

```python
# Inner face: 93 rows × 44 columns = 4092 SiPMs
INNER_INDEX_MAP = np.arange(0, 4092).reshape(93, 44)

# US/DS faces: 24 rows × 6 columns = 144 SiPMs each
US_INDEX_MAP = np.arange(4308, 4452).reshape(24, 6)
DS_INDEX_MAP = np.arange(4452, 4596).reshape(24, 6)

# Outer coarse: 9 rows × 24 columns = 216 SiPMs
OUTER_COARSE_FULL_INDEX_MAP = np.arange(4092, 4308).reshape(9, 24)

# Outer center: 5 columns × 6 rows = 30 SiPMs (higher granularity)
OUTER_CENTER_INDEX_MAP = np.array([...]).T  # Shape: (5, 6)
```

## C. Outer Face Fine Grid Construction

The outer face has two sensor grids that are combined into a unified fine grid:

```
┌──────────────────────────────────────────────────────────────────────┐
│                     Outer Coarse Grid (9×24)                         │
│                        216 sensors total                             │
│   ┌─────────────────────────────────────────────────────────────┐    │
│   │                                                             │    │
│   │    Each coarse cell covers 5×3 fine grid positions          │    │
│   │                                                             │    │
│   │         ┌─────────────────────┐                             │    │
│   │         │  Center Patch (5×6) │  ← Higher granularity       │    │
│   │         │  30 sensors         │    Each cell = 3×2 fine     │    │
│   │         │  at rows 3-4,       │                             │    │
│   │         │  cols 10-13         │                             │    │
│   │         └─────────────────────┘                             │    │
│   │                                                             │    │
│   └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
                    build_outer_fine_grid_tensor()
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│                     Fine Grid (45×72)                                │
│   - Coarse upsampled: 9×5=45 rows, 24×3=72 cols                      │
│   - Center upsampled: 6×3=18 rows, 5×2=10 cols                       │
│   - Center overlaid at position (15, 30) to (33, 40)                 │
│   - Npho divided by scale factor (extensive quantity)                │
│   - Time unchanged (intensive quantity)                              │
└──────────────────────────────────────────────────────────────────────┘
                              ↓
                    Optional: avg_pool2d(kernel=3×3)
                              ↓
┌──────────────────────────────────────────────────────────────────────┐
│                   Pooled Grid (15×24)                                │
│   - Used as model input when outer_fine_pool=[3,3]                   │
│   - Reduces computation while preserving spatial structure           │
└──────────────────────────────────────────────────────────────────────┘
```

**Scale Factors:**
| Grid | Coarse Scale | Center Scale | Position |
|------|--------------|--------------|----------|
| Fine (45×72) | 5×3 | 3×2 | Center starts at (15, 30) |
| Pooled (15×24) | pool 3×3 | pool 3×3 | - |

**Key Function:** `build_outer_fine_grid_tensor(x_batch, pool_kernel)` in `lib/geom_utils.py`

## D. Hexagonal PMT Layout

The Top and Bottom PMT arrays use a hexagonal lattice structure:

```
     Top / Bottom PMT Array (73 nodes)

Row 0:    ● ● ● ● ● ● ● ● ● ● ●     (11)
Row 1:   ● ● ● ● ● ● ● ● ● ● ● ●    (12)
Row 2:    ● ● ● ● ● ● ● ● ● ● ●     (11)
Row 3:   ● ● ● ● ● ● ● ● ● ● ● ●    (12)
Row 4:  ● ● ● ● ● ● ● ● ● ● ● ● ●   (13)
Row 5: ● ● ● ● ● ● ● ● ● ● ● ● ● ●  (14)
```

**Row Lengths:**
- [11, 12, 11, 12, 13, 14] → 73 PMTs per face (indices 4596–4668 for Top, 4669–4741 for Bottom)

**Hexagonal Adjacency Graph:**

The `build_hex_edge_index()` function creates a graph where each PMT connects to its 6 hexagonal neighbors:

```python
# For even rows: neighbors at relative positions
neigh_even = [(r, c-1), (r, c+1), (r-1, c-1), (r-1, c), (r+1, c-1), (r+1, c)]

# For odd rows: neighbors shifted
neigh_odd = [(r, c-1), (r, c+1), (r-1, c), (r-1, c+1), (r+1, c), (r+1, c+1)]
```

The edge index tensor has shape `(3, num_edges)` with:
- Row 0: Source node
- Row 1: Destination node
- Row 2: Edge type (0=self, 1-6=neighbor direction)

## E. Utility Functions (`lib/geom_utils.py`)

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `gather_face(x_batch, index_map)` | (B, 4760, 2), (H, W) | (B, 2, H, W) | Extract rectangular face from flat tensor |
| `gather_hex_nodes(x_batch, indices)` | (B, 4760, 2), (N,) | (B, N, 2) | Extract hex nodes from flat tensor |
| `build_outer_fine_grid_tensor(x_batch, pool)` | (B, 4760, 2), kernel | (B, 2, H, W) | Build outer fine grid with optional pooling |
| `flatten_hex_rows(rows)` | list of arrays | (N,) | Flatten hex row arrays to single index array |

## F. Normalization Constants

There are **two normalization schemes** currently in use. See [Data Pipeline](data-pipeline.md) for detailed explanation.

**Legacy Scheme** (in `lib/geom_defs.py` and `config/train_config.yaml`):
```python
DEFAULT_NPHO_SCALE     = 0.58      # Npho log transform scale
DEFAULT_NPHO_SCALE2    = 1.0       # Secondary npho scale
DEFAULT_TIME_SCALE     = 6.5e-8    # Time normalization (seconds)
DEFAULT_TIME_SHIFT     = 0.5       # Time offset after scaling
DEFAULT_SENTINEL_TIME = -1.0      # Marker for invalid/masked sensors
```

**New Scheme** (in `config/mae_config.yaml` and `config/inpainter_config.yaml`):
```python
npho_scale     = 1000       # Npho log transform scale
npho_scale2    = 4.08       # Secondary npho scale
time_scale     = 1.14e-7    # Time normalization (seconds)
time_shift     = -0.46      # Time offset after scaling
sentinel_time = -1.0       # Marker for invalid/masked sensors
```

**Important:** Models trained with different normalization schemes are **not compatible**. MAE pretraining and downstream fine-tuning must use the same scheme.

**Normalization Formulas:**
```python
# Npho (photon count) - log transform for wide dynamic range
npho_norm = log1p(npho_raw / npho_scale) / npho_scale2

# Time - linear transform
time_norm = (time_raw / time_scale) - time_shift

# Invalid sensors
if invalid:
    npho_norm = 0.0
    time_norm = sentinel_time
```
