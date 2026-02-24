# Data Pipeline

This document covers data format, input normalization, and the data loading pipeline.

## Data Format

ROOT files with TTree structure. Default tree name: `tree`.

### Input Branches

Shape: 4760 per event

| Branch | Description |
|--------|-------------|
| `npho` | Photon counts per sensor (default `npho_branch`) |
| `relative_time` | Timing per sensor (default `time_branch`) |

**Note:** The branch names are configurable via `data.npho_branch` and `data.time_branch` in the config file. The legacy branch name `relative_npho` is deprecated; use `npho` for newer data formats.

### Truth Branches

| Branch | Shape | Description |
|--------|-------|-------------|
| `emiAng` | (2,) | Emission angle (θ, φ) |
| `energyTruth` | (1,) | True gamma energy |
| `timeTruth` | (1,) | True gamma timing |
| `uvwTruth` | (3,) | First interaction position (u, v, w) |
| `xyzTruth` | (3,) | First interaction position (x, y, z) |
| `emiVec` | (3,) | Emission direction unit vector |
| `xyzVTX` | (3,) | Vertex position (gamma origin) |
| `run` | (1,) | Run number |
| `event` | (1,) | Event number |

---

## Input Normalization

All training paths (Regressor, MAE, Inpainter) use the same normalization pipeline. **Critical:** The encoder learns features based on the input distribution, so models trained with different normalization schemes are **not compatible**.

### 1. Normalization Schemes

There are currently **two normalization schemes** in use:

| Scheme | Use Case | When to Use |
|--------|----------|-------------|
| **Legacy** | Regressor training, existing models | Backward compatibility with older checkpoints |
| **New** | MAE/Inpainter, new experiments | Better numerical stability, recommended for new work |

**Parameter Comparison:**

| Parameter | Legacy Scheme | New Scheme | Notes |
|-----------|---------------|------------|-------|
| `npho_scheme` | `log1p` | `log1p` (or `sqrt` for inpainter) | Normalization scheme |
| `npho_scale` | 0.58 | 1000 | Npho scale factor |
| `npho_scale2` | 1.0 | 4.08 | Secondary scale (log1p only) |
| `time_scale` | 6.5e-8 | 1.14e-7 | Time normalization (seconds) |
| `time_shift` | 0.5 | -0.46 | Time offset after scaling |
| `sentinel_time` | -1.0 | -1.0 | Invalid sensor marker |

**Important:** When using MAE pretraining for fine-tuning, ensure all downstream models use the **same** normalization scheme as the MAE.

### 2. Normalization Formulas

**Photon Count (Npho) - Extensive Quantity:**

The npho normalization supports multiple schemes via the `npho_scheme` config parameter:

| Scheme | Formula | Use Case |
|--------|---------|----------|
| `log1p` (default) | `log1p(x/scale)/scale2` | Wide dynamic range (0 to ~10⁶ photons) |
| `anscombe` | `2√(x + 3/8) / (2√(scale + 3/8))` | Poisson variance stabilization |
| `sqrt` | `√x / √scale` | Simpler variance stabilization |
| `linear` | `x / scale` | No transform (baseline) |

All schemes produce output values around 1 when `x = npho_scale`, making them interchangeable for model training.

**Default scheme (log1p):**
```python
# Log-transform to handle wide dynamic range (0 to ~10^6 photons)
npho_norm = log1p(raw_npho / npho_scale) / npho_scale2
```

$$N_{\text{norm}} = \frac{\ln(1 + N_{\text{raw}} / s_1)}{s_2}$$

Where:
- $s_1$ = `npho_scale`
- $s_2$ = `npho_scale2`

**Anscombe scheme (for Poisson data):**
```python
# Variance-stabilizing transform with scaling
npho_norm = 2 * sqrt(raw_npho + 0.375) / (2 * sqrt(npho_scale + 0.375))
```

The Anscombe transform converts Poisson-distributed data to approximately Gaussian with unit variance. The scaling ensures the output range matches other schemes.

**Timing - Intensive Quantity:**
```python
# Linear transform to center around 0
time_norm = (raw_time / time_scale) - time_shift
```

$$t_{\text{norm}} = \frac{t_{\text{raw}}}{s_t} - \delta_t$$

Where:
- $s_t$ = `time_scale`
- $\delta_t$ = `time_shift`

### 3. Invalid Sensor Detection

Sensors are marked as **invalid** based on these conditions:

```python
# Step 1: True invalids (data-level sentinels)
mask_npho_invalid = (raw_npho > 9e9) | isnan(raw_npho)

# Step 2: Domain-breaking values (scheme-dependent)
# NphoTransform.domain_min() returns the minimum raw value the scheme can handle:
#   log1p:    -0.999 * npho_scale  (log1p requires x > -1)
#   sqrt:     0.0                  (sqrt requires x >= 0)
#   anscombe: -0.375              (anscombe requires x > -3/8)
#   linear:   -inf                (no lower bound)
domain_min = npho_transform.domain_min()
mask_domain_break = (~mask_npho_invalid) & (raw_npho < domain_min)
# Domain-breaking sensors are set to sentinel (not normalized)

# Step 3: Time invalids
# - npho is invalid: can't trust timing either
# - raw_npho < npho_threshold: timing unreliable (uncertainty ~ 1/sqrt(npho))
# - raw_time > 9e9: sentinel in data
# - isnan: corrupted data
mask_time_invalid = mask_npho_invalid | (raw_npho < npho_threshold) | (abs(raw_time) > 9e9) | isnan(raw_time)
```

**Why this scheme?**
- The `domain_min()` check is **scheme-dependent**: `log1p` requires `x > -scale`, `sqrt` requires `x >= 0`, etc.
- Small negative npho values (e.g., -10) are physically possible due to baseline subtraction and are handled correctly by `log1p` (but NOT by `sqrt`)
- Timing uncertainty scales as `1/sqrt(npho)`, so low-npho sensors have unreliable timing
- The `npho_threshold` (default: 100, configured via `training.time.npho_threshold`) determines when timing becomes unreliable

**Invalid Sensor Handling:**
| Channel | Invalid Value | Reason |
|---------|---------------|--------|
| Npho | `sentinel_npho` | Invalid sensors marked distinctively |
| Time | `sentinel_time` | Distinctive value far from valid range (~0 after normalization) |

**Low-Npho Sensors (0 < npho < threshold):**
| Channel | Treatment | Reason |
|---------|-----------|--------|
| Npho | Normalized normally | Npho measurement is still valid |
| Time | `sentinel_time` | Timing unreliable at low photon counts |

### 4. Sentinel Value System

The **sentinel value** marks sensors where timing information is unavailable:

**Why use a sentinel value far from valid range?**
- Valid normalized time is typically in range [-1, 1] after shifting
- A value like -5.0 (legacy) or -1.0 (new) is far outside this range
- Convolution operations will "see" this as a strong negative signal

**Detection in Models:**
```python
# Identify already-invalid sensors
already_invalid = (x[:, :, 1] == sentinel_time)  # Check time channel
```

**Masking (Invalid-Aware) (MAE/Inpainter):**
- Already-invalid sensors are excluded from random masking pool
- Loss is computed only on randomly-masked positions (where ground truth exists)
- See `actual_mask_ratio` metric for effective masking after exclusions

### 5. Typical Value Ranges

After normalization with the **legacy scheme** (npho_scale=0.58):

| Channel | Valid Range | Mean | Std |
|---------|-------------|------|-----|
| Npho (normalized) | [0, ~3] | ~0.5 | ~0.5 |
| Npho (invalid) | -1.0 | - | - |
| Time (normalized) | [-1, 1] | ~0 | ~0.3 |
| Time (invalid) | -5.0 | - | - |

After normalization with the **new scheme** (npho_scale=1000):

| Channel | Valid Range | Mean | Std |
|---------|-------------|------|-----|
| Npho (normalized) | [0, ~2.5] | ~1.0 | ~0.7 |
| Npho (invalid) | -1.0 | - | - |
| Time (normalized) | [-1.5, 1.5] | ~0 | ~0.4 |
| Time (invalid) | -1.0 | - | - |

### 6. Configuration Parameters

| Parameter | Config Key | Legacy | New | Description |
|-----------|------------|--------|-----|-------------|
| `npho_scale` | `normalization.npho_scale` | 0.58 | 1000 | Npho normalization scale |
| `npho_scale2` | `normalization.npho_scale2` | 1.0 | 4.08 | Secondary npho scale (log1p only) |
| `npho_scheme` | `normalization.npho_scheme` | - | log1p | Normalization scheme (log1p/anscombe/sqrt/linear) |
| `time_scale` | `normalization.time_scale` | 6.5e-8 | 1.14e-7 | Time scale (seconds) |
| `time_shift` | `normalization.time_shift` | 0.5 | -0.46 | Time offset after scaling |
| `sentinel_time` | `normalization.sentinel_time` | -1.0 | -1.0 | Invalid time sensor marker |
| `sentinel_npho` | `normalization.sentinel_npho` | -1.0 | -1.0 | Invalid npho sensor marker |
| `npho_threshold` | `normalization.npho_threshold` | - | 100 | Min npho for valid timing |

**Important:** All training paths (Regressor, MAE, Inpainter) must use the **same normalization parameters** for the encoder to work correctly. The inpainter must match the MAE's normalization.

**Note on `npho_threshold`:** Sensors with `raw_npho < npho_threshold` will have valid npho values but their time channel will be set to sentinel. This is because timing precision degrades as `1/sqrt(npho)`, making low-npho timing measurements unreliable. The default of 100 is conservative; can be adjusted based on physics requirements.

### 7. Inverse Transform (for Inference)

To convert predictions back to physical units:

```python
# Npho: inverse depends on scheme
# For log1p (default):
raw_npho = npho_scale * (exp(npho_norm * npho_scale2) - 1)

# For anscombe:
scale_factor = 2 * sqrt(npho_scale + 0.375)
raw_npho = (npho_norm * scale_factor / 2) ** 2 - 0.375

# For sqrt:
raw_npho = (npho_norm * sqrt(npho_scale)) ** 2

# For linear:
raw_npho = npho_norm * npho_scale

# Time: inverse of linear transform (same for all schemes)
raw_time = (time_norm + time_shift) * time_scale
```

Use the `NphoTransform` class from `lib/normalization.py` for automatic scheme-aware transforms:

```python
from lib.normalization import NphoTransform

transform = NphoTransform(scheme="log1p", npho_scale=1000, npho_scale2=4.08)
npho_norm = transform.forward(raw_npho)
raw_npho = transform.inverse(npho_norm)
```

---

## Data Loading Pipeline

The streaming data loader uses a multi-level parallelism strategy to efficiently load large ROOT files:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ROOT File (millions of events)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    chunksize (step_size) = 256000                       │
│         Load 256k events at a time from disk into CPU memory            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
          ┌─────────────────────────┼─────────────────────────┐
          ▼                         ▼                         ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│   Worker 0      │       │   Worker 1      │       │   Worker N      │
│  (process)      │       │  (process)      │       │  (process)      │
│                 │       │                 │       │                 │
│  num_workers=8  │  ...  │  Each worker    │  ...  │  Loads chunks   │
│                 │       │  is a separate  │       │  in parallel    │
│                 │       │  Python process │       │                 │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                   num_threads=4 (within each worker)                    │
│        ThreadPool splits chunk into 4 parts for normalization           │
│        (log transform, scaling, sentinel values, etc.)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         batch_size = 4096                               │
│              DataLoader collects samples into GPU batches               │
└─────────────────────────────────────────────────────────────────────────┘
```

### Parameter Reference

| Parameter | Type | What it does | Memory impact |
|-----------|------|--------------|---------------|
| `chunksize` | Disk → CPU | Events loaded from ROOT file at once | **High** - each chunk held in memory |
| `num_workers` | Processes | Parallel DataLoader workers, each loads its own chunks | **High** - N workers × chunk memory |
| `num_threads` | Threads | Threads for CPU preprocessing within each worker | Low - shares memory |
| `batch_size` | CPU → GPU | Samples sent to GPU per iteration | GPU memory |

### Memory Estimation

```
CPU Memory ≈ num_workers × chunksize × ~50 bytes/event × prefetch_factor(2)
           ≈ 8 × 256000 × 50 × 2 ≈ 200 MB per worker ≈ 1.6 GB total
```

### Multi-GPU Data Sharding

When using DDP (multi-GPU training), ROOT file lists are sharded across ranks using round-robin assignment:

```
Files: [A.root, B.root, C.root, D.root, E.root, F.root]

Rank 0: [A.root, C.root, E.root]
Rank 1: [B.root, D.root, F.root]
```

Each rank independently streams its assigned files through the same pipeline above. This avoids `DistributedSampler` (which doesn't work with `IterableDataset`). The regressor uses `get_dataloader(rank=, world_size=)` for sharding; MAE and inpainter shard file lists before passing to engine functions via `shard_file_list()`.

**Important:** Ensure you have at least as many ROOT files as GPUs. With fewer files than ranks, some ranks will have no data.

### Recommended Settings

**Batch jobs (full resources):**
```yaml
data:
  chunksize: 256000
  num_workers: 8
  num_threads: 4
  batch_size: 4096
```

**Interactive sessions (limited memory):**
```yaml
data:
  chunksize: 64000     # 4x smaller chunks
  num_workers: 2       # 4x fewer workers
  num_threads: 4       # Keep same (threads share memory)
  batch_size: 1024     # Smaller GPU batches
```
