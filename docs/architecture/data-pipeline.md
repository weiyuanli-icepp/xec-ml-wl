# Data Pipeline

This document covers data format, input normalization, and the data loading pipeline.

## Data Format

ROOT files with TTree structure. Default tree name: `tree`.

### Input Branches

Shape: 4760 per event

| Branch | Description |
|--------|-------------|
| `relative_npho` | Normalized photon counts per sensor |
| `relative_time` | Normalized timing per sensor |

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
| `npho_scale` | 0.58 | 1000 | First log1p scale factor |
| `npho_scale2` | 1.0 | 4.08 | Second scale factor |
| `time_scale` | 6.5e-8 | 1.14e-7 | Time normalization (seconds) |
| `time_shift` | 0.5 | -0.46 | Time offset after scaling |
| `sentinel_value` | -5.0 | -1.0 | Invalid sensor marker |

**Important:** When using MAE pretraining for fine-tuning, ensure all downstream models use the **same** normalization scheme as the MAE.

### 2. Normalization Formulas

**Photon Count (Npho) - Extensive Quantity:**
```python
# Log-transform to handle wide dynamic range (0 to ~10^6 photons)
npho_norm = log1p(raw_npho / npho_scale) / npho_scale2
```

$$N_{\text{norm}} = \frac{\ln(1 + N_{\text{raw}} / s_1)}{s_2}$$

Where:
- $s_1$ = `npho_scale`
- $s_2$ = `npho_scale2`

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
# Npho invalid if:
mask_npho_bad = (raw_npho <= 0.0) | (raw_npho > 9e9) | isnan(raw_npho)

# Time invalid if npho is bad OR time itself is bad:
mask_time_bad = mask_npho_bad | (abs(raw_time) > 9e9) | isnan(raw_time)
```

**Invalid Sensor Handling:**
| Channel | Invalid Value | Reason |
|---------|---------------|--------|
| Npho | `0.0` | Zero photons is physically valid, acts as "no signal" |
| Time | `sentinel_value` | Distinctive value far from valid range (~0 after normalization) |

### 4. Sentinel Value System

The **sentinel value** marks sensors where timing information is unavailable:

**Why use a sentinel value far from valid range?**
- Valid normalized time is typically in range [-1, 1] after shifting
- A value like -5.0 (legacy) or -1.0 (new) is far outside this range
- Convolution operations will "see" this as a strong negative signal

**Detection in Models:**
```python
# Identify already-invalid sensors
already_invalid = (x[:, :, 1] == sentinel_value)  # Check time channel
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
| Time (normalized) | [-1, 1] | ~0 | ~0.3 |
| Time (invalid) | -5.0 | - | - |

After normalization with the **new scheme** (npho_scale=1000):

| Channel | Valid Range | Mean | Std |
|---------|-------------|------|-----|
| Npho (normalized) | [0, ~2.5] | ~1.0 | ~0.7 |
| Time (normalized) | [-1.5, 1.5] | ~0 | ~0.4 |
| Time (invalid) | -1.0 | - | - |

### 6. Configuration Parameters

| Parameter | Config Key | Legacy | New | Description |
|-----------|------------|--------|-----|-------------|
| `npho_scale` | `normalization.npho_scale` | 0.58 | 1000 | Npho log transform scale |
| `npho_scale2` | `normalization.npho_scale2` | 1.0 | 4.08 | Secondary npho scale |
| `time_scale` | `normalization.time_scale` | 6.5e-8 | 1.14e-7 | Time scale (seconds) |
| `time_shift` | `normalization.time_shift` | 0.5 | -0.46 | Time offset after scaling |
| `sentinel_value` | `normalization.sentinel_value` | -5.0 | -1.0 | Invalid sensor marker |

**Important:** All training paths (Regressor, MAE, Inpainter) must use the **same normalization parameters** for the encoder to work correctly. The inpainter must match the MAE's normalization.

### 7. Inverse Transform (for Inference)

To convert predictions back to physical units:

```python
# Npho: inverse of log1p transform
raw_npho = npho_scale * (exp(npho_norm * npho_scale2) - 1)

# Time: inverse of linear transform
raw_time = (time_norm + time_shift) * time_scale
```

---

## Data Loading Pipeline

The streaming data loader uses a multi-level parallelism strategy to efficiently load large ROOT files:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         ROOT File (millions of events)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    chunksize (step_size) = 256000                        │
│         Load 256k events at a time from disk into CPU memory             │
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
│                   num_threads=4 (within each worker)                     │
│        ThreadPool splits chunk into 4 parts for normalization            │
│        (log transform, scaling, sentinel values, etc.)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         batch_size = 4096                                │
│              DataLoader collects samples into GPU batches                │
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
