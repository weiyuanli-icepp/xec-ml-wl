# Multi-Task Learning

The model supports simultaneous regression of multiple physical observables.

## Task Configuration

Configure tasks in `config/reg/train_config.yaml`:

```yaml
tasks:
  angle:
    enabled: true
    loss_fn: "smooth_l1"    # See loss functions below
    loss_beta: 1.0
    weight: 1.0
  energy:
    enabled: false
    loss_fn: "relative_l1"  # Scale-invariant for energy
    weight: 1.0
    log_transform: false    # Train on log(energy) if true
  timing:
    enabled: false
    loss_fn: "l1"
    weight: 1.0
  uvwFI:
    enabled: false
    loss_fn: "mse"
    weight: 1.0
```

## Loss Functions

| Loss Function | Formula | Use Case |
|---------------|---------|----------|
| `smooth_l1` (default) | Huber loss (smooth_l1 variant) | General purpose, robust to outliers |
| `huber` | Huber loss (PyTorch native) | Same as smooth_l1 with configurable `loss_beta` |
| `l1` | \|pred - target\| | Median regression |
| `mse` / `l2` | (pred - target)² | Mean regression |
| `relative_l1` | \|pred - target\| / \|target\| | Scale-invariant, good for energy |
| `relative_smooth_l1` | smooth_l1 / \|target\| | Robust scale-invariant loss |
| `relative_mse` | (pred - target)² / target² | Relative MSE |

### Relative Loss Functions

For energy regression where σ ∝ √E (stochastic term), standard L1/MSE losses penalize high-energy errors more heavily. Relative losses normalize by the target value, providing scale-invariant training.

### Log Transform

When `log_transform: true` is set, the model learns to predict log(value) instead of value directly. This can improve training stability for quantities spanning multiple orders of magnitude. Predictions are automatically converted back to linear space for validation and artifact generation.

## Models

- `XECEncoder`: Single-task (angle-only, legacy)
- `XECMultiHeadModel`: Multi-task with shared backbone and task-specific heads

## Task Output Dimensions

| Task | Output | Description |
|------|--------|-------------|
| `angle` | 2 | (θ, φ) emission angles |
| `energy` | 1 | Energy |
| `timing` | 1 | Timing |
| `uvwFI` | 3 | (u, v, w) position coordinates |

## Experimental Heads

Available but not fully tested:

| Task | Output | Description |
|------|--------|-------------|
| `angleVec` | 3 | Emission direction unit vector (x, y, z) |
| `n_gamma` | 5 | Number of gammas classification (0-4) |

---

# Sample Reweighting

Balance training distributions using histogram-based reweighting. This helps when certain regions (e.g., specific angles or energies) are underrepresented in training data.

## Configuration

Configure in `config/reg/train_config.yaml`:

```yaml
reweighting:
  angle:
    enabled: true
    nbins_2d: [20, 20]    # (theta_bins, phi_bins)
  energy:
    enabled: false
    nbins: 30
  timing:
    enabled: false
    nbins: 30
  uvwFI:
    enabled: false
    nbins_2d: [10, 10]    # (u_bins, v_bins) - uses same for w
```

## Implementation

The `SampleReweighter` class (`lib/reweighting.py`) fits histograms on training data and computes per-sample weights to balance underrepresented regions during training.

### How It Works

1. **Range Detection**: First pass through training data to find min/max values for each enabled task
2. **Histogram Building**: Build histogram with specified `nbins` within the auto-detected range
3. **Weight Computation**: Weights = 1 / (bin_count + ε), normalized to mean = 1

The histogram range is automatically determined from your training data - you only specify the number of bins.

---

# Validation Metrics

Task-specific metrics are logged to MLflow during validation.

## Angle Task Metrics

| Metric | MLflow Key | Description |
|--------|------------|-------------|
| Cosine Loss | `val_cos` | `1 - cos_sim` where cos_sim is dot product of predicted and true emission direction unit vectors. Range: 0 (perfect) to 2 (opposite). |
| Opening Angle | `angle_resolution_68pct` | 68th percentile of opening angle between pred/true directions (degrees). |
| Theta Bias | `theta_bias` | Mean of θ residuals. |
| Theta RMS | `theta_rms` | Standard deviation of θ residuals. |

## Position Task Metrics

| Metric | MLflow Key | Description |
|--------|------------|-------------|
| Cosine Loss | `val_cos_pos` | `1 - cos_sim` for position vectors. Measures if pred/true positions point in same direction from origin. |
| Distance 68% | `uvw_dist_68pct` | 68th percentile of Euclidean distance between pred/true positions. |
| U/V/W Resolution | `uvw_{u,v,w}_res_68pct` | 68th percentile of absolute residual for each axis. |

## Energy/Timing Task Metrics

Energy and timing tasks use residual-based metrics tracked in resolution plots rather than dedicated MLflow metrics.

## Regenerating Plots

Regenerate resolution plots from saved predictions CSV files:

```bash
# Regenerate all available plots
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/

# Regenerate specific tasks
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --tasks energy angle

# Save to different directory
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --output_dir plots/
```
