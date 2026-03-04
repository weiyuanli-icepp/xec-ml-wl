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
    loss_fn: "smooth_l1"    # or gaussian_nll for uncertainty estimation
    loss_beta: 0.002
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
| `gaussian_nll` | β-NLL (see below) | Uncertainty estimation |

### Gaussian NLL Loss (β-NLL)

Enables per-event uncertainty estimation. The model predicts both mean (μ) and log-variance (log σ²), and the loss penalizes both prediction accuracy and uncertainty calibration:

$$\mathcal{L} = [\sigma^2]^{\beta} \cdot \frac{1}{2}\left(\log\sigma^2 + \frac{(y - \mu)^2}{\sigma^2}\right)$$

Where `[·]` denotes stop-gradient (detached). The `loss_beta` parameter controls β:

| β | Behavior | Use case |
|---|----------|----------|
| 0.0 | Standard NLL — variance can dominate early training | Mathematically pure but can be unstable |
| **0.5** | **Balanced (recommended)** — equal gradient contribution from μ and σ² | Best default |
| 1.0 | μ-gradients match MSE — variance learning is slow but stable | When accuracy matters most |

```yaml
tasks:
  energy:
    enabled: true
    loss_fn: "gaussian_nll"
    loss_beta: 0.5          # β parameter (0=NLL, 0.5=balanced, 1=MSE-like)
    weight: 1.0
```

When `gaussian_nll` is configured, the task head output dimension increases by 1 (e.g., energy: 1 → 2 for `[mu, log_var]`). During validation, log_var is stripped so downstream handlers and plots see the standard prediction shape.

See [ML Techniques](../notes/ml-techniques.md#heteroscedastic-regression-faithful-loss) for background on heteroscedastic regression.

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
| `energy` | 1 (or 2 with `gaussian_nll`) | Energy (+ log_var for uncertainty) |
| `timing` | 1 (or 2 with `gaussian_nll`) | Timing (+ log_var for uncertainty) |
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

Task-specific metrics are logged to MLflow during validation. Per-task loss configuration is also logged as MLflow parameters (`task/{name}_loss_fn`, `task/{name}_loss_beta`, `task/{name}_weight`) for experiment tracking.

## Angle Task Metrics

| Metric | MLflow Key | Description |
|--------|------------|-------------|
| Cosine Loss | `val/cos` | `1 - cos_sim` where cos_sim is dot product of predicted and true emission direction unit vectors. Range: 0 (perfect) to 2 (opposite). |
| Opening Angle | `angle_resolution_68pct` | 68th percentile of opening angle between pred/true directions (degrees). |

## Position Task Metrics

| Metric | MLflow Key | Description |
|--------|------------|-------------|
| Cosine Loss | `val/cos_pos` | `1 - cos_sim` for position vectors. Measures if pred/true positions point in same direction from origin. |
| Distance 68% | `uvw_dist_68pct` | 68th percentile of Euclidean distance between pred/true positions. |
| U/V/W Resolution | `uvw_{u,v,w}_res_68pct` | 68th percentile of absolute residual for each axis. |

## Energy Task Metrics

| Metric | MLflow Key | Description |
|--------|------------|-------------|
| Bias | `energy_bias` | Mean residual (pred - true) |
| Resolution 68% | `energy_res_68pct` | 68th percentile of absolute residuals |
| Pred Std Mean | `energy_pred_std_mean` | Mean predicted σ (only with `gaussian_nll`) |
| Calibration Ratio | `energy_calibration_ratio` | Mean predicted σ / actual residual std (only with `gaussian_nll`; ideal = 1.0) |

## Timing Task Metrics

Timing uses residual-based metrics tracked in resolution plots.

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
