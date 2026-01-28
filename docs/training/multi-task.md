# Multi-Task Learning

The model supports simultaneous regression of multiple physical observables.

## Task Configuration

Configure tasks in `config/train_config.yaml`:

```yaml
tasks:
  angle:
    enabled: true
    loss_fn: "smooth_l1"    # smooth_l1, l1, mse, huber
    loss_beta: 1.0
    weight: 1.0
  energy:
    enabled: false
    loss_fn: "l1"
    weight: 1.0
  timing:
    enabled: false
    loss_fn: "l1"
    weight: 1.0
  uvwFI:
    enabled: false
    loss_fn: "mse"
    weight: 1.0
```

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

Configure in `config/train_config.yaml`:

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
