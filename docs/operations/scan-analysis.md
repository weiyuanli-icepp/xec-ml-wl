# Hyperparameter Scan Analysis

## Scan Summary Tool

`macro/scan_summary.py` extracts metrics from the MLflow database and produces a formatted summary table plus a comparison PDF.

### Usage

```bash
# Angle regressor scan
python macro/scan_summary.py --experiment gamma_angle --prefix ang_scan

# Timing regressor scan
python macro/scan_summary.py --experiment gamma_timing --prefix tim_scan

# Energy regressor scan
python macro/scan_summary.py --experiment gamma_energy

# Table only (skip PDF generation)
python macro/scan_summary.py --experiment gamma_angle --prefix ang_scan --no-plot

# Custom output path
python macro/scan_summary.py --experiment gamma_angle --output my_summary.pdf

# Custom tracking URI
python macro/scan_summary.py --experiment gamma_angle --tracking-uri sqlite:///mlruns.db
```

### Output: Terminal Tables

The script prints four tables:

1. **Configuration** — Only shows parameters that differ across runs, so you immediately see what changed.
2. **Performance** — Sorted by best validation loss. Includes train/val loss, overfit gap, cosine loss, and angular resolution (for angle task).
3. **Bias & Skew** (angle only) — Theta/phi bias (mean residual), RMS, and skewness.
4. **Diagnostics** — Gradient norm max, final learning rate, overfitting ratio.
5. **Rankings** — Numbered list from best to worst by validation loss.

### Output: PDF Plots

The PDF contains the following pages:

| Page | Content | Purpose |
|------|---------|---------|
| 1 | Validation loss curves | Compare convergence speed and final performance |
| 2 | Train (dashed) vs Val (solid) | Detect overfitting (large gap = overfitting) |
| 3 | Angular resolution vs epoch | Track 68th percentile opening angle (angle task only) |
| 4 | Cosine loss vs epoch | Geometric angle accuracy (angle task only) |
| 5 | Theta bias & RMS vs epoch | Systematic offset and spread in theta (angle task only) |
| 6 | Phi bias & RMS vs epoch | Systematic offset and spread in phi (angle task only) |
| 7 | Bar chart ranking | Visual ranking of all runs by best validation loss |

---

## How to Analyze Scan Results

### Key Metrics

| Metric | What it tells you | Good sign | Bad sign |
|--------|-------------------|-----------|----------|
| `best_val_loss` | Overall model quality | Lowest wins | — |
| `overfit_gap` (val - train) | Generalization | Small positive gap | Large gap → overfitting |
| `overfit_ratio` (val / train) | Relative generalization | Close to 1.0 | >> 1 → overfitting |
| `angle_resolution_68pct` | Physics resolution [deg] | Lower is better | — |
| `theta_bias`, `phi_bias` | Systematic offset | Close to 0 | Nonzero → predictions shifted |
| `theta_skew`, `phi_skew` | Residual asymmetry | Close to 0 | Nonzero → asymmetric errors |
| `theta_rms`, `phi_rms` | Residual spread | Lower is better | — |
| `grad_norm_max` | Gradient health | Comparable to grad_clip | >> grad_clip → heavy clipping |
| `final_lr` | LR schedule completion | At `lr_min` if schedule finished | Stuck at high LR → didn't converge |

### Decision Framework

#### 1. Group steps by what they test

Categorize your scan steps into groups (e.g., loss function, regularization, optimization, model capacity, training duration). Compare within groups to draw conclusions.

#### 2. Check convergence first

Look at the loss curves (PDF page 1). If any run's loss is still decreasing at the final epoch, it hasn't converged — its result is not directly comparable to converged runs. Extend training before drawing conclusions.

#### 3. Diagnose the bottleneck

| Observation | Diagnosis | Next step |
|-------------|-----------|-----------|
| Small train/val gap | Underfitting (capacity-limited) | Bigger model, more data, less regularization |
| Large train/val gap | Overfitting | More regularization, more data, smaller model |
| Bias ≠ 0 | Systematic error | Check data distribution, loss function, architecture |
| Skew ≠ 0 | Asymmetric errors | Check angular coverage in training data |
| grad_norm >> grad_clip | Heavy gradient clipping | Increase grad_clip or lower LR |
| Loss plateaus early | LR schedule issue | Try OneCycle, lower initial LR, or longer warmup |

#### 4. Decide next steps

- **One step clearly dominates** → Adopt its settings as the new baseline, do a finer sweep around it.
- **Gains are orthogonal** (e.g., higher grad_clip helped AND lower weight_decay helped) → Combine them in a single run.
- **Extended training helped** → Re-run the best config with more epochs before comparing other hyperparameters.
- **Bias/skew persists** → This is a data or architecture issue, not just hyperparameters.
- **Train/val gap is small everywhere** → Bottleneck is model capacity or data size, not regularization.

---

## Scan Configuration

### Angle Regressor Scan

Configs in `config/reg/ang_scan/`, submitted via `jobs/run_angle_scan.sh`.

```bash
./jobs/run_angle_scan.sh              # Submit all default steps
./jobs/run_angle_scan.sh 8 9          # Submit specific steps
DRY_RUN=1 ./jobs/run_angle_scan.sh    # Preview without submitting
```

### Timing Regressor Scan

Configs in `config/reg/tim_scan/`, submitted via `jobs/run_timing_scan.sh`.

### Energy Regressor Scan

Configs in `config/reg/scan/`, submitted via `jobs/run_scan.sh`.

---

## MLflow Tracked Metrics

These metrics are logged per epoch and available for analysis:

### Common (all tasks)
- `train/loss`, `val/loss` — Optimization loss
- `val/l1`, `val/smooth_l1`, `val/mse` — Validation metrics
- `train/grad_norm_max` — Maximum gradient norm before clipping
- `system/lr` — Current learning rate
- `system/vram_peak_GB` — Peak GPU memory usage
- `system/epoch_time_sec` — Time per epoch

### Angle task
- `val/cos` — Cosine similarity loss (1 - cos_sim)
- `theta_bias`, `theta_rms`, `theta_skew` — Theta residual statistics
- `phi_bias`, `phi_rms`, `phi_skew` — Phi residual statistics
- `angle_resolution_68pct` — 68th percentile opening angle [deg]
