# Output & Artifacts

All results are logged to **MLflow** and stored in the `artifacts/<RUN_NAME>/` directory. The specific outputs depend on the training type (regressor, MAE, or inpainter) and which tasks are enabled.

---

## 1. Regressor Training Artifacts

Output directory: `artifacts/<RUN_NAME>/`

### Checkpoint Files

| File | Description | Contents |
|------|-------------|----------|
| `checkpoint_best.pth` | Best model (lowest validation loss) | model_state_dict, ema_state_dict, optimizer_state_dict, scheduler_state_dict, best_val, mlflow_run_id |
| `checkpoint_last.pth` | Latest checkpoint (after each epoch) | Same as above |

### Task-Specific CSV Predictions

Generated at best epoch and final evaluation. Only created for enabled tasks.

| Task | File | Columns |
|------|------|---------|
| **angle** | `predictions_angle_{run_name}.csv` | true_theta, true_phi, pred_theta, pred_phi |
| **energy** | `predictions_energy_{run_name}.csv` | true_energy, pred_energy |
| **timing** | `predictions_timing_{run_name}.csv` | true_timing, pred_timing |
| **uvwFI** | `predictions_uvwFI_{run_name}.csv` | true_u, true_v, true_w, pred_u, pred_v, pred_w |

### Task-Specific Plots (PDF)

Only generated for enabled tasks. Scatter plots are included within resolution plots.

**Angle Task:**
| File | Description |
|------|-------------|
| `resolution_angle_{run_name}.pdf` | Theta/Phi resolution profiles (binned), 68% width vs angle, opening angle vs angle |

**Energy Task:**
| File | Description |
|------|-------------|
| `resolution_energy_{run_name}.pdf` | Row 1: Residual histogram, resolution vs energy, relative resolution (σ/E) vs energy, pred vs true scatter. Row 2 (if uvwFI enabled): Resolution vs U/V/W first interaction point |

**Timing Task:**
| File | Description |
|------|-------------|
| `resolution_timing_{run_name}.pdf` | Timing resolution profile with residual histogram and pred vs true scatter |

**Position Task:**
| File | Description |
|------|-------------|
| `resolution_uvwFI_{run_name}.pdf` | U/V/W position resolution profiles with residual histograms |

### General Plots (All Tasks)

| File | Description |
|------|-------------|
| `face_weights_{run_name}.pdf` | Model feature importance across detector faces |
| `worst_events/worst_01_{run_name}.pdf` | Top-5 worst predictions with detector visualization |
| `worst_events/worst_02_{run_name}.pdf` | (comprehensive 2-panel npho + timing display) |
| ... up to `worst_05_{run_name}.pdf` | |

**Worst Case Event Display Details:**
- Title shows: Run/Event ID, loss (×1000 scale), truth energy (MeV), first interaction point (u,v,w in cm)
- Energy predictions displayed in MeV units (internal GeV × 1000)
- Two-panel layout: Photon counts (top) and timing (bottom) across all detector faces

### ONNX Export

| File | Description |
|------|-------------|
| `{onnx_filename}` (configurable) | Serialized model for C++ inference, includes all active task heads |

---

## 2. MAE Pre-training Artifacts

Output directory: `{save_path}/` (typically `artifacts/<RUN_NAME>/`)

### Checkpoint Files

| File | Description | Contents |
|------|-------------|----------|
| `mae_checkpoint_best.pth` | Best MAE checkpoint | epoch, model_state_dict, optimizer_state_dict, scaler_state_dict, ema_state_dict, scheduler_state_dict, best_val_loss, mlflow_run_id, config |
| `mae_checkpoint_last.pth` | Latest checkpoint (every 10 epochs + final) | Same as above |

### Encoder Weights for Transfer Learning

| File | Description |
|------|-------------|
| `mae_encoder_epoch_{epoch}.pth` | Standalone encoder weights extracted from EMA model |

Saved at: every 10 epochs, best epoch, and final epoch. Use these for fine-tuning downstream regressor.

### ROOT Predictions (if `--save_predictions` flag)

| File | Description |
|------|-------------|
| `mae_predictions_epoch_{epoch}.root` | Validation set sensor-level predictions |

**Main Tree Branches (`predictions`):**
- `event_id` - Event identifier
- `truth_npho` - Ground truth npho values
- `truth_time` - Ground truth time values (only if `"time"` in `predict_channels`)
- `mask` - Binary mask (1 = masked sensor)
- `masked_npho`, `masked_time` - Input values (masked sensors have sentinel)
- `pred_npho` - Model predictions for npho
- `pred_time` - Model predictions for time (only if `"time"` in `predict_channels`)
- `err_npho` - Prediction errors for npho
- `err_time` - Prediction errors for time (only if `"time"` in `predict_channels`)
- `run_id` - Run number

**Metadata Tree (`metadata`):**

The ROOT file includes a `metadata` tree with a single entry containing model configuration:
- `predict_channels` - Comma-separated list of predicted channels (e.g., `"npho"` or `"npho,time"`)
- `npho_scale` - Normalization scale for npho
- `time_scale` - Normalization scale for time
- `sentinel_time` - Value used for invalid/masked sensors

Downstream macros auto-detect the prediction mode from this metadata.

Saved at: every 10 epochs + final epoch.

### MLflow Metrics

- `train/total_loss`, `train/npho_loss`
- `train/time_loss` (only if `"time"` in `predict_channels`)
- `val/total_loss`, `val/npho_loss`
- `val/time_loss` (only if `"time"` in `predict_channels`)
- `val/mae_npho`, `val/rmse_npho` (if `track_mae_rmse` enabled)
- `val/mae_time`, `val/rmse_time` (if `track_mae_rmse` enabled and predicting time)
- Per-face losses: `val/loss_{face}_npho`
- Per-face time losses: `val/loss_{face}_time` (only if predicting time)
- Channel variances (if `auto_channel_weight` enabled)

---

## 3. Inpainter Training Artifacts

Output directory: `{save_path}/` (typically `artifacts/<RUN_NAME>/`)

### Checkpoint Files

| File | Description | Contents |
|------|-------------|----------|
| `inpainter_checkpoint_best.pth` | Best inpainter checkpoint | epoch, model_state_dict, optimizer_state_dict, scaler_state_dict, scheduler_state_dict, best_val_loss, mlflow_run_id, config |
| `inpainter_checkpoint_last.pth` | Latest checkpoint (every 10 epochs + final) | Same as above |

### ROOT Predictions (if `save_root_predictions=True`)

| File | Description |
|------|-------------|
| `inpainter_predictions_epoch_{epoch}.root` | Validation predictions with masked channel recovery |

**Main Tree Branches (`predictions`):**
- `event_idx` - Event index
- `sensor_id` - Sensor identifier
- `face` - Detector face name
- `truth_npho` - Ground truth npho values
- `truth_time` - Ground truth time values (only if `"time"` in `predict_channels`)
- `pred_npho` - Model predictions for npho
- `pred_time` - Model predictions for time (only if `"time"` in `predict_channels`)
- `error_npho` - Prediction errors for npho
- `error_time` - Prediction errors for time (only if `"time"` in `predict_channels`)

**Metadata Tree (`metadata`):**

The ROOT file includes a `metadata` tree with a single entry containing model configuration:
- `predict_channels` - Comma-separated list of predicted channels (e.g., `"npho"` or `"npho,time"`)
- `npho_scale` - Normalization scale for npho
- `time_scale` - Normalization scale for time
- `sentinel_time` - Value used for invalid/masked sensors

Downstream macros (e.g., `macro/analyze_inpainter.py`, `val_data/analyze_inpainter.py`) auto-detect the prediction mode from this metadata and skip time-related analysis when only npho is predicted.

Saved at: every 10 epochs + final epoch.

### MLflow Metrics

- `train/total_loss`, `train/npho_loss`
- `train/time_loss` (only if `"time"` in `predict_channels`)
- `val/total_loss`, `val/npho_loss`
- `val/time_loss` (only if `"time"` in `predict_channels`)
- Per-face losses: `val/loss_{face}_npho`
- Per-face time losses: `val/loss_{face}_time` (only if predicting time)
- `val/mae_npho`, `val/rmse_npho` (if `track_mae_rmse` enabled)
- `val/mae_time`, `val/rmse_time` (if `track_mae_rmse` enabled and predicting time)
- `actual_mask_ratio`, `n_masked_total`

---

## Directory Structure Summary

```
artifacts/
└── {run_name}/
    │
    │   # Regressor outputs
    ├── checkpoint_best.pth
    ├── checkpoint_last.pth
    ├── model.onnx                        # ONNX export (if enabled)
    ├── face_weights_{run_name}.pdf
    │
    │   # Angle task (if enabled)
    ├── predictions_angle_{run_name}.csv
    ├── resolution_angle_{run_name}.pdf
    │
    │   # Energy task (if enabled)
    ├── predictions_energy_{run_name}.csv
    ├── resolution_energy_{run_name}.pdf   # Includes position-profiled plots if uvwFI enabled
    │
    │   # Timing task (if enabled)
    ├── predictions_timing_{run_name}.csv
    ├── resolution_timing_{run_name}.pdf
    │
    │   # Position task (if enabled)
    ├── predictions_uvwFI_{run_name}.csv
    ├── resolution_uvwFI_{run_name}.pdf
    │
    │   # Worst case events
    └── worst_events/
        ├── worst_01_{run_name}.pdf
        ├── worst_02_{run_name}.pdf
        └── ...

    │   # MAE outputs (if MAE training)
    ├── mae_checkpoint_best.pth
    ├── mae_checkpoint_last.pth
    ├── mae_encoder_epoch_{epoch}.pth
    └── mae_predictions_epoch_{epoch}.root

    │   # Inpainter outputs (if inpainter training)
    ├── inpainter_checkpoint_best.pth
    ├── inpainter_checkpoint_last.pth
    └── inpainter_predictions_epoch_{epoch}.root

mlruns/                                  # MLflow experiment tracking
```

---

## Controlling Artifact Generation

Resolution plots and prediction CSVs are generated **automatically** after training. This behavior can be controlled via the config file:

```yaml
# config/train_config.yaml
checkpoint:
  save_artifacts: true  # Set to false to disable plot/CSV generation
```

Or via CLI:
```bash
python -m lib.train_regressor --config config.yaml --save_artifacts false
```

**When artifacts are saved:**
- At best validation epoch (with `_ep{N}` suffix)
- At end of training (without epoch suffix, using best model)

---

## Visualization Tools

### MLflow UI

Real-time tracking is available with MLflow.

```bash
# Start MLflow UI
$ cd /path/to/xec-ml-wl
$ (activate xec-ml-wl conda environment)
$ mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns.db --host 127.0.0.1 --port 5000
```

### Regenerating Plots from CSV

If you need to regenerate resolution plots (e.g., after updating plotting code) without re-running inference, use the `regenerate_resolution_plots.py` macro:

```bash
# Regenerate all available plots
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/

# Regenerate only specific tasks
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --tasks energy angle

# Save to a different directory
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --output_dir plots/

# Custom suffix for output files
python macro/regenerate_resolution_plots.py artifacts/<RUN_NAME>/ --suffix _v2
```

**Requirements:** The artifact directory must contain `predictions_*.csv` files from a previous training run.

### Standalone Validation

To run validation on a checkpoint without training (generates plots and predictions):

```bash
python macro/validate_regressor.py artifacts/<RUN>/checkpoint_best.pth \
    --val_path data/val/ \
    --tasks energy angle \
    --output_dir artifacts/<RUN>/
```

---

## Resolution Plot Details

### Energy Resolution Fit

The "Relative Resolution vs Energy" plot includes a fit to the standard calorimeter resolution formula:

$$\frac{\sigma}{E} = \sqrt{\left(\frac{a}{\sqrt{E}}\right)^2 + b^2 + \left(\frac{c}{E}\right)^2}$$

Where:
- **a** = stochastic term (statistical fluctuations, scales as $1/\sqrt{E}$)
- **b** = constant term (systematic effects like calibration, leakage)
- **c** = noise term (electronic noise, scales as $1/E$)

Fit parameters are displayed in the plot title as percentages (e.g., `a=2.50%/√E, b=1.20%, c=0.50%/E`).

---

## Metrics Definition

### 1. Physics Performance Metrics (Regressor)

These metrics evaluate the quality of the photon direction reconstruction. Calculated during validation using `eval_stats` and `eval_resolution`.

| Metric | Definition | Formula |
| ------ | ---------- | ------- |
| Theta Bias (`theta_bias`) | The arithmetic mean of the residuals. | $\mu = \text{Mean}(\theta_{\mathrm{pred}} - \theta_{\mathrm{true}})$ |
| Theta RMS (`theta_rms`) | The standard deviation of the residuals. | $\sigma = \text{Std}(\theta_{\mathrm{pred}} - \theta_{\mathrm{true}})$ |
| Theta Skewness (`theta_skew`) | A measure of the asymmetry of the error distribution. | $$\text{Skew} = \frac{\frac{1}{N} \sum_{i=1}^{N} (\Delta \theta_i - \mu)^3}{\left( \frac{1}{N} \sum_{i=1}^{N} (\Delta \theta_i - \mu)^2 \right)^{3/2}}$$ |
| Opening Angle Resolution (`val_resolution_deg`) | The 68th percentile of the 3D opening angle $\psi$ between the predicted and true vectors. | $\psi = \arccos(v_{\mathrm{pred}} \cdot v_{\mathrm{true}})$ |

### 2. MAE/Inpainter Metrics

| Metric | Description |
|--------|-------------|
| `total_loss` | Combined weighted loss (npho + time if predicting both) |
| `loss_npho` | Npho channel loss |
| `loss_time` | Time channel loss (only if `"time"` in `predict_channels`) |
| `mae_npho` | Mean Absolute Error for npho on masked positions |
| `mae_time` | Mean Absolute Error for time on masked positions (only if predicting time) |
| `rmse_npho` | Root Mean Square Error for npho on masked positions |
| `rmse_time` | Root Mean Square Error for time on masked positions (only if predicting time) |
| `actual_mask_ratio` | Effective mask ratio after excluding invalid sensors |

**Npho-Only Mode:** When `predict_channels: ["npho"]`, time-related metrics (`loss_time`, `mae_time`, `rmse_time`) are not tracked.

### 3. System Engineering Metrics

These metrics monitor the health of the training infrastructure.

| Metric | Key in MLflow | Interpretation |
| ------ | ------------- | -------------- |
| Allocated Memory | `system/memory_allocated_GB` | Actual tensor size on GPU. Steady growth indicates memory leak. |
| Reserved Memory | `system/memory_reserved_GB` | Total memory PyTorch requested from OS. OOM if hits limit. |
| Peak Memory | `system/memory_peak_GB` | Highest memory usage (usually during backward). Use to tune batch_size. |
| GPU Utilization | `system/gpu_utilization_pct` | Ratio of Allocated to Total VRAM. Low (<50%) = increase batch size. |
| Fragmentation | `system/memory_fragmentation` | Empty space within reserved blocks. High (>0.5) = inefficient. |
| RAM Usage | `system/ram_used_gb` | System RAM used. High = reduce chunksize. |
| Throughput | `system/epoch_duration_sec` | Wall-clock time per epoch. |

### 4. System Performance Metrics

| Metric | Key in MLflow | Description |
| ------ | ------------- | ----------- |
| Throughput | `system/throughput_events_per_sec` | Events processed per second. |
| Data Load Time | `system/avg_data_load_sec` | Time GPU waits for CPU. If high, increase chunksize. |
| Compute Efficiency | `system/compute_efficiency` | % of time GPU is computing. |

---

## Resuming Training

The script supports resumption. It detects if an EMA state exists in the checkpoint and loads it; otherwise, it syncs the EMA model with the loaded weights.

```bash
--resume_from "artifacts/<run_name>/checkpoint_last.pth"
```
or
```bash
--resume_from "artifacts/<run_name>/checkpoint_best.pth"
```

### Resume Options

| Option | Config Path | CLI | Description |
|--------|-------------|-----|-------------|
| Refresh LR | `checkpoint.refresh_lr` | `--refresh_lr` | Reset scheduler for remaining epochs. If resuming at epoch 15 with total 50, creates new scheduler with T_max=35. |
| Reset Epoch | `checkpoint.reset_epoch` | `--reset_epoch` | Start from epoch 1 (only load model weights, fresh optimizer/scheduler). |
| New MLflow Run | `checkpoint.new_mlflow_run` | `--new_mlflow_run` | Force new MLflow run instead of resuming previous run ID. |

**Example: Resume with fresh learning rate**
```bash
python -m lib.train_regressor --config config.yaml \
    --resume_from artifacts/my_run/checkpoint_best.pth \
    --refresh_lr --lr 1e-5 --epochs 100
```

**Example: Fine-tune from checkpoint with fresh training state**
```bash
python -m lib.train_regressor --config config.yaml \
    --resume_from artifacts/my_run/checkpoint_best.pth \
    --reset_epoch --new_mlflow_run --epochs 50
```

If the run was configured with a scheduler and stopped mid-training (without `--refresh_lr`), it resumes from the learning rate where it stopped:

$$\mathrm{LR} = \mathrm{LR}_\mathrm{min} + \frac{1}{2} \Big(\mathrm{LR}_{\mathrm{max}} - \mathrm{LR}_{\mathrm{min}}\Big) \Bigg(1 + \cos \Big(\frac{\mathrm{epoch} - \mathrm{warmup}}{\mathrm{total} - \mathrm{warmup}} \pi\Big)\Bigg)$$

### Checkpoint Compatibility

| Source | Target | Compatible? | Notes |
|--------|--------|-------------|-------|
| Regressor checkpoint | Regressor | Yes | Full state restored |
| MAE checkpoint | Regressor | Yes | Encoder weights only, optimizer reset |
| MAE checkpoint | MAE | Yes | Full state restored |
| Inpainter checkpoint | Inpainter | Yes | Full state restored |
| Regressor checkpoint | MAE/Inpainter | No | Different model architecture |
| Single-GPU checkpoint | Multi-GPU training | Yes | DDP saves unwrapped state_dict |
| Multi-GPU checkpoint | Single-GPU training | Yes | No `module.` prefix in saved weights |
