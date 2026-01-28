# Dead Channel Inpainting

The library includes a **dead channel inpainting** module for recovering sensor values at malfunctioning or dead channels. This is useful for:
- **Data recovery**: Interpolate missing sensor readings using surrounding context
- **Robustness training**: Train models to handle incomplete detector data
- **Preprocessing**: Clean up data before regression tasks

## 1. Architecture Overview

The inpainter (`XEC_Inpainter`) uses a frozen encoder from MAE pretraining combined with lightweight inpainting heads:

```
Input (with dead channels marked as sentinel)
    ↓
┌─────────────────────────────────────────┐
│  Frozen XECEncoder (from MAE)           │
│  - Extracts latent tokens per face      │
│  - Global context from transformer      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Face-Specific Inpainting Heads         │
│                                         │
│  Rectangular (Inner, US, DS, Outer):    │
│  - FaceInpaintingHead                   │
│  - Local CNN (2× ConvNeXtV2 blocks)     │
│  - Global conditioning from latent      │
│  - Hidden dim: 64                       │
│                                         │
│  Hexagonal (Top, Bottom):               │
│  - HexInpaintingHead                    │
│  - Local GNN (3× HexNeXt blocks)        │
│  - Global conditioning from latent      │
│  - Hidden dim: 96                       │
└─────────────────────────────────────────┘
    ↓
Output: Predicted (npho, time) at masked positions only
```

## 2. Masking Strategy

The inpainter uses **invalid-aware masking** to handle already-invalid sensors in the data:

- **Already-invalid sensors** (where `time == sentinel_value`) are excluded from the random masking pool
- **Random masking** is applied only to valid sensors at the specified `mask_ratio`
- **Loss computation** uses only randomly-masked positions (where ground truth exists)
- **actual_mask_ratio** metric tracks the effective masking: `randomly_masked / valid_sensors`

This ensures:
1. No loss is computed on sensors without ground truth (already-invalid in MC)
2. The model learns to predict from real neighboring context, not from sentinel values

## 3. Training Modes

**Option A: With MAE Pre-training (Recommended)**

```bash
# First, train MAE
python -m lib.train_mae --config config/mae_config.yaml

# Then, train inpainter with frozen MAE encoder
python -m lib.train_inpainter --config config/inpainter_config.yaml \
    --mae_checkpoint artifacts/mae/mae_checkpoint_best.pth
```

**Option B: Without MAE Pre-training (From Scratch)**

```bash
# Train inpainter without MAE (encoder trained jointly)
python -m lib.train_inpainter --config config/inpainter_config.yaml \
    --mae_checkpoint ""
```

**Interactive Training Script:**

```bash
# Edit configuration in the script first
./macro/interactive_inpainter_train_config.sh
```

## 4. Configuration

Configure in `config/inpainter_config.yaml`:

```yaml
# Model
model:
  outer_mode: "finegrid"        # Must match MAE encoder config
  outer_fine_pool: [3, 3]       # Must match MAE encoder config
  mask_ratio: 0.05              # Realistic dead channel density (1-10%)
  freeze_encoder: true          # Freeze encoder, train only heads

# Training
training:
  mae_checkpoint: "artifacts/mae/checkpoint_best.pth"  # or null
  epochs: 50
  lr: 1.0e-4
  lr_scheduler: "cosine"
  loss_fn: "smooth_l1"          # smooth_l1, mse, l1
  npho_weight: 1.0
  time_weight: 1.0
```

## 5. Metrics

| Metric | Description |
|--------|-------------|
| `total_loss` | Combined weighted loss (npho + time) |
| `loss_npho` / `loss_time` | Per-channel losses (sum across faces) |
| `loss_{face}_npho/time` | Per-face, per-channel losses |
| `mae_npho` / `mae_time` | Mean Absolute Error on masked positions |
| `rmse_npho` / `rmse_time` | Root Mean Square Error on masked positions |
| `actual_mask_ratio` | Effective mask ratio after excluding invalid sensors |
| `n_masked_total` | Total number of masked sensors in batch |

## 6. Output

- **Checkpoints**: `artifacts/inpainter/checkpoint_best.pth`, `checkpoint_last.pth`
- **ROOT Predictions**: `inpainter_predictions_epoch_*.root` (every 10 epochs)
  - Branches: `event_idx`, `sensor_id`, `face`, `truth_npho`, `truth_time`, `pred_npho`, `pred_time`, `error_npho`, `error_time`

## 7. Analysis

Use `macro/analyze_inpainter.py` to evaluate predictions:

```bash
python macro/analyze_inpainter.py artifacts/inpainter/inpainter_predictions_epoch_50.root \
    --output analysis_output/
```

**Generated outputs:**
- `global_metrics.csv` - MAE, RMSE, bias, 68th/95th percentiles
- `face_metrics.csv` - Per-face breakdown
- `outliers.csv` - Predictions with large errors
- `residual_distributions.pdf` - Histograms with Gaussian fit
- `residual_per_face_*.pdf` - Per-face residual distributions
- `scatter_truth_vs_pred.pdf` - 2D density plots
- `resolution_vs_signal.pdf` - Resolution/bias vs truth magnitude
- `metrics_summary.pdf` - Bar chart comparison across faces

---

## 8. Real Data Validation

Validate the inpainter on real detector data that already contains dead channels.

### 8.1 Workflow

```
Real Data (with dead channels)
    │
    ▼
┌─────────────────────────────────────┐
│ 1. Identify dead channels (from DB) │
│ 2. Detect additional dead from data │
│ 3. Randomly mask healthy sensors    │
│ 4. Create combined mask             │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Inpainter Inference (CPU)           │
│ - Input: masked real data           │
│ - Output: predictions for all masked│
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│ Output ROOT File                    │
│ - mask_type=0: artificial (has      │
│   ground truth for evaluation)      │
│ - mask_type=1: originally dead      │
│   (no truth, inference only)        │
└─────────────────────────────────────┘
```

### 8.2 Preparing Real Data

Use `macro/PrepareRealData.C` to generate input ROOT files:

```bash
# In meganalyzer environment
./meganalyzer -b -q -I /path/to/PrepareRealData.C+
```

**Output branches:**
| Branch | Description |
|--------|-------------|
| `run`, `event` | Run and event numbers |
| `relative_npho[4760]` | Normalized npho (÷ max) |
| `relative_time[4760]` | Normalized time (- min) |
| `energyReco`, `timeReco` | Reconstructed energy/time |
| `xyzRecoFI[3]`, `uvwRecoFI[3]` | First interaction position |

**Note:** PrepareRealData.C uses `1e10` as sentinel for invalid values. The validation script converts this to `-5.0` (model sentinel) automatically.

### 8.3 Running Validation

```bash
# Using database for dead channels (recommended)
python macro/validate_inpainter_real.py \
    --checkpoint artifacts/inpainter/checkpoint_best.pth \
    --input DataGammaAngle_430000-431000.root \
    --run 430000 \
    --output validation_real/

# Using pre-saved dead channel list
python macro/validate_inpainter_real.py \
    --checkpoint artifacts/inpainter/checkpoint_best.pth \
    --input real_data.root \
    --dead-channel-file dead_channels_430000.txt \
    --output validation_real/

# Customize artificial masking
python macro/validate_inpainter_real.py \
    --checkpoint checkpoint.pth \
    --input real_data.root \
    --run 430000 \
    --n-mask-inner 10 \
    --n-mask-other 1 \
    --seed 42 \
    --output validation_real/
```

**Options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--n-mask-inner` | 10 | Healthy sensors to mask in inner face |
| `--n-mask-other` | 1 | Healthy sensors to mask in other faces |
| `--seed` | 42 | Random seed for reproducibility |
| `--batch-size` | 64 | Batch size for inference |
| `--max-events` | all | Limit number of events |

### 8.4 Output Format

The output ROOT file has additional columns compared to training predictions:

| Branch | Type | Description |
|--------|------|-------------|
| `mask_type` | int32 | 0=artificial (has truth), 1=dead (no truth) |
| `run` | int32 | Run number |
| `event` | int32 | Event number |

**Invalid values:** For dead channels (`mask_type=1`), truth and error are set to `-999`.

### 8.5 Event Display

Visualize individual events with `macro/show_inpainter_real.py`:

```bash
# By event index
python macro/show_inpainter_real.py 0 \
    --predictions validation_real/real_data_predictions.root \
    --original DataGammaAngle_430000-431000.root \
    --channel npho --save event_0.pdf

# By run/event number
python macro/show_inpainter_real.py \
    --predictions validation_real/real_data_predictions.root \
    --original DataGammaAngle_430000-431000.root \
    --run 430123 --event 456 \
    --channel both --save event_430123_456.pdf
```

**Display layout (3 rows × 4 faces):**

| Row | Content | Description |
|-----|---------|-------------|
| 1 | Original | Raw data with dead channels marked (hatched red) |
| 2 | Filled | Original + predictions inserted at masked positions |
| 3 | Residual | pred - truth for artificial masks only (dead = N/A) |

**Visual markers:**
- **Red hatching**: Dead channels (no ground truth)
- **Blue border**: Artificially masked (has ground truth)

**Console output:**
```
============================================================
Event Summary: Run 430123, Event 456
============================================================

Predictions by face:
Face         Artificial       Dead      Total
--------------------------------------------
inner              10         20         30
us                  1          3          4
ds                  1          2          3
outer               1          8          9
top                 1          7          8
bot                 1          5          6
--------------------------------------------
Total              15         45         60

Metrics (artificial masks only):
  npho: MAE=0.0234, RMSE=0.0312, Bias=-0.0012
  time: MAE=0.0156, RMSE=0.0198, Bias=0.0003
============================================================
```

### 8.6 Analyzing Results

```bash
# Analyze predictions (automatically filters by mask_type for metrics)
python macro/analyze_inpainter.py validation_real/real_data_predictions.root \
    --output validation_real/analysis/
```

For quantitative metrics (MAE, RMSE, etc.), only `mask_type=0` (artificial) predictions are used since they have ground truth.

---

## 9. Database Utilities

Query dead channel information from the MEG2 database.

### 9.1 Database Hierarchy

```
RunCatalog (run id) → XECConf_id
    → XECConf (id) → XECPMStatusDB_id
        → XECPMStatusDB (id) → XECPMStatus_id
            → XECPMStatus (idx: 0-4759, IsBad: 0/1)
```

### 9.2 Command Line Usage

```bash
# Print summary for a run
python -m lib.db_utils 430000

# List all dead channel indices
python -m lib.db_utils 430000 --list

# Save to file
python -m lib.db_utils 430000 --output dead_channels_430000.txt
```

**Example output:**
```
==================================================
Dead Channel Summary for Run 430000
==================================================
XECPMStatus_id: 123
Total dead: 45 / 4760 (0.95%)

Dead by face:
   inner: 20
      us: 3
      ds: 2
   outer: 8
     top: 7
     bot: 5
==================================================
```

### 9.3 Python API

```python
from lib.db_utils import (
    get_dead_channels,
    get_dead_channel_mask,
    get_dead_channel_info,
    print_dead_channel_summary
)

# Get dead channel indices
dead_channels = get_dead_channels(run_number=430000)
print(f"Dead channels: {len(dead_channels)}")  # [12, 45, 789, ...]

# Get boolean mask
dead_mask = get_dead_channel_mask(run_number=430000)  # shape: (4760,)

# Get detailed info with per-face breakdown
info = get_dead_channel_info(run_number=430000)
print(info['dead_by_face'])  # {'inner': 20, 'us': 3, ...}

# Print summary
print_dead_channel_summary(run_number=430000)
```

### 9.4 Requirements

- MySQL client (`mysql` command) must be installed and in PATH
- Login credentials configured via `mysql_config_editor`:
  ```bash
  mysql_config_editor set --login-path=meg_ro --host=<host> --user=<user> --password
  ```
- Access to MEG2 database from the machine running the script
