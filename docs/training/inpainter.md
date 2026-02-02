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

## 8. Model Export for Fast Inference

For production use and faster inference, export the trained model to TorchScript format.

### 8.1 Export Format: TorchScript Only

**ONNX export is NOT supported** for the inpainter model. PyTorch's `nn.TransformerEncoder` uses a fused CUDA kernel (`aten::_transformer_encoder_layer_fwd`) that cannot be exported to ONNX.

TorchScript export works correctly and provides:
- Exact numerical match with PyTorch
- C++ inference via libtorch
- ~3x speedup over eager mode (on GPU)

#### Why ONNX Export Fails

The error occurs during `torch.onnx.export()`:
```
UnsupportedOperatorError: Exporting the operator 'aten::_transformer_encoder_layer_fwd'
to ONNX opset version 17 is not supported.
```

**Root cause:** PyTorch's `nn.TransformerEncoder` has two internal paths:
- **Slow path**: Standard ops (matmul, softmax, layer norm) - ONNX compatible
- **Fast path**: Fused CUDA kernel (`_transformer_encoder_layer_fwd`) - NOT ONNX compatible

The fast path is automatically selected when `batch_first=True` (which our model uses).

#### Workarounds Attempted (All Failed)

1. **Disable SDP backends** - Did not affect TransformerEncoder's fused kernel
   ```python
   torch.backends.cuda.enable_flash_sdp(False)
   torch.backends.cuda.enable_mem_efficient_sdp(False)
   torch.backends.cuda.enable_math_sdp(True)
   ```
   **Result:** Still uses `_transformer_encoder_layer_fwd`

2. **Disable nested tensor** - Set `enable_nested_tensor=False` on TransformerEncoder
   ```python
   for module in model.modules():
       if isinstance(module, nn.TransformerEncoder):
           module.enable_nested_tensor = False
   ```
   **Result:** Still uses fused kernel

3. **Set TransformerEncoder to training mode** - Fast path only used in eval mode
   ```python
   for module in model.modules():
       if isinstance(module, nn.TransformerEncoder):
           module.train()
   ```
   **Result:** PyTorch's ONNX exporter uses JIT tracing internally, which still triggers the fused kernel at the C++ level regardless of Python-level training mode.

#### Why Other Transformer Models Work with ONNX

Some transformer models export to ONNX successfully because they:
- Use `batch_first=False` (older default, triggers slow path)
- Use custom transformer implementations with basic ops
- Use Hugging Face transformers (different implementation, not `nn.TransformerEncoder`)
- Were built with older PyTorch versions (before fast path was added)

#### Possible Solutions (Not Implemented)

1. **Replace `nn.TransformerEncoder`** with a custom implementation using basic ops
2. **Use `torch.onnx.dynamo_export`** (newer PyTorch 2.0+ API, may have better support)
3. **Rebuild with `batch_first=False`** and transpose inputs/outputs

### 8.2 Export Commands

```bash
# Export to TorchScript
python macro/export_onnx_inpainter.py \
    artifacts/inpainter/inpainter_checkpoint_best.pth \
    --output artifacts/inpainter/inpainter.pt

# Use standard weights instead of EMA
python macro/export_onnx_inpainter.py checkpoint.pth --no-ema --output model.pt
```

### 8.3 Exported Model Interface

**Inputs:**
| Name | Shape | Description |
|------|-------|-------------|
| `input` | `(B, 4760, 2)` | Sensor data `[npho, time]` with dead channels as sentinel (-5.0) |
| `mask` | `(B, 4760)` | Binary mask: 1 = masked/dead, 0 = valid |

**Output:**
| Name | Shape | Description |
|------|-------|-------------|
| `output` | `(B, 4760, 2)` | Full tensor with predictions at masked positions |

**Usage in C++/Python:**
```python
# TorchScript
model = torch.jit.load("inpainter.pt")
output = model(input_tensor, mask_tensor)

# Apply predictions only at masked positions
final = torch.where(mask.unsqueeze(-1).bool(), output, input_tensor)
```

### 8.4 Fixed-Size Output for TorchScript Export

The inpainter supports two forward methods:

| Method | Output Shape | Use Case |
|--------|--------------|----------|
| `forward()` | Variable (only masked positions) | Training (efficient memory) |
| `forward_full_output()` | Fixed `(B, 4760, 2)` | Export/inference (TorchScript compatible) |

The `forward_full_output()` method predicts ALL sensor positions, enabling clean TorchScript tracing:

```python
# Training (variable-size, more efficient)
results, original_values, mask = model.forward(x_batch, mask=mask)

# Export/inference (fixed-size, TorchScript compatible)
pred_all = model.forward_full_output(x_batch, mask)  # (B, 4760, 2)
```

**Why this works:**
- The model internally computes predictions for all positions anyway
- The `forward()` method extracts only masked positions for efficiency
- The `forward_full_output()` method skips the extraction, returning the full tensor
- No retraining needed - same weights work for both methods

**Note on prediction quality:**
- Predictions at masked positions are the same regardless of which method is used
- Predictions at unmasked positions are computed but meaningless (use original values instead)
- The export wrapper automatically combines: predictions at masked positions, originals elsewhere

#### Internal Implementation: Vectorized Outer Face Head

The `OuterSensorInpaintingHead` (for outer face with finegrid mode) uses **vectorized attention pooling** to compute predictions for all 234 outer sensors in a single batched operation:

```python
# Vectorized: single batched gather + attention pooling
gathered = torch.gather(features_flat, dim=1, index=idx.expand(...))  # All sensors at once
attn_weights = F.softmax(attn_logits.masked_fill(~valid_mask, -inf), dim=-1)
all_sensor_features = (gathered * attn_weights).sum(dim=2)  # (B, 234, hidden_dim)
```

**Why both forward paths compute all sensors:**

Both `forward()` and `forward_full()` call the same `_compute_all_sensor_preds_vectorized()` internally:

| Method | Internal computation | Post-processing |
|--------|---------------------|-----------------|
| `forward()` | All 234 sensors | Extract masked → scatter to variable output |
| `forward_full()` | All 234 sensors | Return directly |

This means `forward()` is actually **slower** than `forward_full()` due to the extraction overhead, regardless of mask ratio.

**Performance implication:** The training engine (`run_epoch_inpainter`) auto-selects `use_fast_forward=True` by default since both paths do the same computation but `forward_full()` avoids the extraction overhead.

**Historical note:** Before vectorization, `forward()` used a Python for-loop over 234 sensors, making it slower for computing all sensors but theoretically faster at low mask ratios. The vectorization eliminated this distinction—batched GPU operations are faster than selective computation with Python loop overhead.

### 8.5 TorchScript Export Workflow

```bash
# Export to TorchScript
python macro/export_onnx_inpainter.py \
    artifacts/inpainter/inpainter_checkpoint_best.pth \
    --output artifacts/inpainter/inpainter.pt

# Use for fast inference
python macro/validate_inpainter_real.py \
    --torchscript artifacts/inpainter/inpainter.pt \
    --input real_data.root \
    --run 430000 \
    --output validation_output/
```

### 8.6 Performance Comparison

| Method | Speed (relative) | Works with any mask? | Notes |
|--------|------------------|---------------------|-------|
| Checkpoint (Python) | 1× | ✅ Yes | For debugging/development |
| TorchScript (fixed output) | ~3× faster | ✅ Yes | Recommended for production |
| Pure C++ libtorch | ~4× faster | ✅ Yes | For future optimization |

### 8.7 Future Research: Direct Masked Prediction

**Current architecture** predicts all positions then extracts masked ones:
```
Encoder → Predict ALL → Extract masked → Variable output
```

**Alternative approach** could use attention to directly predict only masked positions:
```
Encoder → Cross-attention (masked queries) → Fixed output
```

This would:
1. Use position embeddings for masked sensor locations as queries
2. Attend to encoder features to gather context
3. Predict only the required positions

**Potential advantages:**
- More efficient (fewer computations)
- Could improve quality by focusing on masked positions
- Natural fixed-size output (always predict N positions)

**Implementation considerations:**
- Backbone (XECEncoder) can be kept unchanged
- New head architecture required:
  ```python
  class DirectMaskedHead(nn.Module):
      def __init__(self, max_masked=200):
          # Position embeddings for masked locations
          self.pos_embed = nn.Embedding(4760, hidden_dim)
          # Cross-attention layers
          self.cross_attn = nn.TransformerDecoder(...)
          # Output projection
          self.output = nn.Linear(hidden_dim, 2)

      def forward(self, encoder_features, masked_indices):
          # Query positions from masked indices
          queries = self.pos_embed(masked_indices)
          # Cross-attend to encoder features
          features = self.cross_attn(queries, encoder_features)
          # Predict (npho, time)
          return self.output(features)
  ```

**Status:** Future research direction, not implemented.

### 8.8 Future Enhancement: Full Sensor Predictions

Since `forward_full_output()` returns predictions for ALL 4760 sensors, we could enrich the output:

**Current output:** Only masked sensor predictions
**Enhanced output:** All sensor predictions (masked + valid)

**Potential benefits:**
1. Compare predicted vs actual for valid sensors (reconstruction quality)
2. Identify sensors where model struggles
3. Detect data quality issues (large pred-truth mismatch on "valid" sensors)
4. Study model behavior across the full detector

**Note:** The unified `validate_inpainter.py` and `analyze_inpainter.py` macros now support both MC and real data validation with consistent output formats.

---

## 9. Real Data Validation

Validate the inpainter on real detector data that already contains dead channels.

### 9.1 Workflow

```
Checkpoint (.pth)
    │
    ▼
┌─────────────────────────────────────┐
│ Export (one-time)                   │
│ macro/export_onnx_inpainter.py      │
│ --format torchscript                │
└─────────────────────────────────────┘
    │
    ▼
TorchScript Model (.pt)
    │
    ▼
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
│ Inpainter Inference (TorchScript)   │
│ - Input: masked real data + mask    │
│ - Output: (B, 4760, 2) predictions  │
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

### 9.2 Preparing Real Data

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

### 9.3 Running Validation

Use the unified `validate_inpainter.py` macro with `--real-data` flag:

**Recommended workflow (TorchScript - faster):**

```bash
# Step 1: Export model to TorchScript (one-time)
python macro/export_onnx_inpainter.py \
    artifacts/inpainter/inpainter_checkpoint_best.pth \
    --format torchscript \
    --output artifacts/inpainter/inpainter.pt

# Step 2: Run validation with TorchScript model
python macro/validate_inpainter.py \
    --torchscript artifacts/inpainter/inpainter.pt \
    --input DataGammaAngle_430000-431000.root \
    --run 430000 \
    --real-data \
    --n-artificial 50 \
    --output validation_real/
```

**Alternative: Using checkpoint directly (slower, for debugging):**

```bash
python macro/validate_inpainter.py \
    --checkpoint artifacts/inpainter/inpainter_checkpoint_best.pth \
    --input DataGammaAngle_430000-431000.root \
    --run 430000 \
    --real-data \
    --output validation_real/
```

**Other options:**

```bash
# Using pre-saved dead channel list (instead of database)
python macro/validate_inpainter.py \
    --torchscript inpainter.pt \
    --input real_data.root \
    --dead-channel-file dead_channels_430000.txt \
    --real-data \
    --output validation_real/

# Customize artificial masking count
python macro/validate_inpainter.py \
    --torchscript inpainter.pt \
    --input real_data.root \
    --run 430000 \
    --real-data \
    --n-artificial 100 \
    --seed 42 \
    --output validation_real/
```

**Model options (mutually exclusive):**
| Option | Description |
|--------|-------------|
| `--torchscript PATH` | TorchScript model (.pt) - **recommended for speed** |
| `--onnx PATH` | ONNX model (.onnx) |
| `--checkpoint PATH` | Checkpoint file (.pth) - slower, for debugging |

**Other options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--n-mask-inner` | 10 | Healthy sensors to mask in inner face |
| `--n-mask-other` | 1 | Healthy sensors to mask in other faces |
| `--seed` | 42 | Random seed for reproducibility |
| `--batch-size` | 64 | Batch size for inference |
| `--max-events` | all | Limit number of events |

### 9.4 Output Format

The output ROOT file has additional columns compared to training predictions:

| Branch | Type | Description |
|--------|------|-------------|
| `mask_type` | int32 | 0=artificial (has truth), 1=dead (no truth) |
| `run` | int32 | Run number |
| `event` | int32 | Event number |

**Invalid values:** For dead channels (`mask_type=1`), truth and error are set to `-999`.

### 9.5 Event Display

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

### 9.6 Analyzing Results

```bash
# Analyze predictions (automatically filters by mask_type for metrics)
python macro/analyze_inpainter.py validation_real/real_data_predictions.root \
    --output validation_real/analysis/
```

The analysis macro automatically detects real data validation mode (when `mask_type` column exists):

**For artificial masks (mask_type=0):**
- Full quantitative metrics: MAE, RMSE, bias, resolution
- Per-face breakdown
- Residual distributions
- Resolution vs signal plots

**For dead channels (mask_type=1):**
- Plausibility checks (npho ≥ 0, reasonable time range)
- Distribution statistics
- Per-face breakdown
- `dead_channel_distributions.pdf` - prediction distributions

**Console output example:**
```
======================================================================
INPAINTER EVALUATION SUMMARY
(Real Data Validation Mode)
======================================================================

--- Global Metrics (Artificial Masks Only) ---
Metric              npho         time
----------------------------------------
MAE              0.023456     0.034567
...

--- Dead Channel Predictions (No Ground Truth) ---
Total dead channel predictions: 4,500
Events with dead channels: 100

Npho predictions:
  Mean: 0.1234, Std: 0.0567
  Range: [-0.0123, 0.4567]
  Negative fraction: 2.34%
...
```

**Output files (real data mode):**
- `global_metrics.csv` - Metrics for artificial masks
- `face_metrics.csv` - Per-face metrics for artificial masks
- `dead_channel_stats.csv` - Statistics for dead channel predictions
- `dead_channel_distributions.pdf` - Dead channel prediction plots

---

## 10. Database Utilities

Query dead channel information from the MEG2 database.

### 10.1 Database Hierarchy

```
RunCatalog (run id) → XECConf_id
    → XECConf (id) → XECPMStatusDB_id
        → XECPMStatusDB (id) → XECPMStatus_id
            → XECPMStatus (idx: 0-4759, IsBad: 0/1)
```

### 10.2 Command Line Usage

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

### 10.3 Python API

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

### 10.4 Requirements

```bash
# Install pymysql (already included in conda environment)
pip install pymysql

# Test connection
python -m lib.db_utils --check

# Query dead channels
python -m lib.db_utils 430000
```

The read-only database credentials are built into the module. No environment variables needed.

---

## 11. MC Pseudo-Experiment

Apply real data dead channel patterns to MC data for baseline comparison.

### 11.1 Purpose

Real data validation (Section 8) has a limitation: dead channels have no ground truth, so we can only evaluate on artificially masked healthy sensors. The MC pseudo-experiment addresses this by:

1. Taking MC data (where all sensors have ground truth)
2. Applying the same dead channel pattern from a real data run
3. Running inpainter inference on the "dead" channels
4. Comparing predictions with ground truth

This provides a performance baseline for dead channel recovery.

### 11.2 Usage

Use the unified `validate_inpainter.py` macro (without `--real-data` flag for MC mode):

```bash
# Basic usage (MC mode - no --real-data flag)
python macro/validate_inpainter.py \
    --checkpoint artifacts/inpainter/checkpoint_best.pth \
    --input mc_validation.root \
    --run 430000 \
    --output validation_mc/

# With TorchScript model (faster)
python macro/validate_inpainter.py \
    --torchscript artifacts/inpainter/inpainter.pt \
    --input mc_validation.root \
    --run 430000 \
    --output validation_mc/

# With dead channel file instead of database
python macro/validate_inpainter.py \
    --checkpoint checkpoint.pth \
    --input mc_validation.root \
    --dead-channel-file dead_channels_430000.txt \
    --output validation_mc/

# Compare multiple runs
for run in 430000 431000 432000; do
    python macro/validate_inpainter.py \
        --checkpoint checkpoint.pth \
        --input mc_validation.root \
        --run $run \
        --output validation_mc_run${run}/
done
```

**Note:** The legacy `pseudo_experiment_mc.py` is also available with similar interface.

### 11.3 Output

**ROOT file:** `pseudo_experiment_run{RUN}.root`
| Branch | Type | Description |
|--------|------|-------------|
| `event_idx` | int32 | Event index |
| `sensor_id` | int32 | Sensor ID (0-4759) |
| `face` | int32 | Face index (0=inner, 1=us, 2=ds, 3=outer, 4=top, 5=bot) |
| `truth_npho` | float32 | Ground truth npho (from MC) |
| `truth_time` | float32 | Ground truth time (from MC) |
| `pred_npho` | float32 | Predicted npho |
| `pred_time` | float32 | Predicted time |
| `error_npho` | float32 | Prediction error (pred - truth) |
| `error_time` | float32 | Prediction error (pred - truth) |
| `dead_pattern_run` | int32 | Run number of dead channel pattern |

**Metrics CSV:** `metrics_run{RUN}.csv` - Summary metrics

**Console output example:**
```
============================================================
PSEUDO-EXPERIMENT RESULTS
============================================================
Dead channel pattern from run: 430000
Number of dead channels: 45
Events processed: 10,000
Total predictions: 450,000
Predictions with ground truth: 448,500

Global Metrics (all dead channel predictions):
  npho: MAE=0.0234, RMSE=0.0312, Bias=-0.0012, 68%=0.0287
  time: MAE=0.0156, RMSE=0.0198, Bias=0.0003, 68%=0.0189

Per-Face Metrics:
   inner: n=200000, npho_MAE=0.0212, time_MAE=0.0145
      us: n= 30000, npho_MAE=0.0256, time_MAE=0.0167
      ds: n= 20000, npho_MAE=0.0267, time_MAE=0.0178
   outer: n= 80000, npho_MAE=0.0245, time_MAE=0.0156
     top: n= 70000, npho_MAE=0.0278, time_MAE=0.0189
     bot: n= 50000, npho_MAE=0.0289, time_MAE=0.0198
============================================================
```

### 11.4 Comparison Workflow

To compare inpainter performance on real data vs MC using the unified `validate_inpainter.py`:

```bash
# 1. Run real data validation (--real-data flag)
python macro/validate_inpainter.py \
    --checkpoint checkpoint.pth \
    --input real_data.root --run 430000 \
    --real-data --n-artificial 50 \
    --output validation_real/

# 2. Run MC pseudo-experiment with same dead pattern (no --real-data flag)
python macro/validate_inpainter.py \
    --checkpoint checkpoint.pth \
    --input mc_validation.root --run 430000 \
    --output validation_mc/

# 3. Analyze both
python macro/analyze_inpainter.py validation_real/predictions_real_run430000.root \
    --output validation_real/analysis/
python macro/analyze_inpainter.py validation_mc/predictions_mc_run430000.root \
    --output validation_mc/analysis/

# 4. Compare metrics (real vs MC)
# - Real data: metrics only for artificial masks (mask_type=0)
# - MC pseudo: metrics for all dead channels (full picture)
```

This comparison helps understand:
- Whether the model generalizes from MC training to real data
- Whether performance on artificially masked sensors (real data) is representative of dead channel recovery
- Face-specific performance differences between real and MC
