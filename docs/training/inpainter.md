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

### 8.1 Why TorchScript (ONNX Not Supported)

**ONNX export fails** for the inpainter model with error:
```
UnsupportedOperatorError: Exporting the operator 'aten::_transformer_encoder_layer_fwd'
to ONNX opset version 17 is not supported.
```

#### Why This Happens

PyTorch's `nn.TransformerEncoder` has **two internal implementations**:

| Path | Implementation | ONNX Compatible |
|------|----------------|-----------------|
| **Slow path** | Standard ops (matmul, softmax, layer norm separately) | ✅ Yes |
| **Fast path** | Fused CUDA kernel (`_transformer_encoder_layer_fwd`) | ❌ No |

PyTorch **automatically selects the fast path** when ALL of these conditions are met:
- `batch_first=True` (our model uses this)
- Training mode is `False` (eval mode)
- Inputs are on CUDA or meet CPU requirements
- No nested tensors
- Certain PyTorch versions (1.11+)

Our XECEncoder uses `batch_first=True` for the TransformerEncoder, so PyTorch uses the optimized fused kernel which cannot be represented in ONNX.

#### Why Other Transformer Models Work with ONNX

Your colleague's model likely works because of one of these reasons:

1. **Different configuration**: Using `batch_first=False` (the older default) triggers the slow path
2. **Custom implementation**: Hand-written transformer using basic ops (Linear, Softmax, etc.)
3. **Hugging Face transformers**: These use their own implementation, not `nn.TransformerEncoder`
4. **Older PyTorch version**: Before the fast path was added
5. **Explicit slow path**: Some code forces `torch.backends.cuda.enable_flash_sdp(False)` etc.

#### Potential Workarounds (Not Recommended)

To force ONNX export, you could:
```python
# Before export, disable fast path (may hurt performance)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

However, this:
- May not work in all PyTorch versions
- Defeats the purpose of using optimized transformers
- TorchScript is simpler and just works

#### TorchScript vs ONNX: Detailed Comparison

| Aspect | TorchScript | ONNX |
|--------|-------------|------|
| **Format** | PyTorch-native IR (Intermediate Representation) | Cross-platform standard IR |
| **Operator coverage** | 100% of PyTorch ops | ~80% (missing some native/fused ops) |
| **Runtime** | libtorch (C++), PyTorch (Python) | ONNX Runtime, TensorRT, CoreML, etc. |
| **File extension** | `.pt` or `.pth` | `.onnx` |
| **Dynamic shapes** | Full support | Limited (requires explicit axes) |
| **Control flow** | Supported via scripting | Limited (unrolled during export) |
| **Numerical precision** | Exact match with PyTorch | May differ slightly |
| **Model size** | Larger (includes full graph) | Smaller (optimized graph) |
| **Cross-platform** | PyTorch ecosystem only | Any ONNX-compatible runtime |
| **GPU support** | CUDA via libtorch | Depends on runtime (TensorRT, etc.) |

#### When to Use Each

**Use TorchScript when:**
- Model uses PyTorch-specific ops (fused kernels, custom CUDA)
- You need exact numerical reproducibility
- Deploying with libtorch (C++) or staying in PyTorch ecosystem
- Model has complex control flow or dynamic shapes

**Use ONNX when:**
- Need cross-platform deployment (mobile, edge, different frameworks)
- Model uses only standard ops (conv, matmul, relu, etc.)
- Want to use TensorRT, CoreML, or other optimized runtimes
- Model architecture is simple and static

#### For the Inpainter

**TorchScript is the only option** because:
1. Uses `nn.TransformerEncoder` with fused kernels
2. Has dynamic output shapes (variable masked positions)
3. Contains conditional logic based on face types

TorchScript works well and provides:
- Exact numerical match with PyTorch
- ~2-3x speedup over eager mode
- Easy deployment with libtorch for C++ inference

### 8.2 Export Commands

```bash
# Export to TorchScript (default)
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

### 8.4 TorchScript Tracing Limitation

**TorchScript tracing does NOT work** for the inpainter due to dynamic tensor sizes:

```python
# In FaceInpaintingHead.forward():
within_batch_idx = torch.arange(len(batch_idx), device=device) - cumsum[batch_idx]
```

During tracing:
- `len(batch_idx)` becomes a **constant** (e.g., 8299)
- But `batch_idx` varies at runtime (e.g., size 4096)
- Size mismatch → RuntimeError

This is fundamental to how tracing works - it captures one execution path with fixed sizes.

### 8.5 Recommended: Use Checkpoint Mode

For Python inference (validation, analysis), use the checkpoint directly:

```bash
python macro/validate_inpainter_real.py \
    --checkpoint artifacts/inpainter/inpainter_checkpoint_best.pth \
    --input real_data.root \
    --run 430000 \
    --output validation_output/
```

This is ~3x slower than optimized inference but **works correctly** with any mask pattern.

### 8.6 Future: Fast C++ Inference Options

If fast C++ inference is needed, here are the options:

#### Option A: Fixed-Size Output Model (Recommended)

Modify the inpainter to output predictions for ALL sensors, not just masked ones:

**Current (variable size - not exportable):**
```python
# Heads extract only masked positions → variable tensor sizes
pred_masked, indices, valid = head(face_tensor, latent, mask)
```

**Fixed-size (exportable):**
```python
# Heads predict ALL positions → fixed tensor size (B, 4760, 2)
pred_all = head_full(face_tensor, latent)
# Mask is applied AFTER inference in C++ code
```

Changes required:
1. Modify `FaceInpaintingHead`, `HexInpaintingHead`, `OuterSensorInpaintingHead` to return full predictions
2. Create new `XEC_Inpainter_FixedOutput` class that returns (B, 4760, 2)
3. Re-train or adapt weights (heads architecture changes slightly)

**Pros:** Clean TorchScript export, fast inference
**Cons:** Requires model modification and possibly retraining

#### Option B: Pure C++ Implementation with libtorch

Implement the entire inpainter in C++ using libtorch:

1. Load checkpoint weights using `torch::load()`
2. Reconstruct model architecture in C++ (Conv2d, TransformerEncoder, etc.)
3. Implement dynamic masking logic natively in C++

```cpp
// Pseudocode
auto checkpoint = torch::load("inpainter_checkpoint_best.pth");
auto encoder = build_xec_encoder(checkpoint);
auto heads = build_inpainting_heads(checkpoint);

// Dynamic masking works naturally in C++
auto latent = encoder->forward(input);
auto predictions = heads->forward(latent, mask);  // C++ handles variable sizes
```

**Pros:** Full control, maximum performance, handles dynamic sizes
**Cons:** Significant development effort, must maintain C++ and Python versions

#### Option C: Hybrid - Export Encoder, C++ Heads

Split the model:
1. Export XECEncoder's face processing (ConvNeXt, HexNeXt blocks) to TorchScript
2. Implement simple inpainting heads in C++ (just Conv2d + Linear layers)

**Problem:** The TransformerEncoder fusion layer still can't export to ONNX, limiting this approach.

### 8.7 Performance Comparison

| Method | Speed (CPU) | Works with any mask? | Effort |
|--------|-------------|---------------------|--------|
| Checkpoint (Python) | ~0.9 sec/event | ✅ Yes | None |
| TorchScript | ~0.3 sec/event | ❌ No (tracing limitation) | - |
| Fixed-size model | ~0.3 sec/event | ✅ Yes | Medium (modify heads) |
| Pure C++ libtorch | ~0.2 sec/event | ✅ Yes | High (full reimplementation) |

**Current recommendation:** Use checkpoint mode for validation. If C++ inference becomes critical, pursue Option A (fixed-size output).

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

**Recommended workflow (TorchScript - faster):**

```bash
# Step 1: Export model to TorchScript (one-time)
python macro/export_onnx_inpainter.py \
    artifacts/inpainter/inpainter_checkpoint_best.pth \
    --format torchscript \
    --output artifacts/inpainter/inpainter.pt

# Step 2: Run validation with TorchScript model
python macro/validate_inpainter_real.py \
    --torchscript artifacts/inpainter/inpainter.pt \
    --input DataGammaAngle_430000-431000.root \
    --run 430000 \
    --output validation_real/
```

**Alternative: Using checkpoint directly (slower, for debugging):**

```bash
python macro/validate_inpainter_real.py \
    --checkpoint artifacts/inpainter/inpainter_checkpoint_best.pth \
    --input DataGammaAngle_430000-431000.root \
    --run 430000 \
    --output validation_real/
```

**Other options:**

```bash
# Using ONNX model
python macro/validate_inpainter_real.py \
    --onnx artifacts/inpainter/inpainter.onnx \
    --input real_data.root \
    --run 430000 \
    --output validation_real/

# Using pre-saved dead channel list (instead of database)
python macro/validate_inpainter_real.py \
    --torchscript inpainter.pt \
    --input real_data.root \
    --dead-channel-file dead_channels_430000.txt \
    --output validation_real/

# Customize artificial masking
python macro/validate_inpainter_real.py \
    --torchscript inpainter.pt \
    --input real_data.root \
    --run 430000 \
    --n-mask-inner 10 \
    --n-mask-other 1 \
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

```bash
# Basic usage
python macro/pseudo_experiment_mc.py \
    --checkpoint artifacts/inpainter/checkpoint_best.pth \
    --input mc_validation.root \
    --run 430000 \
    --output pseudo_experiment/

# With dead channel file instead of database
python macro/pseudo_experiment_mc.py \
    --checkpoint checkpoint.pth \
    --input mc_validation.root \
    --dead-channel-file dead_channels_430000.txt \
    --output pseudo_experiment/

# Compare multiple runs
for run in 430000 431000 432000; do
    python macro/pseudo_experiment_mc.py \
        --checkpoint checkpoint.pth \
        --input mc_validation.root \
        --run $run \
        --output pseudo_experiment_run${run}/
done
```

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

To compare inpainter performance on real data vs MC:

```bash
# 1. Run real data validation
python macro/validate_inpainter_real.py \
    --checkpoint checkpoint.pth \
    --input real_data.root --run 430000 \
    --output validation_real/

# 2. Run MC pseudo-experiment with same dead pattern
python macro/pseudo_experiment_mc.py \
    --checkpoint checkpoint.pth \
    --input mc_validation.root --run 430000 \
    --output validation_mc/

# 3. Analyze both
python macro/analyze_inpainter.py validation_real/real_data_predictions.root \
    --output validation_real/analysis/
python macro/analyze_inpainter.py validation_mc/pseudo_experiment_run430000.root \
    --output validation_mc/analysis/

# 4. Compare metrics (real vs MC)
# - Real data: metrics only for artificial masks (Section 8)
# - MC pseudo: metrics for all dead channels (full picture)
```

This comparison helps understand:
- Whether the model generalizes from MC training to real data
- Whether performance on artificially masked sensors (real data) is representative of dead channel recovery
- Face-specific performance differences between real and MC
