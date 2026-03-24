# Dead Channel Inpainting

The library includes a **dead channel inpainting** module for recovering sensor values at malfunctioning or dead channels. This is useful for:
- **Data recovery**: Interpolate missing sensor readings using surrounding context
- **Robustness training**: Train models to handle incomplete detector data
- **Preprocessing**: Clean up data before regression tasks

---

## Summary Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPAINTER TRAINING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │  MC ROOT Files  │
                              │  (clean data)   │
                              └────────┬────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
          ┌──────────────────┐                  ┌──────────────────┐
          │   MAE Training   │                  │ Direct Training  │
          │  (Optional but   │                  │  (From scratch)  │
          │   recommended)   │                  │                  │
          └────────┬─────────┘                  └────────┬─────────┘
                   │                                     │
                   │  Pretrained encoder                 │
                   ▼                                     │
          ┌──────────────────┐                           │
          │    Inpainter     │◄──────────────────────────┘
          │    Training      │
          │                  │
          │  • Random mask   │
          │  • Predict npho, │
          │    time at mask  │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │   Checkpoint     │
          │   (.pth file)    │
          └────────┬─────────┘
                   │
                   ▼
          ┌──────────────────┐
          │  TorchScript     │  (macro/export_onnx_inpainter.py)
          │  Export (.pt)    │
          └────────┬─────────┘
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
 ┌─────────────┐        ┌─────────────┐
 │ MC Pseudo-  │        │  Real Data  │
 │ Experiment  │        │ Validation  │
 └──────┬──────┘        └──────┬──────┘
        │                      │
        ▼                      ▼

┌─────────────────────────────────────────────────────────────────────────────┐
│                          VALIDATION WORKFLOWS                               │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────┐   ┌─────────────────────────────────┐
│      MC PSEUDO-EXPERIMENT       │   │       REAL DATA VALIDATION      │
│         (Section 11)            │   │          (Section 9)            │
├─────────────────────────────────┤   ├─────────────────────────────────┤
│                                 │   │                                 │
│  MC Data (all sensors valid)    │   │  Real Data (has dead channels)  │
│            │                    │   │            │                    │
│            ▼                    │   │            ▼                    │
│  ┌─────────────────────┐        │   │  ┌─────────────────────┐        │
│  │ Apply dead channel  │        │   │  │ Detect dead from DB │        │
│  │ pattern from real   │        │   │  │ + sentinel values   │        │
│  │ run (e.g., 430000)  │        │   │  └──────────┬──────────┘        │
│  └──────────┬──────────┘        │   │            │                    │
│             │                   │   │            ▼                    │
│             ▼                   │   │  ┌─────────────────────┐        │
│  ┌─────────────────────┐        │   │  │ Artificially mask   │        │
│  │ Mask = dead pattern │        │   │  │ some healthy sensors│        │
│  │ (full ground truth) │        │   │  │ (for evaluation)    │        │
│  └──────────┬──────────┘        │   │  └──────────┬──────────┘        │
│             │                   │   │            │                    │
│             ▼                   │   │            ▼                    │
│  ┌─────────────────────┐        │   │  ┌─────────────────────┐        │
│  │    Run Inpainter    │        │   │  │    Run Inpainter    │        │
│  └──────────┬──────────┘        │   │  └──────────┬──────────┘        │
│             │                   │   │            │                    │
│             ▼                   │   │            ▼                    │
│  ┌─────────────────────┐        │   │  ┌─────────────────────┐        │
│  │  Compare pred vs    │        │   │  │ Output with mask_type│       │
│  │  truth at ALL dead  │        │   │  │ 0=artificial (truth) │       │
│  │  channel positions  │        │   │  │ 1=dead (no truth)    │       │
│  └─────────────────────┘        │   │  └─────────────────────┘        │
│                                 │   │                                 │
│  OUTPUT:                        │   │  OUTPUT:                        │
│  • Full metrics for dead        │   │  • Metrics only for artificial  │
│    channel recovery             │   │  • Plausibility for dead        │
│  • Baseline performance         │   │  • Event displays               │
│                                 │   │                                 │
└─────────────────────────────────┘   └─────────────────────────────────┘

                    ┌─────────────────────────────────────┐
                    │         ANALYSIS PIPELINE           │
                    │    (macro/analyze_inpainter.py)     │
                    ├─────────────────────────────────────┤
                    │                                     │
                    │  Input: predictions ROOT file       │
                    │            │                        │
                    │            ▼                        │
                    │  ┌───────────────────┐              │
                    │  │ Compute metrics   │              │
                    │  │ • MAE, RMSE, bias │              │
                    │  │ • Per-face        │              │
                    │  │ • Percentiles     │              │
                    │  └─────────┬─────────┘              │
                    │            │                        │
                    │            ▼                        │
                    │  ┌───────────────────┐              │
                    │  │ Generate plots    │              │
                    │  │ • Residuals       │              │
                    │  │ • Scatter         │              │
                    │  │ • Resolution      │              │
                    │  └─────────┬─────────┘              │
                    │            │                        │
                    │            ▼                        │
                    │  OUTPUT:                            │
                    │  • global_metrics.csv               │
                    │  • face_metrics.csv                 │
                    │  • residual_distributions.pdf       │
                    │  • scatter_truth_vs_pred.pdf        │
                    │  • resolution_vs_signal.pdf         │
                    │                                     │
                    └─────────────────────────────────────┘


QUICK REFERENCE - Command Summary:
──────────────────────────────────────────────────────────────────────────────

# 1. Train MAE (optional, recommended)
python -m lib.train_mae --config config/mae/mae_config.yaml

# 2. Train Inpainter
python -m lib.train_inpainter --config config/inp/inpainter_config.yaml \
    --mae_checkpoint artifacts/mae/mae_checkpoint_best.pth

# 3. Export to TorchScript
python macro/export_onnx_inpainter.py checkpoint.pth --output inpainter.pt

# 4a. MC Validation (single checkpoint, ML inference only)
python macro/validate_inpainter.py \
    --checkpoint checkpoint.pth --input mc_data.root \
    --run 430000 --output validation_mc/

# 4b. Baselines (independent of any model, run once)
python macro/compute_inpainter_baselines.py \
    --input mc_data.root --run 430000 \
    --output baselines_mc_run430000.root

# 4c. Real Data Baselines (with on-the-fly solid angles)
python macro/compute_inpainter_baselines.py \
    --input real_data.root --dead-channel-file dead_channels.txt \
    --real-data --compute-solid-angles xyzRecoFI \
    --output baselines_real.root

# 4d. Real Data LocalFit (uses same artificial masking as baselines)
python macro/run_localfit_realdata.py \
    --input real_data.root --dead-channel-file dead_channels.txt \
    --output localfit_real.root

# 4e. Sensorfront Validation (single checkpoint)
python macro/validate_inpainter_sensorfront.py \
    --checkpoint checkpoint.pth --input mc_data.root \
    --solid-angle-branch solid_angle --output validation_sensorfront/

# 4f. Real Data Validation
python val_data/validate_inpainter_real.py \
    --torchscript inpainter.pt --input real_data.root \
    --run 430000 --output validation_data/

# 5. Batch Validation (scan steps, SLURM)
./jobs/run_validate_inpainter.sh 1 2 3 4 5 6            # MC validation

# 5b. Sensorfront: prepare shared data, then parallel localfit + inference
bash macro/submit_sensorfront_prepare_scan.sh            # Manifest + baselines (once)
bash macro/submit_localfit_sensorfront.sh \
    artifacts/sensorfront_shared/_sensorfront_manifest.npz  # LocalFit (parallel)
SHARED_DIR=artifacts/sensorfront_shared \
    bash macro/submit_validate_sensorfront_scan.sh       # ML inference (parallel)

# 6. Cross-Configuration Comparison
python macro/compare_inpainter.py --mode mc \
    --baselines baselines_mc_run430000.root               # MC comparison PDF
python macro/compare_inpainter.py --mode sensorfront     # Sensorfront comparison
python macro/compare_inpainter.py --mode data \
    --baselines baselines_real.root --localfit localfit.root  # Real data comparison

# 7. Single-Run Analysis
python macro/analyze_inpainter.py predictions.root --output analysis/
```

---

## 1. Architecture Overview

The inpainter (`XEC_Inpainter`) uses an encoder (optionally frozen from MAE pretraining) combined with lightweight inpainting heads:

```
Input (with dead channels marked as sentinel)
    ↓
┌─────────────────────────────────────────┐
│  XECEncoder (optionally frozen)         │
│  - Extracts latent tokens per face      │
│  - Global context from transformer      │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Inpainting Heads (configurable)        │
│                                         │
│  head_type="per_face" (default):        │
│    Rectangular (Inner, US, DS):         │
│    - FaceInpaintingHead                 │
│    - Local CNN (2× ConvNeXtV2 blocks)   │
│    - Global conditioning from latent    │
│    - Hidden dim: 64                     │
│                                         │
│    Outer (finegrid mode):               │
│    - OuterSensorInpaintingHead          │
│    - Per-sensor attention pooling       │
│    - Vectorized over all 234 sensors    │
│                                         │
│    Hexagonal (Top, Bottom):             │
│    - HexInpaintingHead                  │
│    - Local GNN (3× HexNeXt blocks)      │
│    - Global conditioning from latent    │
│    - Hidden dim: 96                     │
│                                         │
│  head_type="cross_attention":           │
│    - CrossAttentionInpaintingHead       │
│    - Unified head for all sensors       │
│    - Queries latent tokens via cross-   │
│      attention + KNN local context      │
│    - See cross-attention-inpainter.md   │
└─────────────────────────────────────────┘
    ↓
Output: Predicted (npho, time) at masked positions only
```

**Head variants (controlled by `model.head_type` and `model.use_masked_attention`):**

| Head Type | `head_type` | `use_masked_attention` | Description |
|-----------|-------------|----------------------|-------------|
| Per-face CNN/GNN | `per_face` | `false` | Default. Face-specific heads with local+global context |
| Masked attention | `per_face` | `true` | Attention-based heads, predict only at masked positions |
| Cross-attention | `cross_attention` | - | Unified head, queries all latents via cross-attention + KNN |

## 2. Masking Strategy

The inpainter uses **invalid-aware masking** to handle already-invalid sensors in the data:

- **Already-invalid sensors** (where `time == sentinel_time`) are excluded from the random masking pool
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
python -m lib.train_mae --config config/mae/mae_config.yaml

# Then, train inpainter with frozen MAE encoder
python -m lib.train_inpainter --config config/inp/inpainter_config.yaml \
    --mae_checkpoint artifacts/mae/mae_checkpoint_best.pth
```

**Option B: Without MAE Pre-training (From Scratch)**

```bash
# Train inpainter without MAE (encoder trained jointly)
python -m lib.train_inpainter --config config/inp/inpainter_config.yaml \
    --mae_checkpoint ""
```

**Interactive Training Script:**

```bash
# Edit configuration in the script first
./macro/interactive_inpainter_train_config.sh
```

### Multi-GPU Training

Inpainter training supports multi-GPU via DDP:

```bash
# Submit with 4 GPUs
NUM_GPUS=4 ./jobs/submit_inpainter.sh

# Direct multi-GPU training
torchrun --nproc_per_node=4 -m lib.train_inpainter --config config/inp/inpainter_config.yaml \
    --mae_checkpoint artifacts/mae/mae_checkpoint_best.pth

# Dry run to verify settings
NUM_GPUS=4 DRY_RUN=1 ./jobs/submit_inpainter.sh
```

ROOT file lists are sharded across GPUs. Only rank 0 logs to MLflow, saves checkpoints, and writes ROOT prediction files. See [Regressor Training](regressor.md#4-multi-gpu-training-ddp) for full DDP details.

## 4. Configuration

Configure in `config/inp/inpainter_config.yaml`:

```yaml
# Normalization (must match MAE pretraining)
normalization:
  npho_scheme: "sqrt"             # Normalization scheme (log1p, anscombe, sqrt, linear)
  npho_scale: 1000
  npho_scale2: 4.08               # Used by log1p only
  sentinel_npho: -1.0
  sentinel_time: -1.0

# Model
model:
  outer_mode: "finegrid"          # Must match MAE encoder config
  outer_fine_pool: null           # Must match MAE encoder config
  mask_ratio: 0.05                # Realistic dead channel density (1-10%)
  mask_npho_flat: false           # CDF-based flat masking (uniform across npho quantiles)
  freeze_encoder: false           # Freeze encoder, train only heads
  use_local_context: true         # Use local neighbor context for inpainting
  predict_channels: ["npho"]      # Output channels: ["npho"] or ["npho", "time"]
  use_masked_attention: false     # Use attention-based heads for masked positions
  head_type: "per_face"           # "per_face" or "cross_attention"
  # Cross-attention settings (only when head_type is "cross_attention"):
  sensor_positions_file: null     # Required for cross_attention (e.g. "lib/sensor_positions.txt")
  cross_attn_k: 16               # Number of KNN neighbors for local attention
  cross_attn_hidden: 64           # Hidden dimension for local attention
  cross_attn_latent_dim: 128      # Projection dimension for latent cross-attention
  cross_attn_pos_dim: 96          # Sinusoidal position encoding dimension

# Training
training:
  mae_checkpoint: null            # Path to pretrained MAE, or null
  epochs: 50
  lr: 1.0e-4
  lr_min: 1.0e-6                  # Minimum learning rate for cosine scheduler
  lr_scheduler: "cosine"
  warmup_epochs: 3                # Linear warmup epochs
  weight_decay: 1.0e-4
  loss_fn: "smooth_l1"            # smooth_l1, mse, l1, huber
  loss_beta: 0.1                  # Beta for smooth_l1/huber
  grad_clip: 1.0                  # Gradient clipping (0 to disable)
  amp: true                       # Automatic Mixed Precision
  compile: "reduce-overhead"      # torch.compile mode (max-autotune, reduce-overhead, default, none)
  grad_accum_steps: 4             # Gradient accumulation steps
  ema_decay: null                 # EMA decay rate (null to disable, 0.999 typical)
  npho_weight: 1.0
  time:                           # Time-specific options (ignored if 'time' not in predict_channels)
    weight: 1.0
    mask_ratio_scale: 1.0
    use_npho_weight: true
    npho_threshold: 100.0
  npho_loss_weight:               # Weight loss by sensor intensity
    enabled: false
    alpha: 0.5
  intensity_reweighting:          # Reweight samples by total event intensity
    enabled: false
    nbins: 5
    target: "uniform"
```

### Npho-Only Mode

Set `model.predict_channels: ["npho"]` to train a model that only predicts photon counts:

```yaml
model:
  predict_channels: ["npho"]  # npho-only mode
```

**Benefits:**
- Faster training (fewer output channels, simplified loss)
- Smaller model footprint
- Useful when timing information is not needed

**Behavior:**
- Input: Still uses both npho and time channels (encoder sees full context)
- Output: Only predicts npho (1 channel instead of 2)
- Time-related loss/metrics are skipped
- Output ROOT files only contain `pred_npho`, `truth_npho`, `error_npho`
- Analysis macros auto-detect mode from metadata tree in ROOT files

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
  - Branches: `event_idx`, `run_number`, `event_number`, `sensor_id`, `face`, `truth_npho`, `truth_time`, `pred_npho`, `pred_time`, `error_npho`, `error_time`

## 7. Analysis

Use `macro/analyze_inpainter.py` to evaluate predictions:

```bash
python macro/analyze_inpainter.py artifacts/inpainter/inpainter_predictions_epoch_50.root \
    --output analysis_output/
```

**Baseline comparison with `--baselines`:**

When predictions are generated with `macro/validate_inpainter.py --baselines`, the output ROOT file includes neighbor-average and solid-angle baseline predictions. These are automatically detected and compared by the analysis script.

**Local Fit baseline with `--local-fit`:**

Compare against the physics-based LocalFitBaseline (`others/LocalFitBaseline.C`), which predicts dead channel npho via position reconstruction and solid angle modeling (inner face only):

```bash
# Run LocalFitBaseline on the same MC file
root -l -b -q 'others/LocalFitBaseline.C("mc_data.root", "localfit.root")'

# Include in analysis
python macro/analyze_inpainter.py predictions.root \
    --local-fit localfit.root \
    --output analysis_output/
```

The LocalFitBaseline covers inner face sensors only (0-4091). Unmatched entries use NaN and are excluded from metrics. Events are matched by `(run_number, event_number, sensor_id)` tuple.

**Generated outputs:**
- `global_metrics.csv` - MAE, RMSE, bias, 68th/95th percentiles
- `face_metrics.csv` - Per-face breakdown
- `residual_distributions.pdf` - Histograms with Gaussian fit
- `residual_per_face_*.pdf` - Per-face residual distributions
- `scatter_truth_vs_pred.pdf` - 2D density plots
- `resolution_vs_signal.pdf` - Resolution/bias vs truth magnitude
- `metrics_summary.pdf` - Bar chart comparison across faces
- `baseline_comparison.pdf` - Overlay of ML vs baselines (when baselines available)

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
| `input` | `(B, 4760, 2)` | Sensor data `[npho, time]` with dead channels as sentinels (sentinel_npho=-1.0 for npho, sentinel_time=-1.0 for time) |
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

**Note:** PrepareRealData.C uses `1e10` as sentinel for invalid values. The validation script converts this to `-1.0` (model sentinel) automatically.

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
| `--checkpoint PATH` | Checkpoint file (.pth) - slower, for debugging |

**Other options:**
| Option | Default | Description |
|--------|---------|-------------|
| `--n-artificial` | 50 | Number of healthy sensors to mask artificially per event |
| `--seed` | 42 | Random seed for reproducibility |
| `--batch-size` | 64 | Batch size for inference |
| `--max-events` | all | Limit number of events |
| `--baselines` | off | Enable rule-based baseline computation (neighbor avg, solid angle) |
| `--baseline-k` | 1 | k-hop parameter for baseline neighbor search |

### 9.4 Output Format

The output ROOT file has additional columns compared to training predictions:

| Branch | Type | Description |
|--------|------|-------------|
| `mask_type` | int32 | 0=artificial (has truth), 1=dead (no truth) |
| `run` | int32 | Run number |
| `event` | int32 | Event number |

**Invalid values:** For dead channels (`mask_type=1`), truth and error are set to `-999`.

### 9.5 Event Display

Visualize individual events with `val_data/show_inpainter_real.py`:

```bash
# By event index
python val_data/show_inpainter_real.py 0 \
    --predictions validation_real/real_data_predictions.root \
    --original DataGammaAngle_430000-431000.root \
    --channel npho --save event_0.pdf

# By run/event number
python val_data/show_inpainter_real.py \
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

**ROOT file:** `predictions_{mc|real}_run{RUN}.root`
| Branch | Type | Description |
|--------|------|-------------|
| `event_idx` | int32 | Event index |
| `run_number` | int64 | Run number |
| `event_number` | int64 | Event number |
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

---

## 12. Batch Validation & Cross-Configuration Comparison

When scanning hyperparameters (e.g., mask ratio, normalization scheme, weighting), validate all scan steps in parallel on SLURM and compare results side-by-side.

### 12.1 Scan Step Layout

Each scan step stores its artifacts under `artifacts/inp_scan_<label>/`:

```
artifacts/
  sensorfront_shared/              # Shared data (created once by --manifest-only)
    _sensorfront_manifest.npz      #   Event matching info (for LocalFit)
    _sensorfront_data.npz          #   Raw matched data (for ML inference reload)
    _baselines_raw.npz             #   Raw baseline predictions (scheme-independent)
    localfit_results/              #   LocalFit batch job outputs
      localfit_file0000.npz
      ...
  inp_scan_s1_baseline/
    inpainter_checkpoint_best.pth
    validation_mc/                 # MC validation output
    validation_sensorfront/        # Sensorfront validation output (ML only)
    validation_data/               # Real data validation output
  inp_scan_s2_flatmask/
    ...
```

### 12.2 MC Validation (All Scan Steps)

ML inference and baseline computation are separate. Baselines are computed once
in raw photon space (no normalization needed), independent of any trained model.

**Baseline algorithms:**
- **Neighbor Average** -- averages valid neighbors within 20 cm on the same face
  (distance-based, matching `MEGTXECEnePMWeight::RecoverDeadChannelFromSurroundings`)
- **Solid-Angle Weighted** -- `sum(npho) * omega_target / sum(omega_neighbors)`,
  with fallback to simple average when `sum(npho) <= 50`

Both baselines use distance-based same-face neighbor lookup (20 cm threshold)
instead of the old k-hop grid neighbor approach.

1. **Baselines** (run once, independent of any model):
   ```bash
   python macro/compute_inpainter_baselines.py \
       --input data/E15to60_AngUni_PosSQ/val/ \
       --run 430000 \
       --output baselines_mc_run430000.root

   # Or with a dead channel list file (no database access needed)
   python macro/compute_inpainter_baselines.py \
       --input data/E15to60_AngUni_PosSQ/val/ \
       --dead-channel-file dead_channels_430000.txt \
       --output baselines_mc_run430000.root

   # With pre-computed solid angles from a ROOT branch
   python macro/compute_inpainter_baselines.py \
       --input data/E15to60_AngUni_PosSQ/val/ \
       --run 430000 \
       --solid-angle-branch solid_angle \
       --output baselines_mc_run430000.root

   # With on-the-fly solid angle computation (uses sensor_directions.txt)
   python macro/compute_inpainter_baselines.py \
       --input data/E15to60_AngUni_PosSQ/val/ \
       --run 430000 \
       --compute-solid-angles xyzRecoFI \
       --output baselines_mc_run430000.root

   # MC mode with dead channels only (no artificial masking)
   python macro/compute_inpainter_baselines.py \
       --input data/E15to60_AngUni_PosSQ/val/ \
       --run 430000 --n-artificial 0 \
       --output baselines_mc_run430000.root
   ```

2. **ML inference** (per scan step, via SLURM):
   ```bash
   # Submit all steps
   ./jobs/run_validate_inpainter.sh

   # Submit specific steps
   ./jobs/run_validate_inpainter.sh 3 5

   # Preview without submitting
   DRY_RUN=1 ./jobs/run_validate_inpainter.sh
   ```

**Environment variables for `run_validate_inpainter.sh`:**

| Variable | Default | Description |
|----------|---------|-------------|
| `PARTITION` | `a100-daily` | SLURM partition |
| `TIME` | `04:00:00` | Job time limit |
| `DEVICE` | `cuda` | `cpu` or `cuda` |
| `RUN_NUM` | `430000` | Dead channel pattern run number |
| `VAL_PATH` | (from config) | Override validation data path |
| `DRY_RUN` | `0` | Preview without submitting |

**MC workflow summary:**
```
1. compute_inpainter_baselines.py (once) 
2. run_validate_inpainter.sh (per step) 
3. compare_inpainter.py --mode mc --baselines
```

### 12.3 Sensorfront Validation (All Scan Steps)

Tests inpainter recovery when the peak-signal sensor is masked.

The sensorfront workflow separates model-independent work (event matching, baselines, LocalFit) from ML inference so they can run in parallel. Since geometric matching and baselines are checkpoint-independent, they run **once** in a shared directory and are reused by all scan steps.

**Parallel workflow (recommended):**

```bash
# Step 1: Prepare shared manifest + baselines (no model needed, fast)
bash macro/submit_sensorfront_prepare_scan.sh
# → creates artifacts/sensorfront_shared/ with manifest, raw data, baselines

# Step 2a: Submit LocalFit batch jobs (uses manifest, one task per ROOT file)
bash macro/submit_localfit_sensorfront.sh \
    artifacts/sensorfront_shared/_sensorfront_manifest.npz

# Step 2b: Submit ML inference per scan step (parallel with 2a)
SHARED_DIR=artifacts/sensorfront_shared \
    bash macro/submit_validate_sensorfront_scan.sh
# → each step loads prepared data, normalizes per its scheme, runs ML inference
# → baselines loaded from raw + renormalized (no recomputation)
# → LocalFit results loaded if available in SHARED_DIR/localfit_results/
```

**Standalone workflow (no preparation step):**

```bash
# Without SHARED_DIR, each step loads ROOT files and computes baselines itself
bash macro/submit_validate_sensorfront_scan.sh 1 2 3 4 5 6
```

**How raw baselines work:** The neighbor-avg and solid-angle baselines denormalize to raw npho space for averaging, making the raw predictions scheme-independent. The `--manifest-only` step saves these raw predictions; each inference job re-normalizes them to its model's scheme.

**Environment variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `SHARED_DIR` | (empty) | Path to prepared data; enables fast mode |
| `PARTITION` | `mu3e` | SLURM partition |
| `VAL_PATH` | `data/E15to60_AngUni_PosSQ/val2/` | Validation data (standalone mode only) |
| `DRY_RUN` | `0` | Preview without submitting |

**Sensorfront workflow summary:**
```
validate_inpainter_sensorfront.py --manifest-only --> run_localfit_sensorfront.py --> run_sensorfront_loginNode.sh --> compare_inpainter.py --mode sensorfront
```

### 12.4 Real Data Validation (All Scan Steps)

Uses `val_data/validate_inpainter_real.py` with artificially masked healthy sensors.
Output is in raw photons (no normalization) with `mask_type` column.

Baselines and LocalFit can be computed independently of any trained model.

**Important:** For real data, dead channels have no ground truth. Evaluation uses
**artificially masked healthy sensors** (mask_type=0). All baseline methods must
use the same artificial masking pattern (seed=42, stratified 15 per event) so
their predictions are comparable.

```bash
# 1. Baselines (neighbor avg + solid-angle weighted, run once)
python macro/compute_inpainter_baselines.py \
    --input val_data/data/DataGammaAngle_430026-430126.root \
    --dead-channel-file dead_channels_run430000.txt \
    --real-data \
    --compute-solid-angles xyzRecoFI \
    --output data/inp_baselines_realdata_dcp430000.root

# 2. LocalFit baseline (generates same artificial mask, runs ROOT macro)
python macro/run_localfit_realdata.py \
    --input val_data/data/DataGammaAngle_430026-430126.root \
    --dead-channel-file dead_channels_run430000.txt \
    --output localfit_realdata.root

# 3. ML inference per scan step
python val_data/validate_inpainter_real.py \
    --torchscript artifacts/inp_scan_s1_baseline/inpainter.pt \
    --input real_data.root --run 430000 \
    --output artifacts/inp_scan_s1_baseline/validation_data/
```

**How `run_localfit_realdata.py` works:** It generates the same artificial mask
pattern as `compute_inpainter_baselines.py` (same seed, same stratified masking),
writes a per-event dead channel file, and runs `LocalFitBaseline.C` with those
masks. This way LocalFit predicts for the same sensors that have ground truth,
making the comparison fair.

**Real data workflow summary:**
```
compute_inpainter_baselines.py --real-data  ──┐
run_localfit_realdata.py                    ──┤── compare_inpainter.py --mode data
val_data/validate_inpainter_real.py (×N)    ──┘     --baselines ... --localfit ...
```

### 12.5 Cross-Configuration Comparison

After validation jobs complete, generate comparison PDFs. Use `--baselines` to
load the standalone baseline file (from `compute_inpainter_baselines.py`).
Use `--localfit` to overlay the LocalFitBaseline result (shown with dashed lines).

```bash
# MC validation comparison with baselines
python macro/compare_inpainter.py --mode mc \
    --baselines baselines_mc_run430000.root

# Sensorfront comparison
python macro/compare_inpainter.py --mode sensorfront

# Real data comparison with baselines + LocalFit
python macro/compare_inpainter.py --mode data \
    --baselines data/inp_baselines_realdata_dcp430000.root \
    --localfit localfit_realdata.root

# Overlay standalone LocalFitBaseline result (MC)
python macro/compare_inpainter.py --mode mc \
    --baselines baselines_mc_run430000.root \
    --localfit path/to/localfit_merged.root

# Custom output path
python macro/compare_inpainter.py --mode mc -o my_comparison.pdf
```

**PDF contents (7 pages):**

| Page | Content |
|------|---------|
| 1 | Summary: global relative MAE bar chart (x-axis capped to prevent outlier baselines from squashing ML entries) + per-face grouped bars (auto-hidden when only 1 active face) |
| 2-4 | Per-face relative MAE / RMS / Bias vs truth npho (log-scale bins, data points connected across empty bins) |
| 5-7 | Per-face relative MAE / RMS / Bias vs pred npho (log-scale bins, data points connected across empty bins) |

**Baselines** (from `--baselines` file or first entry that has embedded baselines):
- Neighbor Avg -- distance-based same-face average (20 cm threshold)
- Solid-Angle Weighted -- solid-angle-based weighted average with low-npho fallback
- Local Fit -- physics-based position fit + solid-angle prediction (`LocalFitBaseline.C`), shown with dashed lines

**Plot details:**
- `--localfit` entries are drawn with dashed lines to distinguish baselines from ML models
- NaN sentinel (instead of -999) is used for baseline errors where truth is unavailable
- Per-face metric plots use y-axis range [0, 1.5]
- Per-face bar chart is auto-hidden when only 1 active face
- Global bar chart x-axis is capped to prevent outlier baselines from squashing the scale

**Stdout output** includes global metrics table, per-face MAE, and per-face relative MAE for all methods.

### 12.6 Typical End-to-End Workflow

Baselines and ML inference are independent. Steps 2a-2d can all run in parallel.

```
                    ┌──────────────────────────────┐
                    │ 1. Train scan steps (s1-s8)  │
                    └──────────────┬───────────────┘
                                   │
     ┌──────────────┬──────────────┼──────────────┬──────────────┐
     │              │              │              │              │
┌────▼─────┐ ┌───-──▼──────┐ ┌───-─▼─────┐ ┌──────▼───────┐ ┌────▼──────┐
│2a. MC    │ │2b. MC val   │ │2c. SF     │ │2d. Real data │ │2e. Real   │
│baselines │ │(ML per      │ │prepare +  │ │baselines     │ │data val   │
│(once)    │ │       step) │ │ inference │ │   + LocalFit │ │(ML per    │
└────┬─────┘ └──-───┬──────┘ └─-───┬─────┘ └──────┬───────┘ │ step)     │
     │              │              │              │         └────┬──────┘
     └──────────────┴──────────────┴──────────────┴──────────────┘
                                   │
                    ┌──────────────▼───────────────┐
                    │ 3. Generate comparison PDFs  │
                    └──────────────────────────────┘
```

```bash
# 1. Train scan steps (s1-s8 with different configs)
#    ... (see training section)

# 2a. MC Baselines (no model needed, run once)
python macro/compute_inpainter_baselines.py \
    --input data/E15to60_AngUni_PosSQ/val/ \
    --run 430000 -o baselines_mc_run430000.root

# 2b. MC validation -- ML inference only (runs in parallel with 2a, 2c-2e)
./jobs/run_validate_inpainter.sh               # default: all s1-s8

# 2c. Sensorfront: prepare shared data (manifest + baselines, no model needed)
bash macro/submit_sensorfront_prepare_scan.sh
#    creates artifacts/sensorfront_shared/

# After 2c completes, run LocalFit and ML inference in parallel:
#    4a. LocalFit batch jobs (one array job, shared across all steps)
bash macro/submit_localfit_sensorfront.sh \
    artifacts/sensorfront_shared/_sensorfront_manifest.npz
#    4b. ML inference per scan step (loads prepared data + baselines, fast)
SHARED_DIR=artifacts/sensorfront_shared \
    bash macro/submit_validate_sensorfront_scan.sh

# 2d. Real data baselines + LocalFit (no model needed, run once)
python macro/compute_inpainter_baselines.py \
    --input real_data.root --run 430000 --real-data \
    --compute-solid-angles xyzRecoFI \
    -o baselines_real_run430000.root
root -l -b -q 'others/LocalFitBaseline.C("real_data.root", "localfit_real.root")'

# 2e. Real data ML inference per scan step
python val_data/validate_inpainter_real.py \
    --torchscript artifacts/inp_scan_s1_baseline/inpainter.pt \
    --input real_data.root --run 430000 \
    --output artifacts/inp_scan_s1_baseline/validation_data/

# 3. After ALL jobs complete, generate comparison PDFs
python macro/compare_inpainter.py --mode mc \
    --baselines baselines_mc_run430000.root -o compare_mc.pdf
python macro/compare_inpainter.py --mode sensorfront -o compare_sf.pdf
python macro/compare_inpainter.py --mode data \
    --baselines baselines_real_run430000.root \
    --localfit localfit_real.root -o compare_data.pdf
```

**Checking completion:**
```bash
# Check which MC validations finished
ls artifacts/inp_scan_*/validation_mc/predictions_mc_run430000.root

# Check which sensorfront validations finished
ls artifacts/inp_scan_*/validation_sensorfront/predictions_sensorfront.root

# Check which real data validations finished
ls artifacts/inp_scan_*/validation_data/

# Check SLURM job status
squeue -u $USER --format="%.10i %.10P %.25j %.8T %.10M"
```

---

## Real Data Fine-Tuning

After training and validating the inpainter on MC data, the model can be fine-tuned on real detector data. This section describes the end-to-end procedure.

### Overview

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  MEG2 Database   │────>│  Run List        │────>│  ROOT Processing │
│  (RunCatalog)    │     │  (train/val)     │     │  (meganalyzer)   │
└──────────────────┘     └──────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Fine-tuned      │<────│  Inpainter       │<────│  DataGamma       │
│  Checkpoint      │     │  Training        │     │  ROOT files      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Step 1: Generate Run Lists

Query the MEG2 database for physics runs and produce train/val run lists.

```bash
# On the cluster (needs access to meg.sql.psi.ch)
python macro/generate_realdata_runlist.py --output-dir data/real_data

# Options:
#   --min-run / --max-run   Restrict run range
#   --no-file-check         Skip checking if rec files exist on disk
#   --skip-train N          Skip first N found files per day for training (default: 1)
#   --dry-run               Print without writing files
#   --check                 Test database connection and exit
```

**Selection criteria:**
- `Physics=1`, `Junk=0`, `StartTime >= 2021-01-01`
- Groups runs by date, finds rec files with suffix priority: `_open.root`, `_selected.root`, `.root`

**Train/val split:**
- **Val:** 1st found rec file on every 3rd day (`day_index % 3 == 0`)
- **Train:** 2nd through 201st found rec files on every day (skips the 1st to avoid overlap with val)

**Output:** `data/real_data/runlist_train.txt` and `data/real_data/runlist_val.txt`

Format (two columns, `#` comments allowed):
```
# 2023-11-15
430123 /data/project/meg/offline/run/430xxx/rec430123_open.root
430124 /data/project/meg/offline/run/430xxx/rec430124_open.root
```

### Step 2: Pre-compile the ROOT Macro

Before submitting batch jobs, compile the macro once to avoid race conditions
(multiple jobs trying to compile simultaneously). ACLiC output is redirected
to `~/.cache/xec-ml-wl/aclic/` to keep the `macro/` directory clean.

```bash
cd ~/meghome/offline/analyzer

# Create a one-shot compile loader
echo 'void compile_prep() { gSystem->SetBuildDir("'$HOME'/.cache/xec-ml-wl/aclic"); gROOT->ProcessLine(".L '$HOME'/meghome/xec-ml-wl/macro/PrepareRealDataInpainter.C+"); }' > $HOME/.cache/xec-ml-wl/compile_prep.C

# Run it
./meganalyzer -b -q -I "$HOME/.cache/xec-ml-wl/compile_prep.C()"
```

If you see compilation artifacts in `macro/` (e.g. `*_C.so`, `*_ACLiC_dict*`), clean them up:
```bash
rm -f macro/PrepareRealDataInpainter_C*
```

### Step 3: Process Rec Files with Meganalyzer

Each day's runs (up to 20) are chained together and processed through
`PrepareRealDataInpainter.C`, which:
- Reads rec trees with minimal selection (trigger mask only, no physics selection or pileup cut)
- Extracts `npho`, `time` arrays from `xeccl` branch (4760 channels)
- Computes `relative_npho` (normalized by max) and `relative_time` (shifted by min)
- Outputs one ROOT file per day: `DataGamma_YYYY-MM-DD.root`

```bash
# Submit train processing (one SLURM task per day)
bash macro/submit_prepare_realdata.sh \
    data/real_data/runlist_train.txt \
    data/real_data/runlist_train_days.txt \
    data/real_data/raw

# Submit val processing
bash macro/submit_prepare_realdata.sh \
    data/real_data/runlist_val.txt \
    data/real_data/runlist_val_days.txt \
    data/real_data/val_raw

# Environment variables:
#   START_FROM=0          Skip first N days (for resuming)
#   PARTITION=meg-short   SLURM partition
```

**Notes:**
- Each SLURM task processes one day's worth of runs (~20 files chained via TChain)
- With ~467 days, this fits in a single SLURM array (max index 1999)
- Time limit is 1 hour per task on `meg-short`

### Step 4: Organize into Train/Val Directories

Move the processed files into `train/` and `val/` directories:

```bash
mkdir -p data/real_data/train data/real_data/val
mv data/real_data/raw/DataGamma_*.root data/real_data/train/
mv data/real_data/val_raw/DataGamma_*.root data/real_data/val/
```

### Step 5: Fine-Tune the Inpainter

Fine-tune from the best MC-trained checkpoint (e.g., scan step 3) using the real data config:

```bash
python -m lib.train_inpainter --config config/inp/finetune_realdata.yaml
```

**Key config settings** (`config/inp/finetune_realdata.yaml`):
- `resume_from`: best MC checkpoint (e.g., `artifacts/inp_scan_s3/best_model.pt`)
- `reset_epoch: true` — start epoch counter from 0
- `refresh_lr: true` — use fresh LR schedule from config
- `new_mlflow_run: true` — create separate MLflow run
- `lr: 2e-5` — 5x lower than MC training for fine-tuning
- `epochs: 30`

**Output branches in processed ROOT files:**

| Branch | Type | Description |
|--------|------|-------------|
| `run` | Int | Run number |
| `event` | Int | Event number |
| `npho` | Float[4760] | Raw photon counts |
| `time` | Float[4760] | Raw timing |
| `relative_npho` | Float[4760] | npho / max(npho) |
| `relative_time` | Float[4760] | time - min(time) |
| `ch_npho_max` | Short | Channel with max npho |
| `ch_time_min` | Short | Channel with min time |
| `npho_max_used` | Float | Max npho value used for normalization |
| `time_min_used` | Float | Min time value used for normalization |

### Step 6: Validate on Real Data

After fine-tuning, validate using artificial masking on held-out real data:

```bash
bash macro/submit_validate_data_scan.sh
```

This runs `macro/validate_inpainter.py` with `--real-data` mode, applying artificial masks to real events and measuring reconstruction quality. See [Validation on Real Data](#validation-on-real-data-with-artificial-masking) for details.

### Troubleshooting

- **0 events selected from `_selected.root` files**: These files have pre-filtered events with empty branches. The `PrepareRealDataInpainter.C` macro uses trigger-only selection to handle both `_open.root` and `_selected.root` files.
- **SLURM array index > 1999**: Increase `RUNS_PER_JOB` or use `START_FROM` to process in batches.
- **meganalyzer function name mismatch**: The script uses a loader macro pattern with `gROOT->ProcessLine` because meganalyzer's `-I` flag calls the function matching the filename, not arbitrary functions.
- **Database connection issues**: Run `python macro/generate_realdata_runlist.py --check` to test connectivity to `meg.sql.psi.ch`.
