# Masked Autoencoder (MAE) Pre-training

The library supports self-supervised pre-training using a Masked Autoencoder (MAE) approach. This allows the model to learn geometric features from the raw detector data without requiring ground-truth labels.

## 1. Architecture Overview

The MAE (`XEC_MAE`) consists of an encoder and face-specific decoders:

```
Input: (B, 4760, 2) sensor values (npho, time)
    ↓
┌─────────────────────────────────────────┐
│  Masking (Invalid-Aware)                          │
│  - Exclude already-invalid sensors      │
│  - Randomly mask `mask_ratio` of valid  │
│  - Set masked positions to sentinel     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  XECEncoder (shared with regression)    │
│  - Per-face ConvNeXt/HexNeXt backbones  │
│  - Transformer fusion                   │
│  - Output: 6 latent tokens (1024-dim)   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Face-Specific Decoders                 │
│                                         │
│  FaceDecoder (Inner, US, DS, Outer):    │
│  - Linear: 1024 → 256×4×4               │
│  - ConvTranspose2d: 256→128 (4→8)       │
│  - ConvTranspose2d: 128→64  (8→16)      │
│  - ConvTranspose2d: 64→2    (16→16)     │
│  - Bilinear interpolate to face size    │
│                                         │
│  GraphFaceDecoder (Top, Bottom):        │
│  - Project latent → all nodes           │
│  - Add learnable positional embedding   │
│  - 2× HexNeXtBlock (graph attention)    │
│  - LayerNorm → Linear → 2 channels      │
└─────────────────────────────────────────┘
    ↓
Output: Reconstructed (npho, time) per face
Loss: Computed only on masked positions
```

**Decoder Output Dimensions:**

| Face | Decoder | Output Shape |
|------|---------|--------------|
| Inner | FaceDecoder | (B, 2, 93, 44) |
| US | FaceDecoder | (B, 2, 24, 6) |
| DS | FaceDecoder | (B, 2, 24, 6) |
| Outer (coarse) | FaceDecoder | (B, 2, 9, 24) |
| Outer (finegrid, pooled 3×3) | FaceDecoder | (B, 2, 15, 24) |
| Top | GraphFaceDecoder | (B, 2, 73) |
| Bottom | GraphFaceDecoder | (B, 2, 73) |

## 2. Masking Strategy

The MAE uses **invalid-aware masking** to properly handle already-invalid sensors in MC data:

```python
# Pseudocode for invalid-aware masking
already_invalid = (time == sentinel_time)  # Sensors without valid data
valid_sensors = ~already_invalid

# Only mask from valid sensors
num_to_mask = int(valid_sensors.sum() * mask_ratio)
mask = random_select(valid_sensors, num_to_mask)

# Apply sentinel to masked positions (per-channel)
x_masked[:, :, 0][mask] = sentinel_npho  # npho channel
x_masked[:, :, 1][mask] = sentinel_time   # time channel

# Loss computed only on `mask` (not on already_invalid)
```

**Key Properties:**
- **Already-invalid sensors** (where `time == sentinel`) are excluded from random masking
- **mask_ratio** applies to valid sensors only (e.g., 60% of ~4500 valid → ~2700 masked)
- **Loss** is computed only on randomly-masked positions (ground truth available)
- **actual_mask_ratio** metric tracks effective masking: `randomly_masked / valid_sensors`

## 3. Loss Computation

Loss is computed **only on masked positions** where ground truth exists:

$$\mathcal{L} = \sum_{\text{face}} \left( w_{\text{npho}} \cdot \mathcal{L}_{\text{npho}}^{\text{face}} + w_{\text{time}} \cdot \mathcal{L}_{\text{time}}^{\text{face}} \right)$$

Where for each face:
$$\mathcal{L}_{\text{channel}}^{\text{face}} = \frac{1}{|\text{mask}|} \sum_{i \in \text{mask}} \ell(y_i^{\text{pred}}, y_i^{\text{true}})$$

Supported loss functions: `mse`, `l1`, `smooth_l1`

**Optional: Homoscedastic Channel Weighting**

When `auto_channel_weight=True`, the model learns per-channel uncertainty:
$$\mathcal{L} = \frac{1}{2\sigma_{\text{npho}}^2} \mathcal{L}_{\text{npho}} + \frac{1}{2\sigma_{\text{time}}^2} \mathcal{L}_{\text{time}} + \log\sigma_{\text{npho}} + \log\sigma_{\text{time}}$$

## 4. Quick Start

```bash
# CLI mode (legacy)
python -m lib.train_mae --train_root /path/to/data.root --save_path mae_pretrained.pth --epochs 20 --batch_size 1024

# Finegrid outer face (optional)
python -m lib.train_mae --train_root /path/to/data.root --save_path mae_pretrained.pth --epochs 20 --batch_size 1024 \
  --outer_mode finegrid --outer_fine_pool 3 3

# Config mode (recommended)
python -m lib.train_mae --config config/mae/mae_config.yaml

# Config + CLI override
python -m lib.train_mae --config config/mae/mae_config.yaml --epochs 50 --train_root /path/to/train
```

## 5. Running Pre-training

Dedicated scripts are under `jobs/` to streamline the MAE workflow.

1. Configure the Run: Edit `jobs/run_mae.sh` to set desired parameters:

    - `ROOT_PATH`: Path to the dataset (wildcards supported)
    - `EPOCHS`: Number of pre-training epochs
    - `MASK_RATIO`: Percentage of valid sensors to mask (default `0.6`)
    - `BATCH`: Batch size

2. Submit the Job:

    ```bash
    cd jobs
    ./run_mae.sh
    ```
    This submits a SLURM job using `submit_mae.sh`

3. Output: Checkpoints will be saved to `~/meghome/xec-ml-wl/artifacts/<RUN_NAME>/`. The weights file is typically named `mae_checkpoint_best.pth`

### Multi-GPU Training

MAE pre-training supports multi-GPU via DDP for faster training:

```bash
# Submit with 4 GPUs
NUM_GPUS=4 ./jobs/submit_mae.sh

# Direct multi-GPU training
torchrun --nproc_per_node=4 -m lib.train_mae --config config/mae/mae_config.yaml

# Dry run to verify settings
NUM_GPUS=4 DRY_RUN=1 ./jobs/submit_mae.sh
```

ROOT file lists are sharded across GPUs (round-robin). Only rank 0 logs to MLflow and saves checkpoints. See [Regressor Training](regressor.md#4-multi-gpu-training-ddp) for full DDP details.

## 6. Metrics

| Metric | Description |
|--------|-------------|
| `total_loss` | Combined weighted loss (sum across faces) |
| `loss_npho` / `loss_time` | Per-channel losses (sum across faces) |
| `loss_{face}` | Per-face total loss |
| `loss_{face}_npho/time` | Per-face, per-channel losses |
| `mae_npho` / `mae_time` | Mean Absolute Error on masked positions |
| `rmse_npho` / `rmse_time` | Root Mean Square Error on masked positions |
| `mae_{face}_npho/time` | Per-face MAE |
| `rmse_{face}_npho/time` | Per-face RMSE |
| `actual_mask_ratio` | Effective mask ratio after excluding invalid sensors |

## 7. Fine-Tuning for Regression

Once pre-training is complete, load the learned encoder weights into the regression model:

1. **Configure Regression**: Edit `jobs/run_scan.sh`.
2. **Set Resume Path**: Point the `RESUME_FROM` variable to your MAE checkpoint:
    ```bash
    RESUME_FROM="$HOME/meghome/xec-ml-wl/artifacts/<RUN_NAME>/mae_checkpoint_best.pth"
    ```
3. **Run Regression**: Submit the training job as usual:
    ```bash
    ./run_scan.sh
    ```

**Note on Weight Loading**: The training script (`lib/train_regressor.py`) automatically detects the type of checkpoint provided:

- **Full checkpoint**: If resuming a regression run, it loads the optimizer state, epoch, and full model to continue exactly where it left off.
- **MAE Weights**: If loading an MAE file, it detects "raw weights", loads only the encoder (skipping the regression head), initializes the EMA model correctly, and resets the epoch counter to 1 for fresh fine-tuning

## 8. Configuration Reference

Key parameters in `config/mae/mae_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.predict_channels` | ["npho", "time"] | Which channels to predict (output). Use ["npho"] for npho-only mode |
| `model.mask_ratio` | 0.6 | Fraction of valid sensors to mask |
| `model.decoder_dim` | 128 | Lightweight decoder dimension (smaller forces encoder to learn more) |
| `model.outer_mode` | "finegrid" | Outer face mode (`finegrid` or `split`) |
| `model.outer_fine_pool` | null | Pooling kernel for finegrid outer (e.g., [3, 3]) |
| `model.encoder_dim` | 1024 | Encoder token dimension (must be divisible by 32) |
| `training.loss_fn` | "smooth_l1" | Loss function (smooth_l1, mse, l1, huber) |
| `training.loss_beta` | 1.0 | Beta for smooth_l1/huber (smaller = more sensitive) |
| `training.npho_weight` | 1.0 | Weight for npho channel loss |
| `training.time.weight` | 1.0 | Weight for time channel loss |
| `training.time.mask_ratio_scale` | 1.0 | Bias masking toward valid-time sensors (>1.0 prefers valid-time) |
| `training.time.use_npho_weight` | true | Weight time loss by sqrt(npho) |
| `training.time.npho_threshold` | 100 | Min npho for time loss (skip low-signal sensors) |
| `training.auto_channel_weight` | false | Learn channel weights automatically |
| `training.track_mae_rmse` | false | Compute MAE/RMSE metrics (slower) |
| `training.track_metrics` | false | Per-face training metrics |
| `training.grad_accum_steps` | 1 | Gradient accumulation steps |
| `training.ema_decay` | null | EMA decay rate (null=disabled, 0.999 typical) |
| `training.npho_loss_weight.enabled` | false | Weight loss by sensor intensity |
| `training.npho_loss_weight.alpha` | 0.5 | Exponent for intensity weighting |
| `training.intensity_reweighting.enabled` | false | Reweight samples by total event intensity |

### Configurable Output Channels

The model can predict either both channels (npho and time) or just npho:

```yaml
model:
  predict_channels: ["npho"]  # npho-only mode (faster training, smaller model)
  # predict_channels: ["npho", "time"]  # default: both channels
```

When `predict_channels: ["npho"]`:
- The decoder output dimension is 1 instead of 2
- Time-related loss/metrics are skipped
- Prediction ROOT files only contain `pred_npho`, `truth_npho`, `error_npho`
- Analysis macros auto-detect the mode from metadata in the ROOT file

### Nested Time Configuration

Time-specific training options are grouped under `training.time:` and only apply when `"time"` is in `predict_channels`:

```yaml
training:
  npho_weight: 1.0
  time:
    weight: 1.0                # Loss weight for time channel
    mask_ratio_scale: 1.0      # Stratified masking scale for valid-time sensors
    use_npho_weight: true      # Weight time loss by sqrt(npho)
    npho_threshold: 100.0      # Min npho for conditional time loss
```

**Note:** Use the **new normalization scheme** (npho_scale=1000, sentinel_time=-1.0) for MAE pretraining. See [Data Pipeline](../architecture/data-pipeline.md) for details.

## MAE/Inpainter Shared Parameters

These parameters are shared between MAE and Inpainter training. All time-related options use the nested `training.time:` config path.

| Parameter | Config Path | MAE Default | Inpainter Default | Description |
|-----------|-------------|-------------|-------------------|-------------|
| `mask_ratio` | `model.mask_ratio` | 0.6 | 0.05 | Fraction of valid sensors to mask |
| `time.mask_ratio_scale` | `training.time.mask_ratio_scale` | 1.0 | 1.0 | Bias masking toward valid-time sensors |
| `time.npho_threshold` | `training.time.npho_threshold` | 100 | 100 | Min npho for time loss computation |
| `time.use_npho_weight` | `training.time.use_npho_weight` | true | true | Chi-square-like time weighting |
| `track_mae_rmse` | `training.track_mae_rmse` | false | false | Compute MAE/RMSE metrics (slower) |
| `track_metrics` | `training.track_metrics` | false | false | Per-face training metrics |
| `freeze_encoder` | `model.freeze_encoder` | - | false | Freeze encoder during inpainter training |

**Note:** Legacy flat config keys (`time_mask_ratio_scale`, `npho_threshold`, `use_npho_time_weight`) are auto-migrated to the nested `training.time:` structure but are deprecated. Use the nested paths above.
