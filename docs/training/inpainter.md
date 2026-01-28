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
