#!/bin/bash
# Usage: ./run_inpainter.sh

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# --- Configuration (Run-specific overrides) ---
export RUN_NAME="inpainter_mask0.07"
export EPOCHS=50
export BATCH=1024
export CHUNK_SIZE=256000
export MASK_RATIO="0.07"
export RESUME_FROM=""
export PARTITION="a100-daily"
export TIME="12:00:00"
export CONFIG_PATH="config/inpainter_config.yaml"

# Normalization (must match MAE pretraining)
export NPHO_SCALE="1000"
export NPHO_SCALE2="4.08"
export TIME_SCALE="1.14e-7"
export TIME_SHIFT="-0.46"
export SENTINEL_VALUE="-1.0"

# Loss configuration
export LOSS_FN="smooth_l1"
export NPHO_WEIGHT="1.0"
export TIME_WEIGHT="1.0"

# Learning rate
export LR="2e-4"
export LR_SCHEDULER="cosine"
export LR_MIN="1e-6"
export WARMUP_EPOCHS="0"
export WEIGHT_DECAY="1e-4"
export GRAD_CLIP="1.0"

# Model configuration
export FREEZE_ENCODER="false"
export MAE_CHECKPOINT=""  # Path to MAE checkpoint (optional)

# Paths (Point to your data files)
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"
export MLFLOW_EXPERIMENT="inpainting"

./submit_inpainter.sh
