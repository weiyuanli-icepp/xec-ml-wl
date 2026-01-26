#!/bin/bash
# Usage: ./run_mae_gh.sh
# MAE pre-training on GH nodes with same normalization as inpainter

# Use SQLite backend
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# --- Common Configuration ---
export EPOCHS=50
export BATCH=1024
export CHUNK_SIZE=256000
export RESUME_FROM=""
export PARTITION="gh-daily"
export TIME="06:00:00"
export CONFIG_PATH="config/mae_config.yaml"

# Normalization (must match inpainter)
export NPHO_SCALE="1000"
export NPHO_SCALE2="4.08"
export TIME_SCALE="1.14e-7"
export TIME_SHIFT="-0.46"
export SENTINEL_VALUE="-1.0"

# Loss configuration
export LOSS_FN="smooth_l1"
export TIME_WEIGHT="1.0"
export AUTO_WEIGHT="false"

# Learning rate
export LR="2.5e-4"
export LR_SCHEDULER="cosine"
export LR_MIN="1e-6"

# Paths
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"
export MLFLOW_EXPERIMENT="mae_pretraining"

# --- Mask ratios to scan ---
MASK_RATIOS=(0.60 0.65 0.70)

for MASK_RATIO in "${MASK_RATIOS[@]}"; do
    export MASK_RATIO
    export RUN_NAME="mae_mask${MASK_RATIO}_gh"

    echo "Submitting: $RUN_NAME (mask_ratio=$MASK_RATIO)"
    ./submit_mae.sh

    sleep 2
done

echo "All MAE jobs submitted!"
