#!/bin/bash
# Usage: ./run_inpainter.sh
# Scans over multiple mask ratios

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# --- Mask ratios to scan ---
MASK_RATIOS=(0.10 0.15)

# --- Common Configuration ---
export EPOCHS=50
export BATCH=512
export CHUNK_SIZE=256000
export GRAD_ACCUM_STEPS=4
export RESUME_FROM=""
export PARTITION="a100-daily"
export TIME="6:00:00"
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
export LR="2.5e-4"
export LR_SCHEDULER="cosine"
export LR_MIN="1e-6"
export WARMUP_EPOCHS="3"
export WEIGHT_DECAY="1e-4"
export GRAD_CLIP="1.0"

# Model configuration
export FREEZE_ENCODER="false"
export MAE_CHECKPOINT="artifacts/mae_mask0.60_gh/mae_checkpoint_best.pth"  # Path to MAE checkpoint (optional)

# Paths (Point to your data files)
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"
export MLFLOW_EXPERIMENT="inpainting"

# --- Submit jobs for each mask ratio ---
for MASK in "${MASK_RATIOS[@]}"; do
    # Format mask ratio for run name (e.g., 0.05 -> mask0.05)
    MASK_STR=$(printf "%.2f" $MASK)
    export RUN_NAME="inpainter_pretrained0.60best_mask${MASK_STR}"
    export MASK_RATIO="$MASK"

    echo "Submitting: $RUN_NAME (mask_ratio=$MASK_RATIO)"
    ./submit_inpainter.sh

    # Small delay to avoid overwhelming scheduler
    sleep 0.5
done

echo "All jobs submitted!"
