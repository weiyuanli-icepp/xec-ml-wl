#!/bin/bash
# Usage:
# 1. Make sure to load anaconda module
# 2. Activate xec-ml-wl conda environment
# 3. salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=07:00:00
# 4. Run this script: ./interactive_mae_train2.sh


# --- Configuration ---
export RUN_NAME="sanity_mae"
export EPOCHS=2
export BATCH=1024
export CHUNK_SIZE=256000
export MASK_RATIO="0.6"

# Physics Constants
export NPHO_SCALE="0.58"
export NPHO_SCALE2="1.0"
export TIME_SCALE="6.5e-8"
export TIME_SHIFT="0.5"
export SENTINEL_VALUE="-5.0"

# Paths (Point to your new merged files)
# Ensure these files exist before running!
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"

export MLFLOW_EXPERIMENT="gamma_mae"

# --- Execution ---
echo "Starting Debug Run..."
echo "Train Data: $TRAIN_PATH"
echo "Val Data:   $VAL_PATH"

cd $HOME/meghome/xec-ml-wl
echo "Moved to directory $(pwd)"
python -m lib.train_mae \
    --train_root "${TRAIN_PATH}" \
    --val_root "${VAL_PATH}" \
    --save_path "artifacts/${RUN_NAME}" \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH} \
    --chunksize ${CHUNK_SIZE} \
    --mask_ratio ${MASK_RATIO} \
    --npho_scale ${NPHO_SCALE} \
    --npho_scale2 ${NPHO_SCALE2} \
    --time_scale ${TIME_SCALE} \
    --time_shift ${TIME_SHIFT} \
    --sentinel_value ${SENTINEL_VALUE} \
    --outer_mode "finegrid" \
    --outer_fine_pool 3 3 \
    --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
    --mlflow_run_name "${RUN_NAME}"