#!/bin/bash
# Usage:
# 1. Make sure to load anaconda module
# 2. Activate xec-ml-wl conda environment
# 3. salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=07:00:00
# 4. Run this script: ./interactive_regressor.sh

# Conda activation for GH nodes (miniforge)
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" =~ ^gpu00[1-9]$ ]]; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "xec-ml-wl-gh" ]; then
        if [ -f "/data/user/ext-li_w1/miniforge-arm/bin/activate" ]; then
            source /data/user/ext-li_w1/miniforge-arm/bin/activate
            conda activate xec-ml-wl-gh
        else
            echo "Conda activate script not found at /data/user/ext-li_w1/miniforge-arm/bin/activate" >&2
            exit 1
        fi
    fi
fi

# Prioritize conda's libstdc++ over system GCC module
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# --- Configuration (Run-specific overrides) ---
export RUN_NAME="regressor_interactive_test"
export EPOCHS=20
export BATCH_SIZE=4096
export LR="3e-4"

# Tasks (space-separated: angle energy timing uvwFI)
export TASKS="angle"

# Model architecture
export OUTER_MODE="finegrid"
export OUTER_FINE_POOL="3 3"
export HIDDEN_DIM=256

# Training settings
export WARMUP_EPOCHS=2
export EMA_DECAY=0.999
export CHANNEL_DROPOUT_RATE=0.1
export GRAD_CLIP=1.0

# Normalization (legacy scheme for regressor)
export NPHO_SCALE="0.58"
export NPHO_SCALE2="1.0"
export TIME_SCALE="6.5e-8"
export TIME_SHIFT="0.5"
export SENTINEL_VALUE="-5.0"

# Paths (Point to your data files)
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"

# MLflow
export MLFLOW_EXPERIMENT="gamma_angle"

# ONNX export (set to "null" to disable)
export ONNX="meg2ang_convnextv2.onnx"

# Resume from checkpoint (leave empty for fresh start)
export RESUME_FROM=""
# export RESUME_FROM="artifacts/mae/mae_checkpoint_best.pth"  # Fine-tune from MAE

# --- Execution ---
echo "Starting Regressor Training..."
echo "Config File: config/train_config.yaml"
echo "Train Data: $TRAIN_PATH"
echo "Val Data:   $VAL_PATH"
echo "Tasks:      $TASKS"
if [ -z "$RESUME_FROM" ]; then
    echo "Resume:     None (training from scratch)"
else
    echo "Resume:     $RESUME_FROM"
fi

cd $HOME/meghome/xec-ml-wl
echo "Moved to directory $(pwd)"

# Build command
CMD="python -m lib.train_regressor \
    --config config/train_config.yaml \
    --run_name ${RUN_NAME} \
    --train_path ${TRAIN_PATH} \
    --val_path ${VAL_PATH} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --tasks ${TASKS} \
    --outer_mode ${OUTER_MODE} \
    --outer_fine_pool ${OUTER_FINE_POOL} \
    --hidden_dim ${HIDDEN_DIM} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --ema_decay ${EMA_DECAY} \
    --channel_dropout_rate ${CHANNEL_DROPOUT_RATE} \
    --grad_clip ${GRAD_CLIP} \
    --npho_scale ${NPHO_SCALE} \
    --npho_scale2 ${NPHO_SCALE2} \
    --time_scale ${TIME_SCALE} \
    --time_shift ${TIME_SHIFT} \
    --sentinel_value ${SENTINEL_VALUE} \
    --mlflow_experiment ${MLFLOW_EXPERIMENT} \
    --onnx ${ONNX}"

# Add resume checkpoint if specified
if [ -n "$RESUME_FROM" ]; then
    CMD="$CMD --resume_from ${RESUME_FROM}"
fi

# Execute
eval $CMD
