#!/bin/bash
# Usage:
# 1. Make sure to load anaconda module
# 2. Activate xec-ml-wl conda environment
# 3. salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=07:00:00
# 4. Run this script: ./interactive_mae_train_config.sh

# Conda activation for GH nodes (miniforge)
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" =~ ^gpu00[1-9]$ ]]; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "xec-ml-wl-gh" ]; then
        if [ -f "/data/user/ext-li_w1/miniforge-arm/bin/activate" ]; then
            # Ensure conda is available in non-interactive shells
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

# Use SQLite backend
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# Configuration
export RUN_NAME="mae_mask0.65"
export EPOCHS=50
export LOSS_FN="smooth_l1"
# export TIME_SCALE="5e-5"
# export TIME_SHIFT="0.0065"
export TIME_SCALE="6.5e-8"
export TIME_SHIFT="0.5"
export TIME_WEIGHT="0.05"
export AUTO_WEIGHT="false"
export MASK_RATIO="0.65"
export LR="2.5e-4"
export SCHEDULER="cosine"
export LR_MIN="1e-8"

export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"

# export RESUME_FROM=""
export RESUME_FROM="artifacts/mae_mask0.65/mae_checkpoint_last.pth"

# Execution
echo "Starting MAE Config Mode Test..."
echo "Config File: config/mae_config.yaml"
echo "Train Data: $TRAIN_PATH"
echo "Val Data:   $VAL_PATH"

cd $HOME/meghome/xec-ml-wl
echo "Moved to directory $(pwd)"

AUTO_CHANNEL_FLAG=""
case "${AUTO_WEIGHT}" in
    true|True|TRUE|1|yes|YES)
        AUTO_CHANNEL_FLAG="--auto_channel_weight"
        ;;
esac

python -m lib.train_mae \
    --config config/mae_config.yaml \
    --train_root "${TRAIN_PATH}" \
    --val_root "${VAL_PATH}" \
    --save_path "artifacts/${RUN_NAME}" \
    --mlflow_experiment "mae_pretraining" \
    --mlflow_run_name "${RUN_NAME}" \
    --epochs ${EPOCHS} \
    --loss_fn "${LOSS_FN}" \
    --time_scale "${TIME_SCALE}" \
    --time_shift "${TIME_SHIFT}" \
    --time_weight "${TIME_WEIGHT}" \
    ${AUTO_CHANNEL_FLAG} \
    --mask_ratio "${MASK_RATIO}" \
    --lr_scheduler "${SCHEDULER}" \
    --lr_min "${LR_MIN}" \
    --resume_from "${RESUME_FROM}"