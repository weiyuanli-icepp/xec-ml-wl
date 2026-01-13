#!/bin/bash
# Usage:
# 1. Make sure to load anaconda module
# 2. Activate xec-ml-wl conda environment
# 3. salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=07:00:00
# 4. Run this script: ./interactive_mae_train_config.sh

# --- Fix for awkward_cpp libstdc++ compatibility on GH nodes ---
# Prioritize conda's libstdc++ over system GCC module
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# --- Configuration (Run-specific overrides) ---
export RUN_NAME="sanity_mae_config"
export EPOCHS=2

# Paths (Point to your data files)
# Ensure these files exist before running!
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"

# --- Execution ---
echo "Starting MAE Config Mode Test..."
echo "Config File: config/mae_config.yaml"
echo "Train Data: $TRAIN_PATH"
echo "Val Data:   $VAL_PATH"

cd $HOME/meghome/xec-ml-wl
echo "Moved to directory $(pwd)"

python -m lib.train_mae \
    --config config/mae_config.yaml \
    --train_root "${TRAIN_PATH}" \
    --val_root "${VAL_PATH}" \
    --save_path "artifacts/${RUN_NAME}" \
    --epochs ${EPOCHS} \
    --mlflow_run_name "${RUN_NAME}"
