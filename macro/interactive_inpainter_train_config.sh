#!/bin/bash
# Usage:
# 1. Make sure to load anaconda module
# 2. Activate xec-ml-wl conda environment
# 3. salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=07:00:00
# 4. Run this script: ./interactive_inpainter_train_config.sh

# --- Fix for awkward_cpp libstdc++ compatibility on GH nodes ---
# Prioritize conda's libstdc++ over system GCC module
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

# --- Configuration (Run-specific overrides) ---
export RUN_NAME="inpainter_no_pretrain"
export EPOCHS=50

# Set to empty string to train without MAE pre-training (from scratch)
# Or set to a checkpoint path to use pre-trained MAE encoder
export MAE_CHECKPOINT=""
# export MAE_CHECKPOINT="artifacts/mae/mae_checkpoint_best.pth"

# Paths (Point to your data files)
export TRAIN_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"

# --- Execution ---
echo "Starting Inpainter Training..."
echo "Config File: config/inpainter_config.yaml"
echo "Train Data: $TRAIN_PATH"
echo "Val Data:   $VAL_PATH"
if [ -z "$MAE_CHECKPOINT" ]; then
    echo "MAE Checkpoint: None (training from scratch)"
else
    echo "MAE Checkpoint: $MAE_CHECKPOINT"
fi

cd $HOME/meghome/xec-ml-wl
echo "Moved to directory $(pwd)"

# Build command with optional MAE checkpoint
CMD="python -m lib.train_inpainter \
    --config config/inpainter_config.yaml \
    --train_root ${TRAIN_PATH} \
    --val_root ${VAL_PATH} \
    --save_path artifacts/${RUN_NAME} \
    --epochs ${EPOCHS} \
    --mlflow_run_name ${RUN_NAME}"

# Add MAE checkpoint if specified
if [ -n "$MAE_CHECKPOINT" ]; then
    CMD="$CMD --mae_checkpoint ${MAE_CHECKPOINT}"
fi

# Execute
eval $CMD
