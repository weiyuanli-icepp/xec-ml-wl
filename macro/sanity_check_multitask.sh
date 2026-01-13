#!/bin/bash
# Sanity Check for Multi-Task XEC Regressor
# Usage:
# 1. Make sure to load anaconda module and activate xec-ml-wl environment
# 2. For interactive GPU session:
#    salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=02:00:00
# 3. Run this script: ./sanity_check_multitask.sh
#
# This script tests the multi-task training pipeline with minimal epochs
# to verify everything works correctly before launching full training.

set -euo pipefail

# --- Configuration ---
RUN_NAME="sanity_multitask"
EPOCHS=2
BATCH=256
CHUNKSIZE=10000

# Paths (Update these to your data location)
TRAIN_PATH="${TRAIN_PATH:-$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root}"
VAL_PATH="${VAL_PATH:-$HOME/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root}"

# Tasks to enable (space-separated)
# Options: angle, energy, timing, uvwFI
TASKS="${TASKS:-angle}"

# MLflow
MLFLOW_EXPERIMENT="sanity_check"

# Create temporary config file
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE=$(mktemp /tmp/sanity_config_XXXXXX.yaml)

# Generate YAML config
cat > "$CONFIG_FILE" << EOF
# Sanity Check Config - Auto-generated
data:
  train_path: "${TRAIN_PATH}"
  val_path: "${VAL_PATH}"
  tree_name: "tree"
  batch_size: ${BATCH}
  chunksize: ${CHUNKSIZE}
  num_workers: 4
  num_threads: 4

normalization:
  npho_scale: 0.58
  npho_scale2: 1.0
  time_scale: 6.5e8
  time_shift: 0.5
  sentinel_value: -5.0

model:
  outer_mode: "finegrid"
  outer_fine_pool: [3, 3]
  hidden_dim: 256
  drop_path_rate: 0.0

tasks:
EOF

# Add task configurations based on TASKS variable
for task in $TASKS; do
    case $task in
        angle)
            cat >> "$CONFIG_FILE" << EOF
  angle:
    enabled: true
    loss_fn: "smooth_l1"
    loss_beta: 1.0
    weight: 1.0
EOF
            ;;
        energy)
            cat >> "$CONFIG_FILE" << EOF
  energy:
    enabled: true
    loss_fn: "l1"
    loss_beta: 1.0
    weight: 1.0
EOF
            ;;
        timing)
            cat >> "$CONFIG_FILE" << EOF
  timing:
    enabled: true
    loss_fn: "l1"
    loss_beta: 1.0
    weight: 1.0
EOF
            ;;
        uvwFI)
            cat >> "$CONFIG_FILE" << EOF
  uvwFI:
    enabled: true
    loss_fn: "mse"
    loss_beta: 1.0
    weight: 1.0
EOF
            ;;
    esac
done

# Add remaining config sections
cat >> "$CONFIG_FILE" << EOF

training:
  epochs: ${EPOCHS}
  lr: 3.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 1
  use_scheduler: true
  amp: true
  ema_decay: 0.999
  channel_dropout_rate: 0.1
  grad_clip: 1.0

loss_balance: "manual"

reweighting:
  angle:
    enabled: false
  energy:
    enabled: false
  timing:
    enabled: false
  uvwFI:
    enabled: false

checkpoint:
  resume_from: null
  save_dir: "artifacts"

mlflow:
  experiment: "${MLFLOW_EXPERIMENT}"
  run_name: "${RUN_NAME}"

export:
  onnx: null
EOF

# --- Execution ---
echo "========================================"
echo "Multi-Task Regressor Sanity Check"
echo "========================================"
echo "Run Name:    ${RUN_NAME}"
echo "Tasks:       ${TASKS}"
echo "Epochs:      ${EPOCHS}"
echo "Batch Size:  ${BATCH}"
echo "Train Data:  ${TRAIN_PATH}"
echo "Val Data:    ${VAL_PATH}"
echo "Config File: ${CONFIG_FILE}"
echo "========================================"

# Verify data files exist
if [ ! -f "$TRAIN_PATH" ] && [ ! -d "$TRAIN_PATH" ]; then
    echo "[ERROR] Training data not found: $TRAIN_PATH"
    echo "Please set TRAIN_PATH environment variable to your data location."
    rm -f "$CONFIG_FILE"
    exit 1
fi

if [ ! -f "$VAL_PATH" ] && [ ! -d "$VAL_PATH" ]; then
    echo "[ERROR] Validation data not found: $VAL_PATH"
    echo "Please set VAL_PATH environment variable to your data location."
    rm -f "$CONFIG_FILE"
    exit 1
fi

cd "$REPO_ROOT"
echo "[INFO] Working directory: $(pwd)"

# Print config for debugging
echo ""
echo "[INFO] Generated config:"
echo "----------------------------------------"
cat "$CONFIG_FILE"
echo "----------------------------------------"
echo ""

# Run training
echo "[INFO] Starting sanity check training..."
python train_xec_regressor.py --config "$CONFIG_FILE"

# Cleanup
rm -f "$CONFIG_FILE"

echo ""
echo "========================================"
echo "Sanity check completed successfully!"
echo "========================================"
