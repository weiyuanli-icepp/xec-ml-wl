#!/usr/bin/env bash
# Config-driven Regressor submission script
# Usage: ./submit_regressor.sh
#
# Environment variables:
#   CONFIG_PATH  - Path to YAML config file (default: config/train_config.yaml)
#   RUN_NAME     - Run name (optional, defaults to config or auto-generated)
#   PARTITION    - SLURM partition (default: a100-daily)
#   TIME         - Job time limit (default: 12:00:00)
#
# Optional CLI overrides (empty = use config value):
#   TRAIN_PATH, VAL_PATH, EPOCHS, LR, BATCH_SIZE, RESUME_FROM, etc.

set -euo pipefail

# Required
CONFIG_PATH="${CONFIG_PATH:-config/train_config.yaml}"

# Job settings
RUN_NAME="${RUN_NAME:-}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"

# Optional overrides (empty string means no override)
TRAIN_PATH="${TRAIN_PATH:-}"
VAL_PATH="${VAL_PATH:-}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-}"
EMA_DECAY="${EMA_DECAY:-}"
GRAD_CLIP="${GRAD_CLIP:-}"
CHANNEL_DROPOUT_RATE="${CHANNEL_DROPOUT_RATE:-}"
TASKS="${TASKS:-}"
RESUME_FROM="${RESUME_FROM:-}"
SAVE_DIR="${SAVE_DIR:-}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-}"
ONNX="${ONNX:-}"
LOSS_BALANCE="${LOSS_BALANCE:-}"
OUTER_MODE="${OUTER_MODE:-}"
OUTER_FINE_POOL="${OUTER_FINE_POOL:-}"
HIDDEN_DIM="${HIDDEN_DIM:-}"
DROP_PATH_RATE="${DROP_PATH_RATE:-}"

# Validate config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

# Helper function to extract value from YAML (simple grep-based, handles most cases)
yaml_get() {
    local key="$1"
    local file="$2"
    grep -E "^\s*${key}:" "$file" 2>/dev/null | head -1 | sed 's/.*:\s*//' | sed 's/\s*#.*//' | tr -d '"' | tr -d "'"
}

# Extract key values from config for display
CFG_EPOCHS=$(yaml_get "epochs" "$CONFIG_PATH")
CFG_BATCH=$(yaml_get "batch_size" "$CONFIG_PATH")
CFG_LR=$(yaml_get "lr" "$CONFIG_PATH")
CFG_EXPERIMENT=$(yaml_get "experiment" "$CONFIG_PATH")
CFG_RUN_NAME=$(yaml_get "run_name" "$CONFIG_PATH")

# Use RUN_NAME from env, or from config, or generate timestamp
if [[ -z "$RUN_NAME" ]]; then
    if [[ -n "$CFG_RUN_NAME" && "$CFG_RUN_NAME" != "null" ]]; then
        RUN_NAME="$CFG_RUN_NAME"
    else
        RUN_NAME="reg_$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Determine environment based on partition
if [[ "$PARTITION" == gh* ]]; then
    ENV_NAME="xec-ml-wl-gh"
else
    ENV_NAME="xec-ml-wl"
fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

# Show effective values (override if set, otherwise config)
EFF_EPOCHS="${EPOCHS:-${CFG_EPOCHS:-?}}"
EFF_BATCH="${BATCH_SIZE:-${CFG_BATCH:-?}}"
EFF_LR="${LR:-${CFG_LR:-?}}"
EFF_EXPERIMENT="${MLFLOW_EXPERIMENT:-${CFG_EXPERIMENT:-gamma_angle}}"

echo "[SUBMIT] Regressor Training"
echo "  Config:     $CONFIG_PATH"
echo "  Run:        $RUN_NAME"
echo "  Experiment: $EFF_EXPERIMENT"
echo "  Epochs:     $EFF_EPOCHS | Batch: $EFF_BATCH | LR: $EFF_LR"
echo "  Partition:  $PARTITION | Time: $TIME"

# Build CLI override arguments
CLI_ARGS=""
[ -n "$TRAIN_PATH" ] && CLI_ARGS+=" --train_path \"$TRAIN_PATH\""
[ -n "$VAL_PATH" ] && CLI_ARGS+=" --val_path \"$VAL_PATH\""
[ -n "$EPOCHS" ] && CLI_ARGS+=" --epochs $EPOCHS"
[ -n "$LR" ] && CLI_ARGS+=" --lr $LR"
[ -n "$BATCH_SIZE" ] && CLI_ARGS+=" --batch_size $BATCH_SIZE"
[ -n "$WEIGHT_DECAY" ] && CLI_ARGS+=" --weight_decay $WEIGHT_DECAY"
[ -n "$WARMUP_EPOCHS" ] && CLI_ARGS+=" --warmup_epochs $WARMUP_EPOCHS"
[ -n "$EMA_DECAY" ] && CLI_ARGS+=" --ema_decay $EMA_DECAY"
[ -n "$GRAD_CLIP" ] && CLI_ARGS+=" --grad_clip $GRAD_CLIP"
[ -n "$CHANNEL_DROPOUT_RATE" ] && CLI_ARGS+=" --channel_dropout_rate $CHANNEL_DROPOUT_RATE"
[ -n "$TASKS" ] && CLI_ARGS+=" --tasks $TASKS"
[ -n "$RESUME_FROM" ] && CLI_ARGS+=" --resume_from \"$RESUME_FROM\""
[ -n "$SAVE_DIR" ] && CLI_ARGS+=" --save_dir \"$SAVE_DIR\""
[ -n "$MLFLOW_EXPERIMENT" ] && CLI_ARGS+=" --mlflow_experiment \"$MLFLOW_EXPERIMENT\""
[ -n "$ONNX" ] && CLI_ARGS+=" --onnx \"$ONNX\""
[ -n "$LOSS_BALANCE" ] && CLI_ARGS+=" --loss_balance $LOSS_BALANCE"
[ -n "$OUTER_MODE" ] && CLI_ARGS+=" --outer_mode $OUTER_MODE"
[ -n "$OUTER_FINE_POOL" ] && CLI_ARGS+=" --outer_fine_pool $OUTER_FINE_POOL"
[ -n "$HIDDEN_DIM" ] && CLI_ARGS+=" --hidden_dim $HIDDEN_DIM"
[ -n "$DROP_PATH_RATE" ] && CLI_ARGS+=" --drop_path_rate $DROP_PATH_RATE"

if [ -n "$CLI_ARGS" ]; then
    echo "  Overrides: $CLI_ARGS"
fi
[[ -n "$RESUME_FROM" ]] && echo "  Resume:     $RESUME_FROM"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=reg_${RUN_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --clusters=gmerlin7

set -e

# Load Environment
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true

ARM_CONDA="\$HOME/miniforge-arm/bin/conda"
X86_CONDA="/opt/psi/Programming/anaconda/2024.08/conda/bin/conda"

# Load module for x86 nodes
module load anaconda/2024.08 2>/dev/null || true

# Initialize Conda based on architecture
if [ -f "\$ARM_CONDA" ] && [ "\$(uname -m)" == "aarch64" ]; then
    echo "[JOB] Detected ARM64 architecture. Using Miniforge."
    eval "\$(\$ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
elif [ -f "\$X86_CONDA" ]; then
    eval "\$(\$X86_CONDA shell.bash hook)"
else
    echo "CRITICAL ERROR: Could not find 'conda' on partition ${PARTITION}."
    exit 1
fi

echo "[JOB] Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Fix awkward_cpp libstdc++ compatibility on GH nodes
if [ -n "\$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
fi

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"
echo "[JOB] Config: ${CONFIG_PATH}"

# Use SQLite backend for MLflow
export MLFLOW_TRACKING_URI="sqlite:///\$(pwd)/mlruns.db"

echo "[JOB] Starting Regressor training..."
python -m lib.train_regressor \\
    --config "${CONFIG_PATH}" \\
    --run_name "${RUN_NAME}" ${CLI_ARGS}

echo "[JOB] Finished."
EOF
