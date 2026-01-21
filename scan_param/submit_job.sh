#!/usr/bin/env bash
# Usage: ./submit_job.sh [RUN_NAME] [CONFIG_FILE] [PARTITION] [TIME]
#
# Optional CLI overrides via environment variables:
#   TRAIN_PATH, VAL_PATH - Override data paths
#   EPOCHS, LR, BATCH - Override training params
#   TASKS - Space-separated list of tasks (e.g., "angle energy")
#
# Example:
#   ./submit_job.sh my_run ../config/train_config.yaml a100-daily 12:00:00
#   EPOCHS=100 LR=1e-4 ./submit_job.sh my_run ../config/train_config.yaml

set -euo pipefail

RUN_NAME="${1:-test_run}"
CONFIG_FILE="${2:-../config/train_config.yaml}"
PARTITION="${3:-a100-daily}"
TIME="${4:-12:00:00}"

# Optional overrides (empty string means no override)
TRAIN_PATH="${TRAIN_PATH:-}"
VAL_PATH="${VAL_PATH:-}"
EPOCHS="${EPOCHS:-}"
LR="${LR:-}"
BATCH="${BATCH:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-}"
EMA_DECAY="${EMA_DECAY:-}"
GRAD_CLIP="${GRAD_CLIP:-}"
TASKS="${TASKS:-}"
RESUME_FROM="${RESUME_FROM:-}"
SAVE_DIR="${SAVE_DIR:-}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-}"
ONNX="${ONNX:-}"
AUTO_WEIGHT="${AUTO_WEIGHT:-false}"

# 1. Determine Environment based on Partition
if [[ "$PARTITION" == gh* ]]; then
    ENV_NAME="xec-ml-wl-gh"
else
    ENV_NAME="xec-ml-wl"
fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

echo "[SUBMIT] Run: $RUN_NAME | Config: $CONFIG_FILE | Partition: $PARTITION"

# Build CLI override arguments
CLI_ARGS=""
[ -n "$TRAIN_PATH" ] && CLI_ARGS+=" --train_path \"$TRAIN_PATH\""
[ -n "$VAL_PATH" ] && CLI_ARGS+=" --val_path \"$VAL_PATH\""
[ -n "$EPOCHS" ] && CLI_ARGS+=" --epochs $EPOCHS"
[ -n "$LR" ] && CLI_ARGS+=" --lr $LR"
[ -n "$BATCH" ] && CLI_ARGS+=" --batch $BATCH"
[ -n "$WEIGHT_DECAY" ] && CLI_ARGS+=" --weight_decay $WEIGHT_DECAY"
[ -n "$WARMUP_EPOCHS" ] && CLI_ARGS+=" --warmup_epochs $WARMUP_EPOCHS"
[ -n "$EMA_DECAY" ] && CLI_ARGS+=" --ema_decay $EMA_DECAY"
[ -n "$GRAD_CLIP" ] && CLI_ARGS+=" --grad_clip $GRAD_CLIP"
[ -n "$TASKS" ] && CLI_ARGS+=" --tasks $TASKS"
[ -n "$RESUME_FROM" ] && CLI_ARGS+=" --resume_from \"$RESUME_FROM\""
[ -n "$SAVE_DIR" ] && CLI_ARGS+=" --save_dir \"$SAVE_DIR\""
[ -n "$MLFLOW_EXPERIMENT" ] && CLI_ARGS+=" --mlflow_experiment \"$MLFLOW_EXPERIMENT\""
[ -n "$ONNX" ] && CLI_ARGS+=" --onnx \"$ONNX\""
AUTO_CHANNEL_FLAG=""
case "${AUTO_WEIGHT}" in
    true|True|TRUE|1|yes|YES)
        AUTO_CHANNEL_FLAG="--auto_channel_weight"
        ;;
esac
[ -n "$AUTO_CHANNEL_FLAG" ] && CLI_ARGS+=" $AUTO_CHANNEL_FLAG"

echo "[SUBMIT] CLI overrides:$CLI_ARGS"

# 2. Submit Job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=xec_${RUN_NAME}
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

# 1. Try loading the module (this handles paths automatically for x86 vs ARM)
module load anaconda/2024.08 2>/dev/null || true

# 2. Initialize Conda Dynamically (Escaped \$(...) to run on compute node)
if [ -f "\$ARM_CONDA" ] && [ "\$(uname -m)" == "aarch64" ]; then
    echo "[JOB] Detected ARM64 architecture. Using Miniforge."
    eval "\$(\$ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    # The module worked and conda is in PATH
    eval "\$(conda shell.bash hook)"
elif [ -f "\$X86_CONDA" ]; then
    # Fallback for A100 if module failed
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
echo "[JOB] Changed directory to: \$(pwd)"

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///\$(pwd)/mlruns.db"

echo "[JOB] Running training with config: ${CONFIG_FILE}..."
python scan_param/run_training_cli.py \\
    --config "${CONFIG_FILE}" \\
    --run_name "${RUN_NAME}" ${CLI_ARGS}

echo "[JOB] Finished."
EOF
