#!/usr/bin/env bash
# Config-driven MAE submission script
# Usage: ./submit_mae.sh
# Environment variables:
#   CONFIG_PATH  - Path to config YAML (required, default: config/mae_config.yaml)
#   RUN_NAME     - Run name (optional, defaults to config or auto-generated)
#   PARTITION    - SLURM partition (default: a100-daily)
#   TIME         - Job time limit (default: 23:00:00)
#   RESUME_FROM  - Checkpoint to resume from (optional)

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/mae_config.yaml}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-23:00:00}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_NAME="${RUN_NAME:-}"

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
CFG_MASK_RATIO=$(yaml_get "mask_ratio" "$CONFIG_PATH")
CFG_LR=$(yaml_get "lr" "$CONFIG_PATH")
CFG_EXPERIMENT=$(yaml_get "experiment" "$CONFIG_PATH")
CFG_RUN_NAME=$(yaml_get "run_name" "$CONFIG_PATH")

# Use RUN_NAME from env, or from config, or generate timestamp
if [[ -z "$RUN_NAME" ]]; then
    if [[ -n "$CFG_RUN_NAME" && "$CFG_RUN_NAME" != "null" ]]; then
        RUN_NAME="$CFG_RUN_NAME"
    else
        RUN_NAME="mae_$(date +%Y%m%d_%H%M%S)"
    fi
fi

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

echo "[SUBMIT] MAE Pre-training"
echo "  Config:     $CONFIG_PATH"
echo "  Run:        $RUN_NAME"
echo "  Experiment: ${CFG_EXPERIMENT:-mae_pretraining}"
echo "  Epochs:     ${CFG_EPOCHS:-?} | Batch: ${CFG_BATCH:-?} | Mask: ${CFG_MASK_RATIO:-?} | LR: ${CFG_LR:-?}"
echo "  Partition:  $PARTITION | Time: $TIME"
[[ -n "$RESUME_FROM" ]] && echo "  Resume:     $RESUME_FROM"

RESUME_FLAG=""
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_FLAG="--resume_from ${RESUME_FROM}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=mae_${RUN_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --clusters=gmerlin7

set -e
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

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///\$(pwd)/mlruns.db"

# Create artifacts dir
mkdir -p artifacts/${RUN_NAME}

echo "[JOB] Starting MAE Pre-training..."
python -m lib.train_mae \\
    --config "${CONFIG_PATH}" \\
    --save_path "artifacts/${RUN_NAME}" \\
    --mlflow_run_name "${RUN_NAME}" \\
    ${RESUME_FLAG}

echo "[JOB] Finished."
EOF
