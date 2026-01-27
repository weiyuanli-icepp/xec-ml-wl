#!/usr/bin/env bash
# Config-driven Inpainter submission script
# Usage: ./submit_inpainter.sh
#        DRY_RUN=1 ./submit_inpainter.sh   # Show config without submitting
#
# Environment variables:
#   CONFIG_PATH    - Path to config YAML (default: config/inpainter_config.yaml)
#   RUN_NAME       - Run name (optional, defaults to config or auto-generated)
#   PARTITION      - SLURM partition (default: a100-daily)
#   TIME           - Job time limit (default: 12:00:00)
#   RESUME_FROM    - Checkpoint to resume from (optional)
#   MAE_CHECKPOINT - MAE checkpoint to initialize encoder (optional)
#   DRY_RUN        - Set to 1 to show config without submitting

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/inpainter_config.yaml}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_NAME="${RUN_NAME:-}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"

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
CFG_CHUNKSIZE=$(yaml_get "chunksize" "$CONFIG_PATH")
CFG_NUM_WORKERS=$(yaml_get "num_workers" "$CONFIG_PATH")
CFG_MASK_RATIO=$(yaml_get "mask_ratio" "$CONFIG_PATH")
CFG_LR=$(yaml_get "lr" "$CONFIG_PATH")
CFG_LR_SCHEDULER=$(yaml_get "lr_scheduler" "$CONFIG_PATH")
CFG_WARMUP=$(yaml_get "warmup_epochs" "$CONFIG_PATH")
CFG_WEIGHT_DECAY=$(yaml_get "weight_decay" "$CONFIG_PATH")
CFG_GRAD_CLIP=$(yaml_get "grad_clip" "$CONFIG_PATH")
CFG_GRAD_ACCUM=$(yaml_get "grad_accum_steps" "$CONFIG_PATH")
CFG_LOSS_FN=$(yaml_get "loss_fn" "$CONFIG_PATH")
CFG_NPHO_WEIGHT=$(yaml_get "npho_weight" "$CONFIG_PATH")
CFG_TIME_WEIGHT=$(yaml_get "time_weight" "$CONFIG_PATH")
CFG_AMP=$(yaml_get "amp" "$CONFIG_PATH")
CFG_COMPILE=$(yaml_get "compile" "$CONFIG_PATH")
CFG_EMA=$(yaml_get "ema_decay" "$CONFIG_PATH")
CFG_FREEZE_ENCODER=$(yaml_get "freeze_encoder" "$CONFIG_PATH")
CFG_EXPERIMENT=$(yaml_get "experiment" "$CONFIG_PATH")
CFG_RUN_NAME=$(yaml_get "run_name" "$CONFIG_PATH")
CFG_TRAIN_PATH=$(yaml_get "train_path" "$CONFIG_PATH")
CFG_VAL_PATH=$(yaml_get "val_path" "$CONFIG_PATH")

# Normalization
CFG_NPHO_SCALE=$(yaml_get "npho_scale" "$CONFIG_PATH")
CFG_NPHO_SCALE2=$(yaml_get "npho_scale2" "$CONFIG_PATH")
CFG_TIME_SCALE=$(yaml_get "time_scale" "$CONFIG_PATH")
CFG_TIME_SHIFT=$(yaml_get "time_shift" "$CONFIG_PATH")
CFG_SENTINEL=$(yaml_get "sentinel_value" "$CONFIG_PATH")

# Use RUN_NAME from env, or from config, or generate timestamp
if [[ -z "$RUN_NAME" ]]; then
    if [[ -n "$CFG_RUN_NAME" && "$CFG_RUN_NAME" != "null" ]]; then
        RUN_NAME="$CFG_RUN_NAME"
    else
        RUN_NAME="inp_$(date +%Y%m%d_%H%M%S)"
    fi
fi

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

# Dry-run: show all config parameters
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "============================================"
    echo "[DRY-RUN] Inpainter Training Configuration"
    echo "============================================"
    echo ""
    echo "Config file: $CONFIG_PATH"
    echo ""
    echo "=== Job Settings ==="
    echo "  Run name:      $RUN_NAME"
    echo "  Partition:     $PARTITION"
    echo "  Time limit:    $TIME"
    echo "  Environment:   $ENV_NAME"
    echo "  Log file:      $LOG_FILE"
    [[ -n "$RESUME_FROM" ]] && echo "  Resume from:   $RESUME_FROM"
    [[ -n "$MAE_CHECKPOINT" ]] && echo "  MAE init:      $MAE_CHECKPOINT"
    echo ""
    echo "=== Data ==="
    echo "  Train path:    ${CFG_TRAIN_PATH:-?}"
    echo "  Val path:      ${CFG_VAL_PATH:-?}"
    echo "  Batch size:    ${CFG_BATCH:-?}"
    echo "  Chunk size:    ${CFG_CHUNKSIZE:-?}"
    echo "  Num workers:   ${CFG_NUM_WORKERS:-?}"
    echo ""
    echo "=== Model ==="
    echo "  Mask ratio:    ${CFG_MASK_RATIO:-?}"
    echo "  Freeze encoder:${CFG_FREEZE_ENCODER:-false}"
    echo ""
    echo "=== Normalization ==="
    echo "  npho_scale:    ${CFG_NPHO_SCALE:-?}"
    echo "  npho_scale2:   ${CFG_NPHO_SCALE2:-?}"
    echo "  time_scale:    ${CFG_TIME_SCALE:-?}"
    echo "  time_shift:    ${CFG_TIME_SHIFT:-?}"
    echo "  sentinel:      ${CFG_SENTINEL:-?}"
    echo ""
    echo "=== Training ==="
    echo "  Epochs:        ${CFG_EPOCHS:-?}"
    echo "  Learning rate: ${CFG_LR:-?}"
    echo "  LR scheduler:  ${CFG_LR_SCHEDULER:-null}"
    echo "  Warmup epochs: ${CFG_WARMUP:-0}"
    echo "  Weight decay:  ${CFG_WEIGHT_DECAY:-?}"
    echo "  Grad clip:     ${CFG_GRAD_CLIP:-?}"
    echo "  Grad accum:    ${CFG_GRAD_ACCUM:-1}"
    echo "  AMP:           ${CFG_AMP:-?}"
    echo "  Compile:       ${CFG_COMPILE:-false}"
    echo "  EMA decay:     ${CFG_EMA:-null}"
    echo ""
    echo "=== Loss ==="
    echo "  Loss fn:       ${CFG_LOSS_FN:-?}"
    echo "  Npho weight:   ${CFG_NPHO_WEIGHT:-?}"
    echo "  Time weight:   ${CFG_TIME_WEIGHT:-?}"
    echo ""
    echo "=== MLflow ==="
    echo "  Experiment:    ${CFG_EXPERIMENT:-inpainting}"
    echo ""
    echo "============================================"
    echo "[DRY-RUN] No job submitted. Remove DRY_RUN=1 to submit."
    echo "============================================"
    exit 0
fi

# Normal submission
echo "[SUBMIT] Inpainter Training"
echo "  Config:     $CONFIG_PATH"
echo "  Run:        $RUN_NAME"
echo "  Experiment: ${CFG_EXPERIMENT:-inpainting}"
echo "  Epochs:     ${CFG_EPOCHS:-?} | Batch: ${CFG_BATCH:-?} | Mask: ${CFG_MASK_RATIO:-?} | LR: ${CFG_LR:-?}"
echo "  Partition:  $PARTITION | Time: $TIME"
[[ -n "$RESUME_FROM" ]] && echo "  Resume:     $RESUME_FROM"
[[ -n "$MAE_CHECKPOINT" ]] && echo "  MAE init:   $MAE_CHECKPOINT"

RESUME_FLAG=""
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_FLAG="--resume_from ${RESUME_FROM}"
fi

MAE_CHECKPOINT_FLAG=""
if [[ -n "${MAE_CHECKPOINT}" ]]; then
    MAE_CHECKPOINT_FLAG="--mae_checkpoint ${MAE_CHECKPOINT}"
fi

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=inp_${RUN_NAME}
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

echo "[JOB] Starting Inpainter Training..."
python -m lib.train_inpainter \\
    --config "${CONFIG_PATH}" \\
    --save_path "artifacts/${RUN_NAME}" \\
    --mlflow_run_name "${RUN_NAME}" \\
    ${MAE_CHECKPOINT_FLAG} \\
    ${RESUME_FLAG}

echo "[JOB] Finished."
EOF
