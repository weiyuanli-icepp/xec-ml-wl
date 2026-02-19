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
#   MAE_CHECKPOINT - MAE checkpoint to initialize encoder (optional, overrides config)
#   DRY_RUN        - Set to 1 to show config without submitting
#   NUM_GPUS       - Number of GPUs for DDP training (overrides config if set)

set -euo pipefail

CONFIG_PATH="${CONFIG_PATH:-config/inpainter_config.yaml}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
RESUME_FROM="${RESUME_FROM:-}"
RUN_NAME="${RUN_NAME:-}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-}"
DRY_RUN="${DRY_RUN:-0}"
NUM_GPUS="${NUM_GPUS:-}"  # Empty = read from config

# Validate config file exists
if [[ ! -f "$CONFIG_PATH" ]]; then
    echo "[ERROR] Config file not found: $CONFIG_PATH"
    exit 1
fi

# Helper function to extract value from YAML (simple grep-based, handles most cases)
yaml_get() {
    local key="$1"
    local file="$2"
    # Match line, extract value after first colon, strip comments and quotes
    grep -E "^\s*${key}:" "$file" 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | sed 's/[[:space:]]*#.*//' | tr -d '"' | tr -d "'" || true
}

# === Extract ALL config values ===

# Data
CFG_TRAIN_PATH=$(yaml_get "train_path" "$CONFIG_PATH")
CFG_VAL_PATH=$(yaml_get "val_path" "$CONFIG_PATH")
CFG_TREE_NAME=$(yaml_get "tree_name" "$CONFIG_PATH")
CFG_BATCH=$(yaml_get "batch_size" "$CONFIG_PATH")
CFG_CHUNKSIZE=$(yaml_get "chunksize" "$CONFIG_PATH")
CFG_NUM_WORKERS=$(yaml_get "num_workers" "$CONFIG_PATH")
CFG_NUM_THREADS=$(yaml_get "num_threads" "$CONFIG_PATH")
CFG_NPHO_BRANCH=$(yaml_get "npho_branch" "$CONFIG_PATH")
CFG_TIME_BRANCH=$(yaml_get "time_branch" "$CONFIG_PATH")

# Normalization
CFG_NPHO_SCALE=$(yaml_get "npho_scale" "$CONFIG_PATH")
CFG_NPHO_SCALE2=$(yaml_get "npho_scale2" "$CONFIG_PATH")
CFG_TIME_SCALE=$(yaml_get "time_scale" "$CONFIG_PATH")
CFG_TIME_SHIFT=$(yaml_get "time_shift" "$CONFIG_PATH")
CFG_SENTINEL=$(yaml_get "sentinel_time" "$CONFIG_PATH")

# Model
CFG_OUTER_MODE=$(yaml_get "outer_mode" "$CONFIG_PATH")
CFG_OUTER_FINE_POOL=$(yaml_get "outer_fine_pool" "$CONFIG_PATH")
CFG_MASK_RATIO=$(yaml_get "mask_ratio" "$CONFIG_PATH")
CFG_TIME_MASK_RATIO_SCALE=$(yaml_get "time_mask_ratio_scale" "$CONFIG_PATH")
CFG_FREEZE_ENCODER=$(yaml_get "freeze_encoder" "$CONFIG_PATH")

# Training
CFG_MAE_CHECKPOINT=$(yaml_get "mae_checkpoint" "$CONFIG_PATH")
CFG_EPOCHS=$(yaml_get "epochs" "$CONFIG_PATH")
CFG_LR=$(yaml_get "lr" "$CONFIG_PATH")
CFG_LR_SCHEDULER=$(yaml_get "lr_scheduler" "$CONFIG_PATH")
CFG_LR_MIN=$(yaml_get "lr_min" "$CONFIG_PATH")
CFG_WARMUP=$(yaml_get "warmup_epochs" "$CONFIG_PATH")
CFG_WEIGHT_DECAY=$(yaml_get "weight_decay" "$CONFIG_PATH")
CFG_LOSS_FN=$(yaml_get "loss_fn" "$CONFIG_PATH")
CFG_NPHO_WEIGHT=$(yaml_get "npho_weight" "$CONFIG_PATH")
CFG_TIME_WEIGHT=$(yaml_get "time_weight" "$CONFIG_PATH")
CFG_GRAD_CLIP=$(yaml_get "grad_clip" "$CONFIG_PATH")
CFG_GRAD_ACCUM=$(yaml_get "grad_accum_steps" "$CONFIG_PATH")
CFG_AMP=$(yaml_get "amp" "$CONFIG_PATH")
CFG_TRACK_MAE_RMSE=$(yaml_get "track_mae_rmse" "$CONFIG_PATH")
CFG_TRACK_METRICS=$(yaml_get "track_metrics" "$CONFIG_PATH")
CFG_NPHO_THRESHOLD=$(yaml_get "npho_threshold" "$CONFIG_PATH")
CFG_USE_NPHO_TIME_WEIGHT=$(yaml_get "use_npho_time_weight" "$CONFIG_PATH")

# Checkpoint
CFG_RESUME_FROM=$(yaml_get "resume_from" "$CONFIG_PATH")
CFG_SAVE_DIR=$(yaml_get "save_dir" "$CONFIG_PATH")
CFG_SAVE_INTERVAL=$(yaml_get "save_interval" "$CONFIG_PATH")
CFG_SAVE_PREDICTIONS=$(yaml_get "save_predictions" "$CONFIG_PATH")

# MLflow
CFG_EXPERIMENT=$(yaml_get "experiment" "$CONFIG_PATH")
CFG_RUN_NAME=$(yaml_get "run_name" "$CONFIG_PATH")

# Distributed
CFG_NUM_GPUS=$(yaml_get "num_gpus" "$CONFIG_PATH")

# Use NUM_GPUS from env, or from config, or default to 1
if [[ -z "$NUM_GPUS" ]]; then
    if [[ -n "$CFG_NUM_GPUS" && "$CFG_NUM_GPUS" != "null" ]]; then
        NUM_GPUS="$CFG_NUM_GPUS"
    else
        NUM_GPUS="1"
    fi
fi

# Use RUN_NAME from env, or from config, or generate timestamp
if [[ -z "$RUN_NAME" ]]; then
    if [[ -n "$CFG_RUN_NAME" && "$CFG_RUN_NAME" != "null" ]]; then
        RUN_NAME="$CFG_RUN_NAME"
    else
        RUN_NAME="inp_$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Use RESUME_FROM from env if set, otherwise from config
if [[ -z "$RESUME_FROM" && -n "$CFG_RESUME_FROM" && "$CFG_RESUME_FROM" != "null" ]]; then
    RESUME_FROM="$CFG_RESUME_FROM"
fi

# Use MAE_CHECKPOINT from env if set, otherwise from config
if [[ -z "$MAE_CHECKPOINT" && -n "$CFG_MAE_CHECKPOINT" && "$CFG_MAE_CHECKPOINT" != "null" ]]; then
    MAE_CHECKPOINT="$CFG_MAE_CHECKPOINT"
fi

# === Path Validation ===
# Helper to expand ~ and check if path exists (file, directory, or glob pattern)
check_path() {
    local path="$1"
    local desc="$2"
    local required="$3"  # "required" or "optional"

    # Skip if empty and optional
    if [[ -z "$path" || "$path" == "null" ]]; then
        if [[ "$required" == "required" ]]; then
            echo "[ERROR] $desc is not set"
            return 1
        fi
        return 0
    fi

    # Expand ~ to $HOME
    local expanded="${path/#\~/$HOME}"

    # Check if file or directory exists, or if glob pattern matches
    if [[ -e "$expanded" ]]; then
        return 0
    elif compgen -G "$expanded" > /dev/null 2>&1; then
        return 0  # Glob pattern matches
    else
        echo "[ERROR] $desc not found: $path"
        return 1
    fi
}

VALIDATION_FAILED=0

# Check train path
if ! check_path "$CFG_TRAIN_PATH" "Train path" "required"; then
    VALIDATION_FAILED=1
fi

# Check val path (optional but warn if missing)
if ! check_path "$CFG_VAL_PATH" "Validation path" "optional"; then
    echo "[WARN] Validation path not found, validation will be skipped"
fi

# Check resume checkpoint if specified
if [[ -n "$RESUME_FROM" ]]; then
    if ! check_path "$RESUME_FROM" "Resume checkpoint" "required"; then
        VALIDATION_FAILED=1
    fi
fi

# Check MAE checkpoint if specified
if [[ -n "$MAE_CHECKPOINT" ]]; then
    if ! check_path "$MAE_CHECKPOINT" "MAE checkpoint" "required"; then
        VALIDATION_FAILED=1
    fi
fi

# Check save directory is writable
SAVE_DIR_EXPANDED="${CFG_SAVE_DIR/#\~/$HOME}"
SAVE_DIR_EXPANDED="${SAVE_DIR_EXPANDED:-artifacts/inpainter}"
if [[ ! -d "$SAVE_DIR_EXPANDED" ]]; then
    mkdir -p "$SAVE_DIR_EXPANDED" 2>/dev/null || {
        echo "[ERROR] Cannot create save directory: $SAVE_DIR_EXPANDED"
        VALIDATION_FAILED=1
    }
fi

if [[ "$VALIDATION_FAILED" -eq 1 ]]; then
    echo ""
    echo "[ABORT] Fix the above errors before submitting."
    exit 1
fi

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="$HOME/meghome/xec-ml-wl/log"
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
    echo "  Partition:     $PARTITION"
    echo "  GPUs:          $NUM_GPUS"
    echo "  Time limit:    $TIME"
    echo "  Environment:   $ENV_NAME"
    echo "  Log file:      $LOG_FILE"
    echo ""
    echo "=== Data ==="
    echo "  Train path:    ${CFG_TRAIN_PATH:-?}"
    echo "  Val path:      ${CFG_VAL_PATH:-?}"
    echo "  Tree name:     ${CFG_TREE_NAME:-tree}"
    echo "  Batch size:    ${CFG_BATCH:-?}"
    echo "  Chunk size:    ${CFG_CHUNKSIZE:-?}"
    echo "  Num workers:   ${CFG_NUM_WORKERS:-?}"
    echo "  Num threads:   ${CFG_NUM_THREADS:-4}"
    echo "  Npho branch:   ${CFG_NPHO_BRANCH:-npho}"
    echo "  Time branch:   ${CFG_TIME_BRANCH:-time}"
    echo ""
    echo "=== Normalization ==="
    echo "  npho_scale:    ${CFG_NPHO_SCALE:-?}"
    echo "  npho_scale2:   ${CFG_NPHO_SCALE2:-?}"
    echo "  time_scale:    ${CFG_TIME_SCALE:-?}"
    echo "  time_shift:    ${CFG_TIME_SHIFT:-?}"
    echo "  sentinel:      ${CFG_SENTINEL:-?}"
    echo ""
    echo "=== Model ==="
    echo "  Outer mode:    ${CFG_OUTER_MODE:-finegrid}"
    echo "  Outer pool:    ${CFG_OUTER_FINE_POOL:-[3, 3]}"
    echo "  Mask ratio:    ${CFG_MASK_RATIO:-0.05}"
    echo "  Time mask scale: ${CFG_TIME_MASK_RATIO_SCALE:-1.0}"
    echo "  Freeze encoder: ${CFG_FREEZE_ENCODER:-false}"
    [[ -n "$MAE_CHECKPOINT" ]] && echo "  MAE checkpoint: $MAE_CHECKPOINT"
    echo ""
    echo "=== Training ==="
    echo "  Epochs:        ${CFG_EPOCHS:-?}"
    echo "  Learning rate: ${CFG_LR:-?}"
    echo "  LR scheduler:  ${CFG_LR_SCHEDULER:-null}"
    # Show scheduler-specific parameters
    if [[ "${CFG_LR_SCHEDULER}" == "cosine" ]]; then
        echo "  Warmup epochs: ${CFG_WARMUP:-3}"
        echo "  LR min:        ${CFG_LR_MIN:-1e-6}"
    elif [[ "${CFG_LR_SCHEDULER}" != "null" && "${CFG_LR_SCHEDULER}" != "none" && -n "${CFG_LR_SCHEDULER}" ]]; then
        echo "  Warmup epochs: ${CFG_WARMUP:-0}"
    fi
    echo "  Weight decay:  ${CFG_WEIGHT_DECAY:-?}"
    echo "  Grad clip:     ${CFG_GRAD_CLIP:-1.0}"
    echo "  Grad accum:    ${CFG_GRAD_ACCUM:-1}"
    echo "  AMP:           ${CFG_AMP:-true}"
    echo ""
    echo "=== Loss ==="
    echo "  Loss fn:       ${CFG_LOSS_FN:-smooth_l1}"
    echo "  Npho weight:   ${CFG_NPHO_WEIGHT:-1.0}"
    echo "  Time weight:   ${CFG_TIME_WEIGHT:-1.0}"
    echo "  Npho threshold:      ${CFG_NPHO_THRESHOLD:-100}"
    echo "  Use npho time weight: ${CFG_USE_NPHO_TIME_WEIGHT:-true}"
    echo ""
    echo "=== Tracking ==="
    echo "  Track MAE/RMSE:      ${CFG_TRACK_MAE_RMSE:-false}"
    echo "  Track train metrics: ${CFG_TRACK_METRICS:-false}"
    echo ""
    echo "=== Checkpoint ==="
    echo "  Save dir:      ${CFG_SAVE_DIR:-artifacts/inpainter}"
    echo "  Save interval: ${CFG_SAVE_INTERVAL:-10} epochs"
    echo "  Save predictions: ${CFG_SAVE_PREDICTIONS:-true}"
    [[ -n "$RESUME_FROM" ]] && echo "  Resume from:   $RESUME_FROM"
    echo ""
    echo "=== MLflow ==="
    echo "  Experiment:    ${CFG_EXPERIMENT:-inpainting}"
    echo "  Run name:      $RUN_NAME"
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
echo "  Partition:  $PARTITION | GPUs: $NUM_GPUS | Time: $TIME"
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
#SBATCH --gres=gpu:${NUM_GPUS}
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

echo "[JOB] Starting Inpainter Training (GPUs: ${NUM_GPUS})..."
if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node=${NUM_GPUS} -m lib.train_inpainter \\
        --config "${CONFIG_PATH}" \\
        --save_path "artifacts/${RUN_NAME}" \\
        --mlflow_run_name "${RUN_NAME}" \\
        ${MAE_CHECKPOINT_FLAG} \\
        ${RESUME_FLAG}
else
    python -m lib.train_inpainter \\
        --config "${CONFIG_PATH}" \\
        --save_path "artifacts/${RUN_NAME}" \\
        --mlflow_run_name "${RUN_NAME}" \\
        ${MAE_CHECKPOINT_FLAG} \\
        ${RESUME_FLAG}
fi

echo "[JOB] Finished."
EOF
