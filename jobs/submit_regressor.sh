#!/usr/bin/env bash
# Config-driven Regressor submission script
# Usage: ./submit_regressor.sh
#        DRY_RUN=1 ./submit_regressor.sh   # Show config without submitting
#
# Environment variables:
#   CONFIG_PATH  - Path to YAML config file (default: config/train_config.yaml)
#   RUN_NAME     - Run name (optional, defaults to config or auto-generated)
#   PARTITION    - SLURM partition (default: a100-daily)
#   TIME         - Job time limit (default: 12:00:00)
#   DRY_RUN      - Set to 1 to show config without submitting
#
# Optional CLI overrides (empty = use config value):
#   TRAIN_PATH, VAL_PATH, EPOCHS, LR, BATCH_SIZE, RESUME_FROM, etc.
#   NUM_GPUS     - Number of GPUs for DDP training (overrides config if set)

set -euo pipefail

# Required
CONFIG_PATH="${CONFIG_PATH:-config/train_config.yaml}"

# Job settings
RUN_NAME="${RUN_NAME:-}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"
NUM_GPUS="${NUM_GPUS:-}"  # Empty = read from config

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
CFG_HIDDEN_DIM=$(yaml_get "hidden_dim" "$CONFIG_PATH")
CFG_DROP_PATH=$(yaml_get "drop_path_rate" "$CONFIG_PATH")

# Training
CFG_EPOCHS=$(yaml_get "epochs" "$CONFIG_PATH")
CFG_LR=$(yaml_get "lr" "$CONFIG_PATH")
CFG_WEIGHT_DECAY=$(yaml_get "weight_decay" "$CONFIG_PATH")
CFG_WARMUP=$(yaml_get "warmup_epochs" "$CONFIG_PATH")
CFG_USE_SCHEDULER=$(yaml_get "use_scheduler" "$CONFIG_PATH")
CFG_SCHEDULER=$(yaml_get "scheduler" "$CONFIG_PATH")
CFG_MAX_LR=$(yaml_get "max_lr" "$CONFIG_PATH")
CFG_PCT_START=$(yaml_get "pct_start" "$CONFIG_PATH")
CFG_LR_PATIENCE=$(yaml_get "lr_patience" "$CONFIG_PATH")
CFG_LR_FACTOR=$(yaml_get "lr_factor" "$CONFIG_PATH")
CFG_LR_MIN=$(yaml_get "lr_min" "$CONFIG_PATH")
CFG_AMP=$(yaml_get "amp" "$CONFIG_PATH")
CFG_EMA=$(yaml_get "ema_decay" "$CONFIG_PATH")
CFG_CHANNEL_DROPOUT=$(yaml_get "channel_dropout_rate" "$CONFIG_PATH")
CFG_GRAD_CLIP=$(yaml_get "grad_clip" "$CONFIG_PATH")
CFG_PROFILE=$(yaml_get "profile" "$CONFIG_PATH")
CFG_COMPILE=$(yaml_get "compile" "$CONFIG_PATH")

# Loss balance
CFG_LOSS_BALANCE=$(yaml_get "loss_balance" "$CONFIG_PATH")

# Checkpoint
CFG_RESUME_FROM=$(yaml_get "resume_from" "$CONFIG_PATH")
CFG_SAVE_DIR=$(yaml_get "save_dir" "$CONFIG_PATH")
CFG_SAVE_INTERVAL=$(yaml_get "save_interval" "$CONFIG_PATH")
CFG_SAVE_ARTIFACTS=$(yaml_get "save_artifacts" "$CONFIG_PATH")

# MLflow
CFG_EXPERIMENT=$(yaml_get "experiment" "$CONFIG_PATH")
CFG_RUN_NAME=$(yaml_get "run_name" "$CONFIG_PATH")

# Export
CFG_ONNX=$(yaml_get "onnx" "$CONFIG_PATH")

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
        RUN_NAME="reg_$(date +%Y%m%d_%H%M%S)"
    fi
fi

# Use RESUME_FROM from env if set, otherwise from config
if [[ -z "$RESUME_FROM" && -n "$CFG_RESUME_FROM" && "$CFG_RESUME_FROM" != "null" ]]; then
    RESUME_FROM="$CFG_RESUME_FROM"
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

# Check save directory is writable
SAVE_DIR_EXPANDED="${CFG_SAVE_DIR/#\~/$HOME}"
SAVE_DIR_EXPANDED="${SAVE_DIR_EXPANDED:-artifacts}"
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

# Determine environment based on partition
if [[ "$PARTITION" == gh* ]]; then
    ENV_NAME="xec-ml-wl-gh"
else
    ENV_NAME="xec-ml-wl"
fi

LOG_DIR="$HOME/meghome/xec-ml-wl/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

# Show effective values (override if set, otherwise config)
EFF_EPOCHS="${EPOCHS:-${CFG_EPOCHS:-?}}"
EFF_BATCH="${BATCH_SIZE:-${CFG_BATCH:-?}}"
EFF_LR="${LR:-${CFG_LR:-?}}"
EFF_EXPERIMENT="${MLFLOW_EXPERIMENT:-${CFG_EXPERIMENT:-gamma_angle}}"

# Dry-run: show all config parameters
if [[ "$DRY_RUN" == "1" || "$DRY_RUN" == "true" ]]; then
    echo "============================================"
    echo "[DRY-RUN] Regressor Training Configuration"
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
    echo "  Npho branch:   ${CFG_NPHO_BRANCH:-relative_npho}"
    echo "  Time branch:   ${CFG_TIME_BRANCH:-relative_time}"
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
    echo "  Hidden dim:    ${CFG_HIDDEN_DIM:-256}"
    echo "  Drop path:     ${CFG_DROP_PATH:-0.0}"
    echo ""
    echo "=== Tasks ==="
    # Extract task-specific values (angle, energy, timing, uvwFI)
    # Note: This is a simplified extraction - actual task config is hierarchical
    echo "  (See config file for detailed task settings)"
    echo ""
    echo "=== Training ==="
    echo "  Epochs:        ${CFG_EPOCHS:-?}"
    echo "  Learning rate: ${CFG_LR:-?}"
    echo "  Use scheduler: ${CFG_USE_SCHEDULER:-true}"
    echo "  Scheduler:     ${CFG_SCHEDULER:-cosine}"
    # Show scheduler-specific parameters
    case "${CFG_SCHEDULER:-cosine}" in
        cosine)
            echo "  Warmup epochs: ${CFG_WARMUP:-2}"
            echo "  LR min:        ${CFG_LR_MIN:-1e-7}"
            ;;
        onecycle)
            echo "  Max LR:        ${CFG_MAX_LR:-${CFG_LR}}"
            echo "  Pct start:     ${CFG_PCT_START:-0.3}"
            ;;
        plateau)
            echo "  LR patience:   ${CFG_LR_PATIENCE:-5}"
            echo "  LR factor:     ${CFG_LR_FACTOR:-0.5}"
            echo "  LR min:        ${CFG_LR_MIN:-1e-7}"
            ;;
        # none: no scheduler-specific params
    esac
    echo "  Weight decay:  ${CFG_WEIGHT_DECAY:-?}"
    echo "  Grad clip:     ${CFG_GRAD_CLIP:-1.0}"
    echo "  Ch dropout:    ${CFG_CHANNEL_DROPOUT:-0.1}"
    echo "  AMP:           ${CFG_AMP:-true}"
    if [[ "$PARTITION" == gh* && "${CFG_COMPILE:-false}" != "false" && "${CFG_COMPILE:-false}" != "none" ]]; then
        echo "  Compile:       ${CFG_COMPILE:-false} (will be auto-disabled on ARM/GH nodes)"
    else
        echo "  Compile:       ${CFG_COMPILE:-false}"
    fi
    echo "  EMA decay:     ${CFG_EMA:-0.999}"
    echo "  Profile:       ${CFG_PROFILE:-false}"
    echo "  Loss balance:  ${CFG_LOSS_BALANCE:-manual}"
    echo ""
    echo "=== Reweighting ==="
    echo "  (See config file for detailed reweighting settings)"
    echo ""
    echo "=== Checkpoint ==="
    echo "  Save dir:      ${CFG_SAVE_DIR:-artifacts}"
    echo "  Save interval: ${CFG_SAVE_INTERVAL:-10} epochs"
    echo "  Save artifacts: ${CFG_SAVE_ARTIFACTS:-true}"
    [[ -n "$RESUME_FROM" ]] && echo "  Resume from:   $RESUME_FROM"
    echo ""
    echo "=== Export ==="
    echo "  ONNX:          ${CFG_ONNX:-null}"
    echo ""
    echo "=== MLflow ==="
    echo "  Experiment:    ${CFG_EXPERIMENT:-gamma_angle}"
    echo "  Run name:      $RUN_NAME"
    echo ""

    # Show CLI overrides if any
    CLI_ARGS=""
    [ -n "$TRAIN_PATH" ] && CLI_ARGS+=" --train_path"
    [ -n "$VAL_PATH" ] && CLI_ARGS+=" --val_path"
    [ -n "$EPOCHS" ] && CLI_ARGS+=" --epochs=$EPOCHS"
    [ -n "$LR" ] && CLI_ARGS+=" --lr=$LR"
    [ -n "$BATCH_SIZE" ] && CLI_ARGS+=" --batch_size=$BATCH_SIZE"
    [ -n "$WEIGHT_DECAY" ] && CLI_ARGS+=" --weight_decay"
    [ -n "$WARMUP_EPOCHS" ] && CLI_ARGS+=" --warmup_epochs"
    [ -n "$EMA_DECAY" ] && CLI_ARGS+=" --ema_decay"
    [ -n "$GRAD_CLIP" ] && CLI_ARGS+=" --grad_clip"
    [ -n "$TASKS" ] && CLI_ARGS+=" --tasks"

    if [ -n "$CLI_ARGS" ]; then
        echo "=== CLI Overrides ==="
        echo "  $CLI_ARGS"
        echo ""
    fi

    echo "============================================"
    echo "[DRY-RUN] No job submitted. Remove DRY_RUN=1 to submit."
    echo "============================================"
    exit 0
fi

# Normal submission
echo "[SUBMIT] Regressor Training"
echo "  Config:     $CONFIG_PATH"
echo "  Run:        $RUN_NAME"
echo "  Experiment: $EFF_EXPERIMENT"
echo "  Epochs:     $EFF_EPOCHS | Batch: $EFF_BATCH | LR: $EFF_LR"
echo "  Partition:  $PARTITION | GPUs: $NUM_GPUS | Time: $TIME"

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
#SBATCH --gres=gpu:${NUM_GPUS}
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

echo "[JOB] Starting Regressor training (GPUs: ${NUM_GPUS})..."
if [ "${NUM_GPUS}" -gt 1 ]; then
    torchrun --nproc_per_node=${NUM_GPUS} -m lib.train_regressor \\
        --config "${CONFIG_PATH}" \\
        --run_name "${RUN_NAME}" ${CLI_ARGS}
else
    python -m lib.train_regressor \\
        --config "${CONFIG_PATH}" \\
        --run_name "${RUN_NAME}" ${CLI_ARGS}
fi

echo "[JOB] Finished."
EOF
