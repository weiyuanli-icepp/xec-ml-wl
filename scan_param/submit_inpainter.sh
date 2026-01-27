#!/usr/bin/env bash
# Usage: ./submit_inpainter.sh
# Environment variables can be set before calling this script

set -euo pipefail

RUN_NAME="${RUN_NAME:-inpainter_default}"
EPOCHS="${EPOCHS:-50}"
BATCH="${BATCH:-1024}"
CHUNK_SIZE="${CHUNK_SIZE:-256000}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
MASK_RATIO="${MASK_RATIO:-0.07}"
RESUME_FROM="${RESUME_FROM:-}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
CONFIG_PATH="${CONFIG_PATH:-config/inpainter_config.yaml}"

# Normalization
NPHO_SCALE="${NPHO_SCALE:-1000}"
NPHO_SCALE2="${NPHO_SCALE2:-4.08}"
TIME_SCALE="${TIME_SCALE:-1.14e-7}"
TIME_SHIFT="${TIME_SHIFT:--0.46}"
SENTINEL_VALUE="${SENTINEL_VALUE:--1.0}"

# Loss
LOSS_FN="${LOSS_FN:-smooth_l1}"
NPHO_WEIGHT="${NPHO_WEIGHT:-1.0}"
TIME_WEIGHT="${TIME_WEIGHT:-1.0}"

# Learning rate
LR="${LR:-2e-4}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
LR_MIN="${LR_MIN:-1e-6}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-0}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"

# Model
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
MAE_CHECKPOINT="${MAE_CHECKPOINT:-}"

# Paths
TRAIN_PATH="${TRAIN_PATH:-~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root}"
VAL_PATH="${VAL_PATH:-~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-inpainting}"

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

# Build optional flags
LOSS_FN_FLAG=""
if [[ -n "${LOSS_FN}" ]]; then LOSS_FN_FLAG="--loss_fn ${LOSS_FN}"; fi

NPHO_WEIGHT_FLAG=""
if [[ -n "${NPHO_WEIGHT}" ]]; then NPHO_WEIGHT_FLAG="--npho_weight ${NPHO_WEIGHT}"; fi

TIME_WEIGHT_FLAG=""
if [[ -n "${TIME_WEIGHT}" ]]; then TIME_WEIGHT_FLAG="--time_weight ${TIME_WEIGHT}"; fi

LR_FLAG=""
if [[ -n "${LR}" ]]; then LR_FLAG="--lr ${LR}"; fi

LR_SCHEDULER_FLAG=""
if [[ -n "${LR_SCHEDULER}" ]]; then LR_SCHEDULER_FLAG="--lr_scheduler ${LR_SCHEDULER}"; fi

LR_MIN_FLAG=""
if [[ -n "${LR_MIN}" ]]; then LR_MIN_FLAG="--lr_min ${LR_MIN}"; fi

WARMUP_FLAG=""
if [[ -n "${WARMUP_EPOCHS}" ]] && [[ "${WARMUP_EPOCHS}" != "0" ]]; then WARMUP_FLAG="--warmup_epochs ${WARMUP_EPOCHS}"; fi

FREEZE_FLAG=""
case "${FREEZE_ENCODER}" in
    true|True|TRUE|1|yes|YES)
        FREEZE_FLAG="--freeze_encoder"
        ;;
    *)
        FREEZE_FLAG="--finetune_encoder"
        ;;
esac

MAE_CHECKPOINT_FLAG=""
if [[ -n "${MAE_CHECKPOINT}" ]]; then MAE_CHECKPOINT_FLAG="--mae_checkpoint ${MAE_CHECKPOINT}"; fi

echo "[SUBMIT] Inpainter Run: $RUN_NAME | Exp: $MLFLOW_EXPERIMENT | Mask: $MASK_RATIO | Config: $CONFIG_PATH"

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

# Use SQLite backend (recommended over deprecated file-based backend)
export MLFLOW_TRACKING_URI="sqlite:///\$(pwd)/mlruns.db"

# Create artifacts dir
mkdir -p artifacts/${RUN_NAME}

echo "[JOB] Starting Inpainter Training with Batch=${BATCH} Chunk=${CHUNK_SIZE} Mask=${MASK_RATIO}..."
python -m lib.train_inpainter \\
    --config "${CONFIG_PATH}" \\
    --train_root "${TRAIN_PATH}" \\
    --val_root "${VAL_PATH}" \\
    --save_path "artifacts/${RUN_NAME}" \\
    --epochs ${EPOCHS} \\
    --batch_size ${BATCH} \\
    --chunksize ${CHUNK_SIZE} \\
    --grad_accum_steps ${GRAD_ACCUM_STEPS} \\
    --mask_ratio ${MASK_RATIO} \\
    --npho_scale ${NPHO_SCALE} \\
    --npho_scale2 ${NPHO_SCALE2} \\
    --time_scale ${TIME_SCALE} \\
    --time_shift ${TIME_SHIFT} \\
    --sentinel_value ${SENTINEL_VALUE} \\
    --weight_decay ${WEIGHT_DECAY} \\
    --grad_clip ${GRAD_CLIP} \\
    --outer_mode "finegrid" \\
    --outer_fine_pool 3 3 \\
    --mlflow_experiment "${MLFLOW_EXPERIMENT}" \\
    --mlflow_run_name "${RUN_NAME}" \\
    ${LOSS_FN_FLAG} \\
    ${NPHO_WEIGHT_FLAG} \\
    ${TIME_WEIGHT_FLAG} \\
    ${LR_FLAG} \\
    ${LR_SCHEDULER_FLAG} \\
    ${LR_MIN_FLAG} \\
    ${WARMUP_FLAG} \\
    ${FREEZE_FLAG} \\
    ${MAE_CHECKPOINT_FLAG} \\
    $( [[ -n "${RESUME_FROM}" ]] && echo "--resume_from ${RESUME_FROM}" )

echo "[JOB] Finished."
EOF
