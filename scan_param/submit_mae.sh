#!/usr/bin/env bash
# Usage: ./submit_mae.sh [RUN_NAME] [EPOCHS] [BATCH] [MASK_RATIO] [RESUME_FROM] [PARTITION] [TIME]

set -euo pipefail

RUN_NAME="${RUN_NAME:-mae_default}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-1024}"
CHUNK_SIZE="${CHUNK_SIZE:-256000}"
MASK_RATIO="${MASK_RATIO:-0.6}"
RESUME_FROM="${RESUME_FROM:-}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-23:00:00}"
NPHO_SCALE="${NPHO_SCALE:-1.0}"
NPHO_SCALE2="${NPHO_SCALE2:-1.0}"
TIME_SCALE="${TIME_SCALE:-2.32e6}"
TIME_SHIFT="${TIME_SHIFT:-0.0}"
SENTINEL_VALUE="${SENTINEL_VALUE:--5.0}"
# ROOT_PATH="${ROOT_PATH:-~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/single_run}"
TRAIN_PATH="${TRAIN_PATH:-~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root}"
VAL_PATH="${VAL_PATH:-~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-gamma_mae}"

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

echo "[SUBMIT] MAE Run: $RUN_NAME | Exp: $MLFLOW_EXPERIMENT | Mask: $MASK_RATIO"

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
module load anaconda/2024.08 2>/dev/null || true

if command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
else
    # Fallback paths
    export PATH="/opt/psi/Programming/anaconda/2024.08/conda/bin:\$PATH"
    eval "\$(conda shell.bash hook)"
fi

echo "[JOB] Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"

# Create artifacts dir
mkdir -p artifacts/${RUN_NAME}

echo "[JOB] Starting MAE Pre-training with Batch=${BATCH} Chunk=${CHUNK_SIZE}..."
python -m lib.train_mae \\
    --train_root "${TRAIN_PATH}" \\
    --val_root "${VAL_PATH}" \\
    --save_path "artifacts/${RUN_NAME}" \\
    --epochs ${EPOCHS} \\
    --batch_size ${BATCH} \\
    --chunksize ${CHUNK_SIZE} \\
    --mask_ratio ${MASK_RATIO} \\
    --npho_scale ${NPHO_SCALE} \\
    --npho_scale2 ${NPHO_SCALE2} \\
    --time_scale ${TIME_SCALE} \\
    --time_shift ${TIME_SHIFT} \\
    --sentinel_value ${SENTINEL_VALUE} \\
    --outer_mode "finegrid" \\
    --outer_fine_pool 3 3 \\
    --mlflow_experiment "${MLFLOW_EXPERIMENT}" \\
    --mlflow_run_name "${RUN_NAME}" \\
    $( [[ -n "${RESUME_FROM}" ]] && echo "--resume_from ${RESUME_FROM}" )

echo "[JOB] Finished."
EOF