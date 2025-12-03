#!/usr/bin/env bash
# Usage: ./submit_job.sh [RUN_NAME] [MODEL] [EPOCHS] [REWEIGHT_MODE] [LOSSTYPE] [LR] [BATCH] [RESUME_FROM] [PARTITION] [TIME]
#
# Optional Env Vars:
#   WEIGHT_DECAY (default 1e-4)
#   DROP_PATH    (default 0.0)
#   SCHEDULER    (default -1 for Cosine, set to 1 for Constant)
#   TIME_SCALE   (default 1e-7)

set -euo pipefail

RUN_NAME="${1:-test_run}"
MODEL="${2:-simple}"
EPOCHS="${3:-50}"
REWEIGHT_MODE="${4:-none}"
LOSSTYPE="${5:-smooth_l1}"
LR="${6:-3e-4}"
BATCH="${7:-1024}"
RESUME_FROM="${8:-}"
PARTITION="${9:-a100-daily}"
TIME="${10:-12:00:00}"

# Read optional env vars or set defaults
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
DROP_PATH="${DROP_PATH:-0.0}"
SCHEDULER="${SCHEDULER:--1}"
TIME_SCALE="${TIME_SCALE:-1e-7}"
TIME_SHIFT="${TIME_SHIFT:-0.0}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-2}"
NPHO_SCALE="${NPHO_SCALE:-1.0}"
ONNX="${ONNX:-}"
MLFLOW_EXPERIMENT="${MLFLOW_EXPERIMENT:-gamma_angle}"

CHUNK_SIZE="${CHUNK_SIZE:-4000}"
TREE_NAME="${TREE_NAME:-tree}"
ROOT_PATH="${ROOT_PATH:-~/meghome/xec-ml-wl/data/MCGammaAngle_0-49.root}"


# 1. Determine Environment based on Partition
if [[ "$PARTITION" == gh* ]]; then
    ENV_NAME="xec-ml-wl-gh"
else
    ENV_NAME="xec-ml-wl"
fi

LOG_DIR="slurm_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/${RUN_NAME}_%j.out"

echo "[SUBMIT] Run: $RUN_NAME | Model: $MODEL | WD: $WEIGHT_DECAY | Drop: $DROP_PATH"

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
module load anaconda/2024.08 2>/dev/null || true
eval "\$(/opt/psi/Programming/anaconda/2024.08/conda/bin/conda shell.bash hook)"

echo "[JOB] Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Changed directory to: \$(pwd)"

export MLFLOW_TRACKING_URI="file://\$(pwd)/mlruns"

echo "[JOB] Running training script for model: ${MODEL}..."
python scan_param/run_training_cli.py \
    --root "${ROOT_PATH}" \
    --tree "${TREE_NAME}" \
    --epochs ${EPOCHS} \
    --batch ${BATCH} \
    --chunksize ${CHUNK_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --drop_path_rate ${DROP_PATH} \
    --time_shift ${TIME_SHIFT} \
    --time_scale ${TIME_SCALE} \
    --use_scheduler ${SCHEDULER} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --onnx "${ONNX}" \
    --mlflow_experiment "${MLFLOW_EXPERIMENT}" \
    --NphoScale ${NPHO_SCALE} \
    --reweight_mode "${REWEIGHT_MODE}" \
    --loss_type "${LOSSTYPE}" \
    --run_name "${RUN_NAME}" \
    --model "${MODEL}" \
    --resume_from "${RESUME_FROM}"

echo "[JOB] Finished."
EOF