#!/bin/bash
# =============================================================================
# Angle Regressor Hyperparameter Scan
# =============================================================================
# Usage:
#   ./jobs/run_angle_scan.sh              # Submit all steps (2,3)
#   ./jobs/run_angle_scan.sh 2            # Submit only step 2
#   DRY_RUN=1 ./jobs/run_angle_scan.sh    # Preview without submitting
#
# Steps:
#   1   - (done) Baseline: 3b settings — grad_clip=0.1 cripples learning
#   2   - 4a training settings (lr=3e-4, warmup=5, grad_clip=1.0)
#   3   - s2 + loss_beta=0.1 (quadratic for fine residuals)
#
# All steps use train_middle, 1 GPU, 50 epochs.
# Compare in MLflow experiment: gamma_angle
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"

SCAN_CONFIG_DIR="config/reg/ang_scan"

declare -A STEP_CONFIG
declare -A STEP_NAME

STEP_CONFIG[2]="step2_4a_settings.yaml"
STEP_CONFIG[3]="step3_beta01.yaml"

STEP_NAME[2]="ang_scan_s2_4a"
STEP_NAME[3]="ang_scan_s3_beta01"

if [ $# -eq 0 ]; then
    STEPS=("2" "3")
    echo "[SCAN] No steps specified. Submitting all: ${STEPS[*]}"
    echo ""
else
    STEPS=("$@")
fi

echo "============================================"
echo "Angle Regressor Hyperparameter Scan"
echo "============================================"
echo "Partition:  $PARTITION"
echo "Time limit: $TIME"
echo "Steps:      ${STEPS[*]}"
echo "Dry run:    $DRY_RUN"
echo "============================================"
echo ""

SUBMITTED=0
for STEP in "${STEPS[@]}"; do
    CONFIG="${STEP_CONFIG[$STEP]:-}"
    NAME="${STEP_NAME[$STEP]:-}"

    if [ -z "$CONFIG" ]; then
        echo "[ERROR] Unknown step: $STEP"
        echo "  Valid steps: 2, 3"
        continue
    fi

    CONFIG_PATH="${SCAN_CONFIG_DIR}/${CONFIG}"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "[ERROR] Config not found: $CONFIG_PATH"
        continue
    fi

    echo "--- Step $STEP: $NAME ---"
    echo "  Config: $CONFIG_PATH"

    export CONFIG_PATH="$CONFIG_PATH"
    export RUN_NAME="$NAME"
    export PARTITION="$PARTITION"
    export TIME="$TIME"
    export DRY_RUN="$DRY_RUN"

    unset TRAIN_PATH VAL_PATH EPOCHS LR BATCH_SIZE WEIGHT_DECAY
    unset WARMUP_EPOCHS EMA_DECAY GRAD_CLIP CHANNEL_DROPOUT_RATE
    unset TASKS RESUME_FROM SAVE_DIR MLFLOW_EXPERIMENT ONNX
    unset LOSS_BALANCE OUTER_MODE OUTER_FINE_POOL HIDDEN_DIM DROP_PATH_RATE
    unset NUM_GPUS

    ./jobs/submit_regressor.sh

    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 1
done

echo "============================================"
echo "[SCAN] Submitted $SUBMITTED / ${#STEPS[@]} jobs"
echo ""
echo "Monitor in MLflow:"
echo "  experiment: gamma_angle"
echo "  Compare val/cos and val/l1 across runs"
echo "============================================"
