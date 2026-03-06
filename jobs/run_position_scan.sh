#!/bin/bash
# =============================================================================
# Position Regressor Hyperparameter Scan
# =============================================================================
# Usage:
#   ./jobs/run_position_scan.sh              # Submit all steps (2,3,4)
#   ./jobs/run_position_scan.sh 2 3          # Submit only step 2 and 3
#   DRY_RUN=1 ./jobs/run_position_scan.sh    # Preview without submitting
#
# Steps:
#   1   - (done) Baseline: 3b settings, loss_beta=1.0
#   2   - (done) 4a training settings (lr=3e-4, warmup=5, grad_clip=1.0)
#   3   - (done) s2 + loss_beta=0.1 (quadratic for fine residuals)
#   4   - (done) s2 + batch_size=2048, grad_accum=2 (use VRAM headroom)
#   5   - s4 + gaussian_nll (beta-NLL=0.5, shared uncertainty)
#   6   - s4 + L1 loss
#   7   - s4 + MSE (L2) loss
#
# All steps use train_middle, 1 GPU, 50 epochs.
# Compare in MLflow experiment: gamma_position
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${PARTITION:-gh-daily}"
TIME="${TIME:-1-00:00:00}"
DRY_RUN="${DRY_RUN:-0}"

SCAN_CONFIG_DIR="config/reg/pos_scan"

declare -A STEP_CONFIG
declare -A STEP_NAME

STEP_CONFIG[2]="step2_4a_settings.yaml"
STEP_CONFIG[3]="step3_beta01.yaml"
STEP_CONFIG[4]="step4_batchx2.yaml"
STEP_CONFIG[5]="step5_gnll.yaml"
STEP_CONFIG[6]="step6_l1.yaml"
STEP_CONFIG[7]="step7_mse.yaml"

STEP_NAME[2]="pos_scan_s2_4a"
STEP_NAME[3]="pos_scan_s3_beta01"
STEP_NAME[4]="pos_scan_s4_batchx2"
STEP_NAME[5]="pos_scan_s5_gnll"
STEP_NAME[6]="pos_scan_s6_l1"
STEP_NAME[7]="pos_scan_s7_mse"

if [ $# -eq 0 ]; then
    STEPS=("5" "6" "7")
    echo "[SCAN] No steps specified. Submitting all: ${STEPS[*]}"
    echo ""
else
    STEPS=("$@")
fi

echo "============================================"
echo "Position Regressor Hyperparameter Scan"
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
        echo "  Valid steps: 2, 3, 4, 5, 6, 7"
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
echo "  experiment: gamma_position"
echo "  Compare uvw_dist_68pct across runs"
echo "============================================"
