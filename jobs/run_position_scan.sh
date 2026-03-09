#!/bin/bash
# =============================================================================
# Position Regressor Hyperparameter Scan
# =============================================================================
# Usage:
#   ./jobs/run_position_scan.sh              # Submit all new steps (8-11)
#   ./jobs/run_position_scan.sh 8 9          # Submit specific steps
#   DRY_RUN=1 ./jobs/run_position_scan.sh    # Preview without submitting
#
# Steps:
#   1   - (done) Baseline: 3b settings, loss_beta=1.0
#   2   - (done) 4a training settings (lr=3e-4, warmup=5, grad_clip=1.0)
#   3   - (done) s2 + loss_beta=0.1 (best resolution: 2.50 cm)
#   4   - (done) s2 + batch_size=2048, grad_accum=2 (best val_loss)
#   5   - (done) s4 + gaussian_nll (failed)
#   6   - (done) s4 + L1 loss
#   7   - (done) s4 + MSE (L2) loss
#   8   - s4 + 100 epochs (still improving at ep50)
#   9   - s3 + batch=2048 + 100 epochs (best resolution + more training)
#  10   - s8 + train_large dataset
#  11   - s8 + OneCycle scheduler (max_lr=6e-4)
#  12   - Full-data from scratch (train_max, 4 GPUs, 100ep, beta=1.0)
#  13   - Full-data from scratch (train_max, 4 GPUs, 100ep, beta=0.1)
#
# All steps use train_middle (except 10,12,13), 1 GPU (except 12,13).
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
STEP_CONFIG[8]="step8_100ep.yaml"
STEP_CONFIG[9]="step9_beta01_100ep.yaml"
STEP_CONFIG[10]="step10_largedata.yaml"
STEP_CONFIG[11]="step11_onecycle.yaml"
STEP_CONFIG[12]="step12_fulldata.yaml"
STEP_CONFIG[13]="step13_fulldata_beta01.yaml"

STEP_NAME[2]="pos_scan_s2_4a"
STEP_NAME[3]="pos_scan_s3_beta01"
STEP_NAME[4]="pos_scan_s4_batchx2"
STEP_NAME[5]="pos_scan_s5_gnll"
STEP_NAME[6]="pos_scan_s6_l1"
STEP_NAME[7]="pos_scan_s7_mse"
STEP_NAME[8]="pos_scan_s8_100ep"
STEP_NAME[9]="pos_scan_s9_beta01_100ep"
STEP_NAME[10]="pos_scan_s10_largedata"
STEP_NAME[11]="pos_scan_s11_onecycle"
STEP_NAME[12]="pos_scan_s12_fulldata"
STEP_NAME[13]="pos_scan_s13_fulldata_beta01"

if [ $# -eq 0 ]; then
    STEPS=("8" "9" "10" "11")
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
        echo "  Valid steps: 2-13"
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
