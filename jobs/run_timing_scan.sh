#!/bin/bash
# =============================================================================
# Timing Regressor Hyperparameter Scan
# =============================================================================
# Usage:
#   ./jobs/run_timing_scan.sh              # Submit all new steps (8-13)
#   ./jobs/run_timing_scan.sh 8 9          # Submit specific steps
#   DRY_RUN=1 ./jobs/run_timing_scan.sh    # Preview without submitting
#
# Steps:
#   1   - (done) Baseline: 3b settings — OVERFITTING (val loss goes up)
#   2   - (done) 4a settings + weight_decay=1e-3 + channel_dropout=0.05
#   3   - (done) s2 + smaller model (enc=512, 1 layer, ffn=2048)
#   4   - (done) s3 + drop_path=0.2 (more stochastic depth)
#   5   - (done) s4 + npho_threshold=10 + sentinel_time=-5.0 (BEST)
#   6   - (done) s5 + npho_threshold=3 + sentinel_time=-10.0
#   7   - (done) s6 + mse loss
#   8   - s5 data + big model (enc=1024, 2 fusion layers), bs=2048
#   9   - s5 + lr=1e-4, bs=2048
#  10   - s5 + no channel dropout, bs=2048
#  11   - Resume s5, lr=1.5e-4, 100 epochs, bs=2048
#  12   - s5 + weight_decay=1e-4, bs=2048
#  13   - s5 + bs=2048 only (control)
#  14   - s5 + train_large dataset, bs=2048
#
# All steps use train_middle (except 14), 1 GPU.
# Compare in MLflow experiment: gamma_timing
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${PARTITION:-gh-daily}"
TIME="${TIME:-1-00:00:00}"
DRY_RUN="${DRY_RUN:-0}"

SCAN_CONFIG_DIR="config/reg/tim_scan"

declare -A STEP_CONFIG
declare -A STEP_NAME

STEP_CONFIG[2]="step2_regularize.yaml"
STEP_CONFIG[3]="step3_smallmodel.yaml"
STEP_CONFIG[4]="step4_droppath.yaml"
STEP_CONFIG[5]="step5_threshold.yaml"
STEP_CONFIG[6]="step6_lower_threshold.yaml"
STEP_CONFIG[7]="step7_mse_loss.yaml"
STEP_CONFIG[8]="step8_bigmodel.yaml"
STEP_CONFIG[9]="step9_lr1e4.yaml"
STEP_CONFIG[10]="step10_nodropout.yaml"
STEP_CONFIG[11]="step11_resume_s5.yaml"
STEP_CONFIG[12]="step12_wd1e4.yaml"
STEP_CONFIG[13]="step13_bs2048.yaml"
STEP_CONFIG[14]="step14_largedata.yaml"

STEP_NAME[2]="tim_scan_s2_regularize"
STEP_NAME[3]="tim_scan_s3_smallmodel"
STEP_NAME[4]="tim_scan_s4_droppath"
STEP_NAME[5]="tim_scan_s5_threshold"
STEP_NAME[6]="tim_scan_s6_lower_threshold"
STEP_NAME[7]="tim_scan_s7_mse_loss"
STEP_NAME[8]="tim_scan_s8_bigmodel"
STEP_NAME[9]="tim_scan_s9_lr1e4"
STEP_NAME[10]="tim_scan_s10_nodropout"
STEP_NAME[11]="tim_scan_s11_resume_s5"
STEP_NAME[12]="tim_scan_s12_wd1e4"
STEP_NAME[13]="tim_scan_s13_bs2048"
STEP_NAME[14]="tim_scan_s14_largedata"

if [ $# -eq 0 ]; then
    STEPS=("8" "9" "10" "11" "12" "13" "14")
    echo "[SCAN] No steps specified. Submitting all: ${STEPS[*]}"
    echo ""
else
    STEPS=("$@")
fi

echo "============================================"
echo "Timing Regressor Hyperparameter Scan"
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
        echo "  Valid steps: 2-14"
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
echo "  experiment: gamma_timing"
echo "  Compare timing_res_68pct across runs"
echo "============================================"
