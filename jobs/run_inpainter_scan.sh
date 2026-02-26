#!/bin/bash
# =============================================================================
# Inpainter Hyperparameter Scan
# =============================================================================
# Usage:
#   ./jobs/run_inpainter_scan.sh              # Submit all initial steps (1-6)
#   ./jobs/run_inpainter_scan.sh 1 2          # Submit only step 1 and 2
#   DRY_RUN=1 ./jobs/run_inpainter_scan.sh    # Preview without submitting
#
# Steps (all independent, can run in parallel):
#   1   - Baseline: log1p, mask=0.10, no flat, no npho_wt
#   2   - +flat masking (CDF-based, isolate flat mask effect)
#   3   - +npho_loss_weight (alpha=0.5, isolate loss weight effect)
#   4   - +flat masking + npho_loss_weight (combination)
#   5   - sqrt normalization (no flat, no npho_wt — isolate scheme)
#   6   - +mask_ratio=0.15 (harder task, longer-range correlations)
#
# All steps use train_middle (~40 runs), 1 GPU, 50 epochs.
# Compare val loss in MLflow experiment: inpainting_scan
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"

SCAN_CONFIG_DIR="config/inp/scan"

declare -A STEP_CONFIG
declare -A STEP_NAME

STEP_CONFIG[1]="step1_baseline.yaml"
STEP_CONFIG[2]="step2_flatmask.yaml"
STEP_CONFIG[3]="step3_nphowt.yaml"
STEP_CONFIG[4]="step4_flat_nphowt.yaml"
STEP_CONFIG[5]="step5_sqrt.yaml"
STEP_CONFIG[6]="step6_mask015.yaml"

STEP_NAME[1]="inp_scan_s1_baseline"
STEP_NAME[2]="inp_scan_s2_flatmask"
STEP_NAME[3]="inp_scan_s3_nphowt"
STEP_NAME[4]="inp_scan_s4_flat_nphowt"
STEP_NAME[5]="inp_scan_s5_sqrt"
STEP_NAME[6]="inp_scan_s6_mask015"

# Default: submit all steps (independent, can run in parallel)
if [ $# -eq 0 ]; then
    STEPS=("1" "2" "3" "4" "5" "6")
    echo "[SCAN] No steps specified. Submitting all: ${STEPS[*]}"
    echo ""
else
    STEPS=("$@")
fi

echo "============================================"
echo "Inpainter Hyperparameter Scan"
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
        echo "  Valid steps: 1, 2, 3, 4, 5, 6"
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

    # Clear any override env vars (use config values only)
    unset RESUME_FROM MAE_CHECKPOINT NUM_GPUS

    ./jobs/submit_inpainter.sh

    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 1
done

echo "============================================"
echo "[SCAN] Submitted $SUBMITTED / ${#STEPS[@]} jobs"
echo ""
echo "Monitor in MLflow:"
echo "  experiment: inpainting_scan"
echo "  Compare val_loss across runs"
echo "============================================"
