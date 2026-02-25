#!/bin/bash
# =============================================================================
# Energy Regressor Hyperparameter Scan
# =============================================================================
# Usage:
#   ./jobs/run_energy_scan.sh              # Submit all steps (1,2,3a,3b)
#   ./jobs/run_energy_scan.sh 1 2          # Submit only step 1 and 2
#   ./jobs/run_energy_scan.sh 4a 4b 5 6    # Submit later steps
#   DRY_RUN=1 ./jobs/run_energy_scan.sh    # Preview without submitting
#
# Steps:
#   1   - Baseline (current config on train_middle)
#   2   - +EMA (ema_decay=0.999)
#   3a  - +Medium model (encoder_dim=768, 1 layer)
#   3b  - +Large model (encoder_dim=1024, 2 layers)
#   2r  - Resume step 2 (cosine warm restart, +50 epochs)
#   3ar - Resume step 3a (cosine warm restart, +50 epochs)
#   3br - Resume step 3b (cosine warm restart, +50 epochs)
#   4a  - +LR=3e-4, warmup=5, grad_clip=1.0  (model=3b winner)
#   4b  - +LR=5e-4, warmup=5, grad_clip=1.0  (model=3b winner)
#   5   - +log_transform for energy          (baseline=3b)
#   5b  - +grad_clip=1.0 (isolate grad_clip, baseline=3b)
#   4ar - Resume step 4a (cosine warm restart, +50 epochs)
#   5r  - Resume step 5 (cosine warm restart, +50 epochs)
#   5br - Resume step 5b (cosine warm restart, +50 epochs)
#   6   - +energy reweighting                (baseline=3b)
#
# NOTE: Steps 2r/3ar/3br require resume_from to be set to the checkpoint path.
#       Steps 4ar/5r/5br require resume_from to be set to the checkpoint path.
#       Steps 5-6 have TODO markers for training params (lr/warmup/grad_clip).
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-12:00:00}"
DRY_RUN="${DRY_RUN:-0}"

SCAN_CONFIG_DIR="config/reg/scan"

# Define all steps: (step_id config_file run_name_suffix)
declare -A STEP_CONFIG
declare -A STEP_NAME
STEP_CONFIG[1]="step1_baseline.yaml"
STEP_CONFIG[2]="step2_ema.yaml"
STEP_CONFIG[3a]="step3a_model_mid.yaml"
STEP_CONFIG[3b]="step3b_model_large.yaml"
STEP_CONFIG[2r]="step2_resume.yaml"
STEP_CONFIG[3ar]="step3a_resume.yaml"
STEP_CONFIG[3br]="step3b_resume.yaml"
STEP_CONFIG[4a]="step4a_lr3e-4.yaml"
STEP_CONFIG[4b]="step4b_lr5e-4.yaml"
STEP_CONFIG[5]="step5_logtransform.yaml"
STEP_CONFIG[5b]="step5b_gradclip.yaml"
STEP_CONFIG[4ar]="step4a_resume.yaml"
STEP_CONFIG[5r]="step5_resume.yaml"
STEP_CONFIG[5br]="step5b_resume.yaml"
STEP_CONFIG[6]="step6_reweight.yaml"

STEP_NAME[1]="scan_s1_baseline"
STEP_NAME[2]="scan_s2_ema"
STEP_NAME[3a]="scan_s3a_model768"
STEP_NAME[3b]="scan_s3b_model1024"
STEP_NAME[2r]="scan_s2_ema_resume"
STEP_NAME[3ar]="scan_s3a_model768_resume"
STEP_NAME[3br]="scan_s3b_model1024_resume"
STEP_NAME[4a]="scan_s4a_lr3e-4"
STEP_NAME[4b]="scan_s4b_lr5e-4"
STEP_NAME[5]="scan_s5_logtransform"
STEP_NAME[5b]="scan_s5b_gradclip1"
STEP_NAME[4ar]="scan_s4a_lr3e-4_resume"
STEP_NAME[5r]="scan_s5_logtransform_resume"
STEP_NAME[5br]="scan_s5b_gradclip1_resume"
STEP_NAME[6]="scan_s6_reweight"

# Default: submit steps 1, 2, 3a, 3b (independent, can run in parallel)
if [ $# -eq 0 ]; then
    STEPS=("1" "2" "3a" "3b")
    echo "[SCAN] No steps specified. Submitting initial batch: ${STEPS[*]}"
    echo "[SCAN] After results, update step4-6 configs and run:"
    echo "       ./jobs/run_energy_scan.sh 4a 4b"
    echo "       ./jobs/run_energy_scan.sh 5"
    echo "       ./jobs/run_energy_scan.sh 6"
    echo ""
else
    STEPS=("$@")
fi

echo "============================================"
echo "Energy Regressor Hyperparameter Scan"
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
        echo "  Valid steps: 1, 2, 3a, 3b, 2r, 3ar, 3br, 4a, 4b, 4ar, 5, 5b, 5r, 5br, 6"
        continue
    fi

    CONFIG_PATH="${SCAN_CONFIG_DIR}/${CONFIG}"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "[ERROR] Config not found: $CONFIG_PATH"
        continue
    fi

    echo "--- Step $STEP: $NAME ---"
    echo "  Config: $CONFIG_PATH"

    # Check for TODO markers (steps 5-6 need manual updates)
    if grep -q "# TODO:" "$CONFIG_PATH"; then
        echo "  [WARN] Config has TODO markers - verify params match previous winner"
    fi

    # Check for FIXME markers (resume configs need checkpoint paths)
    if grep -q "FIXME" "$CONFIG_PATH"; then
        echo "  [ERROR] Config has FIXME - set resume_from checkpoint path before running"
        continue
    fi

    export CONFIG_PATH="$CONFIG_PATH"
    export RUN_NAME="$NAME"
    export PARTITION="$PARTITION"
    export TIME="$TIME"
    export DRY_RUN="$DRY_RUN"

    # Clear any override env vars (use config values only)
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
echo "  experiment: gamma_energy"
echo "  Compare val/l1 across runs"
echo "============================================"
