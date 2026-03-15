#!/bin/bash
# =============================================================================
# Run sensorfront validation on login node (no SLURM)
# =============================================================================
# For small datasets (~7k events) where submitting jobs is overkill.
# Requires artifacts/sensorfront_shared/ to be prepared first:
#   python macro/validate_inpainter_sensorfront.py --manifest-only \
#       --input data/E15to60_AngUni_PosSQ/val2/ \
#       --output artifacts/sensorfront_shared \
#       --solid-angle-branch solid_angle
#
# Usage:
#   ./jobs/run_sensorfront_loginNode.sh              # All steps 1-6
#   ./jobs/run_sensorfront_loginNode.sh 3 5          # Only steps 3 and 5
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

SHARED_DIR="${SHARED_DIR:-artifacts/sensorfront_shared}"

declare -A STEP_LABEL
STEP_LABEL[1]="s1_baseline"
STEP_LABEL[2]="s2_flatmask"
STEP_LABEL[3]="s3_nphowt"
STEP_LABEL[4]="s4_flat_nphowt"
STEP_LABEL[5]="s5_sqrt"
STEP_LABEL[6]="s6_mask015"
STEP_LABEL[7]="s7_sqrt_nphowt_mask015"
STEP_LABEL[8]="s8_mask020"

if [ $# -eq 0 ]; then
    STEPS=(1 2 3 4 5 6)
else
    STEPS=("$@")
fi

# Check shared dir exists
if [ ! -f "${SHARED_DIR}/_sensorfront_manifest.npz" ]; then
    echo "[ERROR] Manifest not found: ${SHARED_DIR}/_sensorfront_manifest.npz"
    echo "Run manifest preparation first (see header comment)."
    exit 1
fi

echo "============================================"
echo "Sensorfront Validation (login node)"
echo "============================================"
echo "Steps:      ${STEPS[*]}"
echo "Shared dir: ${SHARED_DIR}"
echo "============================================"
echo ""

for step in "${STEPS[@]}"; do
    label="${STEP_LABEL[$step]:-}"
    if [ -z "$label" ]; then
        echo "[ERROR] Unknown step: $step"
        continue
    fi

    ckpt="artifacts/inp_scan_${label}/inpainter_checkpoint_best.pth"
    outdir="artifacts/inp_scan_${label}/validation_sensorfront/"

    if [ ! -f "$ckpt" ]; then
        echo "[WARN] Checkpoint not found: $ckpt (skipping step $step)"
        continue
    fi

    LOCALFIT_ARG=""
    if [ -d "${SHARED_DIR}/localfit_results" ]; then
        LOCALFIT_ARG="--local-fit-results ${SHARED_DIR}/localfit_results"
    fi

    echo "=== Step ${step}: ${label} ==="
    python macro/validate_inpainter_sensorfront.py \
        --checkpoint "$ckpt" \
        --load-manifest "${SHARED_DIR}" \
        --baselines-from "${SHARED_DIR}" \
        ${LOCALFIT_ARG} \
        --output "$outdir" \
        --device cpu
    echo ""
done

echo "============================================"
echo "Done. Compare with:"
echo "  python macro/compare_inpainter.py --mode sensorfront"
echo "============================================"
