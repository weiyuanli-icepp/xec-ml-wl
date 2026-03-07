#!/bin/bash
#
# Submit SLURM jobs for sensorfront validation of scan checkpoints.
# Tests inpainter recovery when the peak-signal sensor is masked.
#
# Two modes:
#   1. With SHARED_DIR: uses pre-computed manifest + baselines + localfit
#      (fast, skips ROOT re-reading and baseline computation)
#   2. Without SHARED_DIR: standalone per-step (original behavior)
#
# Usage:
#   # Fast mode (after submit_sensorfront_prepare_scan.sh + localfit):
#   SHARED_DIR=artifacts/sensorfront_shared bash macro/submit_validate_sensorfront_scan.sh
#
#   # Standalone mode (original):
#   bash macro/submit_validate_sensorfront_scan.sh          # Submit all steps
#   bash macro/submit_validate_sensorfront_scan.sh 1 6      # Submit only s1, s6
#   DRY_RUN=1 bash macro/submit_validate_sensorfront_scan.sh
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
VAL_PATH="${VAL_PATH:-data/E15to60_AngUni_PosSQ/val2/}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PARTITION="${PARTITION:-mu3e}"
SHARED_DIR="${SHARED_DIR:-}"  # set to enable fast mode

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

# Steps to submit (default: all)
if [ $# -eq 0 ]; then
    STEPS=(1 2 3 4 5 6 7 8)
else
    STEPS=("$@")
fi

STEP_LABELS=(
    [1]="s1_baseline"
    [2]="s2_flatmask"
    [3]="s3_nphowt"
    [4]="s4_flat_nphowt"
    [5]="s5_sqrt"
    [6]="s6_mask015"
    [7]="s7_sqrt_nphowt_mask015"
    [8]="s8_mask020"
)

echo "============================================"
echo "Sensorfront Validation (CPU)"
echo "============================================"
echo "Steps:      ${STEPS[*]}"
if [ -n "$SHARED_DIR" ]; then
    echo "Mode:       fast (prepared data from ${SHARED_DIR})"
else
    echo "Mode:       standalone (per-step ROOT loading)"
    echo "Val data:   ${VAL_PATH}"
fi
echo "Partition:  ${PARTITION}"
echo "Dry run:    ${DRY_RUN}"
echo "============================================"
echo ""

# Build extra args for fast mode
EXTRA_ARGS=""
if [ -n "$SHARED_DIR" ]; then
    EXTRA_ARGS="--load-manifest ${SHARED_DIR} --baselines-from ${SHARED_DIR} --no-manifest"
    LF_DIR="${SHARED_DIR}/localfit_results"
    if [ -d "$LF_DIR" ]; then
        EXTRA_ARGS="${EXTRA_ARGS} --local-fit-results ${LF_DIR}"
        echo "[INFO] LocalFit results found at ${LF_DIR}"
    else
        echo "[WARN] No localfit_results/ in ${SHARED_DIR} — LocalFit baseline will be skipped"
    fi
    echo ""
fi

mkdir -p log

SUBMITTED=0
for STEP in "${STEPS[@]}"; do
    LABEL="${STEP_LABELS[$STEP]:-}"
    if [ -z "$LABEL" ]; then
        echo "[ERROR] Unknown step: $STEP (valid: 1-8)"
        continue
    fi

    CKPT="artifacts/inp_scan_${LABEL}/inpainter_checkpoint_best.pth"
    OUTDIR="artifacts/inp_scan_${LABEL}/validation_sensorfront/"

    echo "--- Step ${STEP}: ${LABEL} ---"
    echo "  Checkpoint: ${CKPT}"
    echo "  Output:     ${OUTDIR}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] Skipping submission"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    mkdir -p "$HOME/.cache/xec-ml-wl"
    BATCH_SCRIPT=$(mktemp "$HOME/.cache/xec-ml-wl/sf_scan_s${STEP}_XXXXXX.sh")

    # In fast mode, no need for --input/--solid-angle-branch (data loaded from manifest)
    if [ -n "$SHARED_DIR" ]; then
        DATA_ARGS="${EXTRA_ARGS}"
    else
        DATA_ARGS="--input ${VAL_PATH} --solid-angle-branch solid_angle"
    fi

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30000
#SBATCH --job-name=sf_scan_${LABEL}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/sf_scan_${LABEL}_%j.log

echo "=== Sensorfront validation: ${LABEL} ==="
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo ""

# Conda setup
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true

ARM_CONDA="\$HOME/miniforge-arm/bin/conda"
X86_CONDA="/opt/psi/Programming/anaconda/2024.08/conda/bin/conda"

module load anaconda/2024.08 2>/dev/null || true

if [ -f "\$ARM_CONDA" ] && [ "\$(uname -m)" == "aarch64" ]; then
    eval "\$(\$ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
elif [ -f "\$X86_CONDA" ]; then
    eval "\$(\$X86_CONDA shell.bash hook)"
else
    echo "CRITICAL ERROR: Could not find conda"
    exit 1
fi

conda activate xec-ml-wl

if [ -n "\$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
fi

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"

python macro/validate_inpainter_sensorfront.py \\
    --checkpoint ${CKPT} \\
    ${DATA_ARGS} \\
    --output ${OUTDIR} \\
    --device cpu \\
    --batch-size ${BATCH_SIZE}

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

    sbatch "${BATCH_SCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 1
done

echo "============================================"
echo "[SCAN] Submitted ${SUBMITTED} / ${#STEPS[@]} sensorfront jobs"
echo "============================================"
