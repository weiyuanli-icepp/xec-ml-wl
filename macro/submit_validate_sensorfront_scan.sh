#!/bin/bash
#
# Submit SLURM jobs for sensorfront validation of scan checkpoints.
# Tests inpainter recovery when the peak-signal sensor is masked.
#
# Usage:
#   bash macro/submit_validate_sensorfront_scan.sh          # Submit s1, s3, s6
#   bash macro/submit_validate_sensorfront_scan.sh 1 6      # Submit only s1, s6
#   DRY_RUN=1 bash macro/submit_validate_sensorfront_scan.sh
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
VAL_PATH="${VAL_PATH:-data/E15to60_AngUni_PosSQ/val2/}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Steps to submit (default: s1, s3, s6)
if [ $# -eq 0 ]; then
    STEPS=(1 3 6)
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
)

echo "============================================"
echo "Sensorfront Validation (CPU)"
echo "============================================"
echo "Steps:    ${STEPS[*]}"
echo "Val data: ${VAL_PATH}"
echo "Dry run:  ${DRY_RUN}"
echo "============================================"
echo ""

mkdir -p log

SUBMITTED=0
for STEP in "${STEPS[@]}"; do
    LABEL="${STEP_LABELS[$STEP]:-}"
    if [ -z "$LABEL" ]; then
        echo "[ERROR] Unknown step: $STEP (valid: 1-6)"
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

    BATCH_SCRIPT=$(mktemp /tmp/sf_scan_s${STEP}_XXXXXX.sh)

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
#SBATCH --account=meg
#SBATCH --partition=mu3e
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
    --input ${VAL_PATH} \\
    --output ${OUTDIR} \\
    --solid-angle-branch solid_angle \\
    --device cpu \\
    --batch-size ${BATCH_SIZE} \\
    --no-manifest

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
