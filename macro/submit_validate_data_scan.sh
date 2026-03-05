#!/bin/bash
#
# Submit SLURM jobs to validate inpainter scan checkpoints on real data.
# Uses --real-data mode with artificial masking for evaluation.
#
# Usage:
#   bash macro/submit_validate_data_scan.sh              # Submit all 8 steps
#   bash macro/submit_validate_data_scan.sh 1 3 5        # Submit only steps 1, 3, 5
#   DRY_RUN=1 bash macro/submit_validate_data_scan.sh    # Preview without submitting
#
# Environment variables:
#   REAL_DATA     - Path to real data ROOT file(s) (default: val_data/data/DataGammaAngle_430026-430035.root)
#   DEAD_FILE     - Path to dead channel list file (default: data/dead_channels_run430000.txt)
#   N_ARTIFICIAL  - Number of artificial masks per event (default: 50)
#   BATCH_SIZE    - Inference batch size (default: 64)
#   MAX_EVENTS    - Max events to process (default: all)
#   PARTITION     - SLURM partition (default: mu3e)
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
REAL_DATA="${REAL_DATA:-val_data/data/DataGammaAngle_430026-430126.root}"
DEAD_FILE="${DEAD_FILE:-data/dead_channels_run430000.txt}"
N_ARTIFICIAL="${N_ARTIFICIAL:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EVENTS="${MAX_EVENTS:-}"
PARTITION="${PARTITION:-mu3e}"

# Determine --account flag based on partition
case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

# Steps to submit
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
echo "Validate Inpainter Scan — Real Data (CPU)"
echo "============================================"
echo "Steps:        ${STEPS[*]}"
echo "Real data:    ${REAL_DATA}"
echo "Dead file:    ${DEAD_FILE}"
echo "N artificial: ${N_ARTIFICIAL}"
echo "Partition:    ${PARTITION}"
echo "Dry run:      ${DRY_RUN}"
echo "============================================"
echo ""

mkdir -p log

SUBMITTED=0
for STEP in "${STEPS[@]}"; do
    LABEL="${STEP_LABELS[$STEP]:-}"
    if [ -z "$LABEL" ]; then
        echo "[ERROR] Unknown step: $STEP (valid: 1-8)"
        continue
    fi

    CKPT="artifacts/inp_scan_${LABEL}/inpainter_checkpoint_best.pth"
    OUTDIR="artifacts/inp_scan_${LABEL}/validation_data/"

    echo "--- Step ${STEP}: ${LABEL} ---"
    echo "  Checkpoint: ${CKPT}"
    echo "  Output:     ${OUTDIR}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] Skipping submission"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    BATCH_SCRIPT=$(mktemp /tmp/validate_data_s${STEP}_XXXXXX.sh)

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30000
#SBATCH --job-name=val_data_${LABEL}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/val_data_${LABEL}_%j.log

echo "=== Validate inpainter (real data): ${LABEL} ==="
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

python macro/validate_inpainter.py \\
    --checkpoint ${CKPT} \\
    --input ${REAL_DATA} \\
    --output ${OUTDIR} \\
    --dead-channel-file ${DEAD_FILE} \\
    --real-data \\
    --n-artificial ${N_ARTIFICIAL} \\
    --baselines \\
    --device cpu \\
    --batch-size ${BATCH_SIZE} \\
    ${MAX_EVENTS:+--max-events ${MAX_EVENTS}}

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

    sbatch "${BATCH_SCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 1
done

echo "============================================"
echo "[SCAN] Submitted ${SUBMITTED} / ${#STEPS[@]} validation jobs"
echo ""
echo "After completion, compare with:"
echo "  python macro/compare_inpainter.py --mode data"
echo "============================================"
