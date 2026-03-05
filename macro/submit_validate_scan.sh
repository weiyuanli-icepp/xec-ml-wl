#!/bin/bash
#
# Submit SLURM job array to validate inpainter scan checkpoints (s1-s6).
# Runs on CPU nodes with a fixed dead channel pattern for fair comparison.
#
# Usage:
#   bash macro/submit_validate_scan.sh              # Submit all 6 steps
#   bash macro/submit_validate_scan.sh 1 3 5        # Submit only steps 1, 3, 5
#   DRY_RUN=1 bash macro/submit_validate_scan.sh    # Preview without submitting
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
RUN_NUMBER="${RUN_NUMBER:-430000}"
VAL_PATH="${VAL_PATH:-data/E15to60_AngUni_PosSQ/val2/}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EVENTS="${MAX_EVENTS:-}"  # empty = all events
LOCAL_FIT="${LOCAL_FIT:-0}"   # set LOCAL_FIT=1 to enable LocalFitBaseline

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
echo "Validate Inpainter Scan (CPU)"
echo "============================================"
echo "Steps:      ${STEPS[*]}"
echo "Run number: ${RUN_NUMBER}"
echo "Val data:   ${VAL_PATH}"
echo "Local fit:  ${LOCAL_FIT}"
echo "Dry run:    ${DRY_RUN}"
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
    OUTDIR="artifacts/inp_scan_${LABEL}/validation_mc/"

    echo "--- Step ${STEP}: ${LABEL} ---"
    echo "  Checkpoint: ${CKPT}"
    echo "  Output:     ${OUTDIR}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] Skipping submission"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    BATCH_SCRIPT=$(mktemp /tmp/validate_scan_s${STEP}_XXXXXX.sh)

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
#SBATCH --account=meg
#SBATCH --partition=mu3e
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30000
#SBATCH --job-name=val_scan_${LABEL}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/val_scan_${LABEL}_%j.log

echo "=== Validate inpainter scan: ${LABEL} ==="
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
    --input ${VAL_PATH} \\
    --output ${OUTDIR} \\
    --run ${RUN_NUMBER} \\
    --baselines \\
    --solid-angle-branch solid_angle \\
    --device cpu \\
    --batch-size ${BATCH_SIZE} \\
    ${MAX_EVENTS:+--max-events ${MAX_EVENTS}} \\
    \$([ "${LOCAL_FIT}" = "1" ] && echo "--local-fit-baseline" || true)

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
echo "After completion, update ENTRIES in macro/compare_inpainter.py"
echo "and run:  python macro/compare_inpainter.py -o scan_comparison.pdf"
echo "============================================"
