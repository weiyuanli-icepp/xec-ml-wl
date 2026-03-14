#!/bin/bash
# =============================================================================
# Inpainter Validation (standalone, with baselines)
# =============================================================================
# Usage:
#   ./jobs/run_validate_inpainter.sh              # Validate all scan steps
#   ./jobs/run_validate_inpainter.sh 3 5          # Validate steps 3 and 5 only
#   DRY_RUN=1 ./jobs/run_validate_inpainter.sh    # Preview without submitting
#
# Environment variables:
#   PARTITION  - SLURM partition (default: a100-daily)
#   TIME       - Job time limit (default: 04:00:00)
#   DEVICE     - cpu or cuda (default: cuda)
#   RUN_NUM    - Dead channel run number (default: 430000)
#   VAL_PATH   - Validation data path (default: auto from config)
#   DRY_RUN    - Set to 1 to preview without submitting
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-04:00:00}"
DEVICE="${DEVICE:-cuda}"
RUN_NUM="${RUN_NUM:-430000}"
DRY_RUN="${DRY_RUN:-0}"
MEM="${MEM:-48G}"

SCAN_CONFIG_DIR="config/inp/scan"

declare -A STEP_NAME
STEP_NAME[1]="inp_scan_s1_baseline"
STEP_NAME[2]="inp_scan_s2_flatmask"
STEP_NAME[3]="inp_scan_s3_nphowt"
STEP_NAME[4]="inp_scan_s4_flat_nphowt"
STEP_NAME[5]="inp_scan_s5_sqrt"
STEP_NAME[6]="inp_scan_s6_mask015"
STEP_NAME[7]="inp_scan_s7_sqrt_nphowt_mask015"
STEP_NAME[8]="inp_scan_s8_mask020"

# Default: validate all steps
if [ $# -eq 0 ]; then
    STEPS=(1 2 3 4 5 6 7 8)
    echo "[VALIDATE] No steps specified. Validating all: ${STEPS[*]}"
else
    STEPS=("$@")
fi

echo "============================================"
echo "Inpainter Validation (with baselines)"
echo "============================================"
echo "Partition:  $PARTITION"
echo "Time limit: $TIME"
echo "Device:     $DEVICE"
echo "Run number: $RUN_NUM"
echo "Steps:      ${STEPS[*]}"
echo "Dry run:    $DRY_RUN"
echo "============================================"
echo ""

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="$HOME/meghome/xec-ml-wl/log"
mkdir -p "$LOG_DIR"

SUBMITTED=0
for STEP in "${STEPS[@]}"; do
    NAME="${STEP_NAME[$STEP]:-}"
    if [ -z "$NAME" ]; then
        echo "[ERROR] Unknown step: $STEP"
        continue
    fi

    ARTIFACT_DIR="artifacts/${NAME}"
    CHECKPOINT="${ARTIFACT_DIR}/inpainter_checkpoint_best.pth"
    OUTPUT_DIR="${ARTIFACT_DIR}/validation_mc"

    # Check checkpoint exists
    CHECKPOINT_EXPANDED="${CHECKPOINT/#\~/$HOME}"
    if [[ ! -f "$HOME/meghome/xec-ml-wl/$CHECKPOINT" ]] && [[ ! -f "$CHECKPOINT_EXPANDED" ]]; then
        echo "[WARN] Checkpoint not found: $CHECKPOINT (skipping step $STEP)"
        continue
    fi

    # Resolve val_path from config if not overridden
    CONFIG_FILE="${SCAN_CONFIG_DIR}/step${STEP}_*.yaml"
    CONFIG_MATCH=$(ls $CONFIG_FILE 2>/dev/null | head -1)
    if [ -z "${VAL_PATH:-}" ] && [ -n "$CONFIG_MATCH" ]; then
        VAL_PATH_CFG=$(grep -E '^\s*val_path:' "$CONFIG_MATCH" 2>/dev/null | head -1 | sed 's/^[^:]*:[[:space:]]*//' | tr -d '"' | tr -d "'")
    fi
    INPUT_PATH="${VAL_PATH:-${VAL_PATH_CFG:-~/meghome/xec-ml-wl/data/E15to60_AngUni_PosSQ/val/}}"

    LOG_FILE="${LOG_DIR}/val_${NAME}_%j.out"

    echo "--- Step $STEP: $NAME ---"
    echo "  Checkpoint: $CHECKPOINT"
    echo "  Input:      $INPUT_PATH"
    echo "  Output:     $OUTPUT_DIR"

    if [ "$DRY_RUN" == "1" ] || [ "$DRY_RUN" == "true" ]; then
        echo "  [DRY-RUN] Would submit validation job"
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=val_${NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=${MEM}
#SBATCH --clusters=gmerlin7

set -e
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
    echo "CRITICAL ERROR: Could not find 'conda'."
    exit 1
fi

conda activate "${ENV_NAME}"

if [ -n "\$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
fi

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"
echo "[JOB] Validating: ${NAME}"

python macro/validate_inpainter.py \\
    --checkpoint "${CHECKPOINT}" \\
    --input "${INPUT_PATH}" \\
    --run ${RUN_NUM} \\
    --output "${OUTPUT_DIR}" \\
    --baselines \\
    --device ${DEVICE}

echo "[JOB] Finished validation for ${NAME}."
EOF

    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 1
done

echo "============================================"
echo "[VALIDATE] Submitted $SUBMITTED / ${#STEPS[@]} jobs"
echo "============================================"
