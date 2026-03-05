#!/bin/bash
#
# Submit SLURM job to prepare sensorfront validation data shared across
# all scan steps.  This creates:
#   1. Manifest (event matching info for LocalFit)
#   2. Raw matched data (so inference jobs skip ROOT re-reading)
#   3. Raw baseline predictions (neighbor avg, solid angle — scheme-independent)
#
# After this job completes, submit:
#   - LocalFit batch jobs (submit_localfit_sensorfront.sh)
#   - ML inference jobs (submit_validate_sensorfront_scan.sh)
# Both can run in parallel.
#
# Usage:
#   bash macro/submit_sensorfront_prepare_scan.sh
#   DRY_RUN=1 bash macro/submit_sensorfront_prepare_scan.sh
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
VAL_PATH="${VAL_PATH:-data/E15to60_AngUni_PosSQ/val2/}"
SHARED_DIR="${SHARED_DIR:-artifacts/sensorfront_shared}"
PARTITION="${PARTITION:-mu3e}"

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

echo "============================================"
echo "Prepare Sensorfront Data (shared across scan)"
echo "============================================"
echo "Val data:    ${VAL_PATH}"
echo "Output:      ${SHARED_DIR}"
echo "Partition:   ${PARTITION}"
echo "Dry run:     ${DRY_RUN}"
echo "============================================"
echo ""

mkdir -p log

if [ "$DRY_RUN" = "1" ]; then
    echo "[DRY RUN] Would submit manifest-only job"
    echo ""
    echo "After completion, run:"
    echo "  bash macro/submit_localfit_sensorfront.sh ${SHARED_DIR}/_sensorfront_manifest.npz"
    echo "  SHARED_DIR=${SHARED_DIR} bash macro/submit_validate_sensorfront_scan.sh"
    exit 0
fi

BATCH_SCRIPT=$(mktemp /tmp/sf_prepare_XXXXXX.sh)

cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=2:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=30000
#SBATCH --job-name=sf_prepare
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/sf_prepare_%j.log

echo "=== Sensorfront: prepare manifest + baselines ==="
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
    --manifest-only \\
    --input ${VAL_PATH} \\
    --output ${SHARED_DIR} \\
    --solid-angle-branch solid_angle

echo ""
echo "=== Done: \$(date) ==="
echo ""
echo "Next steps:"
echo "  1. bash macro/submit_localfit_sensorfront.sh ${SHARED_DIR}/_sensorfront_manifest.npz"
echo "  2. SHARED_DIR=${SHARED_DIR} bash macro/submit_validate_sensorfront_scan.sh"
SLURM_EOF

sbatch "${BATCH_SCRIPT}"
echo ""
echo "============================================"
echo "After this job completes, run in parallel:"
echo "  bash macro/submit_localfit_sensorfront.sh ${SHARED_DIR}/_sensorfront_manifest.npz"
echo "  SHARED_DIR=${SHARED_DIR} bash macro/submit_validate_sensorfront_scan.sh"
echo "============================================"
