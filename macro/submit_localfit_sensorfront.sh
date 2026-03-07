#!/bin/bash
#
# Submit SLURM job array for local-fit sensor-front validation.
# Each array task processes one ROOT file through the LocalFitBaseline macro.
#
# Usage:
#   bash macro/submit_localfit_sensorfront.sh <manifest.npz> [output_dir]
#
# Example:
#   bash macro/submit_localfit_sensorfront.sh \
#       artifacts/sensorfront_shared/_sensorfront_manifest.npz
#

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <manifest.npz> [output_dir]"
    echo ""
    echo "  manifest.npz  Path to _sensorfront_manifest.npz"
    echo "  output_dir    Optional output directory for results"
    echo "                (default: <manifest_dir>/localfit_results/)"
    exit 1
fi

MANIFEST="$(realpath "$1")"
PARTITION="${PARTITION:-mu3e}"

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

# Read number of files from the manifest
N_FILES=$(python3 -c "
import numpy as np
m = np.load('${MANIFEST}', allow_pickle=True)
print(len(m['file_list']))
")

if [ "$N_FILES" -le 0 ]; then
    echo "[ERROR] No files found in manifest"
    exit 1
fi

MAX_IDX=$((N_FILES - 1))
echo "[INFO] Manifest: ${MANIFEST}"
echo "[INFO] Files: ${N_FILES} (indices 0..${MAX_IDX})"
echo "[INFO] Partition: ${PARTITION}"

# Output directory
if [ $# -ge 2 ]; then
    OUTPUT_DIR="$(realpath "$2")"
    OUTPUT_ARG="--output-dir ${OUTPUT_DIR}"
else
    OUTPUT_ARG=""
fi

mkdir -p log

# Create a temporary SLURM batch script
mkdir -p "$HOME/.cache/xec-ml-wl"
BATCH_SCRIPT=$(mktemp "$HOME/.cache/xec-ml-wl/localfit_sensorfront_XXXXXX.sh")

cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3700
#SBATCH --job-name=localfit_sf
#SBATCH --array=0-${MAX_IDX}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/localfit_sf_%A_%a.log

echo "=== LocalFit sensor-front: file \${SLURM_ARRAY_TASK_ID} / ${MAX_IDX} ==="
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

python macro/run_localfit_sensorfront.py \\
    --manifest ${MANIFEST} \\
    --file-index \${SLURM_ARRAY_TASK_ID} \\
    ${OUTPUT_ARG}

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

echo "[INFO] Batch script: ${BATCH_SCRIPT}"
echo ""

# Submit
sbatch "${BATCH_SCRIPT}"
echo "[INFO] Submitted job array with ${N_FILES} tasks"
