#!/bin/bash
#
# Submit SLURM job array for local-fit sensor-front validation.
#
# Usage:
#   bash macro/submit_localfit_sensorfront.sh <manifest.npz> [output_dir]
#
# Example:
#   bash macro/submit_localfit_sensorfront.sh \
#       artifacts/sensorfront_validation/_sensorfront_manifest.npz
#

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <manifest.npz> [output_dir]"
    echo ""
    echo "  manifest.npz  Path to _sensorfront_manifest.npz"
    echo "  output_dir    Optional output directory for results"
    echo "                (default: <manifest_dir>/localfit_results/)"
    exit 1
fi

MANIFEST="$(realpath "$1")"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUN_SCRIPT="${SCRIPT_DIR}/run_localfit_sensorfront.py"

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

# Output directory
if [ $# -ge 2 ]; then
    OUTPUT_DIR="$(realpath "$2")"
    OUTPUT_ARG="--output-dir ${OUTPUT_DIR}"
else
    OUTPUT_ARG=""
fi

# Create a temporary SLURM batch script
BATCH_SCRIPT=$(mktemp /tmp/localfit_sensorfront_XXXXXX.sh)

cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
#SBATCH --account=meg
#SBATCH --partition=mu3e
#SBATCH --time=5:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3700
#SBATCH --job-name=localfit_sf
#SBATCH --array=0-${MAX_IDX}
#SBATCH --output=localfit_sf_%A_%a.log

echo "=== LocalFit sensor-front: file \${SLURM_ARRAY_TASK_ID} / ${MAX_IDX} ==="
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo ""

python3 ${RUN_SCRIPT} \\
    --manifest ${MANIFEST} \\
    --file-index \${SLURM_ARRAY_TASK_ID} \\
    ${OUTPUT_ARG}

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

echo "[INFO] Batch script: ${BATCH_SCRIPT}"
echo ""
cat "${BATCH_SCRIPT}"
echo ""

# Submit
sbatch "${BATCH_SCRIPT}"
echo "[INFO] Submitted job array with ${N_FILES} tasks"
