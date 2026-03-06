#!/bin/bash
#
# Submit SLURM job array to prepare real data for inpainter fine-tuning.
# Each array task processes one run through PrepareRealData.C.
#
# Usage:
#   bash macro/submit_prepare_realdata.sh <runlist.txt> [output_dir]
#
# Run list format (one per line, # comments allowed):
#   <run_number> <full_path_to_rec_file>
#
# Example runlist.txt:
#   # DAQ day 1
#   430123 /data/project/meg/offline/run/430xxx/rec430123_open.root
#   # DAQ day 2
#   431456 /data/project/meg/offline/run/431xxx/rec431456_open.root
#
# Output: one ROOT file per run in output_dir/DataGammaAngle_RRRRRR.root
#

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <runlist.txt> [output_dir]"
    echo ""
    echo "  runlist.txt   Two-column file: <run_number> <file_path>"
    echo "  output_dir    Output directory (default: data/real_data/raw/)"
    exit 1
fi

RUNLIST="$(realpath "$1")"
OUTPUT_DIR="${2:-data/real_data/raw}"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
PARTITION="${PARTITION:-mu3e}"

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

# Count valid (non-comment, non-empty) lines
N_JOBS=$(grep -cvE '^\s*($|#)' "$RUNLIST")

if [ "$N_JOBS" -le 0 ]; then
    echo "[ERROR] No valid entries in run list"
    exit 1
fi

MAX_IDX=$((N_JOBS - 1))
echo "[INFO] Run list: ${RUNLIST}"
echo "[INFO] Jobs: ${N_JOBS} (indices 0..${MAX_IDX})"
echo "[INFO] Output: ${OUTPUT_DIR}"
echo "[INFO] Partition: ${PARTITION}"

mkdir -p log "$OUTPUT_DIR"

# Meganalyzer path (must cd to analyzer dir before running)
ANALYZER_DIR="${ANALYZER_DIR:-$HOME/meghome/offline/analyzer}"
MACRO_PATH="$HOME/meghome/xec-ml-wl/macro/PrepareRealData.C"

BATCH_SCRIPT=$(mktemp /tmp/prepare_realdata_XXXXXX.sh)

cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=4:00:00
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3700
#SBATCH --job-name=prep_real
#SBATCH --array=0-${MAX_IDX}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/prep_real_%A_%a.log

echo "=== PrepareRealData: job \${SLURM_ARRAY_TASK_ID} / ${MAX_IDX} ==="
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo ""

cd ${ANALYZER_DIR}

./meganalyzer -b -q '${MACRO_PATH}+("${RUNLIST}", \${SLURM_ARRAY_TASK_ID}, "${OUTPUT_DIR}")'

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

echo "[INFO] Batch script: ${BATCH_SCRIPT}"
echo ""
cat "${BATCH_SCRIPT}"
echo ""

# Submit
sbatch "${BATCH_SCRIPT}"
echo "[INFO] Submitted job array with ${N_JOBS} tasks"
