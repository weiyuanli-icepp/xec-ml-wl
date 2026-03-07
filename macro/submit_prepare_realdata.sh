#!/bin/bash
#
# Submit SLURM job array to prepare real data for inpainter fine-tuning.
# Each array task processes RUNS_PER_JOB consecutive runs through
# PrepareRealDataInpainter.C, allowing large runlists to fit within
# SLURM's max array index limit (1999).
#
# Usage:
#   bash macro/submit_prepare_realdata.sh <runlist.txt> [output_dir]
#
# Run list format (one per line, # comments allowed):
#   <run_number> <full_path_to_rec_file>
#
# Environment variables:
#   RUNS_PER_JOB  - Number of runlist entries per SLURM task (default: 42)
#   PARTITION     - SLURM partition (default: meg-short)
#   ANALYZER_DIR  - Path to meganalyzer (default: $HOME/meghome/offline/analyzer)
#   START_FROM    - Skip the first N runlist entries (default: 0)
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
    echo ""
    echo "Environment variables:"
    echo "  RUNS_PER_JOB  Runs per SLURM task (default: 42)"
    echo "  START_FROM    Skip first N entries (default: 0)"
    echo "  PARTITION     SLURM partition (default: meg-short)"
    exit 1
fi

RUNLIST="$(realpath "$1")"
OUTPUT_DIR="${2:-data/real_data/raw}"
OUTPUT_DIR="$(realpath -m "$OUTPUT_DIR")"
PARTITION="${PARTITION:-meg-short}"
RUNS_PER_JOB="${RUNS_PER_JOB:-42}"
START_FROM="${START_FROM:-0}"

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

# Count valid (non-comment, non-empty) lines
N_TOTAL=$(grep -cvE '^\s*($|#)' "$RUNLIST")

if [ "$N_TOTAL" -le 0 ]; then
    echo "[ERROR] No valid entries in run list"
    exit 1
fi

# Apply START_FROM offset
N_RUNS=$((N_TOTAL - START_FROM))
if [ "$N_RUNS" -le 0 ]; then
    echo "[ERROR] START_FROM=${START_FROM} >= total entries ${N_TOTAL}"
    exit 1
fi

# Number of SLURM tasks needed
N_JOBS=$(( (N_RUNS + RUNS_PER_JOB - 1) / RUNS_PER_JOB ))

echo "[INFO] Run list: ${RUNLIST}"
echo "[INFO] Total entries: ${N_TOTAL}, starting from: ${START_FROM}, remaining: ${N_RUNS}"
echo "[INFO] Runs per job: ${RUNS_PER_JOB}"
echo "[INFO] SLURM tasks: ${N_JOBS}"
echo "[INFO] Output: ${OUTPUT_DIR}"
echo "[INFO] Partition: ${PARTITION}"

mkdir -p log "$OUTPUT_DIR"

# Meganalyzer path (must cd to analyzer dir before running)
ANALYZER_DIR="${ANALYZER_DIR:-$HOME/meghome/offline/analyzer}"
MACRO_PATH="$HOME/meghome/xec-ml-wl/macro/PrepareRealDataInpainter.C"

# Submit in chunks of 2000 (SLURM max array size, indices 0-1999)
MAX_ARRAY=2000
N_CHUNKS=$(( (N_JOBS + MAX_ARRAY - 1) / MAX_ARRAY ))
SUBMITTED=0

mkdir -p "$HOME/.cache/xec-ml-wl"

for ((CHUNK=0; CHUNK<N_CHUNKS; CHUNK++)); do
    GLOBAL_START=$((CHUNK * MAX_ARRAY))
    GLOBAL_END=$(( (CHUNK + 1) * MAX_ARRAY - 1 ))
    if [ $GLOBAL_END -ge $N_JOBS ]; then
        GLOBAL_END=$((N_JOBS - 1))
    fi
    CHUNK_SIZE=$(( GLOBAL_END - GLOBAL_START + 1 ))
    ARRAY_END=$(( CHUNK_SIZE - 1 ))

    # Time scales with RUNS_PER_JOB: ~1.5 min per run, plus margin
    TIME_MINUTES=$(( RUNS_PER_JOB * 2 + 10 ))
    TIME_HH=$(( TIME_MINUTES / 60 ))
    TIME_MM=$(( TIME_MINUTES % 60 ))
    TIME_STR=$(printf "%02d:%02d:00" $TIME_HH $TIME_MM)

    BATCH_SCRIPT=$(mktemp "$HOME/.cache/xec-ml-wl/prepare_realdata_chunk${CHUNK}_XXXXXX.sh")

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_STR}
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=3700
#SBATCH --job-name=prep_real
#SBATCH --array=0-${ARRAY_END}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/prep_real_%A_%a.log

RUNS_PER_JOB=${RUNS_PER_JOB}
START_FROM=${START_FROM}
CHUNK_OFFSET=${GLOBAL_START}

echo "=== PrepareRealData: task \${SLURM_ARRAY_TASK_ID} (${RUNS_PER_JOB} runs/task) ==="
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo ""

cd ${ANALYZER_DIR}
mkdir -p "\$HOME/.cache/xec-ml-wl"

# Global task index (across chunks)
GLOBAL_TASK=\$(( CHUNK_OFFSET + SLURM_ARRAY_TASK_ID ))

# Process RUNS_PER_JOB consecutive runlist entries
for ((I=0; I<RUNS_PER_JOB; I++)); do
    RUN_IDX=\$(( START_FROM + GLOBAL_TASK * RUNS_PER_JOB + I ))

    # Check if index is beyond the runlist
    if [ \${RUN_IDX} -ge ${N_TOTAL} ]; then
        echo "[INFO] Index \${RUN_IDX} >= ${N_TOTAL}, stopping"
        break
    fi

    echo ""
    echo "--- Run index \${RUN_IDX} (local \$((I+1))/${RUNS_PER_JOB}) ---"

    LOADER="\$HOME/.cache/xec-ml-wl/prep_real_loader_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${I}.C"
    FUNC_NAME="prep_real_loader_\${SLURM_ARRAY_JOB_ID}_\${SLURM_ARRAY_TASK_ID}_\${I}"
    cat > "\${LOADER}" <<MACRO_EOF
void \${FUNC_NAME}() {
    gROOT->ProcessLine(".L ${MACRO_PATH}+");
    gROOT->ProcessLine("PrepareRealDataInpainterFromList(\"${RUNLIST}\", \${RUN_IDX}, \"${OUTPUT_DIR}\")");
}
MACRO_EOF

    ./meganalyzer -b -q -I "\${LOADER}()"
    rm -f "\${LOADER}"
done

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

    echo "[INFO] Chunk $((CHUNK+1))/${N_CHUNKS}: ${CHUNK_SIZE} tasks (global indices ${GLOBAL_START}-${GLOBAL_END})"
    echo "[INFO] Batch script: ${BATCH_SCRIPT}"
    sbatch "${BATCH_SCRIPT}"
    SUBMITTED=$((SUBMITTED + CHUNK_SIZE))
    echo ""
    sleep 1
done

echo "[INFO] Submitted ${SUBMITTED} tasks in ${N_CHUNKS} chunk(s)"
echo "[INFO] Each task processes up to ${RUNS_PER_JOB} runs (${N_RUNS} total)"
