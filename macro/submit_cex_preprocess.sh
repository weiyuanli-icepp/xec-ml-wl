#!/bin/bash
#
# Submit SLURM jobs to run CEXPreprocess.py for all CEX 2023 patches.
# One job per patch. For patches with non-consecutive runs, multiple
# CEXPreprocess.py invocations are chained within the same job.
#
# Usage:
#   bash macro/submit_cex_preprocess.sh                    # All patches
#   bash macro/submit_cex_preprocess.sh 13 12 21           # Only these patches
#   DRY_RUN=1 bash macro/submit_cex_preprocess.sh          # Preview
#
# Environment variables:
#   PARTITION   — SLURM partition (default: daily)
#   TIME        — time limit (default: 12:00:00)
#   MEM         — memory (default: 16G)
#   DEAD_DIR    — directory with per-run dead channel files (default: data/dead_channels)
#   OUTPUT_DIR  — output directory for ROOT files (default: data/cex)
#   RUNLIST     — path to runlist file (default: data/cex/cex2023_runlist.txt)
#   MAX_RUNS    — max runs per CEXPreprocess.py invocation (default: 10)
#   DRY_RUN     — set to 1 to preview without submitting
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
PARTITION="${PARTITION:-daily}"
TIME="${TIME:-12:00:00}"
MEM="${MEM:-16G}"
DEAD_DIR="${DEAD_DIR:-data/dead_channels}"
OUTPUT_DIR="${OUTPUT_DIR:-data/cex}"
RUNLIST="${RUNLIST:-data/cex/cex2023_runlist.txt}"
MAX_RUNS="${MAX_RUNS:-10}"

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

echo "============================================"
echo "CEX Preprocessing"
echo "============================================"
echo "Runlist:   ${RUNLIST}"
echo "Output:    ${OUTPUT_DIR}"
echo "Dead dir:  ${DEAD_DIR}"
echo "Partition: ${PARTITION}"
echo "Max runs:  ${MAX_RUNS}"
echo "Dry run:   ${DRY_RUN}"
echo "============================================"
echo ""

if [ ! -f "${RUNLIST}" ]; then
    echo "[ERROR] Runlist not found: ${RUNLIST}"
    exit 1
fi

# Parse the runlist: extract per-patch consecutive ranges
# Output format: PATCH SRUN NFILES (one line per consecutive range)
parse_runlist() {
    python3 -c "
import sys

patch = None
runs = []
ranges = []

def flush_patch(p, rs):
    if not rs:
        return
    rs.sort()
    start = rs[0]
    count = 1
    for i in range(1, len(rs)):
        if rs[i] == rs[i-1] + 1:
            count += 1
        else:
            print(f'{p} {start} {count}')
            start = rs[i]
            count = 1
    print(f'{p} {start} {count}')

with open('${RUNLIST}') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        if line.startswith('# patch'):
            if patch is not None:
                flush_patch(patch, runs)
            import re
            m = re.search(r'patch\s+(\d+)', line)
            patch = int(m.group(1)) if m else None
            runs = []
        elif line.startswith('#'):
            continue
        else:
            runs.append(int(line))
    if patch is not None:
        flush_patch(patch, runs)
"
}

# Read all ranges into an associative array keyed by patch
declare -A PATCH_RANGES
while IFS=' ' read -r p srun nfiles; do
    PATCH_RANGES[$p]+="${srun}:${nfiles} "
done < <(parse_runlist)

# Determine which patches to process
if [ $# -eq 0 ]; then
    PATCHES=($(echo "${!PATCH_RANGES[@]}" | tr ' ' '\n' | sort -n))
else
    PATCHES=("$@")
fi

echo "Patches: ${PATCHES[*]}"
echo ""

mkdir -p log/cex_preprocess "${OUTPUT_DIR}"

SUBMITTED=0

for PATCH in "${PATCHES[@]}"; do
    ranges="${PATCH_RANGES[$PATCH]:-}"
    if [ -z "$ranges" ]; then
        echo "--- Patch ${PATCH}: NOT IN RUNLIST, skipping ---"
        continue
    fi

    # Count total runs
    total_runs=0
    n_ranges=0
    for r in $ranges; do
        nf="${r#*:}"
        total_runs=$((total_runs + nf))
        n_ranges=$((n_ranges + 1))
    done

    echo "--- Patch ${PATCH}: ${total_runs} runs in ${n_ranges} range(s) ---"

    # Split each range into sub-ranges of at most MAX_RUNS to limit memory.
    # 80 runs accumulate ~9 GB of output arrays; 10 runs ≈ 1.1 GB.
    split_ranges=""
    for r in $ranges; do
        sr="${r%%:*}"
        nf="${r#*:}"
        while [ "$nf" -gt 0 ]; do
            chunk=$(( nf > MAX_RUNS ? MAX_RUNS : nf ))
            split_ranges+="${sr}:${chunk} "
            sr=$(( sr + chunk ))
            nf=$(( nf - chunk ))
        done
    done

    if [ "$DRY_RUN" = "1" ]; then
        for r in $split_ranges; do
            sr="${r%%:*}"
            nf="${r#*:}"
            echo "  CEXPreprocess.py --srun ${sr} --nfiles ${nf} --patch ${PATCH}"
        done
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    mkdir -p "$HOME/.cache/xec-ml-wl"
    BATCH_SCRIPT=$(mktemp "$HOME/.cache/xec-ml-wl/cex_preproc_patch${PATCH}_XXXXXX.sh")

    # Build the CEXPreprocess commands
    PREPROCESS_CMDS=""
    for r in $split_ranges; do
        sr="${r%%:*}"
        nf="${r#*:}"
        PREPROCESS_CMDS+="
echo \"--- Range: srun=${sr} nfiles=${nf} ---\"
python -u others/CEXPreprocess.py \\
    --srun ${sr} --nfiles ${nf} --patch ${PATCH} \\
    --output-dir ${OUTPUT_DIR} \\
    --dead-dir ${DEAD_DIR}
"
    done

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=${MEM}
#SBATCH --job-name=cex_pre_p${PATCH}
#SBATCH --output=${HOME}/meghome/xec-ml-wl/log/cex_preprocess/patch${PATCH}_%j.log

echo "=== CEX Preprocessing: Patch ${PATCH} ==="
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

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"
echo ""
${PREPROCESS_CMDS}
echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

    sbatch "${BATCH_SCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 0.2
done

echo "============================================"
echo "[CEX] Submitted ${SUBMITTED} jobs"
echo "Logs:   log/cex_preprocess/patch*_*.log"
echo "Output: ${OUTPUT_DIR}/"
echo "============================================"
