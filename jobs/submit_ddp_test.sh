#!/usr/bin/env bash
# DDP sanity check submission script
# Validates multi-GPU distributed training setup before running actual training.
#
# Usage:
#   NUM_GPUS=2 ./jobs/submit_ddp_test.sh
#   NUM_GPUS=4 ./jobs/submit_ddp_test.sh
#   NUM_GPUS=4 PARTITION=gh-daily ./jobs/submit_ddp_test.sh
#
# Environment variables:
#   NUM_GPUS   - Number of GPUs to test (required, must be >= 2 for meaningful test)
#   PARTITION  - SLURM partition (default: a100-daily)
#   TIME       - Job time limit (default: 00:10:00)

set -euo pipefail

NUM_GPUS="${NUM_GPUS:-}"
PARTITION="${PARTITION:-a100-daily}"
TIME="${TIME:-00:10:00}"

if [[ -z "$NUM_GPUS" ]]; then
    echo "[ERROR] NUM_GPUS is required. Example: NUM_GPUS=4 ./jobs/submit_ddp_test.sh"
    exit 1
fi

if [[ "$NUM_GPUS" -lt 2 ]]; then
    echo "[WARN] NUM_GPUS=$NUM_GPUS. DDP test is most useful with >= 2 GPUs."
    echo "       Single-GPU mode tests no-op behavior only."
fi

# === Path Validation ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
TEST_FILE="$REPO_ROOT/tests/test_ddp.py"

if [[ ! -f "$TEST_FILE" ]]; then
    echo "[ERROR] DDP test file not found: $TEST_FILE"
    exit 1
fi

ENV_NAME="xec-ml-wl"
if [[ "$PARTITION" == gh* ]]; then ENV_NAME="xec-ml-wl-gh"; fi

LOG_DIR="$HOME/meghome/xec-ml-wl/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/ddp_test_%j.out"

echo "[SUBMIT] DDP Sanity Check"
echo "  GPUs:       $NUM_GPUS"
echo "  Partition:  $PARTITION"
echo "  Time:       $TIME"
echo "  Log:        $LOG_FILE"

sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ddp_test
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:${NUM_GPUS}
#SBATCH --mem=32G
#SBATCH --clusters=gmerlin7

set -e
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true

ARM_CONDA="\$HOME/miniforge-arm/bin/conda"
X86_CONDA="/opt/psi/Programming/anaconda/2024.08/conda/bin/conda"

# Load module for x86 nodes
module load anaconda/2024.08 2>/dev/null || true

# Initialize Conda based on architecture
if [ -f "\$ARM_CONDA" ] && [ "\$(uname -m)" == "aarch64" ]; then
    echo "[JOB] Detected ARM64 architecture. Using Miniforge."
    eval "\$(\$ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
elif [ -f "\$X86_CONDA" ]; then
    eval "\$(\$X86_CONDA shell.bash hook)"
else
    echo "CRITICAL ERROR: Could not find 'conda' on partition ${PARTITION}."
    exit 1
fi

echo "[JOB] Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Fix awkward_cpp libstdc++ compatibility on GH nodes
if [ -n "\$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
fi

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"
echo "[JOB] Running DDP sanity check with ${NUM_GPUS} GPUs..."
echo ""

torchrun --nproc_per_node=${NUM_GPUS} tests/test_ddp.py

echo ""
echo "[JOB] DDP sanity check completed successfully!"
echo "[JOB] Multi-GPU training should work. You can now run:"
echo "      NUM_GPUS=${NUM_GPUS} ./jobs/submit_mae.sh"
echo "      NUM_GPUS=${NUM_GPUS} ./jobs/submit_regressor.sh"
echo "      NUM_GPUS=${NUM_GPUS} ./jobs/submit_inpainter.sh"
EOF
