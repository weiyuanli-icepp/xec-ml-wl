#!/usr/bin/env bash
# Usage: ./submit_inference.sh [RUN_NAME] [ONNX_MODEL] [INPUT_ROOT] [OUTPUT_ROOT] [PARTITION] [TIME]
# Example: ./submit_inference.sh \
#                "run123" "meg2ang.onnx" "Data.root" "Output.root" "gh-daily" "01:00:00"

# This handles all the path exports, environment switching, and library loading for you automatically.

set -euo pipefail

RUN_NAME="${1:-inference_run}"
ONNX_MODEL="${2:-}"
INPUT_ROOT="${3:-}"
OUTPUT_ROOT="${4:-}"
PARTITION="${5:-a100-daily}"
TIME="${6:-02:00:00}"

# === Argument Validation ===
VALIDATION_FAILED=0

if [[ -z "$ONNX_MODEL" ]]; then
    echo "[ERROR] ONNX_MODEL is required (argument 2)"
    VALIDATION_FAILED=1
elif [[ ! -f "$ONNX_MODEL" ]]; then
    echo "[ERROR] ONNX model not found: $ONNX_MODEL"
    VALIDATION_FAILED=1
fi

if [[ -z "$INPUT_ROOT" ]]; then
    echo "[ERROR] INPUT_ROOT is required (argument 3)"
    VALIDATION_FAILED=1
elif [[ ! -f "$INPUT_ROOT" && ! -d "$INPUT_ROOT" ]]; then
    # Check if it's a glob pattern
    if ! compgen -G "$INPUT_ROOT" > /dev/null 2>&1; then
        echo "[ERROR] Input ROOT file not found: $INPUT_ROOT"
        VALIDATION_FAILED=1
    fi
fi

if [[ -z "$OUTPUT_ROOT" ]]; then
    echo "[ERROR] OUTPUT_ROOT is required (argument 4)"
    VALIDATION_FAILED=1
else
    # Check output directory is writable
    OUTPUT_DIR="$(dirname "$OUTPUT_ROOT")"
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        mkdir -p "$OUTPUT_DIR" 2>/dev/null || {
            echo "[ERROR] Cannot create output directory: $OUTPUT_DIR"
            VALIDATION_FAILED=1
        }
    fi
fi

if [[ "$VALIDATION_FAILED" -eq 1 ]]; then
    echo ""
    echo "Usage: ./submit_inference.sh RUN_NAME ONNX_MODEL INPUT_ROOT OUTPUT_ROOT [PARTITION] [TIME]"
    echo "[ABORT] Fix the above errors before submitting."
    exit 1
fi

# Optional: Npho/Time scaling (Must match training!)
NPHO_SCALE="${NPHO_SCALE:-1.0}"
TIME_SCALE="${TIME_SCALE:-2.32e6}"
TIME_SHIFT="${TIME_SHIFT:-0.0}"

# Allow environment overrides for Conda paths
ARM_CONDA_PATH="${ARM_CONDA:-$HOME/miniforge-arm/bin/conda}"
X86_CONDA_PATH="${X86_CONDA:-/opt/psi/Programming/anaconda/2024.08/conda/bin/conda}"

# Determine Environment
if [[ "$PARTITION" == gh* ]]; then
    ENV_NAME="xec-ml-wl-gh"
else
    ENV_NAME="xec-ml-wl"
fi

LOG_DIR="$HOME/meghome/xec-ml-wl/log"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/infer_${RUN_NAME}_%j.out"

echo "[SUBMIT] Inference: $RUN_NAME | Model: $ONNX_MODEL"

# Submit Job
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=inf_${RUN_NAME}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --time=${TIME}
#SBATCH --partition=${PARTITION}
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --clusters=gmerlin7

set -e

# Load Environment
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true

# Define Paths
ARM_CONDA="${ARM_CONDA_PATH}"
X86_CONDA="${X86_CONDA_PATH}"

# 1. Try loading module
module load anaconda/2024.08 2>/dev/null || true

# 2. Initialize Conda Dynamically (Escaped \$ for compute node execution)
if [ -f "\$ARM_CONDA" ] && [ "\$(uname -m)" == "aarch64" ]; then
    echo "[JOB] Detected ARM64 architecture. Using Miniforge."
    eval "\$(\$ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
elif [ -f "\$X86_CONDA" ]; then
    eval "\$(\$X86_CONDA shell.bash hook)"
else
    echo "CRITICAL ERROR: Could not find 'conda'."
    exit 1
fi

echo "[JOB] Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Setup CUDA Libraries (Critical for ONNX Runtime on Cluster)
export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH
# For some setups, we need to be explicit about nvidia libs
export LD_LIBRARY_PATH=\$(find \$CONDA_PREFIX/lib/python*/site-packages/nvidia -name "lib" -type d | paste -sd ":" -):\$LD_LIBRARY_PATH

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Running inference..."

python inference_real_data.py \\
    --onnx "${ONNX_MODEL}" \\
    --input "${INPUT_ROOT}" \\
    --output "${OUTPUT_ROOT}" \\
    --NphoScale ${NPHO_SCALE} \\
    --time_scale ${TIME_SCALE} \\
    --time_shift ${TIME_SHIFT}

echo "[JOB] Finished."
EOF

