#!/bin/bash
# Wrapper script for sanity_check.py with automatic environment activation
#
# Usage:
#   ./macro/run_sanity_check.sh --data /path/to/data.root --pipeline all --device cuda
#
# This script handles conda environment activation for both A100 and GH nodes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Detect node type and activate appropriate environment
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"

if [[ "$HOSTNAME_SHORT" =~ ^gpu00[1-9]$ ]]; then
    # GH nodes (gpu001-gpu009) - use miniforge with ARM environment
    ENV_NAME="xec-ml-wl-gh"
    CONDA_BASE="/data/user/${USER}/miniforge-arm"

    if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
        if [ -f "$CONDA_BASE/bin/activate" ]; then
            echo "[INFO] Activating $ENV_NAME for GH node..."
            source "$CONDA_BASE/bin/activate"
            conda activate "$ENV_NAME"
        else
            echo "[ERROR] Conda not found at $CONDA_BASE" >&2
            echo "Please install miniforge or set up the environment manually." >&2
            exit 1
        fi
    fi
else
    # A100/other nodes - use standard anaconda
    ENV_NAME="xec-ml-wl"

    if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
        # Try module load if available
        if command -v module &> /dev/null; then
            module load anaconda/2024.08 2>/dev/null || true
        fi

        if command -v conda &> /dev/null; then
            echo "[INFO] Activating $ENV_NAME..."
            conda activate "$ENV_NAME" 2>/dev/null || {
                echo "[WARN] Could not activate $ENV_NAME, using current environment"
            }
        fi
    fi
fi

# Prioritize conda's libraries
if [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

# Run the sanity check
cd "$REPO_ROOT"
echo "[INFO] Running sanity_check.py with args: $*"
python macro/sanity_check.py "$@"
