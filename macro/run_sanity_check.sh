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

    # Try multiple possible miniforge locations
    CONDA_PATHS=(
        "/data/user/${USER}/miniforge3"
        "/data/user/${USER}/miniforge-arm"
        "/data/user/${USER}/mambaforge"
        "$HOME/miniforge3"
        "$HOME/mambaforge"
    )

    CONDA_BASE=""
    for path in "${CONDA_PATHS[@]}"; do
        if [ -f "$path/bin/activate" ]; then
            CONDA_BASE="$path"
            break
        fi
    done

    if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
        if [ -n "$CONDA_BASE" ]; then
            echo "[INFO] Found conda at $CONDA_BASE"
            echo "[INFO] Activating $ENV_NAME for GH node..."
            source "$CONDA_BASE/bin/activate"
            conda activate "$ENV_NAME"
        else
            echo "[ERROR] Miniforge/mambaforge not found. Searched:" >&2
            for path in "${CONDA_PATHS[@]}"; do
                echo "  - $path" >&2
            done
            echo "" >&2
            echo "Please install miniforge for ARM64:" >&2
            echo "  wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh" >&2
            echo "  bash Miniforge3-Linux-aarch64.sh -b -p /data/user/\${USER}/miniforge3" >&2
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
