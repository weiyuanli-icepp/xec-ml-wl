#!/usr/bin/env bash
# Usage: ./start_jupyter_xec_gpu.sh [PARTITION] [TIME] [PORT]
# Example: ./start_jupyter_xec_gpu.sh a100-interactive 12:00:00 8888

set -euo pipefail
PARTITION="${1:-a100-interactive}"
TIME="${2:-12:00:00}"
PORT="${3:-8888}"
ENV_NAME="xec-ml-wl"

# 2. Switch Environment based on Partition
# If partition name starts with "gh" (Grace-Hopper), use the ARM environment.
# Otherwise (A100), use the standard x86 environment.
if [[ "$PARTITION" == gh* ]]; then
    ENV_NAME="xec-ml-wl-gh"
else
    ENV_NAME="xec-ml-wl"
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] Submitting job to partition: $PARTITION"
  echo "[INFO] Time limit: $TIME"
  echo "[INFO] Target Environment: $ENV_NAME"
  
  # Not inside a Slurm job yet → allocate an interactive GPU shell and run this same script ON the node.
  # exec srun --mpi=none --pty --clusters=gmerlin7 -p "${PARTITION}" --gres=gpu:1 --mem=48G --time "${TIME}" bash -lc \
  #   "HOSTNAME=\$(hostname); echo \"[INFO] Allocated: \$HOSTNAME\"; $(printf %q "$0") ${PORT} ${TIME} ${PARTITION}"
  exec srun --mpi=none --pty --clusters=gmerlin7 \
    -p "${PARTITION}" \
    --gres=gpu:1 \
    --mem=48G \
    --time "${TIME}" \
    bash -lc "HOSTNAME=\$(hostname); echo \"[INFO] Allocated: \$HOSTNAME\"; $(printf %q "$0") ${PARTITION} ${TIME} ${PORT}"
fi

# ---- From here, we are on the compute node ----

# Load modules (if available) and initialize conda in THIS shell
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
module load anaconda/2024.08 2>/dev/null || true

# Bash: load conda hook; (use shell.zsh hook if you run this script with zsh)
eval "$(/opt/psi/Programming/anaconda/2024.08/conda/bin/conda shell.bash hook)"

# Activate env
echo "[INFO] Activating environment: ${ENV_NAME}"
conda activate "${ENV_NAME}"

# Register kernel (harmless if already present)
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})" || true

echo "[INFO] Jupyter on node: $(hostname)"
echo "[INFO] Launching JupyterLab on port ${PORT}"

# Bind to loopback for security; your browser connects via SSH tunnel
exec jupyter lab --no-browser --ip=127.0.0.1 --port="${PORT}" --ServerApp.root_dir="$HOME"
