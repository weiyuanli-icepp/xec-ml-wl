#!/usr/bin/env bash
# Usage: ./start_jupyter_xec_gpu.sh [PORT] [TIME] [PARTITION]
# Example: ./start_jupyter_xec_gpu.sh 8888 02:00:00 a100-interactive

set -euo pipefail
PORT="${1:-8888}"
PARTITION="${3:-a100-interactive}"
ENV_NAME="xec-ml-wl"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  # Not inside a Slurm job yet → allocate an interactive GPU shell and run this same script ON the node.
  # exec srun --mpi=none --pty --clusters=gmerlin7 -p "${PARTITION}" --gres=gpu:1 --mem=48G --time "${TIME}" bash -lc \
  #   "HOSTNAME=\$(hostname); echo \"[INFO] Allocated: \$HOSTNAME\"; $(printf %q "$0") ${PORT} ${TIME} ${PARTITION}"
  exec srun --mpi=none --pty --clusters=gmerlin7 -p "${PARTITION}" --gres=gpu:1 --mem=48G bash -lc \
    "HOSTNAME=\$(hostname); echo \"[INFO] Allocated: \$HOSTNAME\"; $(printf %q "$0") ${PORT} ${PARTITION}"
fi

# ---- From here, we are on the compute node ----

# Load modules (if available) and initialize conda in THIS shell
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
module load anaconda/2024.08 2>/dev/null || true

# Bash: load conda hook; (use shell.zsh hook if you run this script with zsh)
eval "$(/opt/psi/Programming/anaconda/2024.08/conda/bin/conda shell.bash hook)"

# Activate env
conda activate "${ENV_NAME}"

# Register kernel (harmless if already present)
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (${ENV_NAME})" || true

echo "[INFO] Jupyter on node: $(hostname)"
echo "[INFO] Launching JupyterLab on port ${PORT}"

# Bind to loopback for security; your browser connects via SSH tunnel
exec jupyter lab --no-browser --ip=127.0.0.1 --port="${PORT}" --ServerApp.root_dir="$HOME"
