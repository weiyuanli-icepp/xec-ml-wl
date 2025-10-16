#!/usr/bin/env bash
# JupyterLab launcher for the xec-ml-wl environment on PSI cluster.
# Usage:
#   chmod +x start_jupyter_xec.sh
#   ./start_jupyter_xec.sh 8888   # optional port (default 8888)
#
# If you're using Slurm, first allocate a node (GPU recommended):
#   srun --pty -p a100-interactive --gres=gpu:1 --time=02:00:00 bash
#
# If Anaconda is provided via modules, load it first:
#   module load anaconda/2024.08
#
# Then run this script on the allocated node.

set -euo pipefail

PORT="${1:-8888}"

# Ensure conda is available (module load if you haven't)
if ! command -v conda >/dev/null 2>&1; then
  echo "[WARN] conda not on PATH; trying to source from PSI Anaconda 2024.08..."
  # Adjust this path if different on your system
  if [ -f /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh ]; then
    source /opt/psi/Programming/anaconda/2024.08/conda/etc/profile.d/conda.sh
  fi
fi

# Activate env (GPU or CPU; change name if you created -cpu variant)
ENV_NAME="xec-ml-wl"
conda activate "${ENV_NAME}" || { echo "[ERROR] Failed to activate ${ENV_NAME}"; exit 1; }

# First run only: register a Jupyter kernel
python -m ipykernel install --user --name "${ENV_NAME}" --display-name "Python (xec-ml-wl)" || true

# Optional: generate a minimal Jupyter config the first time
JDIR="${HOME}/.jupyter"
mkdir -p "${JDIR}"
if [ ! -f "${JDIR}/jupyter_lab_config.py" ]; then
  cat > "${JDIR}/jupyter_lab_config.py" << 'EOF'
c.ServerApp.ip = '127.0.0.1'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_remote_access = False
c.ServerApp.iopub_data_rate_limit = 1.0e10
EOF
fi

# Launch Lab on the chosen port
jupyter lab --no-browser --port="${PORT}"
