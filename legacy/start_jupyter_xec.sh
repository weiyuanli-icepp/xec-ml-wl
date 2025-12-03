#!/usr/bin/env bash
set -euo pipefail

# If you need the module system in non-interactive shells:
# (harmless if modules are already available)
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
module load anaconda/2024.08 2>/dev/null || true

# Initialize conda for *this* shell
eval "$(/opt/psi/Programming/anaconda/2024.08/conda/bin/conda shell.bash hook)"

# Now activation works
conda activate xec-ml-wl

# ...your commands...
jupyter lab --no-browser --port 8888
