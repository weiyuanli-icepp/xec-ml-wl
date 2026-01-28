#!/bin/bash
# Usage:
# 1. Make sure to load anaconda module
# 2. Activate xec-ml-wl conda environment
# 3. salloc --cluster=gmerlin7 --partition=a100-interactive --gres=gpu:1 --mem=64G --time=07:00:00
# 4. Run this script: ./interactive_regressor.sh

# Conda activation for GH nodes (miniforge)
HOSTNAME_SHORT="$(hostname -s 2>/dev/null || hostname)"
if [[ "$HOSTNAME_SHORT" =~ ^gpu00[1-9]$ ]]; then
    if [ "${CONDA_DEFAULT_ENV:-}" != "xec-ml-wl-gh" ]; then
        if [ -f "/data/user/ext-li_w1/miniforge-arm/bin/activate" ]; then
            source /data/user/ext-li_w1/miniforge-arm/bin/activate
            conda activate xec-ml-wl-gh
        else
            echo "Conda activate script not found at /data/user/ext-li_w1/miniforge-arm/bin/activate" >&2
            exit 1
        fi
    fi
fi

# Prioritize conda's libstdc++ over system GCC module
if [ -n "$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
fi

export CONFIG_PATH="config/reg/ene_reg_test.yaml"

cd $HOME/meghome/xec-ml-wl
echo "Moved to directory $(pwd)"

# Build command
python -m lib.train_regressor --config ${CONFIG_PATH}
