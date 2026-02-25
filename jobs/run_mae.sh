#!/bin/bash
# Usage: ./jobs/run_mae.sh
#        DRY_RUN=1 ./jobs/run_mae.sh   # Preview without submitting
#
# Inpainter training with MAE pretrained encoder (frozen).
# All settings are in the YAML config; this script just sets job-level overrides.

export CONFIG_PATH="config/inp/inp_mask0.10_mae_frozen.yaml"
export PARTITION="a100-daily"
export TIME="12:00:00"

# Use SQLite backend
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"

cd "$(dirname "$0")" && ./submit_inpainter.sh
