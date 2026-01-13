#!/bin/bash
# Usage: ./run_mae.sh

export RUN_NAME="sanity_test"
export EPOCHS=2
export BATCH=16384
export CHUNK_SIZE=256000
export MASK_RATIO=0.6  # 75% masking is standard for MAE
export PARTITION="a100-daily"
export TIME="1:00:00"
export NPHO_SCALE="0.58"
export NPHO_SCALE2="1.0"
export TIME_SCALE="6.5e-8"
export TIME_SHIFT="0.5"
export SENTINEL_VALUE="-5.0"
export TRAIN_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"
export MLFLOW_EXPERIMENT="gamma_mae"

./submit_mae.sh