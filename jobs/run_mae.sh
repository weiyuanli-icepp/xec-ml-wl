#!/bin/bash
# Usage: ./run_mae.sh

export RUN_NAME="sanity_test"
export EPOCHS=2
export BATCH=1024
export CHUNK_SIZE=256000
export MASK_RATIO=0.2  # 75% masking is standard for MAE
export RESUME_FROM=""
# export PARTITION="a100-daily"
export PARTITION="a100-hourly"
export TIME="0:20:00"
export CONFIG_PATH="config/mae_config.yaml"
# export NPHO_SCALE="0.58"
# export NPHO_SCALE2="1.0"
export TIME_SCALE="5e-5"
export TIME_SHIFT="0.0065"
# export SENTINEL_VALUE="-5.0"
export LOSS_FN="smooth_l1"
export TIME_WEIGHT="0.05"
export AUTO_WEIGHT="false"
export LR_SCHEDULER=""
export MASK_RATIO="0.2"
export TRAIN_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"
export MLFLOW_EXPERIMENT="mae-pretraining"

./submit_mae.sh
