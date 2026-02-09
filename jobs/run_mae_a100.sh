#!/bin/bash
# Usage: ./run_mae.sh

export EPOCHS=50
# export BATCH=1024
# export CHUNK_SIZE=256000
export MASK_RATIO=0.65  # 75% masking is standard for MAE
export RESUME_FROM=""
# export PARTITION="a100-daily"
export PARTITION="a100-daily"
export TIME="03:30:00"
export CONFIG_PATH="config/mae_config.yaml"
# export NPHO_SCALE="0.58"
# export NPHO_SCALE2="1.0"
export TIME_SCALE="6.5e-8"
export TIME_SHIFT="0.5"
# export SENTINEL_VALUE="-5.0"
export LOSS_FN="smooth_l1"
export TIME_WEIGHT="0.05"
export AUTO_WEIGHT="false"
export LR="2.5e-4"
export LR_SCHEDULER="cosine"
export LR_MIN="1e-8"
# export TRAIN_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
# export VAL_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"
export MLFLOW_EXPERIMENT="mae_pretraining"

# export RUN_NAME="mae_tw0.05_tsc6.5e-8_tsh0.5_lr2.5e-4_lrmin1e-8_epochs50_mask0.2_cosine"
export RUN_NAME="mae_mask0.65"

./submit_mae.sh
