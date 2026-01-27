#!/bin/bash
# Usage: ./run_regressor.sh
#
# Example script for running regressor training.
# Modify the parameters below and run this script.

# Job settings
export RUN_NAME="angle_test"
export PARTITION="a100-hourly"
export TIME="1:00:00"

# Config file
export CONFIG_PATH="config/train_config.yaml"

# Data paths (override config if needed)
export TRAIN_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_train.root"
export VAL_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/large_val.root"

# Training parameters (override config if needed)
export EPOCHS=20
export BATCH_SIZE=4096
export LR=3e-4
# export WEIGHT_DECAY=1e-4
# export WARMUP_EPOCHS=2
# export EMA_DECAY=0.999
# export CHANNEL_DROPOUT_RATE=0.1
# export GRAD_CLIP=1.0

# Tasks (space-separated: angle energy timing uvwFI)
export TASKS="angle"

# Model architecture
# export OUTER_MODE="finegrid"
# export OUTER_FINE_POOL="3 3"
# export HIDDEN_DIM=256
# export DROP_PATH_RATE=0.0

# Loss balancing
# export LOSS_BALANCE="manual"  # or "auto"

# Checkpoint/MLflow
# export RESUME_FROM=""
# export SAVE_DIR="artifacts"
export MLFLOW_EXPERIMENT="gamma_angle"

# ONNX export (set to "null" to disable)
# export ONNX="meg2ang_convnextv2.onnx"

./submit_regressor.sh
