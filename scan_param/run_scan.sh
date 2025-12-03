#!/bin/bash

# Example: Scan Drop Path Rates
# Usage: ./run_scan.sh

# Common Settings
# RUN_NAME="run_scan_test_01"
MODEL="convnextv2"
EPOCHS=100
REWEIGHT_MODE="none"
LOSSTYPE="smooth_l1"
LR=3e-4
BATCH=2024
RESUME_FROM=""
PARTITION="a100-daily"
TIME="23:00:00"

WEIGHT_DECAY=1e-4
DROP_PATH=0.25
SCHEDULER=-1
TIME_SCALE=1e-7
TIME_SHIFT=0.0
WARMUP_EPOCHS=2
NPHO_SCALE=1.0
ONNX=""
MLFLOW_EXPERIMENT="gamma_angle"

CHUNK_SIZE=32000
TREE_NAME="tree"
ROOT_PATH="~/meghome/xec-ml-wl/data/MCGammaAngle_0-49.root"

# ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"

# Scan Drop Path Rates
# DROPS=("0.0" "0.1" "0.2")
# DROPS=("0.3" "0.4")

# for DP in "${DROPS[@]}"; do
#     RUN_NAME="scan_dp${DP}_sched${SCHEDULER}"
        # CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
        
        # if [ -f "$CHECK_FILE" ]; then
        #     RESUME_FROM="$CHECK_FILE"
        #     echo "  -> Found checkpoint: Resuming run."
        # else
        #     RESUME_FROM=""
        #     echo "  -> No checkpoint found: Starting fresh."
        # fi
    
#     echo "Submitting job for DropPath: $DP"
    
#     # Pass the new param as an env var
#     DROP_PATH=$DP ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
    
#     sleep 1
# done

# WEIGHTDECAYS=("3e-5" "3e-4" "1e-3")

# for WD in "${WEIGHTDECAYS[@]}"; do
#     RUN_NAME="scan_wd${WD}_sched${SCHEDULER}"
        # CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
        
        # if [ -f "$CHECK_FILE" ]; then
        #     RESUME_FROM="$CHECK_FILE"
        #     echo "  -> Found checkpoint: Resuming run."
        # else
        #     RESUME_FROM=""
        #     echo "  -> No checkpoint found: Starting fresh."
        # fi
    
#     echo "Submitting job for Weight Decay: $WD"
    
#     # Pass the new param as an env var
#     WEIGHT_DECAY=$WD ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
    
#     sleep 1
# done

REWEIGHT_MODES=("theta" "phi" "theta_phi")
LOSSTYPES=("smooth_l1" "cos" "l1" "mse")

for RM in "${REWEIGHT_MODES[@]}"; do
    for LT in "${LOSSTYPES[@]}"; do
        RUN_NAME="scan_rw${RM}_loss${LT}"
        CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
        
        if [ -f "$CHECK_FILE" ]; then
            RESUME_FROM="$CHECK_FILE"
            echo "  -> Found checkpoint: Resuming run."
        else
            RESUME_FROM=""
            echo "  -> No checkpoint found: Starting fresh."
        fi
        
        echo "Submitting job for Reweight Mode: $RM | Loss Type: $LT"
        
        # Pass the new params as env vars
        REWEIGHT_MODE=$RM LOSSTYPE=$LT ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
        
        sleep 1
    done
done

echo "All jobs submitted."