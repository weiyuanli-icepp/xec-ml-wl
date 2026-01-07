#!/bin/bash

# Example: Scan Drop Path Rates
# Usage: ./run_scan.sh

# Common Settings
RUN_NAME="run_scan_test_01"
MODEL="convnextv2"
EPOCHS=200
REWEIGHT_MODE="none"
LOSSTYPE="smooth_l1"
LR="8e-4"
BATCH=16384
# BATCH=8096
RESUME_FROM=""
PARTITION="a100-daily"
# PARTITION="gh-daily"
TIME="23:00:00"

export WEIGHT_DECAY="1e-4"
export DROP_PATH=0.0
export SCHEDULER=1 # 1: constant, -1: cosine annealing
export WARMUP_EPOCHS=0
export TIME_SCALE="6.5e-8"
export TIME_SHIFT="0.5"
export SENTINEL_VALUE="-5.0"
export NPHO_SCALE="0.58"
export NPHO_SCALE2="1.0"
export ONNX=""

export EMA_DECAY=-1
export LOSS_BETA=0.1
export MLFLOW_EXPERIMENT="gamma_angle"

export CHUNK_SIZE=5242880 # BATCH * Integer (320)
export TREE_NAME="tree"
export ROOT_PATH="~/meghome/xec-ml-wl/data/E52.8_AngUni_PosSQ/single_run"

# RUN_NAME="runs100_GH_test_sched${SCHEDULER}"
# ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"

# Items to scan
# use constant learning rate for the test phase
# 1. time and npho normalization -> done
#    Best:
#    NphoScale = 1.0, NphoScale2 = 1.0
#    time shift = 0.0, time scale = 2.32e6
#    Comment: time scale = 1e7 can be still tested
# 2. Learning Rates 
#    Ranking: [8e-4], 5e-3, 2e-3, 8e-3, (2e-2, 4e-4), 2e-4, 1e-4
# 3. Loss Types -> smooth_l1 wins!
#    sub-scan: if smooth_l1 wins, scan beta values [0.1, 0.5, 1.0, 2.0]
#        -> not much difference for val_loss, 
#           but for each angle's rms it looks 0.1 performs better
# 4. Drop Path Rates & Weight Decays & ON/OFF EMA
# 5. Reweight Modes, bins (might skip this)
# 6. channel sizes of the model, GNN depths
# 7. Multi-Task Learning

# ===========================================================================
# Scan Loss functions
# LOSSTYPES=("smooth_l1" "cos" "l1" "mse")
# WEIGHT_DECAYS=("1e-5" "3e-5" "1e-4" "3e-4")
# EPOCHS=100
# DP="0.0"
# # WD="5e-5"
# EMA="0.999"

# for LT in "${LOSSTYPES[@]}"; do
#     for WD in "${WEIGHT_DECAYS[@]}"; do
#         echo "----------------------------------------"
#         echo "Preparing job for Loss Type=$LT | WD=$WD"

#         # RUN_NAME="runs100_loss${LT}"
#         RUN_NAME="angle_E52.8_NoRandomization_lt${LT}_dp${DP}_wd${WD}_ema${EMA}"
#         CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
    
#         if [ -f "$CHECK_FILE" ]; then
#             RESUME_FROM="$CHECK_FILE"
#             echo "  -> Found checkpoint: Resuming run."
#         else
#             RESUME_FROM=""
#             echo "  -> No checkpoint found: Starting fresh."
#         fi

#         echo "Submitting job for Loss Type: $LT"

#         # Pass the new params as env vars
#         # LOSS_BETA=$LB ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LT" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
#         export DROP_PATH=$DP
#         export WEIGHT_DECAY=$WD
#         export EMA_DECAY=$EMA

#         # Submit the job
#         ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LT" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"

#         sleep 1
#     done
# done

# ===========================================================================
# Scan Loss functions
# Test
# EPOCHS=5
# LOSSTYPE="smooth_l1"
# LB="0.1"
# RUN_NAME="runs100_smoothl1_beta${LB}_test"
# RESUME_FROM=""
# LOSS_BETA=$LB ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"

# Full Scan
# LOSS_BETAS=("0.1" "0.5" "1.0" "2.0")
# LOSSTYPE="smooth_l1"

# for LB in "${LOSS_BETAS[@]}"; do
#     RUN_NAME="runs100_smoothl1_beta${LB}"
#     CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
    
#     if [ -f "$CHECK_FILE" ]; then
#         RESUME_FROM="$CHECK_FILE"
#         echo "  -> Found checkpoint: Resuming run."
#     else
#         RESUME_FROM=""
#         echo "  -> No checkpoint found: Starting fresh."
#     fi
    
#     echo "Submitting job for Loss Type: $LOSSTYPE with Beta: $LB"
    
#     # Pass the new params as env vars
#     LOSS_BETA=$LB ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
    
#     sleep 1
# done

# ===========================================================================
# 4. Scan Drop Path Rates & Weight Decays & ON/OFF EMA
# Drop Path Rates to scan
# /////////////////////////////////////////////////////////
# DROP_PATHS=("0.0" "0.1" "0.2" "0.3")
EPOCHS=3
# PARTITION="gh-general"
# TIME="3-00:00:00"
# DROP_PATHS=("0.0" "0.1") # <-
DROP_PATHS=("0.0")
# Weight Decays to scan
# WEIGHT_DECAYS=("1e-4" "1e-3" "1e-2")
# WEIGHT_DECAYS=("0.0" "1e-6" "1e-5" "1e-4") 
# WEIGHT_DECAYS=("1e-7" "1e-6" "1e-5" "1e-4") # <-
WEIGHT_DECAYS=("5e-5")
# WEIGHT_DECAYS=("5e-5" "0.0")
# WEIGHT_DECAYS=("1e-6") 
# WEIGHT_DECAYS=("1e-4") 
# EMA Decay settings: -1 (OFF), 0.999 (ON)
# EMA_SETTINGS=("0.99" "0.999") # <-
EMA_SETTINGS=("-1")
# EMA_SETTINGS=("-1" "0.999" "0.9999")
# export TIME_SCALE="1e7"
# LOSSTYPES=("smooth_l1" "cos" "l1" "mse") # <-
LOSSTYPES=("smooth_l1")
# LOSSTYPE="cos"

# Set base Learning Rate from previous best result
LR="8e-4"
echo "Starting Loss type, Drop Path, Weight Decay, EMA scan..."
for LT in "${LOSSTYPES[@]}"; do
    LOSSTYPE=$LT
    echo "========================================"
    echo "Scanning Loss Type: $LOSSTYPE"
    for DP in "${DROP_PATHS[@]}"; do
        echo "Scanning Drop Path Rate: $DP"
        for WD in "${WEIGHT_DECAYS[@]}"; do
            echo "Scanning Weight Decay: $WD"
            for EMA in "${EMA_SETTINGS[@]}"; do
                echo "----------------------------------------"
                echo "Preparing job for DP=$DP | WD=$WD | EMA=$EMA"
                # Construct a descriptive run name
                # e.g., runs100_reg_dp0.1_wd1e-3_ema0.999
                EMA_LABEL="off"
                if [ "$EMA" != "-1" ]; then
                    EMA_LABEL="$EMA"
                fi
                
                # RUN_NAME="runs100_reg_dp${DP}_wd${WD}_ema${EMA_LABEL}"
                # RUN_NAME="test_runs100_filesplit_trans"
                # RUN_NAME="runs100_trans_lt${LOSSTYPE}_wd${WD}_ema${EMA_LABEL}"
                # RUN_NAME="runs100_trans_ts${TIME_SCALE}_wd${WD}_ema${EMA_LABEL}"
                # RUN_NAME="angle_E52.8_NoRandomization_lt${LOSSTYPE}_dp${DP}_wd${WD}_ema${EMA_LABEL}" # <-
                RUN_NAME="sanity_check"
                
                # Check for existing checkpoint to resume
                CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
                # if RUN_NAME is "sanity_check", start fresh
                if [ "$RUN_NAME" == "sanity_check" ]; then
                    RESUME_FROM=""
                    echo "  -> Sanity check run: Starting fresh."
                else
                    if [ -f "$CHECK_FILE" ]; then
                        RESUME_FROM="$CHECK_FILE"
                        echo "  -> Found checkpoint: Resuming run."
                    else
                        RESUME_FROM=""
                        echo "  -> No checkpoint found: Starting fresh."
                    fi
                fi
                
                echo "Submitting job: LT=$LOSSTYPE | DP=$DP | WD=$WD | EMA=$EMA"
                
                # Export variables for this specific job
                export DROP_PATH=$DP
                export WEIGHT_DECAY=$WD
                export EMA_DECAY=$EMA
                
                # Submit the job
                ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
                
                sleep 1
            done
        done
    done
done
# ===========================================================================
# REWEIGHT_MODES=("theta" "phi" "theta_phi")
# LOSSTYPES=("smooth_l1" "cos" "l1" "mse")

# for RM in "${REWEIGHT_MODES[@]}"; do
#     for LT in "${LOSSTYPES[@]}"; do
#         RUN_NAME="scan_rw${RM}_loss${LT}"
#         CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
        
#         if [ -f "$CHECK_FILE" ]; then
#             RESUME_FROM="$CHECK_FILE"
#             echo "  -> Found checkpoint: Resuming run."
#         else
#             RESUME_FROM=""
#             echo "  -> No checkpoint found: Starting fresh."
#         fi
        
#         echo "Submitting job for Reweight Mode: $RM | Loss Type: $LT"
        
#         # Pass the new params as env vars
#         REWEIGHT_MODE=$RM LOSSTYPE=$LT ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LR" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
        
#         sleep 1
#     done
# done

# # ===========================================================================
# # Scan learning rates
# LEARNING_RATES=("1e-5" "5e-5" "1e-4" "2e-4" "4e-4" "8e-4" "2e-3" "5e-3")
# LEARNING_RATES=("8e-3" "2e-2" "6e-2" "1e-1" "3e-1")
# LEARNING_RATES=("8e-4" "5e-4" "1e-3")
# Copy the checkpoint file to the safe place before running this scan!
# LEARNING_RATES=("8e-4")
# EPOCHS=1500

# for LEARNING_RATE in "${LEARNING_RATES[@]}"; do
#    #  RUN_NAME="runs100_NphoScale1_lr${LEARNING_RATE}"
#     RUN_NAME="runs100_NoTimeShift_NphoScale1_lr${LEARNING_RATE}"

#         # ===================================================
#         # Uncomment following part for resuming run
#         CHECK_FILE="$HOME/meghome/xec-ml-wl/artifacts/${RUN_NAME}/checkpoint_last.pth"
        
#         if [ -f "$CHECK_FILE" ]; then
#             RESUME_FROM="$CHECK_FILE"
#             echo "  -> Found checkpoint: Resuming run."
#         else
#             RESUME_FROM=""
#             echo "  -> No checkpoint found: Starting fresh."
#         fi
#         # ====================================================
    
#     LEARNING_RATE="1e-4"
#     echo "Submitting job for Learning Rate: $LEARNING_RATE"
    
#     # Pass the new param as an env var
#     ./submit_job.sh "$RUN_NAME" "$MODEL" "$EPOCHS" "$REWEIGHT_MODE" "$LOSSTYPE" "$LEARNING_RATE" "$BATCH" "$RESUME_FROM" "$PARTITION" "$TIME"
    
#     sleep 1
# done

echo "All jobs submitted."
