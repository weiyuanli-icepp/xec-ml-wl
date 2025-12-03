# xec-ml-wl

**Deep Learning Analysis for the MEG II Liquid Xenon (LXe) Detector**

This repository contains machine learning models (CNNs and Graph Neural Networks) to regress the emission angle (theta, phi) of photons detected by the LXe detector, utilizing both photon count (N_pho) and timing information.

## 1. Environment Setup

The project supports both **x86 (A100)** and **ARM (Grace-Hopper)** architectures on the Merlin7 cluster.

### First Time Setup

**1. Load Base Tools:**

    module load anaconda/2024.08

**2. Create Conda Environments:**
You need separate environments for different node types due to architecture differences (x86 vs ARM).

* **For Standard GPU Nodes (A100):**

    conda env create -f env_setting/xec-ml-wl.yml

* **For Grace-Hopper Nodes (gh-interactive):**

    conda env create -f env_setting/environment_gh.yml

**3. Prepare Scripts:**

    chmod +x start_jupyter_xec_gpu.sh submit_job.sh run_scan.sh

## 2. Usage

### A. Interactive Jupyter Session

To start a JupyterLab session on a compute node. The script automatically selects the correct environment based on the partition.

**Syntax:** `./start_jupyter_xec_gpu.sh [PARTITION] [TIME] [PORT]`

* **Standard A100 Node:**

    ./start_jupyter_xec_gpu.sh a100-interactive 02:00:00 8888

* **Grace-Hopper Node:**

    ./start_jupyter_xec_gpu.sh gh-interactive 02:00:00 8888

**Connecting to Jupyter:**

1.  Wait for the script to output the allocated node (e.g., `gpu105`) and the token URL.
2.  On your local machine, open an SSH tunnel:

    ssh -N -L 8888:localhost:8888 -J <user>@login001 <user>@gpuXXX

    *(Replace `gpuXXX` with the allocated node name).*
3.  Open `http://localhost:8888` in your browser.

### B. Batch Training (Slurm Jobs)

To submit training jobs to the queue without opening Jupyter.

* **Submit Single Job:**

    ./submit_job.sh <RUN_NAME> <LR> <BATCH_SIZE> <PARTITION> <TIME>
    #### Example:
    ./submit_job.sh test_run_01 3e-4 2048 a100-hourly 04:00:00

* **Run Hyperparameter Scan:**
    Edit `run_scan.sh` to define parameters, then run:

    ./run_scan.sh

## 3. Output & Artifacts

All results are stored in the `artifacts/` and `runs/` directories.

### Directory Structure

```text
xec-ml-wl/
├── artifacts/
│   └── <run_name>/              # Created automatically per run
│       ├── checkpoint_last.pth  # Latest model state (for resuming)
│       ├── checkpoint_best.pth  # Best model state (lowest val_loss)
│       ├── predictions.csv      # CSV with Pred vs Truth for analysis
│       ├── meg2ang.onnx         # Exported model for C++ inference
│       └── *.pdf                # Plots (Scatter, Residuals, Event Displays)
├── runs/                        # TensorBoard logs
├── mlruns/                      # MLflow logs
```

### Visualization

To track training progress and view logged artifacts:

#### 1. Start MLflow UI (Tracks parameters, metrics, and PDF artifacts)
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000 &

#### 2. Start TensorBoard (Tracks loss curves)
tensorboard --logdir runs --host 0.0.0.0 --port 6006 &

*Open http://localhost:5000 and http://localhost:6006 locally (requires tunneling).*

`ssh -N -L 5000(or 6006):localhost:5000(or 6006) <user_name>@login00X.merlin7.psi.ch`

## 4. Model Architectures

This repository implements two main architectures. Both use a **Geometry-Aware** preprocessing step that maps the 4760 sensors onto 2D grids (Inner, Upstream, Downstream, Outer) and Graph structures (Top, Bottom).

### Input Data

* **Channels:** 2 (N_pho, Time)
* **Preprocessing:**
    * N_pho: Log-scaled log(1 + N_pho).
    * Time: Normalized (T / 100ns) and masked (set to 0 where N_pho=0).

### 1. Simple CNN (`angle_model_geom.py`)

* **Backbone:** Shallow CNN (2 Convolutional layers + Pooling).
* **Outer Face:** Uses "Finegrid" mode to stitch Coarse and Fine PMTs into a single image.
* **Hex Faces:** Processed using a custom Graph Convolution (`HexGraphConv`).
* **Speed:** Fast training (~20s / epoch), good baseline.

### 2. ConvNeXt V2 (`angle_model_geom_convnextv2.py`)

* **Backbone:** Deep modern architecture based on [ConvNeXt V2](https://arxiv.org/abs/2301.00808).
* **Key Features:**
    * **GRN (Global Response Normalization):** Enhances feature competition, crucial for sparse photon data.
    * **LayerNorm & GELU:** Stable training for regression.
    * **Depthwise Separable Convs:** Large receptive field (7 x 7).
* **Performance:** Lower loss (~13 deg error vs 18 deg baseline), robust to noise.

## 5. Resuming Training

You can interrupt and resume training at any time. Checkpoints are saved automatically.

To resume, simply pass the path to the checkpoint file in your python script arguments:

main_angle_convnextv2_with_args(
    # ... other args ...
    resume_from="artifacts/my_previous_run/checkpoint_last.pth"
)

This will:
1.  Load model weights and optimizer state.
2.  Restore the correct epoch number.
3.  Continue logging to the **same** MLflow/TensorBoard run ID.