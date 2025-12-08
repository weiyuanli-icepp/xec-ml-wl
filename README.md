# xec-ml-wl

## Deep Learning Analysis for the MEG II Liquid Xenon (LXe) Detector

This repository contains machine learning models (CNNs and Graph Neural Networks) to regress the emission angle (**$\theta$**, **$\phi$**) of photons detected by the LXe detector, utilizing both photon count (**$N_{\mathrm{pho}}$**) and timing information (**$t_{\mathrm{pm}}$**) in each photo-sensor (4092 SiPMs and 668 PMTs).

---

## 1. Environment Setup

The repository supports both **x86 (A100)** and **ARM (Grace-Hopper)** architectures on the Merlin7 cluster. Due to binary incompatibility, I have prepared **two separate environments**.

### First-Time Setup

### 1. A100 Nodes (a100-* partition)

These x86-based nodes use the system Anaconda module:

```bash
$ module load anaconda/2024.08
$ conda env create -f env_setting/xec-ml-wl.yml
```

### 2. Grace-Hopper Nodes (gh-* partition)

These ARM64-based nodes require a custom Miniforge installation.

#### Log in to a GH node:

```bash
$ srun --cluster=gmerlin7 --partition=gh-interactive --gres=gpu:1 --pty /bin/bash
```

#### Install Miniforge:

```bash
$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

$ bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge-arm
```

#### Create Environment:

```bash
$ source $HOME/miniforge-arm/bin/activate

# 1. Create base with system libs
$ mamba create -n xec-ml-wl-gh python=3.10 numpy scipy pandas matplotlib scikit-learn \
    tqdm pyarrow pyyaml jupyterlab ipykernel uproot awkward vector \
    pytorch-lightning torchmetrics tensorboard onnx mlflow \
    -c conda-forge -y

# 2. Activate
$ conda activate xec-ml-wl-gh

# 3. Install PyTorch (GPU)
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install ONNX Runtime GPU (optional)
$ pip install onnxruntime-gpu
```

### 3. Prepare Batch Job

```bash
$ chmod +x start_jupyter_xec_gpu.sh submit_job.sh run_scan.sh
```

---

## 2. Usage

### A. Batch Training (Recommended)

We use `submit_job.sh`, which automatically detects the CPU architecture of the allocated node and activates the correct environment (x86 or ARM).

#### 1. Quick Submission

```bash
# Usage:
# ./submit_job.sh [RUN_NAME] [MODEL] [EPOCHS] [REWEIGHT] [LOSS] [LR] [BATCH] [RESUME] [PARTITION] [TIME]
$ ./submit_job.sh test_run_01 convnextv2 20 none smooth_l1 3e-4 1024 "" a100-hourly 04:00:00
```

#### 2. Hyperparameter Scanning

```bash
# Inside run_scan.sh:
export EMA_DECAY=0.999
export LOSS_BETA=1.0

./submit_job.sh scan_run_01 ...
```

### B. Interactive Jupyter Session

```bash
# Syntax:
# ./start_jupyter_xec_gpu.sh [PARTITION] [TIME] [PORT]
./start_jupyter_xec_gpu.sh gh-interactive 02:00:00 8888
```
1. Wait for the connection URL.
2. Tunnel ports locally: `ssh -N -L 8888:localhost:8888 -J <user>@login001 <user>@gpuXXX`
3. Paste the URL with token in the browser.

---

## 3. Output & Artifacts

All results are logged to **MLflow** and stored in the `artifacts/` directory.

### Key Artifacts

* `checkpoint_best.pth` — Best model weights (includes EMA state).
* `checkpoint_last.pth` — Last epoch's model weights (includes EMA state).
* `predictions_*.csv` — Validation predictions vs truth.
* `meg2ang_*.onnx` — Exported ONNX model for C++ inference.
* `validation_results_*.root` — ROOT file containing event-by-event predictions and truth variables.

### Plots

* `resolution_profile_*.pdf`: 68% width resolution vs $\theta$/$\phi$.
* `saliency_profile_*.pdf`: Physics Sensitivity analysis (Gradient of output w.r.t input Npho/Time).
* `worst_event_*.pdf`: Event displays of the highest-loss events.

### Visualization Tools

```bash
# Start MLflow (Track metrics & PDFs)
$ mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port XXXX

# Start TensorBoard (Track Loss Curves)
$ tensorboard --logdir runs --host 0.0.0.0 --port YYYY
```

---

## 4. Model & Training Features

The primary model is **ConvNeXt V2**, adapted for LXe geometry.

### Key Features

* **Geometry-aware projections**: Maps PMTs to split faces (Inner/Outer/Side) and Hex grids (Top/Bottom).
* **EMA (Exponential Moving Average)**: Maintains a "shadow" model with smoothed weights ($W_{ema} = \beta W_{ema} + (1-\beta)W_{live}$). Significantly improves stability on noisy physics data.
* **Physics saliency**: Calculates $\partial \theta / \partial N_{pho}$ to visualize which detector faces drive the decision.
* **Reliable ONNX export**: Automatically exports the EMA model (if active) and replaces dynamic pooling with Resize(bilinear) for compatibility.

### Argument List (`run_training_cli.py`)

| Argument          | Default         | Description                      |
| ----------------- | --------------- | -------------------------------- |
| `--root`          | Required        | Path to input ROOT file          |
| `--model`         | `convnextv2`    | Model architecture               |
| `--epochs`        | `20`            | Training epochs                  |
| `--batch`         | `256`           | Batch size                       |
| `--lr`            | `3e-4`          | Learning rate                    |
| `--loss_type`     | `smooth_l1`     | Loss function                    |
| `--loss_beta`     | `1.0`           | SmoothL1 parameter               |
| `--ema_decay`     | `0.999`         | EMA decay rate (-1 disables EMA) |
| `--reweight_mode` | `none`          | theta / phi / theta_phi          |
| `--use_scheduler` | `-1`            | -1 for cosine, 1 for constant    |
| `--npho_branch`   | `relative_npho` | Photon count branch              |
| `--time_branch`   | `relative_time` | Timing branch                    |
| `--NphoScale`     | `1e5`           | Photon count normalization       |
| `--time_scale`    | `2.32e6`        | Time normalization               |
| `--time_shift`    | `-0.29`         | Offset shift                     |
| `--onnx`          | `*.onnx`        | Output ONNX filename             |

---

## 5. Resuming Training
The script supports resumption. It detects if an EMA state exists in the checkpoint and loads it; otherwise, it syncs the EMA model with the loaded weights to prevent training divergence.

```bash
--resume_from "artifacts/<run name>/checkpoint_last.pth"
```
or
```bash
--resume_from "artifacts/<run name>/checkpoint_best.pth"
```