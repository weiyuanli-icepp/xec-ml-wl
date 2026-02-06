# Environment Setup

The repository supports both **x86 (A100)** and **ARM (Grace-Hopper)** architectures on the Merlin7 cluster. Due to binary incompatibility, **two separate environments** are prepared.

## First-Time Setup

### 1. A100 Nodes (a100-* partition)

These x86-based nodes use the system Anaconda module:

```bash
$ module load anaconda/2024.08
$ conda env create -f env_setting/xec-ml-wl-gpu.yml

# To update
$ conda env update -f env_setting/xec-ml-wl-gpu.yml --prune

# When changing the python version
$ conda env remove -n xec-ml-wl
$ conda env create -f xec-ml-wl-gpu.yml
```

### 2. Grace-Hopper Nodes (gh-* partition)

These ARM64-based nodes require a custom Miniforge installation. PyTorch is installed via pip (ARM64+CUDA wheels not available on conda channels).

#### Log in to a GH node:

```bash
$ srun --cluster=gmerlin7 --partition=gh-interactive --gres=gpu:1 --mem=40GB --pty /bin/bash
```

#### Install Miniforge (first-time only):

```bash
$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
$ bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge-arm
```

#### Create Environment:

```bash
$ source $HOME/miniforge-arm/bin/activate
$ mamba env create -f env_setting/environment_gh.yml

# Activate and verify
$ conda activate xec-ml-wl-gh
$ python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# -> Expected: PyTorch: 2.5.1, CUDA: 12.4, GPU: NVIDIA GH200 120GB
```

#### Update Environment:

```bash
$ source $HOME/miniforge-arm/bin/activate

# Update existing environment
$ mamba env update -f env_setting/environment_gh.yml --prune

# Or recreate from scratch (when changing Python version)
$ mamba env remove -n xec-ml-wl-gh
$ mamba clean -a -y  # Optional: clear cache
$ mamba env create -f env_setting/environment_gh.yml
```

### 3. Prepare Batch Job

```bash
$ chmod +x start_jupyter_xec_gpu.sh scan_param/*.sh
```
