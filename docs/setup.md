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

These ARM64-based nodes require a custom Miniforge installation.

#### Log in to a GH node:

```bash
$ srun --cluster=gmerlin7 --partition=gh-interactive --gres=gpu:1 --mem=40GB --pty /bin/bash
```

#### Install Miniforge:

```bash
$ wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh

$ bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge-arm
```

#### Create Environment:

```bash
# 1. Prepare the environment
$ source $HOME/miniforge-arm/bin/activate

# 2. Create base with system libs
$ mamba create -n xec-ml-wl-gh python=3.10 numpy scipy pandas matplotlib scikit-learn \
    tqdm pyarrow pyyaml jupyterlab ipykernel uproot awkward vector \
    pytorch-lightning torchmetrics tensorboard onnx mlflow \
    -c conda-forge -y

# 3. Activate
$ conda activate xec-ml-wl-gh

# 4. Install PyTorch (GPU)
$ pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 5. Install ONNX Runtime GPU (optional)
$ pip install onnxruntime-gpu
```

#### Update Environment:

```bash
(Assumes already logged into GH node)
# 1. Prepare the environment
$ source $HOME/miniforge-arm/bin/activate
$ mamba env remove -n xec-ml-wl-gh # <- recreating env when changing python version
(optional) $ mamba clean -a -y # Clear cache to free space/remove corrupt tarballs

# 2. Create base environment
mamba create -n xec-ml-wl-gh python=3.12 \
    numpy scipy pandas matplotlib scikit-learn \
    tqdm pyarrow pyyaml jupyterlab ipykernel \
    uproot awkward vector \
    pytorch-lightning=2.4.0 torchmetrics=1.5.0 tensorboard \
    onnx=1.17.0 mlflow \
    -c conda-forge -y

# 3. Activate
$ conda activate xec-ml-wl-gh

# 5. Install PyTorch (GPU)
$ pip install --upgrade pip
$ pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# 6. Install PyG (skip this if PyG packages are not used in Hex face)
$ pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric

# 7. Install ONNX Runtime GPU and others
$ pip install pytorch-lightning==2.4.0 torchmetrics==1.5.0 onnx==1.17.0 onnxruntime

# 8. Verification
$ python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
# -> Expected PyTorch: 2.5.1, CUDA: 12.4, GPU: NVIDIA GH200 120GB

```

### 3. Prepare Batch Job

```bash
$ chmod +x start_jupyter_xec_gpu.sh scan_param/*.sh
```
