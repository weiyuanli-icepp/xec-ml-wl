# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML model for the MEG II Liquid Xenon (LXe) detector that regresses physical observables from photon sensor data:
- **Input:** 4760 sensors (4092 SiPMs + 668 PMTs) providing photon count (Npho) and timing data
- **Tasks:** Emission angle (θ, φ), energy, timing, position (uvwFI) regression

## Commands

### Environment Setup
```bash
# x86/A100 nodes
module load anaconda/2024.08
conda env create -f env_setting/xec-ml-wl-gpu.yml

# ARM/Grace-Hopper nodes
mamba create -n xec-ml-wl-gh python=3.10 ...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Training
```bash
# Config-based training (direct)
python -m lib.train_regressor --config config/train_config.yaml

# With CLI overrides
python -m lib.train_regressor --config config/train_config.yaml --lr 1e-4 --epochs 30

# SLURM batch submission (set env vars, then run submit script)
export RUN_NAME="my_run" CONFIG_PATH="config/train_config.yaml" PARTITION="a100-daily"
./scan_param/submit_regressor.sh

# Or use the example run script
./scan_param/run_regressor.sh
```

### MAE Pre-training
```bash
python lib/train_mae.py --train_root /path/data.root --save_path mae_pretrained.pth --epochs 20 --batch_size 1024
```

### Export & Inference
```bash
# ONNX export (single-task)
python macro/export_onnx.py artifacts/<RUN>/checkpoint_best.pth --output model.onnx

# ONNX export (multi-task)
python macro/export_onnx.py artifacts/<RUN>/checkpoint_best.pth --multi-task --output model.onnx

# Real data inference
python val_data/inference_real_data.py --onnx model.onnx --input data.root --output output.root
```

### Monitoring
```bash
mlflow ui --backend-store-uri mlruns --host 0.0.0.0 --port 5000
tensorboard --logdir runs --host 0.0.0.0 --port 6006
```

## Architecture

### Training Pipeline
```
Data Loading (XECStreamingDataset) → Preprocessing (Normalization + Reweighting)
    → Model Forward (XECEncoder/XECMultiHeadModel)
    → Loss (Multi-task) → EMA Update → Validation → Checkpoint/ONNX Export
```

### Model Architecture
Multi-branch design processing 6 detector faces:

1. **Rectangular Faces (ConvNeXt V2):** Inner (93×44), Upstream (24×6), Downstream (24×6), Outer (9×24 with optional fine grid)
   - Stem → 2× ConvNeXtV2 blocks (dim=32) → Downsample → 3× blocks (dim=64) → 1024-dim token

2. **Hexagonal Faces (HexNeXt Graph Attention):** Top/Bottom PMT arrays (668 total)
   - Stem → 4× HexNeXt blocks (dim=96) → Global Pool → 1024-dim token

3. **Fusion (Transformer):** 6 face tokens → 2-layer Transformer Encoder (8 heads) → 6144-dim → Task heads

### Key Components
- `lib/models/`: Model architectures
  - `regressor.py`: XECEncoder, XECMultiHeadModel, FaceBackbone, DeepHexEncoder
  - `mae.py`: XEC_MAE for masked autoencoder pretraining
  - `inpainter.py`: XEC_Inpainter for dead channel recovery
  - `blocks.py`: ConvNeXtV2Block, HexNeXtBlock, GRN, DropPath
- `lib/engines/`: Training/validation loops
  - `regressor.py`: run_epoch_stream for multi-task regression
  - `mae.py`: run_epoch_mae, run_eval_mae
  - `inpainter.py`: run_epoch_inpainter, run_eval_inpainter
- `lib/tasks/`: Task-specific handlers (angle, energy, timing, position)
- `lib/dataset.py`: XECStreamingDataset (ROOT file streaming, chunked loading)
- `lib/config.py`: Configuration dataclasses, YAML loading
- `lib/geom_defs.py`: Detector geometry index maps
- `lib/reweighting.py`: SampleReweighter for distribution balancing

## Configuration

`config/train_config.yaml` is the master configuration:
- **data:** train_path, val_path, batch_size, chunksize, num_workers
- **normalization:** npho_scale (0.58), time_scale (6.5e8), sentinel_value (-5.0)
- **model:** outer_mode ("finegrid"/"split"), outer_fine_pool, drop_path_rate
- **tasks:** Enable/disable angle, energy, timing, uvwFI with per-task loss_fn, loss_beta, weight
- **training:** epochs, lr, weight_decay, warmup_epochs, ema_decay, grad_clip
- **reweighting:** Per-task histogram-based sample reweighting

## Data Format

ROOT files with branches:
- **Input:** `relative_npho`, `relative_time` (shape: 4760)
- **Truth:** `angleVec` (θ,φ), `energy`, `timing`, `xyzVTX` (position)
- **Metadata:** `emiVec` (emission direction), `run`, `event`

Preprocessing normalizes Npho/time and marks bad values with sentinel (-5.0).

## Multi-Task Learning

Tasks configured in YAML:
```yaml
tasks:
  angle:
    enabled: true
    loss_fn: "smooth_l1"  # smooth_l1, l1, mse, huber
    weight: 1.0
  energy:
    enabled: false
    ...
```

Models: `XECEncoder` (angle-only legacy), `XECMultiHeadModel` (multi-task)

## Checkpoints

- `checkpoint_best.pth`, `checkpoint_last.pth`: Full state (model, optimizer, EMA, epoch, config)
- MAE checkpoints: Encoder weights only (no heads/EMA)
- Auto-detection: Script differentiates full vs MAE checkpoints and resumes appropriately

## Key Artifacts

- `artifacts/<RUN>/checkpoint_*.pth`: Model weights
- `artifacts/<RUN>/predictions_*.csv`: Validation predictions
- `artifacts/<RUN>/*.onnx`: ONNX export for C++ inference
- `mlruns/`: MLflow experiment tracking
- `runs/`: TensorBoard logs
