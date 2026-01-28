# xec-ml-wl

## Machine Learning Model for the MEG II Liquid Xenon (LXe) Detector

This repository contains machine learning model (CNN + Graph Transformer) to regress physical observables including emission angle (**$\theta$**, **$\phi$**), energy, timing, and position (**uvwFI**) from photons detected by the LXe detector, utilizing both photon count (**$N_{\mathrm{pho}}$**) and timing information (**$t_{\mathrm{pm}}$**) in each photo-sensor (4092 SiPMs and 668 PMTs).

This model respects the complex topology of the detector by combining:
1.  **ConvNeXt V2** for rectangular faces (Inner, Outer, US, DS).
2.  **HexNeXt (Graph Attention)** for hexagonal PMT faces (Top, Bottom).
3.  **Transformer Fusion** to correlate signals across disjoint detector faces.

---

## Quick Reference

| Task | Command | Config |
|------|---------|--------|
| **Train Regressor** | `./scan_param/run_regressor.sh` | `config/train_config.yaml` |
| **MAE Pretraining** | `./scan_param/run_mae.sh` | `config/mae_config.yaml` |
| **Inpainter Training** | `./scan_param/run_inpainter.sh` | `config/inpainter_config.yaml` |
| **ONNX Export** | `python macro/export_onnx.py checkpoint.pth --output model.onnx` | - |
| **MLflow UI** | `mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000` | - |

**Common Config Overrides (set env vars before running submit script):**
```bash
# Reduce batch size (OOM fix)
BATCH_SIZE=512 ./scan_param/submit_regressor.sh

# Enable multiple tasks
TASKS="angle energy" ./scan_param/submit_regressor.sh

# Use MAE weights for fine-tuning
RESUME_FROM="mae_checkpoint.pth" ./scan_param/submit_regressor.sh
```

**Important:** There are two [normalization schemes](docs/architecture/data-pipeline.md#1-normalization-schemes). Use **legacy** for existing regressor models, **new** for MAE/inpainter experiments.

---

## Documentation

### Getting Started
- [Environment Setup](docs/setup.md) - A100 and Grace-Hopper node configuration

### Training
- [Regressor Training](docs/training/regressor.md) - Batch training and Jupyter sessions
- [MAE Pre-training](docs/training/mae.md) - Self-supervised masked autoencoder
- [Dead Channel Inpainting](docs/training/inpainter.md) - Sensor value recovery
- [Multi-Task Learning](docs/training/multi-task.md) - Multi-task configuration and sample reweighting

### Architecture
- [Model Architecture](docs/architecture/model.md) - ConvNeXt, HexNeXt, Transformer fusion
- [Data Pipeline](docs/architecture/data-pipeline.md) - Data format, normalization, loading
- [Detector Geometry](docs/architecture/detector-geometry.md) - Sensor mapping and index definitions
- [File Dependency](docs/architecture/file-dependency.md) - Codebase structure diagram

### Operations
- [Output & Artifacts](docs/operations/artifacts.md) - Checkpoints, metrics, visualization
- [Real Data Inference](docs/operations/inference.md) - ONNX export and validation
- [Troubleshooting](docs/operations/troubleshooting.md) - Common issues and FAQ

### Notes
- [ML Techniques](docs/notes/ml-techniques.md) - EMA, schedulers, positional encoding, FCMAE
- [Future Work](docs/notes/future-work.md) - Prospects and improvement ideas

---

## Key Features

- **Multi-branch architecture** handling heterogeneous sensor geometry
- **Self-supervised pretraining** via Masked Autoencoder (MAE)
- **Dead channel recovery** through inpainting heads
- **Multi-task learning** for simultaneous regression of multiple observables
- **Streaming data loading** for large ROOT files
- **Config-based training** with YAML files and CLI overrides
- **MLflow + TensorBoard** integration for experiment tracking

---

## Model Overview

```
Input: (B, 4760, 2) sensor values (npho, time)
    ↓
┌─────────────────────────────────────────┐
│  Per-Face Processing                    │
│  - Rectangular: ConvNeXt V2 → 1024-dim  │
│  - Hexagonal: HexNeXt Graph → 1024-dim  │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Transformer Fusion (2 layers)          │
│  - 6 face tokens × 1024 dim             │
│  - 8-head self-attention                │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Task-Specific Heads                    │
│  - Angle: (θ, φ)                        │
│  - Energy, Timing, Position             │
└─────────────────────────────────────────┘
```

See [Model Architecture](docs/architecture/model.md) for detailed diagrams.

---

*Last updated: January 2026*
