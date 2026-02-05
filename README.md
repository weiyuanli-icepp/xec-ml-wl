# xec-ml-wl

## Machine Learning Model for the MEG II Liquid Xenon (LXe) Detector

This repository contains machine learning models (CNN + Graph Transformer) for the LXe detector, utilizing both photon count (**$N_{\mathrm{pho}}$**) and timing information (**$t_{\mathrm{pm}}$**) from 4760 photo-sensors (4092 SiPMs + 668 PMTs):

- **Regressor:** Predicts physical observables including emission angle (**$\theta$**, **$\phi$**), energy, timing, and position (**uvwFI**)
- **MAE (Masked Autoencoder):** Self-supervised pretraining for encoder weights
- **Inpainter:** Dead channel recovery through sensor value prediction

This model respects the complex topology of the detector by combining:
1.  **ConvNeXt V2** for rectangular faces (Inner, Outer, US, DS).
2.  **HexNeXt (Graph Attention)** for hexagonal PMT faces (Top, Bottom).
3.  **Transformer Fusion** to correlate signals across disjoint detector faces.

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

*Last updated: February 2026*
