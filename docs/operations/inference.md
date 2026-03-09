# Inference & Validation

Run trained models on simulation or real detector data.

## Standalone Validation

Validate a checkpoint or ONNX model against simulation data without training. Generates the same resolution plots and prediction CSVs as training-time validation.

### PyTorch Checkpoint

```bash
# Auto-detect tasks and config from checkpoint
python macro/validate_regressor.py artifacts/<RUN>/checkpoint_best.pth \
    --val_path data/val/

# Specify tasks and output directory
python macro/validate_regressor.py artifacts/<RUN>/checkpoint_best.pth \
    --val_path data/val/ \
    --tasks energy angle \
    --output_dir results/
```

When the checkpoint embeds a config dict (all recent checkpoints do), model architecture and normalization parameters are restored automatically. Use `--config` to override.

### ONNX Model

```bash
# ONNX validation (requires --config for normalization params)
python macro/validate_regressor.py model.onnx \
    --val_path data/cex/ \
    --config config/reg/scan/step3b_model_large.yaml \
    --tasks energy
```

The script auto-detects `.onnx` vs `.pth` by file extension. ONNX inference uses `onnxruntime` and runs on CPU.

**Note:** Models trained with `gaussian_nll` produce 2-column energy output `[mu, log_var]`; the validation script extracts column 0 as the prediction automatically.

### Dead-Channel Recovery Mode

Add `--dead-channel` to compare three dead-channel recovery strategies side by side (ONNX only):

```bash
# All three strategies (raw + neighbor avg + inpainted)
python macro/validate_regressor.py regressor.onnx \
    --val_path data/cex/CEX23_patch13_r557545_n100.root \
    --config config.yaml --tasks energy \
    --dead-channel \
    --inpainter-torchscript inpainter.pt

# Without inpainter (raw + neighbor avg only)
python macro/validate_regressor.py regressor.onnx \
    --val_path data/cex/CEX23_patch13_r557545_n100.root \
    --config config.yaml --tasks energy \
    --dead-channel
```

The script:
1. Reads events from the CEX ROOT file via uproot (npho, time, dead mask)
2. Normalizes using the configurable `NphoTransform` (respects `npho_scheme` from config)
3. For each event, creates three input variants:
   - **Raw** — dead channels left as sentinel values
   - **Neighbor average** — dead channels filled by averaging valid k-hop neighbors (`lib/inpainter_baselines.py`)
   - **Inpainted** — dead channels filled by the ML inpainter model
4. Runs the ONNX regressor on each variant
5. Writes all predictions to a ROOT output file

**Output branches:** `energy_raw`, `energy_neighavg`, `energy_inpainted` (plus metadata: `run`, `event`, `energyTruth`, `energyReco`, etc.)

**Dead-channel options:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--dead-channel` | off | Enable dead-channel recovery mode |
| `--inpainter-torchscript` | none | TorchScript inpainter (`.pt`) |
| `--inpainter-checkpoint` | none | PyTorch inpainter checkpoint (`.pth`, alternative) |
| `--neighbor-k` | 1 | Number of neighbor hops for averaging |
| `--dead-from-db` | none | Query dead channels from DB for this run number |
| `--dead-from-file` | none | Load dead channels from text file |
| `--chunksize` | 2048 | Events to read per chunk from ROOT file |
| `--tree-name` | tree | TTree name in input ROOT file |

Dead channels are detected from the `dead` branch in the ROOT file by default. Use `--dead-from-db` or `--dead-from-file` to override.

**Dead channel recovery methods:**

| Method | Source | Description |
|--------|--------|-------------|
| Raw | Built-in | Dead channels left as sentinels; regressor sees them as-is |
| Neighbor avg | `lib/inpainter_baselines.py` | Geometry-aware averaging of valid k-hop neighbors (supports all 6 faces) |
| Inpainted | Exported inpainter model | ML-based prediction using surrounding channel context |

**Prerequisites** — export the regressor and (optionally) the inpainter:

```bash
# Export regressor to ONNX
python macro/export_onnx.py artifacts/<RUN>/checkpoint_best.pth \
    --tasks energy --output regressor.onnx

# Export inpainter to TorchScript
python macro/export_onnx_inpainter.py artifacts/<RUN>/inpainter_checkpoint_best.pth \
    --output inpainter.pt
```

CEX data must be preprocessed with dead channel info (the `dead` branch). See [Preprocess CEX data](#1-preprocess-cex-data) below.

---

## Real Data Inference

Run trained models on real detector data for physics analysis.

### 1. Export checkpoint to ONNX

```bash
# Auto-detect tasks from checkpoint
python macro/export_onnx.py \
    artifacts/<RUN_NAME>/checkpoint_best.pth \
    --output model.onnx

# Specify tasks explicitly
python macro/export_onnx.py \
    artifacts/<RUN_NAME>/checkpoint_best.pth \
    --tasks energy timing uvwFI angle \
    --output model.onnx
```

## 2. Prepare real data input file

Process rec files using the MEG analyzer:

```bash
cd $MEG2SYS/analyzer
./meganalyzer -b -q -I '$HOME/meghome/xec-ml-wl/macro/PrepareRealData.C+(start_run, n_runs, "rec_suffix", "rec_dir")'

# Output: DataGammaAngle_<start_run>-<end_run>.root
# ~2000 runs -> ~100k events
mv DataGammaAngle_*.root $HOME/xec-ml-wl/val_data/
```

## 3. Run inference

Login to GPU node:

```bash
srun --cluster=gmerlin7 -p a100-interactive --time=02:00:00 --gres=gpu:1 --pty /bin/bash
```

Set up CUDA libraries:

```bash
export LD_LIBRARY_PATH=$(find $CONDA_PREFIX/lib/python3.10/site-packages/nvidia -name "lib" -type d | paste -sd ":" -):$LD_LIBRARY_PATH
```

Run inference:

```bash
python val_data/inference_real_data.py \
    --onnx model.onnx \
    --input val_data/DataGammaAngle_<start_run>-<end_run>.root \
    --output val_data/inference_results.root
```

The script auto-detects tasks from the ONNX model outputs and saves predictions for each task.

**Normalization parameters** are loaded from `lib/geom_defs.py` by default:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `npho_scale` | 1000.0 | Npho scale for log1p transform |
| `npho_scale2` | 4.08 | Npho scale2 divisor |
| `time_scale` | 1.14e-7 | Time normalization scale |
| `time_shift` | -0.46 | Time shift after scaling |
| `sentinel_time` | -1.0 | Value for invalid channels |

Override if your model was trained with different values:

```bash
python val_data/inference_real_data.py \
    --onnx model.onnx \
    --input val_data/DataGammaAngle_*.root \
    --npho_scale 1000.0 --npho_scale2 4.08 \
    --time_scale 1.14e-7 --time_shift -0.46
```

## 4. Generate analysis plots

```bash
python val_data/plot_real_data_analysis.py \
    --input val_data/inference_results.root \
    --checkpoint artifacts/<RUN_NAME>/checkpoint_best.pth \
    --output_dir plots_real_data/
```

Generates plots for each detected task:
- **Angle**: scatter plots, opening angle residual, resolution profiles
- **Energy**: pred vs truth comparison, residual histogram
- **Timing**: pred vs truth comparison, residual histogram
- **Position (uvwFI)**: per-coordinate comparisons

## CEX Real Data Validation (Batch)

Validate the energy regressor on CEX23 charge exchange data (24 detector patches).

### 1. Preprocess CEX data

Convert raw rec files into ML-ready ROOT files with event selection cuts:

```bash
python others/CEXPreprocess.py --srun 557545 --nfiles 100 --patch 13 \
    --output-dir data/cex --dead-dir data/dead_channels/
```

**Event selection cuts:**
- Physics triggers only (mask == 50 or 51)
- Energy window: 54–57 MeV (`EGamma`)
- `|EGamma + bgoEnergy - Epi0| < 20 MeV` (energy conservation)
- Physical opening angle (sqrtarg ≥ 0)
- Valid npho and time channels

Output: `data/cex/CEX23_patch{PATCH}_r{SRUN}_n{NFILES}.root` (5 files per patch, 24 patches total)

### 2. Run batch inference

Submit 24 parallel SLURM jobs (one per patch), each running ONNX inference on CPU:

```bash
# Preview first
DRY_RUN=1 bash macro/submit_validate_regressor_cex.sh

# Submit all patches
bash macro/submit_validate_regressor_cex.sh

# Submit specific patches
bash macro/submit_validate_regressor_cex.sh 13 12 21

# Dead-channel recovery mode (with inpainter)
DEAD_CHANNEL=1 INPAINTER=inpainter.pt bash macro/submit_validate_regressor_cex.sh

# Dead-channel recovery mode (without inpainter)
DEAD_CHANNEL=1 bash macro/submit_validate_regressor_cex.sh
```

**Defaults** (override via env vars):

| Variable | Default | Description |
|----------|---------|-------------|
| `CHECKPOINT` | `~/meghome/xec-ml-wl/artifacts/ene_50epoch_sqrt_DSmax/meg2ene.onnx` | ONNX model |
| `CONFIG` | `config/reg/ene_reg_50epoch_sqrt_DSmax.yaml` | Normalization config |
| `CEX_DIR` | `data/cex` | Input CEX ROOT files |
| `OUTPUT_BASE` | `val_data/cex` | Output directory |
| `PARTITION` | `meg-long` | SLURM partition |
| `TIME` | `06:00:00` | Time limit (6 hours) |
| `DEAD_CHANNEL` | `0` | Set to `1` to enable dead-channel recovery |
| `INPAINTER` | (none) | Path to inpainter TorchScript (`.pt`) |
| `NEIGHBOR_K` | `1` | Neighbor hops for averaging |

**Important:** Inference is **not resumable**. If a job is killed by the time limit, the entire patch must be re-run. Increase `TIME` if needed.

Output: `val_data/cex/patch{1..24}/predictions_energy_*.csv` (standard mode) or `val_data/cex/patch{1..24}/regressor_*.root` (dead-channel mode)

### 3. Combine results and plot

Merge all patches and produce resolution plots with Gaussian fits:

```bash
python macro/combine_cex_results.py
```

**Output files:**
- `val_data/cex/CEX23_combined.root` — all events with `patch` column
- `val_data/cex/CEX23_resolution.pdf` — multi-page PDF:
  - Page 1: Resolution (σ) and bias (μ) vs patch with Gaussian-fit error bars
  - Page 2: Combined residual histogram with Gaussian fit overlay
  - Pages 3+: Per-patch residual histograms with individual Gaussian fits

Options:
```bash
# Specific patches only
python macro/combine_cex_results.py --patches 13 12 21

# Data only, skip plots
python macro/combine_cex_results.py --no-plots
```

### 4. Compare with conventional method

Compare ML energy resolution with the conventional peak-fitting method:

```bash
python macro/compare_cex_conventional.py \
    --conv /data/project/meg/shared/subprojects/xec/shared/peakpos_nsum2_2023CEX.txt \
    --ml-base val_data/cex
```

The script:
- Parses the conventional `peakpos` text file (run number, peak position, resolution in %)
- Maps each run to its CEX23 patch using the run ranges from `others/submit_cex.py`
- Aggregates conventional results per patch (weighted average across runs)
- Loads ML predictions from per-patch CSVs and fits a double Gaussian to the residual
- Compares core σ from both methods side by side

**Output:**
- Terminal table: per-patch resolution (σ in MeV and %), bias, ML/Conv ratio
- PDF (`val_data/cex/CEX23_ml_vs_conventional.pdf`):
  - Page 1: Resolution and bias vs patch for both methods
  - Page 2: ML/Conv resolution ratio vs patch

**Conventional resolution format** (`peakpos_nsum2_2023CEX.txt`):
- `peak <value> <error>` — fitted core mean in ADC counts (× 0.0000015634 → MeV)
- `reso <value> <error>` — fitted core σ / fitted core mean in percent

Options:
```bash
# Table only
python macro/compare_cex_conventional.py --no-plot

# Specific patches
python macro/compare_cex_conventional.py --patches 13 12 21

# Custom output path
python macro/compare_cex_conventional.py --output path/to/comparison.pdf
```

---

## Output branches

The inference script outputs a ROOT file with `val_tree` containing:

| Branch | Description |
|--------|-------------|
| `run_id`, `event_id` | Event identification |
| `pred_theta`, `pred_phi` | Angle predictions (if angle task) |
| `true_theta`, `true_phi` | Angle truth values |
| `opening_angle` | Opening angle between pred and true |
| `pred_energy`, `true_energy` | Energy (if energy task) |
| `pred_timing`, `true_timing` | Timing (if timing task) |
| `pred_u/v/w`, `true_u/v/w` | Position (if uvwFI task) |
| `x_vtx`, `y_vtx`, `z_vtx` | Vertex position |

---

## Inpainter Export (TorchScript)

The inpainter model uses fused CUDA kernels that are not compatible with ONNX export. Use TorchScript instead:

```bash
python macro/export_onnx_inpainter.py \
    artifacts/<RUN>/inpainter_checkpoint_best.pth \
    --output inpainter_model.pt
```

The exported TorchScript model bakes normalization constants inside, so it accepts **raw** sensor values + binary mask as input and outputs raw npho with inpainted values at masked positions.

| Format | Regressor | Inpainter |
|--------|-----------|-----------|
| ONNX | `macro/export_onnx.py` | Not supported |
| TorchScript | Not needed | `macro/export_onnx_inpainter.py` |
