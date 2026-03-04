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
