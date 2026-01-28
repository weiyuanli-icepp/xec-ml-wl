# Real Data Validation

Validation using real data can be performed in the following procedure.

## 1. Convert checkpoint files to ONNX files (`macro/export_onnx.py`)

```bash
# Single-task (angle-only)
$ python macro/export_onnx.py \
artifacts/<RUN_NAME>/checkpoint_best.pth \
--output model.onnx

# Multi-task (auto-detect from checkpoint)
$ python macro/export_onnx.py \
artifacts/<RUN_NAME>/checkpoint_best.pth \
--multi-task --output model.onnx

# Multi-task (specify tasks)
$ python macro/export_onnx.py \
artifacts/<RUN_NAME>/checkpoint_best.pth \
--multi-task --tasks angle energy --output model.onnx
```

## 2. Process rec files to a input file for ONNX run time script (`macro/PrepareRealData.C`)

```bash
$ cd $MEG2SYS/analyzer
$ ./meganalyzer -b -q -I '$HOME/meghome/xec-ml-wl/macro/PrepareRealData.C+(start_runnumber, number_of_runs, "rec_suffix", "rec_dir")'
# DataGammaAngle_<start_runnumber>-<end_runnumber>.root will be generated. 2000 runs -> 100k events
$ mv DataGammaAngle_<start_runnumber>-<end_runnumber>.root $HOME/xec-ml-wl/val_data/
```

## 3. Use inference script to output the prediction and "truth" (`inference_real_data.py`)

First login to interactive gpu node.

```bash
$ srun --cluster=gmerlin7 -p a100-interactive --time=02:00:00 --gres=gpu:1 --pty /bin/bash
```

Before executing the script, we need to export some path to `$LD_LIBRARY_PATH`

```bash
$ export LD_LIBRARY_PATH=$(find $CONDA_PREFIX/lib/python3.10/site-packages/nvidia -name "lib" -type d | paste -sd ":" -):$LD_LIBRARY_PATH
```

Check if it worked:

```bash
$ echo $LD_LIBRARY_PATH
```

Now we can start inference:

```bash
$ python inference_real_data.py \
    --onnx onnx/<RUN_NAME>.onnx \
    --input val_data/DataGammaAngle_<start_runnumber>-<end_runnumber>.root \
    --output Output_Run<start_runnumber>-<end_runnumber>.root \
    --npho_scale 0.58 --npho_scale2 1.0 \
    --time_scale 6.5e-8 --time_shift 0.5 \
    --sentinel_value -5.0
```

## 4. Check inference result with plotting macro

```bash
$ plot_real_data_analysis.py \
    --input val_data/Output_Run<start_runnumber>-<end_runnumber>.root \
    --checkpoint \
    --output_dir \
    --outer_mode
```
