# Regressor Training

This document covers batch training and interactive Jupyter sessions for the regression model.

## Batch Training

All training scripts follow a consistent pattern:
- `run_*.sh` - Set environment variables and call submit script
- `submit_*.sh` - Creates and submits SLURM job

The submit scripts automatically detect CPU architecture and activate the correct conda environment (x86 or ARM).

### 1. Quick Submission (Regressor)

```bash
$ cd scan_param

# Edit run_regressor.sh to set your parameters, then:
$ ./run_regressor.sh

# Or set env vars directly:
$ export RUN_NAME="my_run" CONFIG_PATH="config/train_config.yaml" PARTITION="a100-daily"
$ ./submit_regressor.sh
```

### 2. Config + CLI Overrides

Override specific parameters via environment variables:

```bash
# Override epochs and learning rate
$ EPOCHS=100 LR=1e-4 ./submit_regressor.sh

# Enable multiple tasks
$ TASKS="angle energy" ./submit_regressor.sh
```

### 3. Direct Python Training

```bash
# Config-based training
$ python -m lib.train_regressor --config config/train_config.yaml

# With CLI overrides
$ python -m lib.train_regressor --config config/train_config.yaml --lr 1e-4 --epochs 30
```

### 4. Optimization Best Practices

For GH nodes, use the following settings to maximize throughput:
* **Batch Size**: 16384 (Uses ~65GB VRAM, ~70% capacity)
* **Chunk Size**: 524880 (320 Batches)
* **Memory**: Normally uses 5-10GB RAM

## Interactive Jupyter Session

```bash
# Syntax:
# ./start_jupyter_xec_gpu.sh [PARTITION] [TIME] [PORT]
./start_jupyter_xec_gpu.sh gh-interactive 02:00:00 8888
```
1. Wait for the connection URL.
2. Tunnel ports locally: `ssh -N -L 8888:localhost:8888 -J <user>@login001 <user>@gpuXXX`
3. Paste the URL with token in the browser.

## Configuration Parameters

Training is now **config-based** using `config/train_config.yaml`. CLI arguments can override config values.

### Core Training Parameters

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `--train_path` | `data.train_path` | Required | Path to training ROOT file(s) |
| `--val_path` | `data.val_path` | Required | Path to validation ROOT file(s) |
| `--batch` | `data.batch_size` | `256` | Batch size |
| `--chunksize` | `data.chunksize` | `256000` | Events per ROOT read |
| `--epochs` | `training.epochs` | `20` | Training epochs |
| `--lr` | `training.lr` | `3e-4` | Learning rate |
| `--weight_decay` | `training.weight_decay` | `1e-4` | Weight decay |
| `--warmup_epochs` | `training.warmup_epochs` | `2` | Warmup epochs |
| `--ema_decay` | `training.ema_decay` | `0.999` | EMA decay rate |
| `--grad_clip` | `training.grad_clip` | `1.0` | Gradient clipping |
| `--grad_accum_steps` | `training.grad_accum_steps` | `1` | Gradient accumulation steps |
| `--outer_mode` | `model.outer_mode` | `finegrid` | Outer face mode (`finegrid` or `split`) |
| `--tasks` | `tasks.*` | angle only | Enable specific tasks (angle, energy, timing, uvwFI) |
| `--resume_from` | `checkpoint.resume_from` | `null` | Path to checkpoint to resume |

### Normalization Parameters

| Parameter | Config Path | Legacy | New | Description |
|-----------|-------------|--------|-----|-------------|
| `--npho_scale` | `normalization.npho_scale` | 0.58 | 1000 | Npho log transform scale |
| `--npho_scale2` | `normalization.npho_scale2` | 1.0 | 4.08 | Secondary npho scale |
| `--time_scale` | `normalization.time_scale` | 6.5e-8 | 1.14e-7 | Time normalization |
| `--time_shift` | `normalization.time_shift` | 0.5 | -0.46 | Time offset shift |
| `--sentinel_value` | `normalization.sentinel_value` | -5.0 | -1.0 | Bad channel marker |

See [Data Pipeline](../architecture/data-pipeline.md) for detailed normalization documentation.
