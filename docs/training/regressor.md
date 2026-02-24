# Regressor Training

This document covers batch training and interactive Jupyter sessions for the regression model.

## Batch Training

All training scripts follow a consistent pattern:
- `run_*.sh` - Set environment variables and call submit script
- `submit_*.sh` - Creates and submits SLURM job

The submit scripts automatically detect CPU architecture and activate the correct conda environment (x86 or ARM).

### 1. Quick Submission (Regressor)

```bash
$ cd jobs

# Edit run_regressor.sh to set your parameters, then:
$ ./run_regressor.sh

# Or set env vars directly:
$ export RUN_NAME="my_run" CONFIG_PATH="config/reg/train_config.yaml" PARTITION="a100-daily"
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
$ python -m lib.train_regressor --config config/reg/train_config.yaml

# With CLI overrides
$ python -m lib.train_regressor --config config/reg/train_config.yaml --lr 1e-4 --epochs 30
```

### 4. Multi-GPU Training (DDP)

All training scripts support multi-GPU training via PyTorch DistributedDataParallel. Set `NUM_GPUS` to enable:

```bash
# Submit with multiple GPUs
$ NUM_GPUS=4 ./jobs/submit_regressor.sh

# Direct multi-GPU training (without SLURM)
$ torchrun --nproc_per_node=4 -m lib.train_regressor --config config/reg/train_config.yaml

# Single GPU still works as before
$ python -m lib.train_regressor --config config/reg/train_config.yaml
```

**Key behaviors with DDP:**
- ROOT file lists are sharded across ranks (round-robin), so each GPU processes different files
- Only rank 0 logs to MLflow, saves checkpoints, and prints progress
- Metrics are all-reduced (averaged) across ranks before logging
- Checkpoints are saved without `module.` prefix, so they are compatible with both single-GPU and multi-GPU
- **Effective batch size** = `per_gpu_batch_size × num_gpus`. Reduce per-GPU batch size if needed
- Gradient accumulation uses `model.no_sync()` on intermediate steps for efficiency

### 5. Optimization Best Practices

For GH nodes, use the following settings to maximize throughput:
* **Batch Size**: 16384 (Uses ~65GB VRAM, ~70% capacity)
* **Chunk Size**: 524880 (320 Batches)
* **Memory**: Normally uses 5-10GB RAM

## Interactive Jupyter Session

To start a Jupyter session on a GPU node:

1. Request a GPU interactive session via SLURM
2. Start Jupyter: `jupyter notebook --no-browser --port=8888`
3. Tunnel ports locally: `ssh -N -L 8888:localhost:8888 -J <user>@login001 <user>@gpuXXX`
4. Paste the URL with token in the browser

## Configuration Parameters

Training is now **config-based** using `config/reg/train_config.yaml`. CLI arguments can override config values.

### Core Training Parameters

| Parameter | Config Path | Default | Description |
|-----------|-------------|---------|-------------|
| `--train_path` | `data.train_path` | Required | Path to training ROOT file(s) |
| `--val_path` | `data.val_path` | Required | Path to validation ROOT file(s) |
| `--batch_size` | `data.batch_size` | `256` | Batch size |
| `--chunksize` | `data.chunksize` | `256000` | Events per ROOT read |
| `--epochs` | `training.epochs` | `20` | Training epochs |
| `--lr` | `training.lr` | `3e-4` | Learning rate |
| `--weight_decay` | `training.weight_decay` | `1e-4` | Weight decay |
| `--warmup_epochs` | `training.warmup_epochs` | `2` | Warmup epochs |
| `--ema_decay` | `training.ema_decay` | `0.999` | EMA decay rate |
| `--grad_clip` | `training.grad_clip` | `1.0` | Gradient clipping |
| `--grad_accum_steps` | `training.grad_accum_steps` | `1` | Gradient accumulation steps |
| `--channel_dropout_rate` | `training.channel_dropout_rate` | `0.1` | Random channel dropout rate |
| `--lr_scheduler` | `training.lr_scheduler` | `null` | LR schedule (`cosine`, `onecycle`, `plateau`, `null`/`none`) |
| `--compile` | `training.compile` | `max-autotune` | torch.compile mode (`max-autotune`, `reduce-overhead`, `default`, `none`) |
| `--outer_mode` | `model.outer_mode` | `finegrid` | Outer face mode (`finegrid` or `split`) |
| `--encoder_dim` | `model.encoder_dim` | `1024` | Encoder token dimension (must be divisible by 32) |
| `--dim_feedforward` | `model.dim_feedforward` | `null` | FFN dimension (null = encoder_dim × 4) |
| `--num_fusion_layers` | `model.num_fusion_layers` | `2` | Number of transformer fusion layers |
| `--tasks` | `tasks.*` | angle only | Enable specific tasks (energy, timing, uvwFI, angle) |
| `--resume_from` | `checkpoint.resume_from` | `null` | Path to checkpoint to resume |
| `--refresh_lr` | `checkpoint.refresh_lr` | `false` | Reset LR scheduler for remaining epochs |
| `--reset_epoch` | `checkpoint.reset_epoch` | `false` | Start from epoch 1 (load weights only) |
| `--new_mlflow_run` | `checkpoint.new_mlflow_run` | `false` | Force new MLflow run when resuming |

### Normalization Parameters

| Parameter | Config Path | Legacy | New | Description |
|-----------|-------------|--------|-----|-------------|
| `--npho_scheme` | `normalization.npho_scheme` | `log1p` | `log1p` | Normalization scheme (`log1p`, `anscombe`, `sqrt`, `linear`) |
| `--npho_scale` | `normalization.npho_scale` | 0.58 | 1000 | Npho normalization scale |
| `--npho_scale2` | `normalization.npho_scale2` | 1.0 | 4.08 | Secondary npho scale (log1p only) |
| `--time_scale` | `normalization.time_scale` | 6.5e-8 | 1.14e-7 | Time normalization |
| `--time_shift` | `normalization.time_shift` | 0.5 | -0.46 | Time offset shift |
| `--sentinel_time` | `normalization.sentinel_time` | -1.0 | -1.0 | Bad channel marker |

See [Data Pipeline](../architecture/data-pipeline.md) for detailed normalization documentation.
