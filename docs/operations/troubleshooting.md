# Troubleshooting

## Common Issues

### 1. CUDA Out of Memory (GPU OOM)

**Symptom:** `RuntimeError: CUDA out of memory`

**Solutions:**
```bash
# Reduce batch size
BATCH_SIZE=512 ./scan_param/submit_regressor.sh

# For MAE (decoder uses more memory)
BATCH=1024 ./scan_param/submit_mae.sh

# Use gradient accumulation for effective larger batch
# effective_batch = batch_size × grad_accum_steps
training:
  batch_size: 512
  grad_accum_steps: 4  # Effective batch = 2048
```

**Recommended batch sizes:**
| Model | A100 (40GB) | GH200 (96GB) |
|-------|-------------|--------------|
| Regressor | 8192-16384 | 16384-32768 |
| MAE | 1024-2048 | 2048-4096 |
| Inpainter | 1024-2048 | 2048-4096 |

### 2. CPU/System Memory OOM (Large Datasets)

**Symptom:** Process killed, `MemoryError`, or system becomes unresponsive when training with large ROOT files (>1M events)

**Root Cause:** The inpainter/MAE with `save_predictions: true` (or `save_root_predictions: true`) collects ALL predictions in memory during validation before writing to ROOT. With 1M events × 5% mask × 238 sensors = 12M prediction dicts.

**Memory estimation:**
```
Per prediction: ~200 bytes (dict with 8 floats + metadata)
1M events × 5% mask × 238 sensors ≈ 12M predictions ≈ 2.4 GB
1M events × 10% mask × 238 sensors ≈ 24M predictions ≈ 4.8 GB
```

**Solutions:**
```yaml
# Option 1: Disable prediction saving
checkpoint:
  save_predictions: false  # or save_root_predictions: false

# Option 2: Reduce validation frequency
checkpoint:
  save_interval: 50  # Save only every 50 epochs

# Option 3: Use smaller validation set
data:
  val_path: "small_val.root"  # Use subset for validation
```

**Best Practice:** For large-scale training, disable `save_predictions` during initial experiments. Enable only for final evaluation runs with smaller validation sets.

### 3. MLflow Database Locked

**Symptom:** `sqlite3.OperationalError: database is locked`

**Solution:**
```bash
# Kill any hanging processes
pkill -f mlflow

# Or use a fresh database
rm mlruns.db
export MLFLOW_TRACKING_URI="sqlite:///mlruns.db"
```

### 4. torch.compile Issues

#### A. Triton Installation Error

**Symptom:** `RuntimeError: Cannot find a working triton installation`

**Solution:** Disable compilation in config:
```yaml
training:
  compile: "none"  # or "false"
```

#### B. CPU OOM During Compilation

**Symptom:** SSH session closes, process killed, or system becomes unresponsive during the first epoch (before actual training starts).

**Root Cause:** `torch.compile` with `max-autotune` mode benchmarks many Triton kernel configurations, consuming significant CPU memory during compilation.

**Solution:** Use a less aggressive compile mode:
```yaml
training:
  compile: "reduce-overhead"  # Recommended balance
  # or
  compile: "none"  # Disable completely
```

#### C. torch.compile Mode Reference

| Mode | Compilation Time | Memory Usage | Runtime Speed | Best For |
|------|------------------|--------------|---------------|----------|
| `max-autotune` | Slowest (5-15 min) | Highest | Fastest | Production with ample resources |
| `reduce-overhead` | Medium (2-5 min) | Medium | Fast | **Recommended default** |
| `default` | Fast (30-60 sec) | Low | Moderate | Quick iteration |
| `false`/`none` | None | None | Baseline | Memory-constrained/debugging |

**Mode Details:**

- **`max-autotune`**: Benchmarks 10-50+ kernel variants per operation to find the optimal configuration. Produces verbose Triton autotuning output. Best final performance but highest compilation overhead.

- **`reduce-overhead`**: Uses CUDA graphs to reduce Python overhead. Less aggressive kernel tuning than max-autotune. Good balance of compilation time and runtime speed (~85-90% of max-autotune performance).

- **`default`**: Basic fusion and optimization passes. Fastest compilation with minimal memory overhead. Still provides meaningful speedup over eager mode.

- **`false`/`none`**: Disables torch.compile entirely (eager execution). No compilation overhead. Useful for debugging or when compilation causes issues.

**Factors Affecting Compile Time:**

| Factor | Impact |
|--------|--------|
| Number of unique operations | More ops = more kernels to compile |
| Number of unique tensor shapes | Each shape triggers separate compilation |
| Branching/conditionals | Dynamic control flow requires more graph analysis |
| `max-autotune` kernel variants | Benchmarks 10-50+ configurations per matmul/conv |

**Estimated Compile Times for XEC Model:**

The XEC architecture has moderate complexity with 6 parallel face branches (4 ConvNeXt + 2 HexNeXt), Transformer fusion, and multiple task heads. This means more unique kernels than a simple ResNet, but not as many as a very deep model.

| Mode | Estimated Time | Notes |
|------|----------------|-------|
| `max-autotune` | 5-10 min | Benchmarks every matmul/conv variant; 6 branches mean ~6× more kernels to tune |
| `reduce-overhead` | 1-3 min | CUDA graph capture + basic tuning; no exhaustive benchmarking |
| `default` | 30-90 sec | Just fusion/optimization passes; no autotuning |

**Practical Notes:**

1. **First epoch appears slow** - compilation happens on first forward pass
2. **Compilation is cached** - subsequent runs in same session reuse compiled kernels
3. **Shape changes retrigger compilation** - if batch size changes (e.g., last incomplete batch), it recompiles for that shape

**Recommendations by Use Case:**

| Use Case | Recommended Mode | Reason |
|----------|------------------|--------|
| Hyperparameter search | `reduce-overhead` or `default` | Fast iteration, compile time dominates short runs |
| Quick experiments | `reduce-overhead` | Good balance of compile time and runtime |
| Final production training | `max-autotune` | 5-10 min compile amortized over many hours of training |
| Debugging | `none` | Skip compilation entirely for faster feedback |

The runtime performance difference between `reduce-overhead` and `max-autotune` is typically ~5-15%, so unless training for many hours, `reduce-overhead` often provides the better time tradeoff.

#### D. Cudagraphs Warning with reduce-overhead Mode

**Symptom:** When using `compile: "reduce-overhead"`, you see warnings like:
```
skipping cudagraphs due to cpu device (primals_2). Found from:
   File "lib/geom_utils.py", line 25, in gather_face
     idx_flat = torch.tensor(index_map.reshape(-1), device=device, dtype=torch.long)
```

**Cause:** The `reduce-overhead` mode attempts to capture CUDA graphs for the forward pass. CUDA graphs require all operations to run purely on GPU with no CPU-GPU synchronization. In `geom_utils.py`, the `gather_face()` function creates index tensors from numpy arrays:

```python
idx_flat = torch.tensor(index_map.reshape(-1), device=device, dtype=torch.long)
```

Even though `device=device` is specified, `torch.tensor()` first creates an intermediate CPU tensor from the numpy array before moving to GPU. This CPU→GPU transfer breaks cudagraph compatibility.

**Is it a problem?** No. The warning is informational, not an error. PyTorch simply falls back to eager mode for that specific operation. Your training will:
- Run correctly with no functional issues
- Still benefit from `torch.compile` optimizations for other parts of the model (convolutions, attention, etc.)
- Use the index caching mechanism (after the first forward pass, tensors are cached on GPU)

**Performance impact:** Minimal. The `gather_face()` operation is a simple index gather, which is fast regardless of cudagraphs. The bulk of compute time is in convolutions and attention layers, which are still optimized.

**Potential fix (not recommended):** To enable full cudagraph capture, all index tensors would need to be pre-registered as model buffers in `__init__` rather than created lazily:

```python
# In model __init__
self.register_buffer('inner_idx', torch.tensor(INNER_INDEX_MAP.reshape(-1), dtype=torch.long))

# In forward
vals = torch.index_select(x_batch, 1, self.inner_idx)
```

However, this adds complexity and the performance gain is negligible for this architecture. The current caching approach in `geom_utils.py` is sufficient.

**Recommendation:** Keep `reduce-overhead` as-is and ignore the warning. If you want to suppress the warning entirely, use `compile: "default"` which doesn't attempt cudagraph capture.

### 5. NaN Loss During Training

**Symptom:** Loss becomes NaN after a few epochs

**Possible causes and solutions:**
1. **Learning rate too high:** Reduce `lr` by 10x
2. **Gradient explosion:** Enable gradient clipping (`grad_clip: 1.0`)
3. **Bad normalization:** Check `npho_scale`, `time_scale` match your data
4. **Data issue:** Check for NaN/Inf in input ROOT files

```bash
# Debug data
python -c "
import uproot
f = uproot.open('your_data.root')
t = f['tree']
npho = t['relative_npho'].array()
print(f'NaN count: {np.isnan(npho).sum()}')
print(f'Inf count: {np.isinf(npho).sum()}')
"
```

**Note:** The training code automatically skips batches that produce NaN/Inf loss values to prevent model corruption.

### 5b. Loss Spike / Gradient Explosion

**Symptom:** Training progresses normally, then loss suddenly increases by 10x or more and never recovers.

**Root Cause:** A single batch with unusual data or numerical instability causes large gradients that corrupt model weights. Once weights are corrupted, the model cannot recover through normal gradient descent.

**Prevention and Diagnosis:**

#### A. Gradient Clipping (`grad_clip`)

Gradient clipping limits how much the model can change in one update:

```
How it works:
1. Compute total gradient norm: ||g|| = sqrt(Σ gᵢ²) across all parameters
2. If ||g|| > grad_clip: scale all gradients by (grad_clip / ||g||)
3. Direction preserved, magnitude capped
```

**Configuration:**
```yaml
training:
  grad_clip: 1.0    # Default - caps gradient norm at 1.0
```

#### B. Using `grad_norm_max` Metric

The training logs `grad_norm_max` to MLflow each epoch - the maximum gradient norm observed during that epoch (before clipping).

**Key concept:** `grad_clip` is a **safety ceiling**, not a target. When gradients are naturally small, a loose ceiling (e.g., 1.0) is perfectly fine - it simply won't activate. You only need to tighten it if you observe instability or want a closer safety margin.

**How to use it:**

1. **Run a few epochs** and observe `grad_norm_max` in MLflow
2. **Interpret the values:**

| `grad_norm_max` | `grad_clip` | Interpretation |
|-----------------|-------------|----------------|
| 0.001 - 0.01 | 1.0 | Gradients very small - training is stable, no action needed |
| 0.01 - 0.1 | 1.0 | Gradients small, clipping never activates - this is healthy |
| 0.5 - 0.8 | 1.0 | Gradients moderate, some headroom for spikes |
| ~1.0 consistently | 1.0 | Clipping every step - may slow learning, consider raising |
| Spike to 1.0 before loss spike | 1.0 | Gradient explosion caught by clipping |

3. **Optionally tighten `grad_clip`** (only if you want a closer safety net):

| Typical `grad_norm_max` | Optional tighter `grad_clip` |
|-------------------------|------------------------------|
| ~0.01 | 0.05 - 0.1 |
| ~0.1 | 0.3 - 0.5 |
| ~0.5 | 1.0 - 2.0 |

**Goal:** Keep `grad_clip` at ~2-5× your typical gradient norm. The default of 1.0 is safe for most cases and will catch explosions without interfering with normal training.

#### C. Recovery from Loss Spike

If a loss spike occurs:

1. **Resume from best checkpoint:**
   ```yaml
   checkpoint:
     resume_from: "artifacts/<run_name>/checkpoint_best.pth"
   ```

2. **Lower learning rate:**
   ```yaml
   training:
     lr: 1.0e-4    # Try 3-10x lower than before
   ```

3. **Lower grad_clip:**
   ```yaml
   training:
     grad_clip: 0.5    # More aggressive clipping
   ```

#### D. Task-Specific Considerations

Different tasks may need different settings:

| Task | Typical Error Scale | `loss_beta` | Notes |
|------|---------------------|-------------|-------|
| Angle | ~1-10° | 1.0 | Default works well |
| Energy (GeV) | ~0.001-0.01 | 0.01 | Small errors need small beta |
| Timing | ~1e-9 | 0.1 | Adjust based on time scale |

For energy regression with GeV units (0.015-0.06 range), errors are typically ~0.005 GeV. With `loss_beta: 1.0`, smooth_l1 is always in quadratic mode (effectively MSE). Consider `loss_beta: 0.01` or use pure `loss_fn: "l1"` for more robustness.

### 6. Slow Data Loading (CPU Bottleneck)

**Symptom:** GPU utilization < 50%, `data_load` takes >20% of epoch time in profiler output

**Understanding the parameters:**

| Parameter | What it does | Memory Impact | Speed Impact |
|-----------|--------------|---------------|--------------|
| `chunksize` | Events loaded into RAM per ROOT read | **High** - directly proportional | Fewer I/O operations |
| `num_workers` | DataLoader worker processes | **High** - each has own memory | Parallel file reading |
| `num_threads` | Threads for preprocessing within worker | **Minimal** - threads share memory | Parallel normalization |

**Solutions by platform:**

For **A100 nodes** (more memory, no worker limitations):
```yaml
data:
  chunksize: 262144     # 256K events
  num_workers: 4        # Parallel prefetching
  num_threads: 4
```

For **Grace-Hopper nodes** (see section 9 for limitations):
```yaml
data:
  chunksize: 65536      # Keep small to avoid CPU OOM
  num_workers: 1        # GH limitation - cannot use >1
  num_threads: 8        # Safe to increase - speeds up preprocessing
```

**Why `num_threads` is safe from OOM:** Threads share memory within the same process. Increasing `num_threads` just parallelizes preprocessing of the *already-loaded* chunk using numpy views (not copies).

### 7. Checkpoint Resume Fails

**Symptom:** `KeyError` or shape mismatch when resuming

**Possible causes:**
1. **Model architecture changed:** Ensure `outer_mode`, `outer_fine_pool` match
2. **MAE vs Full checkpoint confusion:** MAE checkpoints don't have optimizer state
3. **Task configuration changed:** Multi-task model expects same enabled tasks

**Solution:** Start fresh or ensure config matches checkpoint:
```bash
# Check what's in the checkpoint
python -c "
import torch
ckpt = torch.load('checkpoint.pth', map_location='cpu', weights_only=False)
print('Keys:', ckpt.keys())
if 'config' in ckpt:
    print('Config:', ckpt['config'])
"
```

### 8. Inconsistent Results Between Runs

**Symptom:** Different results with same configuration

**Solution:** Set random seeds:
```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

### 9. Grace-Hopper (GH) Node Specific Limitations

Grace-Hopper nodes have several platform-specific constraints:

#### A. DataLoader `num_workers` Must Be 1

**Symptom:** Training hangs or crashes with `num_workers > 1`

**Cause:** GH nodes have limited multiprocessing capacity due to ARM architecture differences and CUDA context handling with unified memory.

**Solution:**
```yaml
data:
  num_workers: 1        # Required for GH nodes
  num_threads: 8        # Compensate with more preprocessing threads
```

#### B. `torch.compile` Auto-Disabled

**Symptom:** Log shows `[INFO] ARM architecture detected - disabling torch.compile`

**Cause:** Triton (used by torch.compile) doesn't support ARM architecture.

**Impact:** Training runs in eager mode, which is slightly slower but fully functional. No action needed.

#### C. `torch.compile` + Multiprocessing Conflict (A100 nodes)

**Symptom:** `pthread_join failed` errors when using both `torch.compile` and `num_workers > 0`

**Cause:** LLVM/Triton conflict with multiprocessing DataLoader workers.

**Solutions:**
```yaml
# Option 1: Disable torch.compile
training:
  compile: false

# Option 2: Use single worker
data:
  num_workers: 0
```

#### D. Recommended GH Configuration

```yaml
data:
  batch_size: 2048
  chunksize: 65536      # Keep small - GH has unified memory constraints
  num_workers: 1        # GH limitation
  num_threads: 8        # Safe to increase for faster preprocessing

training:
  compile: "reduce-overhead"  # Will auto-disable on ARM anyway
```

---

## FAQ

**Q: Can I train on CPU?**
A: Technically yes, but not recommended. Training is 50-100x slower. For debugging:
```bash
CUDA_VISIBLE_DEVICES="" python -m lib.train_mae --config config.yaml
```

**Q: How do I know if MAE pretraining helped?**
A: Compare validation metrics:
1. Train regressor from scratch (no `--resume_from`)
2. Train regressor with MAE weights (`--resume_from mae_checkpoint.pth`)
3. Compare `val_resolution_deg` after same number of epochs

**Q: What mask_ratio should I use for MAE?**
A: Typical values: 0.5-0.75. Higher masking forces the model to learn better representations but may hurt reconstruction quality. Start with 0.6 (65%).

**Q: How do I export for C++ inference?**
A: Use the ONNX export script:
```bash
python macro/export_onnx.py artifacts/my_run/checkpoint_best.pth --output model.onnx

# Verify the export
python -c "
import onnxruntime as ort
sess = ort.InferenceSession('model.onnx')
print('Inputs:', [i.name for i in sess.get_inputs()])
print('Outputs:', [o.name for o in sess.get_outputs()])
"
```

**Q: Why is `actual_mask_ratio` different from `mask_ratio`?**
A: `actual_mask_ratio` accounts for already-invalid sensors in the data. If 10% of sensors are already invalid (time == sentinel), and you set `mask_ratio=0.6`, then:
- Valid sensors: 90% × 4760 = 4284
- Randomly masked: 60% × 4284 = 2570
- `actual_mask_ratio` = 2570 / 4284 ≈ 0.60

**Q: Can I use different normalization for MAE and regression?**
A: **No.** The encoder learns features based on the input distribution. If you change normalization, the learned features won't transfer correctly. Always use the same `npho_scale`, `time_scale`, etc. See [Data Pipeline](../architecture/data-pipeline.md) for the two available schemes.

**Q: What is gradient accumulation and when should I use it?**
A: Gradient accumulation simulates larger batch sizes when GPU memory is limited. Set `grad_accum_steps: 4` with `batch_size: 512` to get an effective batch of 2048. Use when you want larger batches for better convergence but can't fit them in GPU memory.

**Q: Why does inpainter training use so much CPU memory with large datasets?**
A: When `save_predictions: true`, the inpainter collects ALL validation predictions in memory before writing to ROOT. With 1M events × 5% mask × 238 sensors, this can use 2-5 GB of RAM. Disable `save_predictions` for large-scale training, or use a smaller validation set.

**Q: How do I visualize what the model learned?**
A: Several options:
1. **Saliency maps:** Generated automatically at end of training (`saliency_profile_*.pdf`)
2. **MAE reconstructions:** Check `mae_predictions_epoch_*.root`
3. **Worst events:** Check `worst_event_*.pdf` for failure modes
4. **MLflow:** `mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns.db`
