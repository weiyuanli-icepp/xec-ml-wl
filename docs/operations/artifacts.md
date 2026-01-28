# Output & Artifacts

All results are logged to **MLflow** and stored in the `artifacts/<RUN_NAME>/` directory.

## Key Artifacts

* `checkpoint_best.pth` — Best model weights (includes EMA state).
* `checkpoint_last.pth` — Last epoch's model weights (includes EMA state).
* `predictions_*.csv` — Validation predictions vs truth.
* `*.onnx` — Exported ONNX model for C++ inference (supports single-task and multi-task).
* `validation_results_*.root` — ROOT file containing event-by-event predictions and truth variables.

## Plots

* `resolution_profile_*.pdf`: 68% width resolution vs $\theta$/$\phi$.
* `saliency_profile_*.pdf`: Physics Sensitivity analysis (Gradient of output w.r.t input Npho/Time).
* `worst_event_*.pdf`: Event displays of the highest-loss events.
* `saliency_profile_*.pdf`: Computes $\nabla_{\text{Input}} \text{Output}$ to quantify how much the model relies on Photon Counts vs Timing for each face (Inner, Top, Hex, etc.) to determine $\theta$ and $\phi$.

## Visualization Tools

The real time tracking of the training is available with MLflow and TensorBoard.
```bash
# Start MLflow (Track metrics & PDFs)
$ cd /path/to/xec-ml-wl
$ (activate xec-ml-wl conda environment)
$ mlflow ui --backend-store-uri sqlite:///$(pwd)/mlruns.db --host 127.0.0.1 --port 5000

# Start TensorBoard (Track Loss Curves)
$ tensorboard --logdir runs --host 0.0.0.0 --port YYYY
```

## Metrics Definition

### 1. Physics Performance Metrics

These metrics evaluate the quality of the photon direction reconstruction. They are calculated during the validation phase using `eval_stats` and `eval_resolution`.

| Metric | Definition | Formula |
| ------ | ---------- | ------- |
| Theta Bias (`theta_bias`) | The arithmetic mean of the residuals. | $\mu = \text{Mean}(\theta_{\mathrm{pred}} - \theta_{\mathrm{true}})$ |
| Theta RMS (`theta_rms`) | The standard deviation of the residuals. | $\sigma = \text{Std}(\theta_{\mathrm{pred}} - \theta_{\mathrm{true}})$ |
| Theta Skewness (`theta_skew`) | A measure of the asymmetry of the error distribution. | $$\text{Skew} = \frac{\frac{1}{N} \sum_{i=1}^{N} (\Delta \theta_i - \mu)^3}{\left( \frac{1}{N} \sum_{i=1}^{N} (\Delta \theta_i - \mu)^2 \right)^{3/2}}$$ |
| Opening Angle Resolution (`val_resolution_deg`) | The 68th percentile of the 3D opening angle $\psi$ between the predicted and true vectors. | $\psi = \arccos(v_{\mathrm{pred}} \cdot v_{\mathrm{true}})$ |

### 2. System Engineering Metrics

These metrics monitor the health of the training infrastructure (GPU/CPU) to detect bottlenecks or imminent crashes.

| Metric           | Key in MLflow                    | Interpretation                                                                            |
| ---------------- | -------------------------------- | --------------------------------- |
| Allocated Memory | `system/` `memory_allocated_GB`  | The actual size of tensors (weights, gradients, data) on the GPU. Steady growth indicates a memory leak.Reserved Memorysystem/memory_reserved_GBThe total memory PyTorch has requested from the OS. If this hits the hardware limit, an OOM crash occurs.                                                      |
| Peak Memory      | `system/` `memory_peak_GB`       | The highest memory usage recorded (usually during the backward pass). Use this to tune batch_size.                          |
| GPU Utilization  | `system/` `gpu_utilization_pct`  | Ratio of Allocated to Total VRAM. Low values (<50%) suggest the batch size can be increased; very high values (>90%) risk OOM.  |
| Fragmentation    | `system/` `memory_fragmentation` | Ratio of empty space within reserved memory blocks. High fragmentation (>0.5) indicates inefficient memory use.                |
| RAM Usage        | `system/` `ram_used_gb`          | System RAM used by the process. High usage warns that step_size for ROOT file reading is too large.                            |
| Throughput       | `system/` `epoch_duration_sec`   | Wall-clock time per epoch. If high while GPU utilization is low, the pipeline is CPU-bound (data loading bottleneck).        |

### 3. System Performance Metrics

These metrics determine if the training pipeline is efficient or bottlenecked.
| Metric             | Key in MLflow                      | Definition & goal                 |
| ------------------ | ---------------------------------- | --------------------------------- |
| Throughput         | `system/throughput_events_per_sec` | Events processed per second.      |
| Data Load Time     | `system/avg_data_load_sec`         | Time GPU waits for CPU. If high, increase CHUNK_SIZE. |
| Compute Efficiency | `system/compute_efficiency`        | % of time GPU is computing.       |

---

## Resuming Training

The script supports resumption. It detects if an EMA state exists in the checkpoint and loads it; otherwise, it syncs the EMA model with the loaded weights to prevent training divergence.

```bash
--resume_from "artifacts/<run name>/checkpoint_last.pth"
```
or
```bash
--resume_from "artifacts/<run name>/checkpoint_best.pth"
```

If the run configurated with a scheduler and stopped in the middle of training, it can be resumed from the learning rate ( $\mathrm{LR}$ ) where it stopped. The learning rate can be calculated with following formula:
 $$\mathrm{LR} = \mathrm{LR}_\mathrm{min} + \frac{1}{2} \Big(\mathrm{LR}_{\mathrm{max}} - \mathrm{LR}_{\mathrm{min}}\Big) \Bigg(1 + \cos \Big(\frac{\mathrm{epoch} - \mathrm{warmup}}{\mathrm{total} - \mathrm{warmup}} \pi\Big)\Bigg)$$
