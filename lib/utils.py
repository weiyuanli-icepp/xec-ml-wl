import torch
import numpy as np
import uproot
import time
import os
import psutil
from .geom_utils import gather_face, gather_hex_nodes
from .geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP, 
    OUTER_COARSE_FULL_INDEX_MAP, OUTER_CENTER_INDEX_MAP,
    TOP_HEX_ROWS, BOTTOM_HEX_ROWS, flatten_hex_rows
)

# ------------------------------------------------------------
# GPU memory monitoring utilities
# ------------------------------------------------------------
def format_mem(bytes_val):
    return f"{bytes_val / (1024**3):.2f} GB"

def get_gpu_memory_stats(device=None):
    if not torch.cuda.is_available():
        return None
    if device is None:
        device = torch.device("cuda")
    torch.cuda.synchronize(device)
    alloc   = torch.cuda.memory_allocated(device)
    reserv  = torch.cuda.memory_reserved(device)
    peak    = torch.cuda.max_memory_allocated(device)
    return {
        "allocated": alloc,
        "reserved": reserv,
        "peak": peak,
    }

def count_model_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# ------------------------------------------------------------
#  Utilities: streaming ROOT reading
# ------------------------------------------------------------
def iterate_chunks(files, tree, branches, step_size=4000):
    """
    files: Can be a single string "file.root", a wildcard "file_*.root", or a list ["file1.root", "file2.root"]
    """
    if isinstance(files, list):
        files_input = {f: tree for f in files}
    else:
        files_input = f"{files}:{tree}"
        
    
    for arrays in uproot.iterate(files_input, branches, step_size=step_size, library="np"):
        yield arrays

# ------------------------------------------------------------
# Loss function helper
# ------------------------------------------------------------
def get_pointwise_loss_fn(loss_name: str):
    """
    Returns a point-wise loss function with reduction='none'.
    Supported: smooth_l1/huber, mse, l1. Defaults to smooth_l1.
    """
    name = (loss_name or "").lower()
    if name in ("smooth_l1", "huber"):
        return lambda pred, target: torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")
    if name == "l1":
        return lambda pred, target: torch.nn.functional.l1_loss(pred, target, reduction="none")
    if name == "mse":
        return lambda pred, target: torch.nn.functional.mse_loss(pred, target, reduction="none")
    return lambda pred, target: torch.nn.functional.smooth_l1_loss(pred, target, reduction="none")

# ------------------------------------------------------------
# Angle conversion utilities
# ------------------------------------------------------------
def angles_deg_to_unit_vec(angles: torch.Tensor) -> torch.Tensor:
    # angles in degrees -> radians
    theta = torch.deg2rad(angles[:, 0])
    phi   = torch.deg2rad(angles[:, 1])

    sin_th = torch.sin(theta)
    cos_th = torch.cos(theta)
    sin_ph = torch.sin(phi)
    cos_ph = torch.cos(phi)

    x = -sin_th * cos_ph
    y =  sin_th * sin_ph
    z =  cos_th

    v = torch.stack([x, y, z], dim=1)  # (B,3)
    return v

# ------------------------------------------------------------
# Physics Analysis: Gradient Saliency
# ------------------------------------------------------------
def compute_face_saliency(model, x_batch, device):
    """
    Computes which faces are most important for determining Theta vs Phi.
    Separates importance by input channel (Npho vs Time).
    """
    model.eval()
    x_in = x_batch.clone().to(device)
    x_in.requires_grad = True
    pred = model(x_in)  # (B, 2)
    
    # Structure: results[angle][channel][face] = score
    saliency_results = {}
    channels = {0: "npho", 1: "time"}
    
    for angle_idx, angle_name in enumerate(["theta", "phi"]):
        saliency_results[angle_name] = {}
        
        if x_in.grad is not None:
            x_in.grad.zero_()
            
        target = pred[:, angle_idx]
        target.sum().backward(retain_graph=True)
        
        for ch_idx, ch_name in channels.items():
            grads = x_in.grad[:, :, ch_idx].abs() 
            
            face_scores = {}
            
            face_scores["inner"] = gather_face(grads.unsqueeze(-1), INNER_INDEX_MAP).mean()
            face_scores["us"]    = gather_face(grads.unsqueeze(-1), US_INDEX_MAP).mean()
            face_scores["ds"]    = gather_face(grads.unsqueeze(-1), DS_INDEX_MAP).mean()
            
            if hasattr(model, "outer_mode") and model.outer_mode == "split":
                face_scores["outer_coarse"] = gather_face(grads.unsqueeze(-1), OUTER_COARSE_FULL_INDEX_MAP).mean()
                face_scores["outer_center"] = gather_face(grads.unsqueeze(-1), OUTER_CENTER_INDEX_MAP).mean()
                
            top_indices = torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long().to(device)
            bot_indices = torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long().to(device)
            
            face_scores["hex_top"] = gather_hex_nodes(grads.unsqueeze(-1), top_indices).mean()
            face_scores["hex_bottom"] = gather_hex_nodes(grads.unsqueeze(-1), bot_indices).mean()
            
            saliency_results[angle_name][ch_name] = {k: v.item() for k, v in face_scores.items()}

    return saliency_results


# ------------------------------------------------------------
# Training Profiler
# ------------------------------------------------------------
class SimpleProfiler:
    """Simple timer-based profiler for identifying training bottlenecks.

    Usage:
        profiler = SimpleProfiler(enabled=True)
        profiler.start("forward")
        # ... forward pass ...
        profiler.stop()
        profiler.start("backward")
        # ... backward pass ...
        profiler.stop()
        print(profiler.report())
    """

    def __init__(self, enabled=False, sync_cuda=True):
        self.enabled = enabled
        self.sync_cuda = sync_cuda
        self.timings = {}
        self.counts = {}
        self._start_time = None
        self._current_name = None

    def start(self, name):
        if not self.enabled:
            return
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self._current_name = name
        self._start_time = time.perf_counter()

    def stop(self):
        if not self.enabled or self._start_time is None:
            return
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._start_time
        name = self._current_name
        if name not in self.timings:
            self.timings[name] = 0.0
            self.counts[name] = 0
        self.timings[name] += elapsed
        self.counts[name] += 1
        self._start_time = None
        self._current_name = None

    def report(self, title="Timing breakdown"):
        if not self.enabled or not self.timings:
            return ""
        lines = [f"[Profiler] {title}:"]
        total = sum(self.timings.values())
        for name, t in sorted(self.timings.items(), key=lambda x: -x[1]):
            pct = 100 * t / total if total > 0 else 0
            avg = t / self.counts[name] if self.counts[name] > 0 else 0
            lines.append(f"  {name}: {t:.2f}s ({pct:.1f}%) | {avg*1000:.2f}ms avg")
        lines.append(f"  TOTAL: {total:.2f}s")
        return "\n".join(lines)

    def reset(self):
        """Reset all timings for a new epoch."""
        self.timings = {}
        self.counts = {}
        self._start_time = None
        self._current_name = None


# ------------------------------------------------------------
# MLflow System Metrics Logging
# ------------------------------------------------------------
def get_system_metrics(device=None):
    """
    Collect standardized system metrics for MLflow logging.

    Returns:
        dict: Metrics dictionary with standardized names:
            - system/vram_allocated_GB: GPU memory allocated
            - system/vram_reserved_GB: GPU memory reserved
            - system/vram_peak_GB: GPU peak memory
            - system/vram_utilization: GPU memory utilization (0-1)
            - system/ram_used_GB: System RAM used
            - system/ram_percent: System RAM percentage
            - system/process_rss_GB: Process resident memory
    """
    metrics = {}

    # GPU metrics
    if torch.cuda.is_available():
        if device is None:
            device = torch.device("cuda")
        torch.cuda.synchronize(device)

        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        peak = torch.cuda.max_memory_allocated(device)

        metrics["system/vram_allocated_GB"] = allocated / 1e9
        metrics["system/vram_reserved_GB"] = reserved / 1e9
        metrics["system/vram_peak_GB"] = peak / 1e9

        # Utilization: allocated / reserved (how efficiently we use reserved memory)
        if reserved > 0:
            metrics["system/vram_utilization"] = allocated / reserved

    # RAM metrics
    try:
        ram = psutil.virtual_memory()
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()

        metrics["system/ram_used_GB"] = ram.used / 1e9
        metrics["system/ram_percent"] = ram.percent
        metrics["system/process_rss_GB"] = mem_info.rss / 1e9
    except Exception:
        pass

    return metrics


def log_system_metrics_to_mlflow(step, device=None, epoch_time_sec=None,
                                  throughput_events_per_sec=None,
                                  lr=None):
    """
    Log standardized system metrics to MLflow.

    Args:
        step: MLflow step (usually epoch number)
        device: torch device for GPU metrics
        epoch_time_sec: Time taken for this epoch
        throughput_events_per_sec: Training throughput
        lr: Current learning rate
    """
    try:
        import mlflow
    except ImportError:
        return

    metrics = get_system_metrics(device)

    # Add optional metrics
    if epoch_time_sec is not None:
        metrics["system/epoch_time_sec"] = epoch_time_sec

    if throughput_events_per_sec is not None:
        metrics["system/throughput_events_per_sec"] = throughput_events_per_sec

    if lr is not None:
        metrics["lr"] = lr

    if metrics:
        mlflow.log_metrics(metrics, step=step)
