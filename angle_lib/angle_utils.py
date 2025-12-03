import torch
import numpy as np
import uproot

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

# ------------------------------------------------------------
#  Utilities: streaming ROOT reading
# ------------------------------------------------------------
def iterate_chunks(path, tree, branches, step_size=4000):
    with uproot.open(path) as f:
        t = f[tree]
        for arrays in t.iterate(branches, step_size=step_size, library="np"):
            yield arrays

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