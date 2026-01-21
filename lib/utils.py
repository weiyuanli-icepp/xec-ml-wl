import torch
import numpy as np
import uproot
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
