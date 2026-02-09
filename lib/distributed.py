"""
Centralized DDP (DistributedDataParallel) utilities.

All training scripts import from here. When launched with `torchrun`, DDP is
activated automatically via RANK/WORLD_SIZE/LOCAL_RANK env vars. When launched
with plain `python`, everything is a no-op and training runs on a single GPU
exactly as before.

Usage in training scripts:
    from .distributed import (
        setup_ddp, cleanup_ddp, is_main_process,
        shard_file_list, reduce_metrics, wrap_ddp, barrier,
    )

    rank, local_rank, world_size = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    ...
    model = wrap_ddp(model, local_rank)
    ...
    cleanup_ddp()
"""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_ddp():
    """
    Initialize DDP from torchrun environment variables.

    Returns:
        (rank, local_rank, world_size): Process identifiers.
            - If torchrun env vars are absent, returns (0, 0, 1) (single-GPU).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")

        return rank, local_rank, world_size

    # Not launched with torchrun — single-GPU mode
    return 0, 0, 1


def cleanup_ddp():
    """Destroy the process group if DDP is initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Return True if this is rank 0 or DDP is not active."""
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def shard_file_list(files, rank, world_size):
    """
    Round-robin shard a list of files across ranks.

    Args:
        files: List of file paths.
        rank: Current process rank.
        world_size: Total number of processes.

    Returns:
        Subset of files assigned to this rank.
    """
    if world_size <= 1:
        return files
    return files[rank::world_size]


def reduce_metrics(metrics_dict, device):
    """
    All-reduce (average) a dict of scalar metrics across all ranks.

    No-op if DDP is not active.

    Args:
        metrics_dict: Dict[str, float] of metric values.
        device: torch.device to create tensors on.

    Returns:
        Dict[str, float] with averaged values (in-place modification + return).
    """
    if not dist.is_initialized():
        return metrics_dict

    world_size = dist.get_world_size()
    for key in metrics_dict:
        val = torch.tensor(metrics_dict[key], dtype=torch.float64, device=device)
        dist.all_reduce(val, op=dist.ReduceOp.SUM)
        metrics_dict[key] = (val / world_size).item()

    return metrics_dict


def wrap_ddp(model, local_rank):
    """
    Wrap a model with DistributedDataParallel.

    No-op if DDP is not active. Should be called AFTER model.to(device)
    and BEFORE torch.compile().

    Args:
        model: The model (already on the correct device).
        local_rank: Local GPU index.

    Returns:
        DDP-wrapped model, or the original model if not distributed.
    """
    if not dist.is_initialized():
        return model
    return DDP(model, device_ids=[local_rank])


def barrier():
    """Synchronize all ranks. No-op if DDP is not active."""
    if dist.is_initialized():
        dist.barrier()
