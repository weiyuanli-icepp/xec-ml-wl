#!/usr/bin/env python3
"""
Standalone DDP validation script.

Run on a cluster node with multiple GPUs to verify that the distributed
utilities work correctly before touching training code.

Usage:
    # Single-GPU (sanity check — everything should be no-op):
    python tests/test_ddp.py

    # Multi-GPU (actual DDP test):
    torchrun --nproc_per_node=2 tests/test_ddp.py
    torchrun --nproc_per_node=4 tests/test_ddp.py
"""

import sys
import os

# Add project root so `lib.*` imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from lib.distributed import (
    setup_ddp,
    cleanup_ddp,
    is_main_process,
    shard_file_list,
    reduce_metrics,
    wrap_ddp,
    barrier,
)


def _log(rank, msg):
    print(f"[rank {rank}] {msg}", flush=True)


def test_setup(rank, local_rank, world_size):
    """Verify setup_ddp returns sensible values."""
    assert 0 <= rank < world_size
    assert 0 <= local_rank < torch.cuda.device_count()
    _log(rank, f"setup_ddp OK  (rank={rank}, local_rank={local_rank}, world_size={world_size})")


def test_is_main_process(rank, world_size):
    """Only rank 0 should report as main."""
    if world_size > 1:
        if rank == 0:
            assert is_main_process()
        else:
            assert not is_main_process()
    else:
        assert is_main_process()
    _log(rank, "is_main_process OK")


def test_shard_file_list(rank, world_size):
    """Verify round-robin sharding gives disjoint, complete coverage."""
    files = [f"file_{i}.root" for i in range(10)]
    shard = shard_file_list(files, rank, world_size)

    # Each rank gets its round-robin slice
    expected = files[rank::world_size]
    assert shard == expected, f"Expected {expected}, got {shard}"

    # Collectively, all shards should cover all files (no duplicates, no gaps)
    barrier()
    _log(rank, f"shard_file_list OK  ({len(shard)} files)")


def test_wrap_ddp_and_gradient_sync(rank, local_rank, world_size):
    """
    Wrap a toy model with DDP, run a forward/backward pass, and verify
    that gradients are synchronized across ranks.
    """
    device = torch.device(f"cuda:{local_rank}")

    # Simple model — same init on all ranks (DDP guarantees this via broadcast)
    torch.manual_seed(42)
    model = nn.Linear(8, 4).to(device)
    model = wrap_ddp(model, local_rank)

    # Each rank uses a DIFFERENT input to produce different local gradients
    torch.manual_seed(rank)
    x = torch.randn(16, 8, device=device)
    loss = model(x).sum()
    loss.backward()

    if world_size > 1:
        # After backward, DDP should have averaged the gradients
        # Collect the gradient from all ranks and verify they're the same
        local_grad = model.module.weight.grad.clone()
        all_grads = [torch.zeros_like(local_grad) for _ in range(world_size)]
        torch.distributed.all_gather(all_grads, local_grad)

        for i in range(1, world_size):
            assert torch.allclose(all_grads[0], all_grads[i], atol=1e-6), (
                f"Gradients differ between rank 0 and rank {i}"
            )
        _log(rank, "DDP gradient sync OK")
    else:
        _log(rank, "DDP gradient sync SKIPPED (single-GPU)")


def test_reduce_metrics(rank, local_rank, world_size):
    """Verify all-reduce averaging of metrics."""
    device = torch.device(f"cuda:{local_rank}")

    # Each rank reports a different metric value
    metrics = {
        "loss": float(rank + 1),      # rank 0 → 1.0, rank 1 → 2.0, ...
        "accuracy": float(rank * 10),  # rank 0 → 0.0, rank 1 → 10.0, ...
    }

    reduced = reduce_metrics(metrics, device)

    if world_size > 1:
        # Expected average: loss = mean(1, 2, ..., world_size)
        expected_loss = sum(range(1, world_size + 1)) / world_size
        expected_acc = sum(r * 10 for r in range(world_size)) / world_size

        assert abs(reduced["loss"] - expected_loss) < 1e-6, (
            f"Expected loss={expected_loss}, got {reduced['loss']}"
        )
        assert abs(reduced["accuracy"] - expected_acc) < 1e-6, (
            f"Expected accuracy={expected_acc}, got {reduced['accuracy']}"
        )
        _log(rank, f"reduce_metrics OK  (loss={reduced['loss']:.4f}, acc={reduced['accuracy']:.4f})")
    else:
        # Single-GPU: metrics should be unchanged
        assert reduced["loss"] == 1.0
        assert reduced["accuracy"] == 0.0
        _log(rank, "reduce_metrics OK (single-GPU, no-op)")


def test_rank_gated_io(rank, world_size):
    """Verify rank-gating logic for I/O operations."""
    if is_main_process():
        _log(rank, "rank_gated_io: I am rank 0, I would do MLflow/checkpoint I/O")
    else:
        _log(rank, "rank_gated_io: I am NOT rank 0, skipping I/O")

    barrier()
    _log(rank, "rank_gated_io OK")


def test_barrier(rank, world_size):
    """Verify barrier synchronization."""
    barrier()
    _log(rank, "barrier OK")


def main():
    rank, local_rank, world_size = setup_ddp()

    if is_main_process():
        print("=" * 60)
        print(f"DDP Validation Test  (world_size={world_size})")
        print("=" * 60)

    try:
        test_setup(rank, local_rank, world_size)
        test_is_main_process(rank, world_size)
        test_shard_file_list(rank, world_size)
        test_wrap_ddp_and_gradient_sync(rank, local_rank, world_size)
        test_reduce_metrics(rank, local_rank, world_size)
        test_rank_gated_io(rank, world_size)
        test_barrier(rank, world_size)

        barrier()
        if is_main_process():
            print("=" * 60)
            print("ALL TESTS PASSED")
            print("=" * 60)
    finally:
        cleanup_ddp()


if __name__ == "__main__":
    main()
