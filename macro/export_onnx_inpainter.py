#!/usr/bin/env python3
"""
Export XEC_Inpainter model to TorchScript format.

NOTE: ONNX export is NOT supported. The model uses nn.TransformerEncoder which
internally uses a fused CUDA kernel (aten::_transformer_encoder_layer_fwd) that
cannot be exported to ONNX. This is a PyTorch limitation, not a model limitation.
TorchScript export works correctly.

Usage:
    python macro/export_onnx_inpainter.py checkpoint.pth --output inpainter.pt

The exported model takes:
    - input: (B, 4760, 2) - sensor values (npho, time) with dead channels as sentinel
    - mask: (B, 4760) - binary mask (1 = dead/masked, 0 = valid)

Returns:
    - output: (B, 4760, 2) - reconstructed values (predictions at masked positions,
              original values at unmasked positions)

For C++ inference with libtorch:
    auto model = torch::jit::load("inpainter.pt");
    auto output = model.forward({input, mask}).toTensor();
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.models import XECEncoder
from lib.models.inpainter import XEC_Inpainter


class InpainterScriptableWrapper(nn.Module):
    """
    TorchScript-compatible wrapper for XEC_Inpainter.

    Uses the forward_full_output() method which returns fixed-size (B, 4760, 2) tensor,
    enabling clean TorchScript export without dynamic tensor size issues.
    """

    def __init__(self, inpainter: XEC_Inpainter):
        super().__init__()
        self.inpainter = inpainter

    def forward(self, x_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_input: (B, 4760, 2) - sensor values with dead channels as sentinel
            mask: (B, 4760) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            output: (B, 4760, 2) - tensor with inpainted values at masked positions,
                    original values preserved at unmasked positions
        """
        # Use fixed-size output method for clean TorchScript export
        pred_all = self.inpainter.forward_full_output(x_input, mask)

        # Combine: use predictions at masked positions, original at unmasked
        mask_expanded = mask.bool().unsqueeze(-1).expand_as(x_input)
        output = torch.where(mask_expanded, pred_all, x_input)

        return output


# Keep old class name as alias for backward compatibility
InpainterONNXWrapper = InpainterScriptableWrapper


def load_inpainter_checkpoint(checkpoint_path, prefer_ema=True):
    """
    Load inpainter checkpoint and return model with weights.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract config
    config = checkpoint.get("config", {})

    # Get model config
    outer_mode = config.get("outer_mode", "finegrid")
    outer_fine_pool = config.get("outer_fine_pool", (3, 3))
    sentinel_value = config.get("sentinel_value", -5.0)
    time_mask_ratio_scale = config.get("time_mask_ratio_scale", 1.0)

    print(f"[INFO] Model config: outer_mode={outer_mode}, outer_fine_pool={outer_fine_pool}")

    # Create encoder
    encoder = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool if isinstance(outer_fine_pool, tuple) else tuple(outer_fine_pool),
        drop_path_rate=0.0  # Always 0 for inference
    )

    # Create inpainter
    inpainter = XEC_Inpainter(
        encoder=encoder,
        freeze_encoder=True,
        sentinel_value=sentinel_value,
        time_mask_ratio_scale=time_mask_ratio_scale
    )

    # Load weights
    if prefer_ema and "ema_state_dict" in checkpoint and checkpoint["ema_state_dict"] is not None:
        print("[INFO] Loading EMA weights")
        raw_dict = checkpoint["ema_state_dict"]
        source = "EMA"
    elif "model_state_dict" in checkpoint:
        print("[INFO] Loading standard model weights")
        raw_dict = checkpoint["model_state_dict"]
        source = "Standard"
    else:
        raise KeyError("Checkpoint missing 'model_state_dict' or 'ema_state_dict'")

    # Clean up keys (remove 'module.' prefix if present)
    clean_dict = {}
    for k, v in raw_dict.items():
        if k.startswith("module."):
            clean_dict[k[7:]] = v
        else:
            clean_dict[k] = v

    # Load state dict
    try:
        inpainter.load_state_dict(clean_dict, strict=True)
        print(f"[SUCCESS] Loaded {source} weights successfully")
    except RuntimeError as e:
        print(f"[WARN] Strict loading failed, trying non-strict: {e}")
        inpainter.load_state_dict(clean_dict, strict=False)

    return inpainter, config


def main():
    parser = argparse.ArgumentParser(
        description="Export XEC_Inpainter to TorchScript format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to TorchScript
  python macro/export_onnx_inpainter.py checkpoint.pth --output inpainter.pt

  # Without EMA weights
  python macro/export_onnx_inpainter.py checkpoint.pth --no-ema --output inpainter.pt

NOTE: ONNX export is NOT supported due to nn.TransformerEncoder's fused kernel.
        """
    )
    parser.add_argument("checkpoint", type=str, help="Path to inpainter checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: inpainter.pt)")
    parser.add_argument("--no-ema", action="store_true", help="Use standard weights instead of EMA")
    parser.add_argument("--trace-batch-size", type=int, default=64,
                        help="Batch size for tracing (default: 64)")
    parser.add_argument("--trace-mask-per-event", type=int, default=150,
                        help="Number of masked sensors per event for tracing (default: 150)")

    args = parser.parse_args()

    # Set default output filename
    if args.output is None:
        args.output = "inpainter.pt"

    # Load model
    inpainter, config = load_inpainter_checkpoint(args.checkpoint, prefer_ema=not args.no_ema)
    inpainter.eval()

    # Create wrapper
    wrapper = InpainterONNXWrapper(inpainter)
    wrapper.eval()

    # Dummy inputs for tracing
    # With forward_full_output(), the model returns fixed-size (B, 4760, 2) tensor,
    # so tracing is clean and batch size can vary at inference time.
    trace_batch = args.trace_batch_size
    trace_mask_per_event = args.trace_mask_per_event

    print(f"[INFO] Tracing with batch_size={trace_batch}, ~{trace_mask_per_event} masked per event")
    print(f"[INFO] Using fixed-size output method (forward_full_output) for clean tracing")

    dummy_input = torch.randn(trace_batch, 4760, 2)
    dummy_mask = torch.zeros(trace_batch, 4760)

    # Create realistic mask pattern for each event in batch
    for b in range(trace_batch):
        # Random mask positions (simulating dead + artificial)
        mask_indices = torch.randperm(4760)[:trace_mask_per_event]
        dummy_mask[b, mask_indices] = 1.0

    # Test forward pass
    print("[INFO] Testing forward pass...")
    with torch.no_grad():
        test_output = wrapper(dummy_input, dummy_mask)
    print(f"[INFO] Output shape: {test_output.shape}")

    # Create output directory
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Export to TorchScript
    print(f"[INFO] Exporting to TorchScript: {args.output}")

    try:
        # Use tracing (simpler than scripting for this model)
        traced = torch.jit.trace(wrapper, (dummy_input, dummy_mask))
        traced.save(args.output)
        print(f"[SUCCESS] Model exported to {args.output}")

        # Print model info
        file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
        print(f"[INFO] File size: {file_size_mb:.1f} MB")

        # Verify
        print("[INFO] Verifying TorchScript model...")
        loaded = torch.jit.load(args.output)
        with torch.no_grad():
            ts_output = loaded(dummy_input, dummy_mask)
            pt_output = wrapper(dummy_input, dummy_mask)

        max_diff = (ts_output - pt_output).abs().max().item()
        print(f"[INFO] Max difference PyTorch vs TorchScript: {max_diff:.6f}")

        if max_diff < 1e-5:
            print("[SUCCESS] TorchScript verification passed")
        else:
            print("[WARN] Large difference detected")

    except Exception as e:
        print(f"[ERROR] TorchScript export failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
