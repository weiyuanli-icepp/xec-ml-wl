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
    - x_raw: (B, 4760, 2) - RAW sensor values [npho_raw, time_raw]
    - mask:  (B, 4760)    - binary mask (1 = dead/masked, 0 = valid)

Returns:
    - npho_out: (B, 4760) - raw npho with inpainted values at masked positions,
                original values preserved at unmasked positions

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

    Handles ALL normalization, sentinel detection, and denormalization internally
    so callers (C++, Python) just pass raw values. Mirrors the training dataset
    logic in lib/dataset.py:195-254.

    Input:  x_raw (B, 4760, 2) with [raw_npho, relative_time]
            mask  (B, 4760)    binary dead-channel mask (1=dead/bad, 0=valid)
    Output: npho_out (B, 4760) raw npho with inpainted values at masked positions
    """

    def __init__(self, inpainter: XEC_Inpainter, config: dict):
        super().__init__()
        self.inpainter = inpainter
        self.out_channels = inpainter.out_channels

        # Normalization constants from config
        self.npho_scheme: str = config.get("npho_scheme", "log1p")
        self.npho_scale: float = config.get("npho_scale", 1000.0)
        self.npho_scale2: float = config.get("npho_scale2", 4.08)
        self.time_scale: float = config.get("time_scale", 1.14e-7)
        self.time_shift: float = config.get("time_shift", -0.46)
        self.sentinel_npho: float = config.get("sentinel_npho", -1.0)
        self.sentinel_time: float = config.get("sentinel_time", -1.0)
        self.npho_threshold: float = config.get("npho_threshold", 100.0)

        # Pre-compute scheme-dependent constants (safe in __init__, not in forward)
        import math
        if self.npho_scheme == "log1p":
            self.domain_min: float = -self.npho_scale * 0.999
        elif self.npho_scheme == "sqrt":
            self.domain_min = 0.0
        elif self.npho_scheme == "anscombe":
            self.domain_min = -0.375
        else:
            self.domain_min = float('-inf')
        self.sqrt_scale: float = math.sqrt(self.npho_scale)
        self.anscombe_sf: float = 2.0 * math.sqrt(self.npho_scale + 0.375)

    def _npho_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.npho_scheme == "log1p":
            return torch.log1p(x / self.npho_scale) / self.npho_scale2
        elif self.npho_scheme == "sqrt":
            return torch.sqrt(x) / self.sqrt_scale
        elif self.npho_scheme == "anscombe":
            return 2.0 * torch.sqrt(x + 0.375) / self.anscombe_sf
        else:
            return x / self.npho_scale

    def _npho_inverse(self, y: torch.Tensor) -> torch.Tensor:
        if self.npho_scheme == "log1p":
            return self.npho_scale * (torch.exp(y * self.npho_scale2) - 1.0)
        elif self.npho_scheme == "sqrt":
            return (y * self.sqrt_scale) ** 2
        elif self.npho_scheme == "anscombe":
            return (y * self.anscombe_sf / 2.0) ** 2 - 0.375
        else:
            return y * self.npho_scale

    def forward(self, x_raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_raw: (B, 4760, 2) - RAW sensor values [raw_npho, relative_time]
            mask:  (B, 4760)    - binary dead-channel mask (1=dead/bad, 0=valid)
        Returns:
            npho_out: (B, 4760) - raw npho with inpainted values at masked positions
        """
        raw_n = x_raw[:, :, 0]
        raw_t = x_raw[:, :, 1]
        mask_bool = mask.bool()

        # Build masks (mirrors dataset.py:214-243)
        npho_invalid = mask_bool | (raw_n > 9e9) | torch.isnan(raw_n) | torch.isinf(raw_n)
        domain_break = (~npho_invalid) & (raw_n < self.domain_min)
        time_invalid = (npho_invalid | (raw_n < self.npho_threshold)
                        | (raw_t.abs() > 9e9) | torch.isnan(raw_t) | torch.isinf(raw_t))

        # Npho normalization (configurable scheme)
        raw_n_safe = torch.where(npho_invalid | domain_break, torch.zeros_like(raw_n), raw_n)
        n_norm = self._npho_forward(raw_n_safe)
        n_norm = torch.where(npho_invalid, torch.full_like(n_norm, self.sentinel_npho), n_norm)
        n_norm = torch.where(domain_break, torch.zeros_like(n_norm), n_norm)

        # Time normalization (linear: raw/scale - shift, matching dataset.py:253)
        t_norm = raw_t / self.time_scale - self.time_shift
        t_norm = torch.where(time_invalid, torch.full_like(t_norm, self.sentinel_time), t_norm)

        x_norm = torch.stack([n_norm, t_norm], dim=-1)
        pred_all = self.inpainter.forward_full_output(x_norm, mask)

        # Denormalize npho output
        npho_pred_raw = self._npho_inverse(pred_all[:, :, 0])

        # Combine: predictions at masked, original at unmasked
        return torch.where(mask_bool, npho_pred_raw, raw_n)


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
    sentinel_time = config.get("sentinel_time", config.get("sentinel_value", -1.0))
    sentinel_npho = config.get("sentinel_npho", config.get("npho_sentinel_value", -1.0))
    time_mask_ratio_scale = config.get("time_mask_ratio_scale", 1.0)
    predict_channels = config.get("predict_channels", ["npho", "time"])
    use_masked_attention = config.get("use_masked_attention", False)
    head_type = config.get("head_type", "per_face")
    sensor_positions_file = config.get("sensor_positions_file", None)
    cross_attn_k = config.get("cross_attn_k", 16)
    cross_attn_hidden = config.get("cross_attn_hidden", 64)
    cross_attn_latent_dim = config.get("cross_attn_latent_dim", 128)
    cross_attn_pos_dim = config.get("cross_attn_pos_dim", 96)

    print(f"[INFO] Model config: outer_mode={outer_mode}, outer_fine_pool={outer_fine_pool}")
    print(f"[INFO]   predict_channels={predict_channels}, head_type={head_type}")
    if head_type == "cross_attention":
        print(f"[INFO]   sensor_positions_file={sensor_positions_file}")
    else:
        print(f"[INFO]   use_masked_attention={use_masked_attention}")

    # Create encoder
    encoder = XECEncoder(
        outer_mode=outer_mode,
        outer_fine_pool=outer_fine_pool if isinstance(outer_fine_pool, tuple) else tuple(outer_fine_pool),
        drop_path_rate=0.0,  # Always 0 for inference
        encoder_dim=config.get('encoder_dim', 1024),
        dim_feedforward=config.get('dim_feedforward', None),
        num_fusion_layers=config.get('num_fusion_layers', 2),
        sentinel_time=sentinel_time,
    )

    # Create inpainter
    inpainter = XEC_Inpainter(
        encoder=encoder,
        freeze_encoder=True,
        sentinel_time=sentinel_time,
        time_mask_ratio_scale=time_mask_ratio_scale,
        predict_channels=predict_channels,
        use_masked_attention=use_masked_attention,
        head_type=head_type,
        sensor_positions_file=sensor_positions_file,
        cross_attn_k=cross_attn_k,
        cross_attn_hidden=cross_attn_hidden,
        cross_attn_latent_dim=cross_attn_latent_dim,
        cross_attn_pos_dim=cross_attn_pos_dim,
        sentinel_npho=sentinel_npho,
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

    # Create wrapper (with config for baked normalization constants)
    wrapper = InpainterONNXWrapper(inpainter, config)
    wrapper.eval()

    print(f"[INFO] Wrapper config: npho_scheme={wrapper.npho_scheme}, "
          f"npho_scale={wrapper.npho_scale}, npho_scale2={wrapper.npho_scale2}")
    print(f"[INFO]   time_scale={wrapper.time_scale}, time_shift={wrapper.time_shift}")
    print(f"[INFO]   sentinel_npho={wrapper.sentinel_npho}, sentinel_time={wrapper.sentinel_time}")
    print(f"[INFO]   npho_threshold={wrapper.npho_threshold}, domain_min={wrapper.domain_min}")

    # Dummy inputs for tracing (raw scale values, not normalized)
    trace_batch = args.trace_batch_size
    trace_mask_per_event = args.trace_mask_per_event

    out_channels = inpainter.out_channels
    print(f"[INFO] Tracing with batch_size={trace_batch}, ~{trace_mask_per_event} masked per event")
    print(f"[INFO] Output channels: {out_channels} ({inpainter.predict_channels})")
    print(f"[INFO] Using fixed-size output method (forward_full_output) for clean tracing")

    dummy_input = torch.zeros(trace_batch, 4760, 2)
    dummy_input[:, :, 0] = torch.rand(trace_batch, 4760) * 2000   # raw npho [0, 2000]
    dummy_input[:, :, 1] = torch.rand(trace_batch, 4760) * 1e-7   # relative time [0, 100ns]
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
