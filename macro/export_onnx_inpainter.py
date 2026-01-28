#!/usr/bin/env python3
"""
Export XEC_Inpainter model to TorchScript format.

NOTE: ONNX export is NOT supported because the model uses nn.TransformerEncoder,
which has a native operator (aten::_transformer_encoder_layer_fwd) that cannot
be exported to ONNX. Use TorchScript instead.

Usage:
    python macro/export_onnx_inpainter.py artifacts/<RUN>/inpainter_checkpoint_best.pth --output inpainter.pt

The exported model takes:
    - input: (B, 4760, 2) - sensor values (npho, time) with dead channels as sentinel
    - mask: (B, 4760) - binary mask (1 = dead/masked, 0 = valid)

Returns:
    - output: (B, 4760, 2) - reconstructed values (only masked positions are meaningful)

For C++ inference with libtorch:
    auto model = torch::jit::load("inpainter.pt");
    auto output = model.forward({input, mask}).toTensor();
    // Apply output only at masked positions
"""

import os
import sys
import argparse
import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lib.models import XECEncoder
from lib.models.inpainter import XEC_Inpainter
from lib.geom_defs import (
    INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
    OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
    flatten_hex_rows, OUTER_ALL_SENSOR_IDS, OUTER_SENSOR_ID_TO_IDX
)


class InpainterScriptableWrapper(nn.Module):
    """
    TorchScript-compatible wrapper for XEC_Inpainter.

    Uses vectorized scatter operations that work with torch.jit.script.
    Converts the complex dict output to a simple (B, 4760, 2) tensor.
    """

    def __init__(self, inpainter: XEC_Inpainter):
        super().__init__()
        self.inpainter = inpainter

        # Pre-register face index maps as buffers
        self.register_buffer("inner_idx_flat", torch.from_numpy(INNER_INDEX_MAP).long().flatten())
        self.register_buffer("us_idx_flat", torch.from_numpy(US_INDEX_MAP).long().flatten())
        self.register_buffer("ds_idx_flat", torch.from_numpy(DS_INDEX_MAP).long().flatten())
        self.register_buffer("outer_coarse_idx_flat", torch.from_numpy(OUTER_COARSE_FULL_INDEX_MAP).long().flatten())
        self.register_buffer("top_hex_idx", torch.from_numpy(flatten_hex_rows(TOP_HEX_ROWS)).long())
        self.register_buffer("bot_hex_idx", torch.from_numpy(flatten_hex_rows(BOTTOM_HEX_ROWS)).long())
        self.register_buffer("outer_sensor_ids_buf", torch.from_numpy(OUTER_ALL_SENSOR_IDS).long())

        # Store face dimensions as tensors for scriptability
        self.inner_W = 44
        self.us_W = 6
        self.ds_W = 6
        self.outer_W = 24

    def forward(self, x_input: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_input: (B, 4760, 2) - sensor values with dead channels as sentinel
            mask: (B, 4760) - binary mask (1 = masked/dead, 0 = valid)

        Returns:
            output: (B, 4760, 2) - full tensor with predictions at masked positions
        """
        # Run inpainter
        results, original_values, _ = self.inpainter(x_input, mask=mask)

        # Start with original input
        output = x_input.clone()

        # Process each face with vectorized scatter
        # Inner face
        if "inner" in results:
            output = self._scatter_rect_face_vectorized(
                output, results["inner"]["pred"], results["inner"]["indices"],
                results["inner"]["valid"], self.inner_idx_flat, self.inner_W
            )

        # US face
        if "us" in results:
            output = self._scatter_rect_face_vectorized(
                output, results["us"]["pred"], results["us"]["indices"],
                results["us"]["valid"], self.us_idx_flat, self.us_W
            )

        # DS face
        if "ds" in results:
            output = self._scatter_rect_face_vectorized(
                output, results["ds"]["pred"], results["ds"]["indices"],
                results["ds"]["valid"], self.ds_idx_flat, self.ds_W
            )

        # Outer face
        if "outer" in results:
            outer_result = results["outer"]
            if outer_result.get("is_sensor_level", False):
                output = self._scatter_sensor_level_vectorized(
                    output, outer_result["pred"], outer_result["sensor_ids"],
                    outer_result["valid"]
                )
            else:
                output = self._scatter_rect_face_vectorized(
                    output, outer_result["pred"], outer_result["indices"],
                    outer_result["valid"], self.outer_coarse_idx_flat, self.outer_W
                )

        # Top hex face
        if "top" in results:
            output = self._scatter_hex_face_vectorized(
                output, results["top"]["pred"], results["top"]["indices"],
                results["top"]["valid"], self.top_hex_idx
            )

        # Bottom hex face
        if "bot" in results:
            output = self._scatter_hex_face_vectorized(
                output, results["bot"]["pred"], results["bot"]["indices"],
                results["bot"]["valid"], self.bot_hex_idx
            )

        return output

    def _scatter_rect_face_vectorized(
        self,
        output: torch.Tensor,
        pred: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        idx_flat: torch.Tensor,
        W: int
    ) -> torch.Tensor:
        """Vectorized scatter for rectangular faces."""
        B = output.shape[0]
        max_masked = pred.shape[1]

        if max_masked == 0:
            return output

        # Convert (h, w) indices to sensor IDs
        h_idx = indices[:, :, 0]  # (B, max_masked)
        w_idx = indices[:, :, 1]  # (B, max_masked)
        flat_pos = h_idx * W + w_idx  # (B, max_masked)

        # Clamp and gather sensor IDs
        flat_pos_clamped = flat_pos.clamp(0, idx_flat.shape[0] - 1)
        sensor_ids = idx_flat[flat_pos_clamped]  # (B, max_masked)

        # Valid mask: valid flag AND valid sensor ID
        valid_mask = valid & (sensor_ids >= 0) & (sensor_ids < 4760)

        # Vectorized scatter using index_put_ with flattened indices
        # Flatten batch and position dimensions
        batch_indices = torch.arange(B, device=output.device).unsqueeze(1).expand(B, max_masked)

        # Get flat indices where valid
        valid_flat = valid_mask.flatten()  # (B * max_masked,)
        batch_flat = batch_indices.flatten()[valid_flat]  # (N_valid,)
        sensor_flat = sensor_ids.flatten()[valid_flat]  # (N_valid,)
        pred_flat = pred.reshape(-1, 2)[valid_flat]  # (N_valid, 2)

        # Scatter predictions
        if pred_flat.shape[0] > 0:
            output[batch_flat, sensor_flat] = pred_flat

        return output

    def _scatter_sensor_level_vectorized(
        self,
        output: torch.Tensor,
        pred: torch.Tensor,
        sensor_ids: torch.Tensor,
        valid: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized scatter for sensor-level predictions (outer finegrid)."""
        B = output.shape[0]
        max_masked = pred.shape[1]

        if max_masked == 0:
            return output

        # Valid mask
        valid_mask = valid & (sensor_ids >= 0) & (sensor_ids < 4760)

        # Flatten and gather valid entries
        batch_indices = torch.arange(B, device=output.device).unsqueeze(1).expand(B, max_masked)

        valid_flat = valid_mask.flatten()
        batch_flat = batch_indices.flatten()[valid_flat]
        sensor_flat = sensor_ids.flatten()[valid_flat]
        pred_flat = pred.reshape(-1, 2)[valid_flat]

        # Scatter
        if pred_flat.shape[0] > 0:
            output[batch_flat, sensor_flat] = pred_flat

        return output

    def _scatter_hex_face_vectorized(
        self,
        output: torch.Tensor,
        pred: torch.Tensor,
        indices: torch.Tensor,
        valid: torch.Tensor,
        hex_indices: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized scatter for hex faces."""
        B = output.shape[0]
        max_masked = pred.shape[1]

        if max_masked == 0:
            return output

        # Map node indices to sensor IDs
        indices_clamped = indices.clamp(0, hex_indices.shape[0] - 1)
        sensor_ids = hex_indices[indices_clamped]

        # Valid mask
        valid_mask = valid & (sensor_ids >= 0)

        # Flatten and gather valid entries
        batch_indices = torch.arange(B, device=output.device).unsqueeze(1).expand(B, max_masked)

        valid_flat = valid_mask.flatten()
        batch_flat = batch_indices.flatten()[valid_flat]
        sensor_flat = sensor_ids.flatten()[valid_flat]
        pred_flat = pred.reshape(-1, 2)[valid_flat]

        # Scatter
        if pred_flat.shape[0] > 0:
            output[batch_flat, sensor_flat] = pred_flat

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
        description="Export XEC_Inpainter to TorchScript (ONNX not supported due to TransformerEncoder)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export to TorchScript (default and recommended)
  python macro/export_onnx_inpainter.py artifacts/inpainter_run/inpainter_checkpoint_best.pth --output inpainter.pt

  # Without EMA weights
  python macro/export_onnx_inpainter.py checkpoint.pth --no-ema --output inpainter.pt

NOTE: ONNX export is NOT supported because the model uses nn.TransformerEncoder,
which cannot be exported to ONNX. Use TorchScript instead.
        """
    )
    parser.add_argument("checkpoint", type=str, help="Path to inpainter checkpoint (.pth)")
    parser.add_argument("--output", type=str, default=None, help="Output file (default: inpainter.pt)")
    parser.add_argument("--format", type=str, choices=["onnx", "torchscript"], default="torchscript",
                        help="Export format (default: torchscript). NOTE: ONNX will fail due to TransformerEncoder.")
    parser.add_argument("--no-ema", action="store_true", help="Use standard weights instead of EMA")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (if attempting ONNX)")

    args = parser.parse_args()

    # Set default output filename based on format
    if args.output is None:
        args.output = "inpainter.pt" if args.format == "torchscript" else "inpainter.onnx"

    # Load model
    inpainter, config = load_inpainter_checkpoint(args.checkpoint, prefer_ema=not args.no_ema)
    inpainter.eval()

    # Create wrapper
    wrapper = InpainterONNXWrapper(inpainter)
    wrapper.eval()

    # Dummy inputs
    dummy_input = torch.randn(1, 4760, 2)
    dummy_mask = torch.zeros(1, 4760)
    # Mask some random sensors
    dummy_mask[0, :100] = 1.0

    # Test forward pass
    print("[INFO] Testing forward pass...")
    with torch.no_grad():
        test_output = wrapper(dummy_input, dummy_mask)
    print(f"[INFO] Output shape: {test_output.shape}")

    # Create output directory
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.format == "torchscript":
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

    else:
        # Export to ONNX
        print(f"[INFO] Exporting to ONNX: {args.output} (opset {args.opset})")
        print("[ERROR] ONNX export is NOT supported for this model!")
        print("[ERROR] The model uses nn.TransformerEncoder which has operator")
        print("[ERROR] 'aten::_transformer_encoder_layer_fwd' that cannot be exported to ONNX.")
        print("[ERROR] Please use --format torchscript instead.")
        print("")
        print("[INFO] Attempting export anyway (will likely fail)...")

        try:
            torch.onnx.export(
                wrapper,
                (dummy_input, dummy_mask),
                args.output,
                export_params=True,
                opset_version=args.opset,
                do_constant_folding=True,
                input_names=["input", "mask"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "mask": {0: "batch_size"},
                    "output": {0: "batch_size"},
                }
            )
            print(f"[SUCCESS] Model exported to {args.output}")

            # Print model info
            file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
            print(f"[INFO] File size: {file_size_mb:.1f} MB")

        except Exception as e:
            print(f"[ERROR] ONNX export failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Verify with ONNX runtime if available
        try:
            import onnxruntime as ort
            print("[INFO] Verifying with ONNX Runtime...")

            sess = ort.InferenceSession(args.output)

            # Run inference
            ort_inputs = {
                "input": dummy_input.numpy(),
                "mask": dummy_mask.numpy()
            }
            ort_output = sess.run(None, ort_inputs)[0]

            # Compare with PyTorch
            with torch.no_grad():
                pt_output = wrapper(dummy_input, dummy_mask).numpy()

            max_diff = abs(ort_output - pt_output).max()
            print(f"[INFO] Max difference PyTorch vs ONNX: {max_diff:.6f}")

            if max_diff < 1e-4:
                print("[SUCCESS] ONNX verification passed")
            else:
                print("[WARN] Large difference detected, but export completed")

        except ImportError:
            print("[INFO] onnxruntime not installed, skipping verification")
        except Exception as e:
            print(f"[WARN] ONNX verification failed: {e}")


if __name__ == "__main__":
    main()
