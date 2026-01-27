"""
Model architectures for the XEC ML pipeline.

This module provides:
- regressor: XECEncoder, XECMultiHeadModel for regression tasks
- mae: XEC_MAE for masked autoencoder pretraining
- inpainter: XEC_Inpainter for dead channel recovery
- blocks: Shared model building blocks (ConvNeXtV2Block, HexNeXtBlock, etc.)
"""

from .regressor import (
    XECEncoder,
    XECMultiHeadModel,
    AutomaticLossScaler,
    DeepHexEncoder,
    FaceBackbone,
)
from .mae import XEC_MAE, FaceDecoder, GraphFaceDecoder
from .inpainter import XEC_Inpainter
from .blocks import (
    ConvNeXtV2Block,
    HexNeXtBlock,
    LayerNorm,
    GRN,
    DropPath,
)

__all__ = [
    # Regressor
    "XECEncoder",
    "XECMultiHeadModel",
    "AutomaticLossScaler",
    "DeepHexEncoder",
    "FaceBackbone",
    # MAE
    "XEC_MAE",
    "FaceDecoder",
    "GraphFaceDecoder",
    # Inpainter
    "XEC_Inpainter",
    # Blocks
    "ConvNeXtV2Block",
    "HexNeXtBlock",
    "LayerNorm",
    "GRN",
    "DropPath",
]
