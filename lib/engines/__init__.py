"""
Training and evaluation engines for the XEC ML pipeline.

This module provides:
- regressor: run_epoch_stream for regression training/validation
- mae: run_epoch_mae, run_eval_mae for MAE training/evaluation
- inpainter: run_epoch_inpainter, run_eval_inpainter for inpainter training/evaluation
"""

from .regressor import run_epoch_stream
from .mae import run_epoch_mae, run_eval_mae
from .inpainter import (
    run_epoch_inpainter,
    run_eval_inpainter,
    compute_inpainting_loss,
    save_predictions_to_root,
)

__all__ = [
    # Regressor
    "run_epoch_stream",
    # MAE
    "run_epoch_mae",
    "run_eval_mae",
    # Inpainter
    "run_epoch_inpainter",
    "run_eval_inpainter",
    "compute_inpainting_loss",
    "save_predictions_to_root",
]
