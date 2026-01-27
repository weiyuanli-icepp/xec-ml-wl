"""
Timing task handler for regressor training.

Handles gamma timing regression task.
"""

from typing import Dict, Any, List
import numpy as np
import torch

from .base import TaskHandler


class TimingTaskHandler(TaskHandler):
    """Handler for timing regression task."""

    task_name = "timing"

    def compute_val_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        criterion_smooth: torch.nn.Module,
        criterion_l1: torch.nn.Module,
        criterion_mse: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute validation losses for timing task."""
        p_timing = preds["timing"]
        t_timing = targets["timing"]

        # Ensure target has same shape as prediction
        if t_timing.ndim == 1:
            t_timing = t_timing.unsqueeze(-1)

        l_smooth = criterion_smooth(p_timing, t_timing).mean(dim=-1)
        l_l1 = criterion_l1(p_timing, t_timing).mean(dim=-1)
        l_mse = criterion_mse(p_timing, t_timing).mean(dim=-1)

        return {
            "smooth_l1": l_smooth,
            "l1": l_l1,
            "mse": l_mse,
            "batch_loss": l_smooth.mean(),
        }

    def collect_predictions(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        val_root_data: Dict[str, List],
    ) -> None:
        """Collect timing predictions for artifact generation."""
        p_timing = preds["timing"]
        t_timing = targets["timing"]

        val_root_data["pred_timing"].append(p_timing.squeeze(-1).cpu().numpy())
        val_root_data["true_timing"].append(t_timing.squeeze(-1).cpu().numpy())

    def compute_metrics(
        self,
        val_root_data: Dict[str, np.ndarray],
        loss_sums: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute timing-specific metrics for MLflow logging."""
        metrics = {}

        pred_timing = val_root_data.get("pred_timing", np.array([]))
        true_timing = val_root_data.get("true_timing", np.array([]))

        if pred_timing.size == 0 or true_timing.size == 0:
            return metrics

        residual = pred_timing - true_timing
        metrics["timing_bias"] = float(np.mean(residual))
        metrics["timing_res_68pct"] = float(np.percentile(np.abs(residual), 68))

        return metrics
