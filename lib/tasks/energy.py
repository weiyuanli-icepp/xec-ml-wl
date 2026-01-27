"""
Energy task handler for regressor training.

Handles gamma energy regression task.
"""

from typing import Dict, Any, List
import numpy as np
import torch

from .base import TaskHandler


class EnergyTaskHandler(TaskHandler):
    """Handler for energy regression task."""

    task_name = "energy"

    def compute_val_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        criterion_smooth: torch.nn.Module,
        criterion_l1: torch.nn.Module,
        criterion_mse: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute validation losses for energy task."""
        p_energy = preds["energy"]
        t_energy = targets["energy"]

        # Ensure target has same shape as prediction
        if t_energy.ndim == 1:
            t_energy = t_energy.unsqueeze(-1)

        l_smooth = criterion_smooth(p_energy, t_energy).mean(dim=-1)
        l_l1 = criterion_l1(p_energy, t_energy).mean(dim=-1)
        l_mse = criterion_mse(p_energy, t_energy).mean(dim=-1)

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
        """Collect energy predictions for artifact generation."""
        p_energy = preds["energy"]
        t_energy = targets["energy"]

        val_root_data["pred_energy"].append(p_energy.squeeze(-1).cpu().numpy())
        val_root_data["true_energy"].append(t_energy.squeeze(-1).cpu().numpy())

    def compute_metrics(
        self,
        val_root_data: Dict[str, np.ndarray],
        loss_sums: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute energy-specific metrics for MLflow logging."""
        metrics = {}

        pred_energy = val_root_data.get("pred_energy", np.array([]))
        true_energy = val_root_data.get("true_energy", np.array([]))

        if pred_energy.size == 0 or true_energy.size == 0:
            return metrics

        residual = pred_energy - true_energy
        metrics["energy_bias"] = float(np.mean(residual))
        metrics["energy_res_68pct"] = float(np.percentile(np.abs(residual), 68))

        return metrics
