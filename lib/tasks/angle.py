"""
Angle task handler for regressor training.

Handles emission angle (theta, phi) regression task.
"""

from typing import Dict, Any, List
import numpy as np
import torch

from .base import TaskHandler
from ..utils import angles_deg_to_unit_vec
from ..metrics import eval_stats, eval_resolution


class AngleTaskHandler(TaskHandler):
    """Handler for angle (theta, phi) regression task."""

    task_name = "angle"

    def compute_val_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        criterion_smooth: torch.nn.Module,
        criterion_l1: torch.nn.Module,
        criterion_mse: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute validation losses for angle task."""
        p_angle = preds["angle"]
        t_angle = targets["angle"]

        l_smooth = criterion_smooth(p_angle, t_angle).mean(dim=-1)
        l_l1 = criterion_l1(p_angle, t_angle).mean(dim=-1)
        l_mse = criterion_mse(p_angle, t_angle).mean(dim=-1)

        return {
            "smooth_l1": l_smooth,
            "l1": l_l1,
            "mse": l_mse,
            "batch_loss": l_smooth.mean(),
        }

    def compute_cosine_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cosine similarity loss (angle-specific).

        Args:
            preds: Dictionary of predictions
            targets: Dictionary of targets (must include "emiVec")

        Returns:
            Dictionary with "cos" loss and "opening_angle" in degrees
        """
        p_angle = preds["angle"]
        t_vec = targets.get("emiVec")

        if t_vec is None:
            return {}

        v_pred = angles_deg_to_unit_vec(p_angle)
        cos_sim = torch.sum(v_pred * t_vec, dim=1).clamp(-1.0, 1.0)
        l_cos = 1.0 - cos_sim
        opening_angle = torch.acos(cos_sim) * (180.0 / np.pi)

        return {
            "cos": l_cos,
            "opening_angle": opening_angle,
        }

    def collect_predictions(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        val_root_data: Dict[str, List],
    ) -> None:
        """Collect angle predictions for artifact generation.

        Note: opening_angle is collected separately via compute_cosine_loss()
        in the engine to avoid duplication.
        """
        p_angle = preds["angle"]
        t_angle = targets["angle"]

        val_root_data["pred_theta"].append(p_angle[:, 0].cpu().numpy())
        val_root_data["pred_phi"].append(p_angle[:, 1].cpu().numpy())
        val_root_data["true_theta"].append(t_angle[:, 0].cpu().numpy())
        val_root_data["true_phi"].append(t_angle[:, 1].cpu().numpy())

    def compute_metrics(
        self,
        val_root_data: Dict[str, np.ndarray],
        loss_sums: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute angle-specific metrics for MLflow logging."""
        metrics = {}

        pred_theta = val_root_data.get("pred_theta", np.array([]))
        pred_phi = val_root_data.get("pred_phi", np.array([]))
        true_theta = val_root_data.get("true_theta", np.array([]))
        true_phi = val_root_data.get("true_phi", np.array([]))

        if pred_theta.size == 0 or pred_phi.size == 0:
            return metrics

        angle_pred_np = np.stack([pred_theta, pred_phi], axis=1)
        angle_true_np = np.stack([true_theta, true_phi], axis=1)

        # Compute angle statistics
        angle_stats = eval_stats(angle_pred_np, angle_true_np, print_out=False)
        metrics.update(angle_stats)

        # Compute resolution
        res_68, psi_deg = eval_resolution(angle_pred_np, angle_true_np)
        metrics["angle_resolution_68pct"] = res_68

        return metrics
