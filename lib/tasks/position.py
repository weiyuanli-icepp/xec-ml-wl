"""
Position task handler for regressor training.

Handles gamma first interaction position (uvwFI) regression task.
"""

from typing import Dict, Any, List
import numpy as np
import torch

from .base import TaskHandler


class PositionTaskHandler(TaskHandler):
    """Handler for position (uvwFI) regression task."""

    task_name = "uvwFI"

    def compute_val_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        criterion_smooth: torch.nn.Module,
        criterion_l1: torch.nn.Module,
        criterion_mse: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """Compute validation losses for position task."""
        p_uvw = preds["uvwFI"]
        t_uvw = targets["uvwFI"]

        l_smooth = criterion_smooth(p_uvw, t_uvw).mean(dim=-1)
        l_l1 = criterion_l1(p_uvw, t_uvw).mean(dim=-1)
        l_mse = criterion_mse(p_uvw, t_uvw).mean(dim=-1)

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
        """Collect position predictions for artifact generation."""
        p_uvw = preds["uvwFI"]
        t_uvw = targets["uvwFI"]

        val_root_data["pred_u"].append(p_uvw[:, 0].cpu().numpy())
        val_root_data["pred_v"].append(p_uvw[:, 1].cpu().numpy())
        val_root_data["pred_w"].append(p_uvw[:, 2].cpu().numpy())
        val_root_data["true_u"].append(t_uvw[:, 0].cpu().numpy())
        val_root_data["true_v"].append(t_uvw[:, 1].cpu().numpy())
        val_root_data["true_w"].append(t_uvw[:, 2].cpu().numpy())

    def collect_residuals(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        loss_sums: Dict[str, Any],
    ) -> None:
        """Collect per-axis residuals for resolution computation."""
        p_uvw = preds["uvwFI"]
        t_uvw = targets["uvwFI"]

        residual = p_uvw - t_uvw

        # Initialize residual lists if not present
        if "uvw_u_res" not in loss_sums:
            loss_sums["uvw_u_res"] = []
            loss_sums["uvw_v_res"] = []
            loss_sums["uvw_w_res"] = []
            loss_sums["uvw_dist"] = []

        loss_sums["uvw_u_res"].append(residual[:, 0].cpu().numpy())
        loss_sums["uvw_v_res"].append(residual[:, 1].cpu().numpy())
        loss_sums["uvw_w_res"].append(residual[:, 2].cpu().numpy())

        dist = torch.norm(residual, dim=1)
        loss_sums["uvw_dist"].append(dist.cpu().numpy())

    def compute_cosine_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute cosine similarity loss for position vectors.

        Cosine similarity measures whether the predicted position vector
        points in the same direction from origin as the true position.
        cos_loss = 1 - cos_sim, where cos_sim = (pred · true) / (|pred| * |true|)

        Returns:
            Dict with 'cos_pos' (cosine loss) and 'pos_angle' (angle in degrees)
        """
        p_uvw = preds["uvwFI"]
        t_uvw = targets["uvwFI"]

        # Compute norms
        p_norm = torch.norm(p_uvw, dim=1, keepdim=True).clamp(min=1e-8)
        t_norm = torch.norm(t_uvw, dim=1, keepdim=True).clamp(min=1e-8)

        # Normalize to unit vectors
        p_unit = p_uvw / p_norm
        t_unit = t_uvw / t_norm

        # Cosine similarity: dot product of unit vectors
        cos_sim = torch.sum(p_unit * t_unit, dim=1).clamp(-1.0, 1.0)

        # Cosine loss: 1 - cos_sim (0 = perfect alignment, 2 = opposite directions)
        cos_loss = 1.0 - cos_sim

        # Angle between vectors in degrees
        pos_angle = torch.acos(cos_sim) * (180.0 / np.pi)

        return {
            "cos_pos": cos_loss,
            "pos_angle": pos_angle,
        }

    def compute_metrics(
        self,
        val_root_data: Dict[str, np.ndarray],
        loss_sums: Dict[str, Any],
    ) -> Dict[str, float]:
        """Compute position-specific metrics for MLflow logging."""
        metrics = {}

        # Check for per-axis residuals in loss_sums
        if "uvw_u_res" in loss_sums and loss_sums["uvw_u_res"]:
            u_res = np.concatenate(loss_sums["uvw_u_res"])
            v_res = np.concatenate(loss_sums["uvw_v_res"])
            w_res = np.concatenate(loss_sums["uvw_w_res"])
            dist = np.concatenate(loss_sums["uvw_dist"])

            metrics["uvw_u_res_68pct"] = float(np.percentile(np.abs(u_res), 68))
            metrics["uvw_v_res_68pct"] = float(np.percentile(np.abs(v_res), 68))
            metrics["uvw_w_res_68pct"] = float(np.percentile(np.abs(w_res), 68))
            metrics["uvw_dist_68pct"] = float(np.percentile(dist, 68))
            metrics["uvw_u_bias"] = float(np.mean(u_res))
            metrics["uvw_v_bias"] = float(np.mean(v_res))
            metrics["uvw_w_bias"] = float(np.mean(w_res))
            metrics["uvw_dist_bias"] = float(np.mean(dist))

        return metrics
