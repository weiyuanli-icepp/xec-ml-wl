"""
Base class for task-specific handlers in the regressor training pipeline.

Each task handler encapsulates:
- Loss computation (validation)
- Prediction collection
- Metrics computation
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import numpy as np
import torch


class TaskHandler(ABC):
    """Base class for task-specific loss/metrics computation."""

    task_name: str = ""

    @abstractmethod
    def compute_val_loss(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        criterion_smooth: torch.nn.Module,
        criterion_l1: torch.nn.Module,
        criterion_mse: torch.nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute validation losses for this task.

        Args:
            preds: Dictionary of predictions, keyed by task name
            targets: Dictionary of targets, keyed by task name
            criterion_smooth: SmoothL1Loss criterion (reduction="none")
            criterion_l1: L1Loss criterion (reduction="none")
            criterion_mse: MSELoss criterion (reduction="none")

        Returns:
            Dictionary containing:
                - "smooth_l1": per-sample smooth L1 loss tensor
                - "l1": per-sample L1 loss tensor
                - "mse": per-sample MSE loss tensor
                - "batch_loss": scalar batch mean loss for total tracking
        """
        raise NotImplementedError

    @abstractmethod
    def collect_predictions(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        val_root_data: Dict[str, List],
    ) -> None:
        """
        Collect predictions and targets for artifact generation.

        Args:
            preds: Dictionary of predictions
            targets: Dictionary of targets
            val_root_data: Dictionary to append predictions/targets to
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(
        self,
        val_root_data: Dict[str, np.ndarray],
        loss_sums: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Compute final metrics for MLflow logging.

        Args:
            val_root_data: Dictionary of concatenated predictions/targets
            loss_sums: Dictionary of accumulated loss values

        Returns:
            Dictionary of metric names to values for MLflow logging
        """
        raise NotImplementedError

    def is_active(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> bool:
        """Check if this task is active in the current batch."""
        return self.task_name in preds and self.task_name in targets
