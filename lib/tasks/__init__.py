"""
Task handlers for the regressor training pipeline.

Provides modular task-specific logic for:
- Angle (theta, phi) regression
- Energy regression
- Timing regression
- Position (uvwFI) regression

Usage:
    from lib.tasks import get_task_handlers

    handlers = get_task_handlers(["angle", "energy"])
    for handler in handlers:
        if handler.is_active(preds, targets):
            losses = handler.compute_val_loss(...)
            handler.collect_predictions(...)
"""

from typing import List, Optional

from .base import TaskHandler
from .angle import AngleTaskHandler
from .energy import EnergyTaskHandler
from .timing import TimingTaskHandler
from .position import PositionTaskHandler


# Registry of available task handlers
TASK_HANDLERS = {
    "angle": AngleTaskHandler,
    "energy": EnergyTaskHandler,
    "timing": TimingTaskHandler,
    "uvwFI": PositionTaskHandler,
}


def get_task_handlers(active_tasks: Optional[List[str]] = None) -> List[TaskHandler]:
    """
    Get task handler instances for the specified tasks.

    Args:
        active_tasks: List of task names to get handlers for.
                     If None, returns handlers for all tasks.

    Returns:
        List of TaskHandler instances

    Raises:
        ValueError: If an unknown task name is provided
    """
    if active_tasks is None:
        active_tasks = list(TASK_HANDLERS.keys())

    handlers = []
    for task_name in active_tasks:
        if task_name not in TASK_HANDLERS:
            raise ValueError(
                f"Unknown task: {task_name}. "
                f"Available tasks: {list(TASK_HANDLERS.keys())}"
            )
        handlers.append(TASK_HANDLERS[task_name]())

    return handlers


def get_task_handler(task_name: str) -> TaskHandler:
    """
    Get a single task handler instance.

    Args:
        task_name: Name of the task

    Returns:
        TaskHandler instance

    Raises:
        ValueError: If an unknown task name is provided
    """
    if task_name not in TASK_HANDLERS:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_HANDLERS.keys())}"
        )
    return TASK_HANDLERS[task_name]()


__all__ = [
    "TaskHandler",
    "AngleTaskHandler",
    "EnergyTaskHandler",
    "TimingTaskHandler",
    "PositionTaskHandler",
    "get_task_handlers",
    "get_task_handler",
    "TASK_HANDLERS",
]
