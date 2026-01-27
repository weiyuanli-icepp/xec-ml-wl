"""
Configuration management for XEC training.
Loads YAML config files and provides defaults.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    enabled: bool = False
    loss_fn: str = "smooth_l1"  # "smooth_l1", "l1", "mse", "huber"
    loss_beta: float = 1.0      # For smooth_l1/huber
    weight: float = 1.0         # Manual loss weight


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: str = ""
    val_path: str = ""
    tree_name: str = "tree"
    batch_size: int = 256
    chunksize: int = 256000
    num_workers: int = 8
    num_threads: int = 4
    npho_branch: str = "relative_npho"  # Input branch for photon counts
    time_branch: str = "relative_time"  # Input branch for timing


@dataclass
class NormalizationConfig:
    """Input normalization parameters."""
    npho_scale: float = 0.58
    npho_scale2: float = 1.0
    time_scale: float = 6.5e-8  # Fixed: was 6.5e8
    time_shift: float = 0.5
    sentinel_value: float = -5.0


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    outer_mode: str = "finegrid"  # "finegrid" or "split"
    outer_fine_pool: Optional[List[int]] = None  # [h, w] or None
    hidden_dim: int = 256
    drop_path_rate: float = 0.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    use_scheduler: bool = True
    amp: bool = True
    ema_decay: float = 0.999
    channel_dropout_rate: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    profile: bool = False  # Enable training profiler to identify bottlenecks


@dataclass
class TaskReweightConfig:
    """Reweighting config for a single task."""
    enabled: bool = False
    nbins: int = 30
    nbins_2d: List[int] = field(default_factory=lambda: [20, 20])


@dataclass
class ReweightingConfig:
    """Sample reweighting configuration."""
    angle: TaskReweightConfig = field(default_factory=TaskReweightConfig)
    energy: TaskReweightConfig = field(default_factory=TaskReweightConfig)
    timing: TaskReweightConfig = field(default_factory=TaskReweightConfig)
    uvwFI: TaskReweightConfig = field(default_factory=TaskReweightConfig)


@dataclass
class CheckpointConfig:
    """Checkpoint and saving configuration."""
    resume_from: Optional[str] = None
    save_dir: str = "artifacts"


@dataclass
class MLflowConfig:
    """MLflow tracking configuration."""
    experiment: str = "gamma_angle"
    run_name: Optional[str] = None


@dataclass
class ExportConfig:
    """Model export configuration."""
    onnx: Optional[str] = "meg2ang_convnextv2.onnx"


@dataclass
class XECConfig:
    """Complete training configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tasks: Dict[str, TaskConfig] = field(default_factory=dict)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    loss_balance: str = "manual"  # "manual" or "auto"
    reweighting: ReweightingConfig = field(default_factory=ReweightingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    export: ExportConfig = field(default_factory=ExportConfig)


def load_config(config_path: str) -> XECConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        XECConfig object with all settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = XECConfig()

    # Data
    if 'data' in raw_config:
        for k, v in raw_config['data'].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)

    # Normalization
    if 'normalization' in raw_config:
        for k, v in raw_config['normalization'].items():
            if hasattr(config.normalization, k):
                setattr(config.normalization, k, v)

    # Model
    if 'model' in raw_config:
        for k, v in raw_config['model'].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)

    # Tasks
    if 'tasks' in raw_config:
        for task_name, task_cfg in raw_config['tasks'].items():
            tc = TaskConfig()
            if isinstance(task_cfg, dict):
                for k, v in task_cfg.items():
                    if hasattr(tc, k):
                        setattr(tc, k, v)
            config.tasks[task_name] = tc

    # Training
    if 'training' in raw_config:
        for k, v in raw_config['training'].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)

    # Loss balance
    if 'loss_balance' in raw_config:
        config.loss_balance = raw_config['loss_balance']

    # Reweighting (nested structure)
    if 'reweighting' in raw_config:
        for task_name in ['angle', 'energy', 'timing', 'uvwFI']:
            if task_name in raw_config['reweighting']:
                task_cfg = raw_config['reweighting'][task_name]
                task_rw = getattr(config.reweighting, task_name)
                if isinstance(task_cfg, dict):
                    for k, v in task_cfg.items():
                        if hasattr(task_rw, k):
                            setattr(task_rw, k, v)

    # Checkpoint
    if 'checkpoint' in raw_config:
        for k, v in raw_config['checkpoint'].items():
            if hasattr(config.checkpoint, k):
                setattr(config.checkpoint, k, v)

    # MLflow
    if 'mlflow' in raw_config:
        for k, v in raw_config['mlflow'].items():
            if hasattr(config.mlflow, k):
                setattr(config.mlflow, k, v)

    # Export
    if 'export' in raw_config:
        for k, v in raw_config['export'].items():
            if hasattr(config.export, k):
                setattr(config.export, k, v)

    return config


def get_active_tasks(config: XECConfig) -> List[str]:
    """Get list of enabled task names."""
    active = [name for name, tc in config.tasks.items() if tc.enabled]
    if not active:
        import warnings
        warnings.warn(
            "No tasks are enabled in the configuration. "
            "Enable at least one task (angle, energy, timing, uvwFI) in the config file.",
            UserWarning
        )
    return active


def get_task_weights(config: XECConfig) -> Dict[str, Dict[str, Any]]:
    """
    Convert task configs to task_weights dict for engine.py.

    Returns:
        Dict like: {"angle": {"loss_fn": "smooth_l1", "weight": 1.0}, ...}
    """
    weights = {}
    for name, tc in config.tasks.items():
        if tc.enabled:
            weights[name] = {
                "loss_fn": tc.loss_fn,
                "loss_beta": tc.loss_beta,
                "weight": tc.weight
            }
    return weights


def save_config(config: XECConfig, save_path: str):
    """Save configuration to YAML file."""
    def dataclass_to_dict(obj):
        if hasattr(obj, '__dataclass_fields__'):
            return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: dataclass_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [dataclass_to_dict(v) for v in obj]
        else:
            return obj

    config_dict = dataclass_to_dict(config)

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)


def create_default_config(save_path: str = None) -> XECConfig:
    """
    Create a default configuration with common settings.
    Optionally save to file.
    """
    config = XECConfig()

    # Set up default tasks
    config.tasks = {
        "angle": TaskConfig(enabled=True, loss_fn="smooth_l1", weight=1.0),
        "energy": TaskConfig(enabled=False, loss_fn="l1", weight=1.0),
        "timing": TaskConfig(enabled=False, loss_fn="l1", weight=1.0),
        "uvwFI": TaskConfig(enabled=False, loss_fn="mse", weight=1.0),
    }

    if save_path:
        save_config(config, save_path)
        print(f"[INFO] Default config saved to: {save_path}")

    return config


# ------------------------------------------------------------
#  MAE (Masked Autoencoder) Configuration
# ------------------------------------------------------------
@dataclass
class MAEDataConfig:
    """Data configuration for MAE pre-training."""
    train_path: str = ""
    val_path: str = ""
    tree_name: str = "tree"
    batch_size: int = 1024
    chunksize: int = 256000
    num_workers: int = 4
    num_threads: int = 4
    npho_branch: str = "relative_npho"  # Input branch for photon counts
    time_branch: str = "relative_time"  # Input branch for timing


@dataclass
class MAEModelConfig:
    """Model configuration for MAE."""
    outer_mode: str = "finegrid"
    outer_fine_pool: Optional[List[int]] = None
    mask_ratio: float = 0.6
    time_mask_ratio_scale: float = 1.0  # Scale factor for masking valid-time sensors (1.0 = uniform)


@dataclass
class MAETrainingConfig:
    """Training configuration for MAE."""
    epochs: int = 20
    lr: float = 1e-4
    lr_scheduler: Optional[str] = None  # "cosine" or None
    lr_min: float = 1e-6
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    loss_fn: str = "smooth_l1"  # smooth_l1, mse, l1, huber
    npho_weight: float = 1.0
    time_weight: float = 1.0
    auto_channel_weight: bool = False
    channel_dropout_rate: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1  # Gradient accumulation steps
    ema_decay: Optional[float] = None  # None = disabled, 0.999 = typical value
    amp: bool = True
    # Conditional time loss: only compute where npho > threshold
    npho_threshold: Optional[float] = None  # None uses DEFAULT_NPHO_THRESHOLD (10.0)
    use_npho_time_weight: bool = True  # Weight time loss by sqrt(npho)


@dataclass
class MAECheckpointConfig:
    """Checkpoint configuration for MAE."""
    resume_from: Optional[str] = None
    save_dir: str = "artifacts"
    save_interval: int = 10  # Save checkpoint every N epochs
    save_predictions: bool = True  # Save ROOT file with sensor predictions


@dataclass
class MAEMLflowConfig:
    """MLflow configuration for MAE."""
    experiment: str = "mae_pretraining"
    run_name: Optional[str] = None


@dataclass
class MAEConfig:
    """Complete MAE pre-training configuration."""
    data: MAEDataConfig = field(default_factory=MAEDataConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    model: MAEModelConfig = field(default_factory=MAEModelConfig)
    training: MAETrainingConfig = field(default_factory=MAETrainingConfig)
    checkpoint: MAECheckpointConfig = field(default_factory=MAECheckpointConfig)
    mlflow: MAEMLflowConfig = field(default_factory=MAEMLflowConfig)


def load_mae_config(config_path: str) -> MAEConfig:
    """
    Load MAE configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        MAEConfig object with all settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = MAEConfig()

    # Data
    if 'data' in raw_config:
        for k, v in raw_config['data'].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)

    # Normalization
    if 'normalization' in raw_config:
        for k, v in raw_config['normalization'].items():
            if hasattr(config.normalization, k):
                setattr(config.normalization, k, v)

    # Model
    if 'model' in raw_config:
        for k, v in raw_config['model'].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)

    # Training
    if 'training' in raw_config:
        for k, v in raw_config['training'].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)

    # Checkpoint
    if 'checkpoint' in raw_config:
        for k, v in raw_config['checkpoint'].items():
            if hasattr(config.checkpoint, k):
                setattr(config.checkpoint, k, v)

    # MLflow
    if 'mlflow' in raw_config:
        for k, v in raw_config['mlflow'].items():
            if hasattr(config.mlflow, k):
                setattr(config.mlflow, k, v)

    return config


# ------------------------------------------------------------
#  Inpainter (Dead Channel Recovery) Configuration
# ------------------------------------------------------------
@dataclass
class InpainterDataConfig:
    """Data configuration for inpainter training."""
    train_path: str = ""
    val_path: str = ""
    tree_name: str = "tree"
    batch_size: int = 1024
    chunksize: int = 256000
    num_workers: int = 4
    num_threads: int = 4
    npho_branch: str = "relative_npho"  # Input branch for photon counts
    time_branch: str = "relative_time"  # Input branch for timing


@dataclass
class InpainterModelConfig:
    """Model configuration for inpainter."""
    outer_mode: str = "finegrid"
    outer_fine_pool: Optional[List[int]] = None
    mask_ratio: float = 0.05  # Default 5% for realistic dead channel density
    time_mask_ratio_scale: float = 1.0  # Scale factor for masking valid-time sensors (1.0 = uniform)
    freeze_encoder: bool = True  # Freeze encoder from MAE


@dataclass
class InpainterTrainingConfig:
    """Training configuration for inpainter."""
    mae_checkpoint: str = ""  # Path to MAE checkpoint for encoder initialization
    epochs: int = 50
    lr: float = 1e-4
    lr_scheduler: Optional[str] = None  # "cosine" or None
    lr_min: float = 1e-6
    warmup_epochs: int = 0
    weight_decay: float = 1e-4
    loss_fn: str = "smooth_l1"  # smooth_l1, mse, l1, huber
    npho_weight: float = 1.0
    time_weight: float = 1.0
    grad_clip: float = 1.0
    amp: bool = True
    track_mae_rmse: bool = True
    save_root_predictions: bool = True
    grad_accum_steps: int = 1
    track_train_metrics: bool = True
    # Conditional time loss: only compute where npho > threshold
    npho_threshold: Optional[float] = None  # None uses DEFAULT_NPHO_THRESHOLD (10.0)
    use_npho_time_weight: bool = True  # Weight time loss by sqrt(npho)


@dataclass
class InpainterCheckpointConfig:
    """Checkpoint configuration for inpainter."""
    resume_from: Optional[str] = None
    save_dir: str = "artifacts"
    save_interval: int = 10


@dataclass
class InpainterMLflowConfig:
    """MLflow configuration for inpainter."""
    experiment: str = "inpainting"
    run_name: Optional[str] = None


@dataclass
class InpainterConfig:
    """Complete inpainter training configuration."""
    data: InpainterDataConfig = field(default_factory=InpainterDataConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    model: InpainterModelConfig = field(default_factory=InpainterModelConfig)
    training: InpainterTrainingConfig = field(default_factory=InpainterTrainingConfig)
    checkpoint: InpainterCheckpointConfig = field(default_factory=InpainterCheckpointConfig)
    mlflow: InpainterMLflowConfig = field(default_factory=InpainterMLflowConfig)


def load_inpainter_config(config_path: str) -> InpainterConfig:
    """
    Load inpainter configuration from YAML file.

    Args:
        config_path: Path to YAML config file.

    Returns:
        InpainterConfig object with all settings.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    config = InpainterConfig()

    # Data
    if 'data' in raw_config:
        for k, v in raw_config['data'].items():
            if hasattr(config.data, k):
                setattr(config.data, k, v)

    # Normalization
    if 'normalization' in raw_config:
        for k, v in raw_config['normalization'].items():
            if hasattr(config.normalization, k):
                setattr(config.normalization, k, v)

    # Model
    if 'model' in raw_config:
        for k, v in raw_config['model'].items():
            if hasattr(config.model, k):
                setattr(config.model, k, v)

    # Training
    if 'training' in raw_config:
        for k, v in raw_config['training'].items():
            if hasattr(config.training, k):
                setattr(config.training, k, v)

    # Checkpoint
    if 'checkpoint' in raw_config:
        for k, v in raw_config['checkpoint'].items():
            if hasattr(config.checkpoint, k):
                setattr(config.checkpoint, k, v)

    # MLflow
    if 'mlflow' in raw_config:
        for k, v in raw_config['mlflow'].items():
            if hasattr(config.mlflow, k):
                setattr(config.mlflow, k, v)

    return config
