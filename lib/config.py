"""
Configuration management for XEC training.
Loads YAML config files and provides defaults.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from .geom_defs import (
    DEFAULT_NPHO_SCALE, DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE, DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_VALUE
)


@dataclass
class TaskConfig:
    """Configuration for a single task."""
    enabled: bool = False
    loss_fn: str = "smooth_l1"  # smooth_l1, l1, mse, relative_l1, relative_smooth_l1, relative_mse
    loss_beta: float = 1.0      # For smooth_l1/huber
    weight: float = 1.0         # Manual loss weight
    log_transform: bool = False  # Train on log(value) for energy/timing tasks


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
    npho_branch: str = "npho"  # Input branch for photon counts
    time_branch: str = "relative_time"  # Input branch for timing
    log_invalid_npho: bool = True  # Log warning when invalid npho values detected


@dataclass
class NormalizationConfig:
    """Input normalization parameters."""
    npho_scale: float = DEFAULT_NPHO_SCALE
    npho_scale2: float = DEFAULT_NPHO_SCALE2
    time_scale: float = DEFAULT_TIME_SCALE
    time_shift: float = DEFAULT_TIME_SHIFT
    sentinel_value: float = DEFAULT_SENTINEL_VALUE


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
    # New unified scheduler naming (preferred)
    lr_scheduler: Optional[str] = None  # "cosine", "onecycle", "plateau", or null to disable
    # Legacy scheduler config (for backward compatibility)
    use_scheduler: bool = True
    scheduler: str = "cosine"  # "cosine", "onecycle", "plateau", or "none"
    # OneCycleLR specific
    max_lr: Optional[float] = None  # Max LR for OneCycleLR (defaults to lr if not set)
    pct_start: float = 0.3  # Fraction of training for LR increase phase
    # ReduceLROnPlateau specific
    lr_patience: int = 5  # Epochs to wait before reducing LR
    lr_factor: float = 0.5  # Factor to reduce LR by
    lr_min: float = 1e-7  # Minimum LR
    # General
    amp: bool = True
    ema_decay: float = 0.999
    channel_dropout_rate: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    profile: bool = False  # Enable training profiler to identify bottlenecks
    compile: str = "max-autotune"  # torch.compile mode: "max-autotune", "reduce-overhead", "default", or "false"/"none" to disable


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
    save_interval: int = 10  # Save checkpoint every N epochs
    save_artifacts: bool = True  # Save plots, CSVs, worst case displays (disable for quick testing)
    refresh_lr: bool = False  # Reset LR scheduler when resuming (schedule runs from current epoch to end)
    reset_epoch: bool = False  # Start from epoch 1 when resuming (only load model weights)
    new_mlflow_run: bool = False  # Force new MLflow run even when resuming from checkpoint


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


def load_config(config_path: str, warn_missing: bool = True, auto_update: bool = True) -> XECConfig:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file.
        warn_missing: If True, print warnings about missing config options.
        auto_update: If True, automatically add missing options to the config file
                     and exit so user can review the changes.

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

    # Validate and optionally update config
    if warn_missing or auto_update:
        missing = validate_config(config_path, config_type="regressor",
                                  auto_update=auto_update, verbose=warn_missing)
        if missing and auto_update:
            print("\n[INFO] Config file has been updated with missing options.")
            print("[INFO] Please review the changes and re-run the script.")
            print(f"[INFO] Config file: {config_path}")
            import sys
            sys.exit(0)

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
    Convert task configs to task_weights dict for engine_regressor.py.

    Returns:
        Dict like: {"angle": {"loss_fn": "smooth_l1", "weight": 1.0, "log_transform": False}, ...}
    """
    weights = {}
    for name, tc in config.tasks.items():
        if tc.enabled:
            weights[name] = {
                "loss_fn": tc.loss_fn,
                "loss_beta": tc.loss_beta,
                "weight": tc.weight,
                "log_transform": tc.log_transform,
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
    npho_branch: str = "npho"  # Input branch for photon counts
    time_branch: str = "relative_time"  # Input branch for timing
    log_invalid_npho: bool = True  # Log warning when invalid npho values detected


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
    compile: str = "reduce-overhead"  # torch.compile mode
    # Conditional time loss: only compute where npho > threshold
    npho_threshold: Optional[float] = None  # None uses DEFAULT_NPHO_THRESHOLD (10.0)
    use_npho_time_weight: bool = True  # Weight time loss by sqrt(npho)
    track_mae_rmse: bool = False  # Compute/log MAE/RMSE metrics
    track_train_metrics: bool = False  # Track per-face loss during training
    profile: bool = False  # Enable training profiler


@dataclass
class MAECheckpointConfig:
    """Checkpoint configuration for MAE."""
    resume_from: Optional[str] = None
    save_dir: str = "artifacts"
    save_interval: int = 10  # Save checkpoint every N epochs
    save_predictions: bool = True  # Save ROOT file with sensor predictions
    new_mlflow_run: bool = False  # Force new MLflow run even when resuming from checkpoint


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


def load_mae_config(config_path: str, warn_missing: bool = True, auto_update: bool = True) -> MAEConfig:
    """
    Load MAE configuration from YAML file.

    Args:
        config_path: Path to YAML config file.
        warn_missing: If True, print warnings about missing config options.
        auto_update: If True, automatically add missing options to the config file
                     and exit so user can review the changes.

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

    # Validate and optionally update config
    if warn_missing or auto_update:
        missing = validate_config(config_path, config_type="mae",
                                  auto_update=auto_update, verbose=warn_missing)
        if missing and auto_update:
            print("\n[INFO] Config file has been updated with missing options.")
            print("[INFO] Please review the changes and re-run the script.")
            print(f"[INFO] Config file: {config_path}")
            import sys
            sys.exit(0)

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
    npho_branch: str = "npho"  # Input branch for photon counts
    time_branch: str = "relative_time"  # Input branch for timing
    log_invalid_npho: bool = True  # Log warning when invalid npho values detected


@dataclass
class InpainterModelConfig:
    """Model configuration for inpainter."""
    outer_mode: str = "finegrid"
    outer_fine_pool: Optional[List[int]] = None
    mask_ratio: float = 0.05  # Default 5% for realistic dead channel density
    time_mask_ratio_scale: float = 1.0  # Scale factor for masking valid-time sensors (1.0 = uniform)
    freeze_encoder: bool = True  # Freeze encoder from MAE
    use_local_context: bool = True  # Use local neighbor context for inpainting


@dataclass
class InpainterTrainingConfig:
    """Training configuration for inpainter."""
    mae_checkpoint: Optional[str] = None  # Path to MAE checkpoint for encoder initialization
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
    compile: str = "reduce-overhead"  # torch.compile mode
    track_mae_rmse: bool = True
    save_root_predictions: bool = True
    grad_accum_steps: int = 1
    track_train_metrics: bool = True
    # Conditional time loss: only compute where npho > threshold
    npho_threshold: Optional[float] = None  # None uses DEFAULT_NPHO_THRESHOLD (10.0)
    use_npho_time_weight: bool = True  # Weight time loss by sqrt(npho)
    ema_decay: Optional[float] = None  # None = disabled, 0.999 = typical value
    profile: bool = False  # Enable training profiler


@dataclass
class InpainterCheckpointConfig:
    """Checkpoint configuration for inpainter."""
    resume_from: Optional[str] = None
    save_dir: str = "artifacts"
    save_interval: int = 10
    save_predictions: bool = True  # Save ROOT file with sensor predictions
    new_mlflow_run: bool = False  # Force new MLflow run even when resuming from checkpoint


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


def load_inpainter_config(config_path: str, warn_missing: bool = True, auto_update: bool = True) -> InpainterConfig:
    """
    Load inpainter configuration from YAML file.

    Args:
        config_path: Path to YAML config file.
        warn_missing: If True, print warnings about missing config options.
        auto_update: If True, automatically add missing options to the config file
                     and exit so user can review the changes.

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

    # Validate and optionally update config
    if warn_missing or auto_update:
        missing = validate_config(config_path, config_type="inpainter",
                                  auto_update=auto_update, verbose=warn_missing)
        if missing and auto_update:
            print("\n[INFO] Config file has been updated with missing options.")
            print("[INFO] Please review the changes and re-run the script.")
            print(f"[INFO] Config file: {config_path}")
            import sys
            sys.exit(0)

    return config


# ------------------------------------------------------------
#  Config Validation and Auto-Update
# ------------------------------------------------------------

def _get_dataclass_defaults(cls) -> Dict[str, Any]:
    """Get default values from a dataclass."""
    import dataclasses
    defaults = {}
    for f in dataclasses.fields(cls):
        if f.default is not dataclasses.MISSING:
            defaults[f.name] = f.default
        elif f.default_factory is not dataclasses.MISSING:
            defaults[f.name] = f.default_factory()
        else:
            defaults[f.name] = None
    return defaults


def _check_missing_keys(raw_section: Dict, dataclass_cls, section_name: str) -> List[tuple]:
    """
    Check for missing keys in a config section.

    Returns:
        List of (key, default_value) tuples for missing keys.
    """
    if raw_section is None:
        raw_section = {}

    defaults = _get_dataclass_defaults(dataclass_cls)
    missing = []

    for key, default_val in defaults.items():
        if key not in raw_section:
            missing.append((key, default_val))

    return missing


def validate_config(config_path: str, config_type: str = "auto",
                   auto_update: bool = False, verbose: bool = True) -> Dict[str, List[tuple]]:
    """
    Validate a config file and optionally auto-update with missing options.

    Args:
        config_path: Path to YAML config file.
        config_type: One of "regressor", "mae", "inpainter", or "auto" (detect from path/content).
        auto_update: If True, update the config file with missing options.
        verbose: If True, print warnings about missing options.

    Returns:
        Dict mapping section names to lists of (key, default_value) tuples for missing keys.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        raw_config = yaml.safe_load(f)

    if raw_config is None:
        raw_config = {}

    # Auto-detect config type
    if config_type == "auto":
        if "mae" in config_path.lower() or raw_config.get('mlflow', {}).get('experiment', '').startswith('mae'):
            config_type = "mae"
        elif "inp" in config_path.lower() or "inpainter" in config_path.lower() or \
             raw_config.get('mlflow', {}).get('experiment', '') == 'inpainting':
            config_type = "inpainter"
        else:
            config_type = "regressor"

    # Define section -> dataclass mapping for each config type
    if config_type == "regressor":
        section_map = {
            'data': DataConfig,
            'normalization': NormalizationConfig,
            'model': ModelConfig,
            'training': TrainingConfig,
            'checkpoint': CheckpointConfig,
            'mlflow': MLflowConfig,
            'export': ExportConfig,
        }
    elif config_type == "mae":
        section_map = {
            'data': MAEDataConfig,
            'normalization': NormalizationConfig,
            'model': MAEModelConfig,
            'training': MAETrainingConfig,
            'checkpoint': MAECheckpointConfig,
            'mlflow': MAEMLflowConfig,
        }
    elif config_type == "inpainter":
        section_map = {
            'data': InpainterDataConfig,
            'normalization': NormalizationConfig,
            'model': InpainterModelConfig,
            'training': InpainterTrainingConfig,
            'checkpoint': InpainterCheckpointConfig,
            'mlflow': InpainterMLflowConfig,
        }
    else:
        raise ValueError(f"Unknown config type: {config_type}")

    all_missing = {}

    for section_name, dataclass_cls in section_map.items():
        raw_section = raw_config.get(section_name, {})
        missing = _check_missing_keys(raw_section, dataclass_cls, section_name)
        if missing:
            all_missing[section_name] = missing

    # Print warnings
    if verbose and all_missing:
        print(f"\n[WARN] Config file '{config_path}' is missing the following options:")
        for section, missing_list in all_missing.items():
            for key, default_val in missing_list:
                val_str = repr(default_val) if not isinstance(default_val, str) else f'"{default_val}"'
                print(f"  {section}.{key}: {val_str} (default)")
        print()

    # Auto-update the config file
    if auto_update and all_missing:
        _update_config_file(config_path, raw_config, all_missing, verbose)

    return all_missing


def _update_config_file(config_path: str, raw_config: Dict,
                        missing: Dict[str, List[tuple]], verbose: bool = True):
    """
    Update a config file with missing options.

    Preserves comments and formatting by appending missing options to each section.
    """
    # Read original file content
    with open(config_path, 'r') as f:
        lines = f.readlines()

    # For each section with missing keys, find the section and add missing keys
    for section_name, missing_list in missing.items():
        if not missing_list:
            continue

        # Find the section in the file
        section_start = -1
        section_end = len(lines)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(f'{section_name}:'):
                section_start = i
            elif section_start >= 0 and stripped and not stripped.startswith('#') and \
                 not stripped.startswith('-') and ':' in stripped and \
                 not line.startswith(' ') and not line.startswith('\t'):
                # Found next top-level section
                section_end = i
                break

        if section_start < 0:
            # Section doesn't exist, create it
            insert_lines = [f"\n{section_name}:\n"]
            for key, default_val in missing_list:
                val_str = _format_yaml_value(default_val)
                insert_lines.append(f"  {key}: {val_str}\n")
            lines.extend(insert_lines)
        else:
            # Find last line of section content (before next section or end)
            insert_pos = section_end
            for i in range(section_end - 1, section_start, -1):
                if lines[i].strip() and not lines[i].strip().startswith('#'):
                    insert_pos = i + 1
                    break

            # Insert missing keys
            insert_lines = []
            for key, default_val in missing_list:
                val_str = _format_yaml_value(default_val)
                insert_lines.append(f"  {key}: {val_str}  # (auto-added)\n")

            for j, line in enumerate(insert_lines):
                lines.insert(insert_pos + j, line)

    # Write updated file
    with open(config_path, 'w') as f:
        f.writelines(lines)

    if verbose:
        print(f"[INFO] Updated config file with missing options: {config_path}")


def _format_yaml_value(value) -> str:
    """Format a Python value for YAML output."""
    if value is None:
        return "null"
    elif isinstance(value, bool):
        return "true" if value else "false"
    elif isinstance(value, str):
        if value == "":
            return '""'
        return f'"{value}"' if ' ' in value or ':' in value else value
    elif isinstance(value, (list, tuple)):
        return '[' + ', '.join(_format_yaml_value(v) for v in value) + ']'
    elif isinstance(value, float):
        # Use scientific notation for very small/large numbers
        if value != 0 and (abs(value) < 1e-4 or abs(value) > 1e6):
            return f"{value:.2e}"
        return str(value)
    else:
        return str(value)
