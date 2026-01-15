#!/usr/bin/env python3
"""
CLI for XEC multi-task regressor training.

Supports two modes:
1. Config-based: Pass a YAML config file with --config
2. CLI overrides: Override specific config values via command line

Examples:
  # Use config file only
  python run_training_cli.py --config ../config/train_config.yaml

  # Override specific values
  python run_training_cli.py --config ../config/train_config.yaml --lr 1e-4 --epochs 30

  # Quick test with minimal args
  python run_training_cli.py --config ../config/train_config.yaml --train_path /path/to/train --val_path /path/to/val
"""

import sys
import os
import argparse
import tempfile
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_parser():
    parser = argparse.ArgumentParser(
        description="XEC Multi-Task Regressor Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config file (required or use defaults)
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (recommended)")

    # Data paths (can override config)
    parser.add_argument("--train_path", type=str, default=None,
                        help="Path to training data (directory, file, or glob pattern)")
    parser.add_argument("--val_path", type=str, default=None,
                        help="Path to validation data (directory, file, or glob pattern)")
    parser.add_argument("--tree", type=str, default=None, help="ROOT tree name")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--chunksize", type=int, default=None, help="Chunk size for streaming")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")

    # Training hyperparameters (can override config)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--ema_decay", type=float, default=None)
    parser.add_argument("--channel_dropout_rate", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    # Model architecture (can override config)
    parser.add_argument("--outer_mode", type=str, default=None, choices=["finegrid", "split"])
    parser.add_argument("--drop_path_rate", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)

    # Task configuration
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        choices=["angle", "energy", "timing", "uvwFI"],
                        help="Enable specific tasks (overrides config)")

    # Preprocessing
    parser.add_argument("--npho_scale", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    # Checkpointing
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save artifacts")

    # MLflow
    parser.add_argument("--run_name", type=str, default=None, help="MLflow run name")
    parser.add_argument("--mlflow_experiment", type=str, default=None, help="MLflow experiment name")

    # Export
    parser.add_argument("--onnx", type=str, default=None, help="ONNX output filename (null to disable)")

    # Loss configuration
    parser.add_argument("--loss_balance", type=str, default=None, choices=["manual", "auto"])

    return parser


def apply_cli_overrides(config_dict, args):
    """Apply CLI argument overrides to config dictionary."""

    # Data overrides
    if args.train_path is not None:
        config_dict.setdefault("data", {})["train_path"] = args.train_path
    if args.val_path is not None:
        config_dict.setdefault("data", {})["val_path"] = args.val_path
    if args.tree is not None:
        config_dict.setdefault("data", {})["tree_name"] = args.tree
    if args.batch is not None:
        config_dict.setdefault("data", {})["batch_size"] = args.batch
    if args.chunksize is not None:
        config_dict.setdefault("data", {})["chunksize"] = args.chunksize
    if args.num_workers is not None:
        config_dict.setdefault("data", {})["num_workers"] = args.num_workers

    # Training overrides
    if args.epochs is not None:
        config_dict.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config_dict.setdefault("training", {})["lr"] = args.lr
    if args.weight_decay is not None:
        config_dict.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.warmup_epochs is not None:
        config_dict.setdefault("training", {})["warmup_epochs"] = args.warmup_epochs
    if args.ema_decay is not None:
        config_dict.setdefault("training", {})["ema_decay"] = args.ema_decay
    if args.channel_dropout_rate is not None:
        config_dict.setdefault("training", {})["channel_dropout_rate"] = args.channel_dropout_rate
    if args.grad_clip is not None:
        config_dict.setdefault("training", {})["grad_clip"] = args.grad_clip

    # Model overrides
    if args.outer_mode is not None:
        config_dict.setdefault("model", {})["outer_mode"] = args.outer_mode
    if args.drop_path_rate is not None:
        config_dict.setdefault("model", {})["drop_path_rate"] = args.drop_path_rate
    if args.hidden_dim is not None:
        config_dict.setdefault("model", {})["hidden_dim"] = args.hidden_dim

    # Task overrides
    if args.tasks is not None:
        config_dict.setdefault("tasks", {})
        # Disable all tasks first
        for task in ["angle", "energy", "timing", "uvwFI"]:
            if task not in config_dict["tasks"]:
                config_dict["tasks"][task] = {"enabled": False}
            else:
                config_dict["tasks"][task]["enabled"] = False
        # Enable specified tasks
        for task in args.tasks:
            if task not in config_dict["tasks"]:
                config_dict["tasks"][task] = {}
            config_dict["tasks"][task]["enabled"] = True

    # Normalization overrides
    if args.npho_scale is not None:
        config_dict.setdefault("normalization", {})["npho_scale"] = args.npho_scale
    if args.time_scale is not None:
        config_dict.setdefault("normalization", {})["time_scale"] = args.time_scale
    if args.time_shift is not None:
        config_dict.setdefault("normalization", {})["time_shift"] = args.time_shift
    if args.sentinel_value is not None:
        config_dict.setdefault("normalization", {})["sentinel_value"] = args.sentinel_value

    # Checkpoint overrides
    if args.resume_from is not None:
        config_dict.setdefault("checkpoint", {})["resume_from"] = args.resume_from
    if args.save_dir is not None:
        config_dict.setdefault("checkpoint", {})["save_dir"] = args.save_dir

    # MLflow overrides
    if args.run_name is not None:
        config_dict.setdefault("mlflow", {})["run_name"] = args.run_name
    if args.mlflow_experiment is not None:
        config_dict.setdefault("mlflow", {})["experiment"] = args.mlflow_experiment

    # Export overrides
    if args.onnx is not None:
        config_dict.setdefault("export", {})["onnx"] = args.onnx if args.onnx.lower() != "null" else None

    # Loss balance override
    if args.loss_balance is not None:
        config_dict["loss_balance"] = args.loss_balance

    return config_dict


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Load base config
    if args.config is not None:
        config_path = os.path.expanduser(args.config)
        if not os.path.exists(config_path):
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        print(f"[CLI] Loaded config from: {config_path}")
    else:
        # Start with empty config (will use defaults)
        config_dict = {}
        print("[CLI] No config file specified, using defaults with CLI overrides")

    # Apply CLI overrides
    config_dict = apply_cli_overrides(config_dict, args)

    # Validate required fields
    if not config_dict.get("data", {}).get("train_path"):
        print("[ERROR] train_path is required. Specify via --train_path or in config file.")
        sys.exit(1)
    if not config_dict.get("data", {}).get("val_path"):
        print("[ERROR] val_path is required. Specify via --val_path or in config file.")
        sys.exit(1)

    # Write merged config to temp file and run training
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        temp_config_path = f.name

    try:
        print(f"[CLI] Starting training with merged config...")
        print(f"[CLI] Train path: {config_dict['data']['train_path']}")
        print(f"[CLI] Val path: {config_dict['data']['val_path']}")

        # Import and run training
        from lib.train_regressor import train_with_config
        train_with_config(temp_config_path)

    finally:
        # Clean up temp file
        if os.path.exists(temp_config_path):
            os.unlink(temp_config_path)


if __name__ == "__main__":
    main()
