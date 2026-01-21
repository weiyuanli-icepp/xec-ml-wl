#!/usr/bin/env python3
"""
CLI for XEC Inpainter (Dead Channel Recovery) training.

Supports two modes:
1. Config-based: Pass a YAML config file with --config
2. CLI overrides: Override specific config values via command line

The --mae_checkpoint is optional. If not provided, the encoder is initialized
from scratch (no pre-training).

Examples:
  # Use config file only (with MAE pre-training)
  python run_inpainter.py --config ../config/inpainter_config.yaml

  # Without MAE pre-training (train from scratch)
  python run_inpainter.py --config ../config/inpainter_config.yaml --no_mae_pretrain

  # Override specific values
  python run_inpainter.py --config ../config/inpainter_config.yaml --lr 1e-4 --epochs 30

  # CLI mode without config
  python run_inpainter.py --train_root /path/train.root --val_root /path/val.root --epochs 50
"""

import sys
import os
import argparse
import tempfile
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def get_parser():
    parser = argparse.ArgumentParser(
        description="XEC Inpainter Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Config file
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (recommended)")

    # Data paths
    parser.add_argument("--train_root", type=str, default=None,
                        help="Path to training ROOT file(s)")
    parser.add_argument("--val_root", type=str, default=None,
                        help="Path to validation ROOT file(s)")
    parser.add_argument("--tree", type=str, default=None, help="ROOT tree name")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--chunksize", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_threads", type=int, default=None)

    # Model configuration
    parser.add_argument("--mae_checkpoint", type=str, default=None,
                        help="Path to MAE checkpoint for encoder (optional)")
    parser.add_argument("--no_mae_pretrain", action="store_true",
                        help="Train without MAE pre-training (initialize from scratch)")
    parser.add_argument("--outer_mode", type=str, default=None,
                        choices=["split", "finegrid"])
    parser.add_argument("--outer_fine_pool", type=int, nargs=2, default=None,
                        help="Outer fine pool kernel, e.g., 3 3")
    parser.add_argument("--mask_ratio", type=float, default=None,
                        help="Mask ratio for training")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights")
    parser.add_argument("--finetune_encoder", action="store_true",
                        help="Fine-tune encoder (not frozen)")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_scheduler", type=str, default=None,
                        choices=["none", "cosine"])
    parser.add_argument("--lr_min", type=float, default=None)
    parser.add_argument("--warmup_epochs", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    # Loss configuration
    parser.add_argument("--loss_fn", type=str, default=None,
                        choices=["smooth_l1", "mse", "l1", "huber"])
    parser.add_argument("--npho_weight", type=float, default=None)
    parser.add_argument("--time_weight", type=float, default=None)

    # Normalization
    parser.add_argument("--npho_scale", type=float, default=None)
    parser.add_argument("--npho_scale2", type=float, default=None)
    parser.add_argument("--time_scale", type=float, default=None)
    parser.add_argument("--time_shift", type=float, default=None)
    parser.add_argument("--sentinel_value", type=float, default=None)

    # Checkpointing
    parser.add_argument("--save_path", type=str, default=None,
                        help="Directory to save checkpoints")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to inpainter checkpoint to resume from")
    parser.add_argument("--save_interval", type=int, default=None)

    # MLflow
    parser.add_argument("--mlflow_experiment", type=str, default=None)
    parser.add_argument("--mlflow_run_name", type=str, default=None)

    return parser


def apply_cli_overrides(config_dict, args):
    """Apply CLI argument overrides to config dictionary."""

    # Data overrides
    if args.train_root is not None:
        config_dict.setdefault("data", {})["train_path"] = args.train_root
    if args.val_root is not None:
        config_dict.setdefault("data", {})["val_path"] = args.val_root
    if args.tree is not None:
        config_dict.setdefault("data", {})["tree_name"] = args.tree
    if args.batch_size is not None:
        config_dict.setdefault("data", {})["batch_size"] = args.batch_size
    if args.chunksize is not None:
        config_dict.setdefault("data", {})["chunksize"] = args.chunksize
    if args.num_workers is not None:
        config_dict.setdefault("data", {})["num_workers"] = args.num_workers
    if args.num_threads is not None:
        config_dict.setdefault("data", {})["num_threads"] = args.num_threads

    # Model overrides
    if args.no_mae_pretrain:
        config_dict.setdefault("training", {})["mae_checkpoint"] = None
    elif args.mae_checkpoint is not None:
        config_dict.setdefault("training", {})["mae_checkpoint"] = args.mae_checkpoint
    if args.outer_mode is not None:
        config_dict.setdefault("model", {})["outer_mode"] = args.outer_mode
    if args.outer_fine_pool is not None:
        config_dict.setdefault("model", {})["outer_fine_pool"] = args.outer_fine_pool
    if args.mask_ratio is not None:
        config_dict.setdefault("model", {})["mask_ratio"] = args.mask_ratio
    if args.freeze_encoder:
        config_dict.setdefault("model", {})["freeze_encoder"] = True
    if args.finetune_encoder:
        config_dict.setdefault("model", {})["freeze_encoder"] = False

    # Training overrides
    if args.epochs is not None:
        config_dict.setdefault("training", {})["epochs"] = args.epochs
    if args.lr is not None:
        config_dict.setdefault("training", {})["lr"] = args.lr
    if args.lr_scheduler is not None:
        config_dict.setdefault("training", {})["lr_scheduler"] = args.lr_scheduler
    if args.lr_min is not None:
        config_dict.setdefault("training", {})["lr_min"] = args.lr_min
    if args.warmup_epochs is not None:
        config_dict.setdefault("training", {})["warmup_epochs"] = args.warmup_epochs
    if args.weight_decay is not None:
        config_dict.setdefault("training", {})["weight_decay"] = args.weight_decay
    if args.grad_clip is not None:
        config_dict.setdefault("training", {})["grad_clip"] = args.grad_clip

    # Loss overrides
    if args.loss_fn is not None:
        config_dict.setdefault("training", {})["loss_fn"] = args.loss_fn
    if args.npho_weight is not None:
        config_dict.setdefault("training", {})["npho_weight"] = args.npho_weight
    if args.time_weight is not None:
        config_dict.setdefault("training", {})["time_weight"] = args.time_weight

    # Normalization overrides
    if args.npho_scale is not None:
        config_dict.setdefault("normalization", {})["npho_scale"] = args.npho_scale
    if args.npho_scale2 is not None:
        config_dict.setdefault("normalization", {})["npho_scale2"] = args.npho_scale2
    if args.time_scale is not None:
        config_dict.setdefault("normalization", {})["time_scale"] = args.time_scale
    if args.time_shift is not None:
        config_dict.setdefault("normalization", {})["time_shift"] = args.time_shift
    if args.sentinel_value is not None:
        config_dict.setdefault("normalization", {})["sentinel_value"] = args.sentinel_value

    # Checkpoint overrides
    if args.save_path is not None:
        config_dict.setdefault("checkpoint", {})["save_dir"] = args.save_path
    if args.resume_from is not None:
        config_dict.setdefault("checkpoint", {})["resume_from"] = args.resume_from
    if args.save_interval is not None:
        config_dict.setdefault("checkpoint", {})["save_interval"] = args.save_interval

    # MLflow overrides
    if args.mlflow_experiment is not None:
        config_dict.setdefault("mlflow", {})["experiment"] = args.mlflow_experiment
    if args.mlflow_run_name is not None:
        config_dict.setdefault("mlflow", {})["run_name"] = args.mlflow_run_name

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
        config_dict = {}
        print("[CLI] No config file specified, using defaults with CLI overrides")

    # Apply CLI overrides
    config_dict = apply_cli_overrides(config_dict, args)

    # Validate required fields
    if not config_dict.get("data", {}).get("train_path"):
        print("[ERROR] train_path is required. Specify via --train_root or in config file.")
        sys.exit(1)

    # Build CLI args for train_inpainter
    cli_args = []

    # Data args
    data = config_dict.get("data", {})
    if data.get("train_path"):
        cli_args.extend(["--train_root", str(data["train_path"])])
    if data.get("val_path"):
        cli_args.extend(["--val_root", str(data["val_path"])])
    if data.get("tree_name"):
        cli_args.extend(["--tree", str(data["tree_name"])])
    if data.get("batch_size"):
        cli_args.extend(["--batch_size", str(data["batch_size"])])
    if data.get("chunksize"):
        cli_args.extend(["--chunksize", str(data["chunksize"])])
    if data.get("num_workers"):
        cli_args.extend(["--num_workers", str(data["num_workers"])])
    if data.get("num_threads"):
        cli_args.extend(["--num_threads", str(data["num_threads"])])

    # Model args
    model = config_dict.get("model", {})
    if model.get("outer_mode"):
        cli_args.extend(["--outer_mode", str(model["outer_mode"])])
    if model.get("outer_fine_pool"):
        pool = model["outer_fine_pool"]
        cli_args.extend(["--outer_fine_pool", str(pool[0]), str(pool[1])])
    if model.get("mask_ratio") is not None:
        cli_args.extend(["--mask_ratio", str(model["mask_ratio"])])
    if model.get("freeze_encoder") is False:
        cli_args.append("--finetune_encoder")
    elif model.get("freeze_encoder") is True:
        cli_args.append("--freeze_encoder")

    # Training args
    training = config_dict.get("training", {})
    mae_ckpt = training.get("mae_checkpoint")
    if mae_ckpt:
        cli_args.extend(["--mae_checkpoint", str(mae_ckpt)])
    if training.get("epochs"):
        cli_args.extend(["--epochs", str(training["epochs"])])
    if training.get("lr"):
        cli_args.extend(["--lr", str(training["lr"])])
    if training.get("lr_scheduler"):
        cli_args.extend(["--lr_scheduler", str(training["lr_scheduler"])])
    if training.get("lr_min"):
        cli_args.extend(["--lr_min", str(training["lr_min"])])
    if training.get("warmup_epochs") is not None:
        cli_args.extend(["--warmup_epochs", str(training["warmup_epochs"])])
    if training.get("weight_decay"):
        cli_args.extend(["--weight_decay", str(training["weight_decay"])])
    if training.get("grad_clip"):
        cli_args.extend(["--grad_clip", str(training["grad_clip"])])
    if training.get("loss_fn"):
        cli_args.extend(["--loss_fn", str(training["loss_fn"])])
    if training.get("npho_weight") is not None:
        cli_args.extend(["--npho_weight", str(training["npho_weight"])])
    if training.get("time_weight") is not None:
        cli_args.extend(["--time_weight", str(training["time_weight"])])

    # Normalization args
    norm = config_dict.get("normalization", {})
    if norm.get("npho_scale"):
        cli_args.extend(["--npho_scale", str(norm["npho_scale"])])
    if norm.get("npho_scale2"):
        cli_args.extend(["--npho_scale2", str(norm["npho_scale2"])])
    if norm.get("time_scale"):
        cli_args.extend(["--time_scale", str(norm["time_scale"])])
    if norm.get("time_shift"):
        cli_args.extend(["--time_shift", str(norm["time_shift"])])
    if norm.get("sentinel_value"):
        cli_args.extend(["--sentinel_value", str(norm["sentinel_value"])])

    # Checkpoint args
    ckpt = config_dict.get("checkpoint", {})
    if ckpt.get("save_dir"):
        cli_args.extend(["--save_path", str(ckpt["save_dir"])])
    if ckpt.get("resume_from"):
        cli_args.extend(["--resume_from", str(ckpt["resume_from"])])
    if ckpt.get("save_interval"):
        cli_args.extend(["--save_interval", str(ckpt["save_interval"])])

    # MLflow args
    mlflow_cfg = config_dict.get("mlflow", {})
    if mlflow_cfg.get("experiment"):
        cli_args.extend(["--mlflow_experiment", str(mlflow_cfg["experiment"])])
    if mlflow_cfg.get("run_name"):
        cli_args.extend(["--mlflow_run_name", str(mlflow_cfg["run_name"])])

    print(f"[CLI] Starting inpainter training...")
    print(f"[CLI] Train path: {data.get('train_path')}")
    print(f"[CLI] Val path: {data.get('val_path')}")
    print(f"[CLI] MAE checkpoint: {mae_ckpt if mae_ckpt else 'None (training from scratch)'}")

    # Run training
    import subprocess
    cmd = [sys.executable, "-m", "lib.train_inpainter"] + cli_args
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
