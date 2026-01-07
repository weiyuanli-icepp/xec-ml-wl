import sys
import os
import argparse
import importlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def get_parser():
    parser = argparse.ArgumentParser(description="LXe Angle Regression Training")
    
    # Data & Paths
    parser.add_argument("--root", type=str, default="~/meghome/xec-ml-wl/data/MCGammaAngle_0-49.root")
    parser.add_argument("--tree", type=str, default="tree")
    parser.add_argument("--run_name", type=str, default=None, help="Name for MLflow run and artifacts")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    
    # Training Hyperparameters
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--chunksize", type=int, default=32000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--drop_path_rate", type=float, default=0.0)
    parser.add_argument("--use_scheduler", type=int, default=-1, help="-1 for Cosine+Warmup, other value for Constant LR")
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    
    # Preprocessing parameters
    parser.add_argument("--NphoScale", type=float, default=0.58)
    parser.add_argument("--NphoScale2", type=float, default=1.0)
    parser.add_argument("--time_scale", type=float, default=6.5e-8)
    parser.add_argument("--time_shift", type=float, default=0.5)
    parser.add_argument("--sentinel_value", type=float, default=-5.0)
    
    # Model & Data Processing
    parser.add_argument("--model", type=str, default="simple", choices=["simple", "convnextv2"], help="Model architecture to use")
    parser.add_argument("--npho_branch", type=str, default="relative_npho")
    parser.add_argument("--time_branch", type=str, default="relative_time")
    parser.add_argument("--reweight_mode", type=str, default="none") 
    parser.add_argument("--loss_type", type=str, default="smooth_l1")
    parser.add_argument("--loss_beta", type=float, default=1.0, help="Beta parameter for Smooth L1 loss if used")
    parser.add_argument("--outer_mode", type=str, default="finegrid")
    
    # Export
    parser.add_argument("--onnx", type=str, default="meg2ang_convnextv2.onnx")
    
    # MLflow
    parser.add_argument("--mlflow_experiment", type=str, default="gamma_angle")
    
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    reweight = args.reweight_mode
    if reweight.lower() in ["none", "null", ""]:
        reweight = ""

    print(f"[CLI] Starting training run: {args.run_name}")
    print(f"[CLI] Model: {args.model}")
    print(f"[CLI] Learning Rate: {args.lr}")
    print(f"[CLI] Scheduler: {'Cosine' if args.use_scheduler == -1 else 'Constant'}")

    if args.model == "simple":
        from legacy.train_angle import main_angle_with_args
        main_angle_with_args(
            root=os.path.expanduser(args.root),
            tree=args.tree,
            epochs=args.epochs,
            batch=args.batch,
            chunksize=args.chunksize,
            lr=args.lr,
            amp=True,
            npho_branch=args.npho_branch,
            time_branch=args.time_branch,
            NphoScale=args.NphoScale,
            reweight_mode=reweight,
            loss_type=args.loss_type,
            outer_mode=args.outer_mode,
            outer_fine_pool=(3,3),
            run_name=args.run_name,
            resume_from=args.resume_from
        )
    elif args.model == "convnextv2":
        from train_xec_regressor import main_xec_regressor_with_args
        main_xec_regressor_with_args(
            root=os.path.expanduser(args.root),
            tree=args.tree,
            epochs=args.epochs,
            batch=args.batch,
            chunksize=args.chunksize,
            lr=args.lr,
            weight_decay=args.weight_decay,
            drop_path_rate=args.drop_path_rate,
            time_shift=args.time_shift,
            time_scale=args.time_scale,
            sentinel_value=args.sentinel_value,
            use_scheduler=args.use_scheduler,
            warmup_epochs=args.warmup_epochs,
            onnx=args.onnx,
            mlflow_experiment=args.mlflow_experiment,
            amp=True,
            npho_branch=args.npho_branch,
            time_branch=args.time_branch,
            NphoScale=args.NphoScale,
            NphoScale2=args.NphoScale2,
            reweight_mode=reweight,
            loss_type=args.loss_type,
            loss_beta=args.loss_beta,
            outer_mode=args.outer_mode,
            outer_fine_pool=(3,3),
            run_name=args.run_name,
            resume_from=args.resume_from,
            ema_decay=args.ema_decay,
        )

if __name__ == "__main__":
    main()