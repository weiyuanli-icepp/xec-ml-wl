#!/usr/bin/env python3
"""
Submit CEXPreprocess.py jobs to SLURM for parallel processing.

Splits the run range into batches and submits one SLURM job per batch.

Usage:
    # Process runs 557545-557644 (100 runs), 10 runs per job → 10 jobs
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 \
        --dead-file data/dead_channels_run430000.txt

    # Dry run (show jobs without submitting)
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 --dry-run

    # Custom batch size
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 --runs-per-job 5

    # Single run per job (maximum parallelism)
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 --runs-per-job 1
"""
import os
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
PREPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "CEXPreprocess.py")

LOG_DIR = os.path.expandvars("$HOME/meghome/xec-ml-wl/log/cex_preprocess")

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name=cex_{srun}_{nfiles}
#SBATCH --output={log_dir}/cex_{srun}_{nfiles}_%j.out
#SBATCH --error={log_dir}/cex_{srun}_{nfiles}_%j.err
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --account=meg
#SBATCH --time={time}
#SBATCH --mem={mem}

set -e

echo "Job starting on host: $(hostname)"
echo "Processing runs {srun} to {srun_end} ({nfiles} runs)"

# Load conda
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true
module load anaconda/2024.08 2>/dev/null || true

ARM_CONDA="$HOME/miniforge-arm/bin/conda"
X86_CONDA="/opt/psi/Programming/anaconda/2024.08/conda/bin/conda"

if [ -f "$ARM_CONDA" ] && [ "$(uname -m)" == "aarch64" ]; then
    eval "$($ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
elif [ -f "$X86_CONDA" ]; then
    eval "$($X86_CONDA shell.bash hook)"
else
    echo "ERROR: conda not found"
    exit 1
fi

conda activate {env_name}
cd {repo_dir}

python {script} \\
    --srun {srun} --nfiles {nfiles} --patch {patch} \\
    --output-dir {output_dir} {dead_flag}

echo "Job finished with exit code: $?"
"""


def main():
    parser = argparse.ArgumentParser(
        description="Submit CEXPreprocess.py jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--srun", type=int, required=True, help="Starting run number")
    parser.add_argument("--nfiles", type=int, required=True, help="Total number of runs")
    parser.add_argument("--patch", type=int, required=True, help="CEX patch number")
    parser.add_argument("--dead-file", default=None, help="Dead channel file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/cex/)")
    parser.add_argument("--runs-per-job", type=int, default=10,
                        help="Runs per SLURM job (default: 10)")
    parser.add_argument("--partition", default="mu3e", help="SLURM partition")
    parser.add_argument("--time", default="04:00:00", help="Time limit per job")
    parser.add_argument("--mem", default="8G", help="Memory per job")
    parser.add_argument("--env-name", default="xec-ml-wl", help="Conda environment")
    parser.add_argument("--dry-run", action="store_true", help="Show jobs without submitting")
    args = parser.parse_args()

    # Default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(REPO_DIR, "data", "cex")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    dead_flag = f"--dead-file {os.path.abspath(args.dead_file)}" if args.dead_file else ""

    # Split into batches
    batches = []
    for start in range(args.srun, args.srun + args.nfiles, args.runs_per_job):
        n = min(args.runs_per_job, args.srun + args.nfiles - start)
        batches.append((start, n))

    print(f"CEX Preprocessing: runs {args.srun}-{args.srun + args.nfiles - 1}")
    print(f"  Patch: {args.patch}")
    print(f"  Output: {args.output_dir}")
    print(f"  Dead file: {args.dead_file or '(none)'}")
    print(f"  Runs per job: {args.runs_per_job}")
    print(f"  Total jobs: {len(batches)}")
    print()

    submitted = 0
    for srun, nfiles in batches:
        script_content = SLURM_TEMPLATE.format(
            srun=srun,
            nfiles=nfiles,
            srun_end=srun + nfiles - 1,
            patch=args.patch,
            partition=args.partition,
            time=args.time,
            mem=args.mem,
            log_dir=LOG_DIR,
            env_name=args.env_name,
            repo_dir=REPO_DIR,
            script=PREPROCESS_SCRIPT,
            output_dir=os.path.abspath(args.output_dir),
            dead_flag=dead_flag,
        )

        if args.dry_run:
            print(f"  [DRY-RUN] srun={srun} nfiles={nfiles} "
                  f"-> CEX23_patch{args.patch}_r{srun}_n{nfiles}.root")
        else:
            result = subprocess.run(
                ["sbatch"], input=script_content, capture_output=True, text=True
            )
            if result.returncode == 0:
                job_id = result.stdout.strip().split()[-1]
                print(f"  Submitted srun={srun} nfiles={nfiles} -> job {job_id}")
                submitted += 1
            else:
                print(f"  FAILED srun={srun}: {result.stderr.strip()}")

    if args.dry_run:
        print(f"\n[DRY-RUN] {len(batches)} jobs would be submitted.")
    else:
        print(f"\nSubmitted {submitted}/{len(batches)} jobs.")
        print(f"Logs: {LOG_DIR}")
        print(f"Output: {args.output_dir}")
        print(f"\nAfter completion, merge with:")
        print(f"  hadd -f data/cex/CEX23_patch{args.patch}_all.root "
              f"{args.output_dir}/CEX23_patch{args.patch}_r*.root")


if __name__ == "__main__":
    main()
