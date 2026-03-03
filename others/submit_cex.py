#!/usr/bin/env python3
"""
Submit CEXPreprocess.py jobs to SLURM for parallel processing.

Splits the run range into batches and submits one SLURM job per batch.
Auto-generates per-run dead channel files from the MEG2 database.

Usage:
    # Process runs 557545-557644 (100 runs), 10 runs per job → 10 jobs
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13

    # Dry run (show jobs without submitting)
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 --dry-run

    # Single run per job (maximum parallelism)
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 --runs-per-job 1

    # Skip dead channel generation (already done or not needed)
    python others/submit_cex.py --srun 557545 --nfiles 100 --patch 13 --no-dead
"""
import os
import sys
import argparse
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
PREPROCESS_SCRIPT = os.path.join(SCRIPT_DIR, "CEXPreprocess.py")
DEAD_DIR = os.path.join(REPO_DIR, "data", "dead_channels")

LOG_DIR = os.path.expandvars("$HOME/meghome/xec-ml-wl/log/cex_preprocess")

# Add repo to path for imports
sys.path.insert(0, REPO_DIR)

# CEX23 run configurations: (srun, nfiles, patch)
CEX23_CONFIGS = [
    (557545, 100, 13),
    (558304, 100, 12),
    (558394, 100, 21),
    (559081, 100, 20),
    (559171, 100,  5),
    (559862, 100,  4),
    (558991, 100, 22),
    (558214, 100, 14),
    (559772, 100,  6),
    (558900, 100, 19),
    (558124, 100, 11),
    (559682, 100,  3),
    (559261, 100,  1),
    (559498, 100,  2),
    (559592, 100,  7),
    (559408, 100,  8),
    (557628, 100,  9),
    (557809, 100, 10),
    (558034, 100, 15),
    (557718, 100, 16),
    (558484, 100, 17),
    (558717, 100, 18),
    (558807, 100, 23),
    (558575, 100, 24),
]

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


def generate_dead_channel_files(srun, nfiles):
    """Query DB and generate per-run dead channel files.

    Caches by XECPMStatus_id — runs sharing the same PM status
    get the same dead channel list without redundant queries.

    Returns the number of unique dead channel configurations found.
    """
    from lib.db_utils import get_xec_pm_status_id, get_dead_channels, save_dead_channel_list

    os.makedirs(DEAD_DIR, exist_ok=True)

    # Cache: pm_status_id → dead channel array (avoid redundant queries)
    status_cache = {}
    generated = 0
    skipped = 0
    failed = 0

    for iRun in range(srun, srun + nfiles):
        outpath = os.path.join(DEAD_DIR, f"dead_run{iRun}.txt")

        # Skip if already exists
        if os.path.exists(outpath):
            skipped += 1
            continue

        try:
            pm_status_id = get_xec_pm_status_id(iRun)
            if pm_status_id is None:
                failed += 1
                continue

            if pm_status_id not in status_cache:
                status_cache[pm_status_id] = get_dead_channels(iRun)

            dead = status_cache[pm_status_id]

            with open(outpath, 'w') as f:
                f.write(f"# Dead channels for run {iRun} "
                        f"(XECPMStatus_id={pm_status_id}, n={len(dead)})\n")
                for ch in dead:
                    f.write(f"{ch}\n")

            generated += 1

        except Exception as e:
            print(f"  WARNING: Failed to query run {iRun}: {e}")
            failed += 1

    print(f"  Dead channel files: {generated} generated, {skipped} cached, "
          f"{failed} failed")
    print(f"  Unique PM status configs: {len(status_cache)}")

    return len(status_cache)


def main():
    parser = argparse.ArgumentParser(
        description="Submit CEXPreprocess.py jobs to SLURM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--srun", type=int, default=None, help="Starting run number")
    parser.add_argument("--nfiles", type=int, default=None, help="Total number of runs")
    parser.add_argument("--patch", type=int, default=None, help="CEX patch number")
    parser.add_argument("--all", action="store_true",
                        help="Process all 24 CEX23 patch configurations")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: data/cex/)")
    parser.add_argument("--runs-per-job", type=int, default=10,
                        help="Runs per SLURM job (default: 10)")
    parser.add_argument("--partition", default="mu3e", help="SLURM partition")
    parser.add_argument("--time", default="04:00:00", help="Time limit per job")
    parser.add_argument("--mem", default="8G", help="Memory per job")
    parser.add_argument("--env-name", default="xec-ml-wl", help="Conda environment")
    parser.add_argument("--no-dead", action="store_true",
                        help="Skip dead channel generation (no --dead-dir passed)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show jobs without submitting")
    args = parser.parse_args()

    # Validate arguments
    if args.all:
        configs = CEX23_CONFIGS
    elif args.srun is not None and args.nfiles is not None and args.patch is not None:
        configs = [(args.srun, args.nfiles, args.patch)]
    else:
        parser.error("Either --all or --srun/--nfiles/--patch are required")

    # Default output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(REPO_DIR, "data", "cex")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- Generate dead channel files for all runs across all configs ---
    if not args.no_dead:
        print(f"Generating dead channel files in {DEAD_DIR} ...")
        for srun, nfiles, patch in configs:
            print(f"  Patch {patch:2d}: runs {srun}-{srun + nfiles - 1}")
            generate_dead_channel_files(srun, nfiles)
        dead_flag = f"--dead-dir {os.path.abspath(DEAD_DIR)}"
    else:
        dead_flag = ""

    # --- Submit jobs for each config ---
    total_submitted = 0
    total_jobs = 0

    for srun, nfiles, patch in configs:
        # Split into batches
        batches = []
        for start in range(srun, srun + nfiles, args.runs_per_job):
            n = min(args.runs_per_job, srun + nfiles - start)
            batches.append((start, n))
        total_jobs += len(batches)

        print(f"\nPatch {patch:2d}: runs {srun}-{srun + nfiles - 1} "
              f"({len(batches)} jobs)")

        for batch_srun, batch_nfiles in batches:
            script_content = SLURM_TEMPLATE.format(
                srun=batch_srun,
                nfiles=batch_nfiles,
                srun_end=batch_srun + batch_nfiles - 1,
                patch=patch,
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
                print(f"  [DRY-RUN] srun={batch_srun} nfiles={batch_nfiles} "
                      f"-> CEX23_patch{patch}_r{batch_srun}_n{batch_nfiles}.root")
            else:
                result = subprocess.run(
                    ["sbatch"], input=script_content, capture_output=True, text=True
                )
                if result.returncode == 0:
                    job_id = result.stdout.strip().split()[-1]
                    print(f"  Submitted srun={batch_srun} nfiles={batch_nfiles} "
                          f"-> job {job_id}")
                    total_submitted += 1
                else:
                    print(f"  FAILED srun={batch_srun}: {result.stderr.strip()}")

    if args.dry_run:
        print(f"\n[DRY-RUN] {total_jobs} jobs would be submitted "
              f"({len(configs)} patches).")
    else:
        print(f"\nSubmitted {total_submitted}/{total_jobs} jobs "
              f"({len(configs)} patches).")
        print(f"Logs: {LOG_DIR}")
        print(f"Output: {args.output_dir}")
        print(f"\nAfter completion, merge per patch:")
        for _, _, patch in configs:
            print(f"  hadd -f data/cex/CEX23_patch{patch}_all.root "
                  f"{args.output_dir}/CEX23_patch{patch}_r*.root")


if __name__ == "__main__":
    main()
