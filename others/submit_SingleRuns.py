#!/usr/bin/env python3

# Script to automate the submission of MCXECPreprocess.C jobs via Slurm.
# Note: Energy_Truth < 15 MeV events are skipped.

# Usage:
# 1  Modify the range_start, range_end, conf, and output_dir variables below.
# 2  Optionally set run_offset to shift output run numbers (e.g., 200 for runs 200-299)
# 3  Create "single_run" directory under xec-ml-wl/data/E***_Ang***_Pos***/
# 4  ./submit_SingleRuns.py [--dry-run]
#
# Example: To process E15to60_2 runs 0-99 but output as runs 200-299:
#   conf       = "E15to60_2"
#   output_dir = "../../xec-ml-wl/data/E15to60_AngUni_PosSQ/single_run"
#   run_offset = 200
#   This will read from E15to60_2/rec00000.root etc. and output MCGamma_200.root etc.
#
# If you want to randomize npho and timing, set ActivateRandomization=1 in the macro
#
# To combine single run files for training and validation, execute
#    hadd -f large_train.root single_run/MCGamma_{0..89}.root
#    hadd -f large_val.root single_run/MCGamma_{90..99}.root


import os
import argparse

range_start = 100
range_end   = 199
run_offset  = 200  # Add offset to output run number and filename (e.g., 200 for runs 200-299)

# Configuration for MCXECPreprocess
conf       = "E15to60_2"  # Input config directory name
output_dir = "../../xec-ml-wl/data/E15to60_AngUni_PosSQ/single_run"  # Output directory

main_macro_path = "/data/project/meg/shared/subprojects/xec/li_w/AngleRec/mc/anaMacro/MCXECPreprocess.C"
analyzer_dir = os.path.expandvars("$MEG2SYS/analyzer")
base_work_dir     = "/data/project/meg/shared/subprojects/xec/li_w/AngleRec/mc/anaMacro"
slurm_scripts_dir = os.path.join(base_work_dir, "slurm_scripts")
loader_macros_dir = os.path.join(base_work_dir, "loaders")
log_dir           = os.path.join(base_work_dir, "logs")
build_dir_base    = os.path.join(base_work_dir, "tmp_builds")

# --- Templates ---
# Note: runOffset is the 7th parameter of MCXECPreprocess (after RandomSeed)
loader_template = """
#include <TSystem.h>
#include <TROOT.h>
#include <TString.h>

void loader_{n}() {{
    gSystem->SetBuildDir("{build_dir}", kTRUE);
    gROOT->ProcessLine(TString::Format(".L {main_macro_path}+"));
    gROOT->ProcessLine("MCXECPreprocess({n}, {n}, \\"{conf}\\", \\"{output_dir}\\", 0, 42, {run_offset})");
}}
"""

slurm_template = """#!/bin/bash
#SBATCH --partition=mu3e
#SBATCH --ntasks=1
#SBATCH --job-name=mcxecpre_{n}
#SBATCH --output="{log_dir}/job_{n}-%A.out"
#SBATCH --error="{log_dir}/job_{n}-%A.err"
#SBATCH --account=meg
#SBATCH --time=12:00:00

echo "Job N={n} starting on host: `hostname`"
ulimit -c 0
cd {analyzer_dir}
./meganalyzer -I '{loader_macro_path}()' -b -q
RET=$?
rm -rf {build_dir}
echo "Job finished with exit code: $RET"
"""

def main():
    parser = argparse.ArgumentParser(description="Submit MCXECPreprocess jobs.")
    parser.add_argument("--dry-run", action="store_true", help="Generate files but do not submit.")
    args = parser.parse_args()

    # Create necessary directories
    for d in [slurm_scripts_dir, loader_macros_dir, log_dir, build_dir_base]:
        os.makedirs(d, exist_ok=True)

    # Check for meganalyzer
    meganalyzer_exec = os.path.join(analyzer_dir, "meganalyzer")
    if not os.path.isfile(meganalyzer_exec):
        print(f"WARNING: meganalyzer not found at {meganalyzer_exec}")
        print("Please check the 'analyzer_dir' variable in the script.")

    print(f"Preparing jobs for N = {range_start} to {range_end}...")
    print(f"  Input config: {conf}")
    print(f"  Output dir:   {output_dir}")
    if run_offset != 0:
        print(f"  Run offset:   {run_offset} (output runs {range_start + run_offset}-{range_end + run_offset})")

    submit_count = 0

    for n in range(range_start, range_end + 1):

        # 1. Setup paths for this specific job
        unique_build_dir = os.path.join(build_dir_base, f"build_{n}")
        loader_filename  = os.path.join(loader_macros_dir, f"loader_{n}.C")
        slurm_filename   = os.path.join(slurm_scripts_dir, f"submit_{n}.sl")

        # 2. Write the Loader Macro
        loader_content = loader_template.format(
            n=n,
            build_dir=unique_build_dir,
            main_macro_path=main_macro_path,
            conf=conf,
            output_dir=output_dir,
            run_offset=run_offset
        )
        with open(loader_filename, "w") as f:
            f.write(loader_content)

        # 3. Write the Slurm Script
        slurm_content = slurm_template.format(
            n=n,
            log_dir=log_dir,
            analyzer_dir=analyzer_dir,
            loader_macro_path=os.path.abspath(loader_filename),
            build_dir=unique_build_dir
        )
        with open(slurm_filename, "w") as f:
            f.write(slurm_content)

        # 4. Submit
        if not args.dry_run:
            print(f"Submitting N={n} -> MCGamma_{n + run_offset}.root ...")
            os.system(f"sbatch {slurm_filename}")
            submit_count += 1
        else:
            print(f"Dry run: N={n} -> MCGamma_{n + run_offset}.root")

    if not args.dry_run:
        print(f"\nSuccess! {submit_count} jobs submitted.")
    else:
        print("\nDry run finished. No jobs submitted.")

if __name__ == "__main__":
    main()
