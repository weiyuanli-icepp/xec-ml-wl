#!/usr/bin/env python3
"""Fix MLflow epoch numbering for resume runs that used reset_epoch=true.

When a training run resumes with reset_epoch=true, epochs 1-50 are logged
again in the same MLflow run, creating duplicate step numbers. This script
shifts the duplicate metrics by +50 so they appear as epochs 51-100.

Usage:
    python macro/fix_mlflow_resume_epochs.py --run-name scan_s2_ema
    python macro/fix_mlflow_resume_epochs.py --run-name scan_s3a_model768
    python macro/fix_mlflow_resume_epochs.py --run-name scan_s3b_model1024
    python macro/fix_mlflow_resume_epochs.py --run-id <mlflow_run_id>

    # Dry run (preview without modifying):
    python macro/fix_mlflow_resume_epochs.py --run-name scan_s2_ema --dry-run
"""

import argparse
import os

import mlflow


def find_run_by_name(experiment_name, run_name):
    """Find MLflow run by experiment and run name."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
    )
    if not runs:
        raise ValueError(
            f"No run with name '{run_name}' in experiment '{experiment_name}'"
        )
    if len(runs) > 1:
        print(f"[WARN] Found {len(runs)} runs with name '{run_name}', using most recent")
    return runs[0].info.run_id


def fix_resume_epochs(run_id, epoch_offset=50, dry_run=False):
    """Shift duplicate epoch metrics by offset in an MLflow run.

    For each metric, if there are duplicate step numbers (from resume with
    reset_epoch=true), the later entries (by timestamp) are shifted by
    epoch_offset.
    """
    client = mlflow.tracking.MlflowClient()

    run = client.get_run(run_id)
    print(f"Run ID: {run_id}")
    print(f"Run name: {run.data.tags.get('mlflow.runName', 'N/A')}")
    print(f"Epoch offset: +{epoch_offset}")
    print(f"Dry run: {dry_run}")
    print()

    # Get all metric keys from the run
    metric_keys = list(run.data.metrics.keys())
    print(f"Found {len(metric_keys)} metric keys: {metric_keys}")
    print()

    total_shifted = 0

    for key in metric_keys:
        history = client.get_metric_history(run_id, key)

        # Group by step to find duplicates
        step_entries = {}
        for m in history:
            step_entries.setdefault(m.step, []).append(m)

        # Find steps with duplicates
        duplicates = {s: entries for s, entries in step_entries.items()
                      if len(entries) > 1}

        if not duplicates:
            continue

        print(f"  {key}: {len(duplicates)} duplicate steps found")

        # For each duplicate step, shift the later entry (by timestamp)
        for step, entries in sorted(duplicates.items()):
            entries_sorted = sorted(entries, key=lambda m: m.timestamp)
            # Keep the first (original), shift the rest (resume)
            for m in entries_sorted[1:]:
                new_step = m.step + epoch_offset
                if dry_run:
                    print(f"    [DRY] {key} step {m.step} -> {new_step} "
                          f"(value={m.value:.6e})")
                else:
                    client.log_metric(
                        run_id, key, m.value,
                        step=new_step, timestamp=m.timestamp,
                    )
                total_shifted += 1

    print()
    if dry_run:
        print(f"[DRY RUN] Would shift {total_shifted} metric entries by +{epoch_offset}")
    else:
        print(f"Shifted {total_shifted} metric entries by +{epoch_offset}")
        if total_shifted > 0:
            print("[NOTE] The original duplicate entries still exist at the old step numbers.")
            print("       MLflow does not support deleting individual metric entries.")
            print("       The shifted entries will show as the latest at the new step numbers.")


def main():
    parser = argparse.ArgumentParser(
        description="Fix MLflow epoch numbering for resume runs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run-id", type=str, help="MLflow run ID")
    group.add_argument("--run-name", type=str, help="MLflow run name to search for")
    parser.add_argument("--experiment", type=str, default="gamma_energy",
                        help="MLflow experiment name (default: gamma_energy)")
    parser.add_argument("--offset", type=int, default=50,
                        help="Epoch offset to add (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview changes without modifying MLflow")
    args = parser.parse_args()

    # Set tracking URI to SQLite database (same as training scripts)
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        db_path = os.path.join(os.getcwd(), "mlruns.db")
        tracking_uri = f"sqlite:///{db_path}"
        mlflow.set_tracking_uri(tracking_uri)
        print(f"MLflow tracking URI: {tracking_uri}")

    if args.run_id:
        run_id = args.run_id
    else:
        run_id = find_run_by_name(args.experiment, args.run_name)

    fix_resume_epochs(run_id, epoch_offset=args.offset, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
