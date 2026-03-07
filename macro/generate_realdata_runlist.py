#!/usr/bin/env python3
"""
Generate train/val run lists for PrepareRealDataInpainter from the MEG2 database.

Queries RunCatalog for physics runs (Physics=1, Junk=0, after 2021),
groups by date, and produces two runlists:
  - val:   1st found rec file on days divisible by 3 (day_index % 3 == 0)
  - train: 2nd through 201st found rec files on every day (up to 200 per day)

Usage:
    python macro/generate_realdata_runlist.py
    python macro/generate_realdata_runlist.py --dry-run
    python macro/generate_realdata_runlist.py --no-file-check
    python macro/generate_realdata_runlist.py --check

Requirements:
    pip install pymysql
"""

import os
import sys
import argparse
from collections import defaultdict

# Add repo root to path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from lib.db_utils import _run_mysql_query, DEFAULT_DATABASE, check_connection

REC_DIR = "/data/project/meg/offline/run"


def get_runs_with_dates(min_run, max_run):
    """
    Query RunCatalog for physics runs (Junk=0, Physics=1, StartTime after 2021).

    Returns list of (run_number, date_string) tuples.
    """
    conditions = [
        "Junk = 0",
        "Physics = 1",
        "StartTime >= '2021-01-01'",
    ]
    if min_run is not None:
        conditions.append(f"id >= {min_run}")
    if max_run is not None:
        conditions.append(f"id <= {max_run}")

    query = (
        f"SELECT id, DATE(StartTime) AS run_date "
        f"FROM RunCatalog "
        f"WHERE {' AND '.join(conditions)} "
        f"ORDER BY id"
    )

    rows = _run_mysql_query(query, database=DEFAULT_DATABASE)
    range_str = f"[{min_run or 'start'}, {max_run or 'end'}]"
    print(f"[INFO] Found {len(rows)} physics runs in range {range_str}")
    return rows


def group_by_date(rows):
    """Group runs by date, returning {date: [run_numbers]} sorted by date."""
    by_date = defaultdict(list)
    for run_id, run_date in rows:
        if run_date and run_date not in ('NULL', ''):
            by_date[run_date].append(int(run_id))
    # Sort runs within each day
    for date in by_date:
        by_date[date].sort()
    print(f"[INFO] {len(by_date)} unique days with physics runs")
    return by_date


def find_rec_file(run_number):
    """
    Check if a rec file exists for the given run number.

    Tries rec{run}_open.root, rec{run}_selected.root, then rec{run}.root.
    Returns the path if found, None otherwise.
    """
    run_dir = f"{run_number // 1000}xxx"
    base = os.path.join(REC_DIR, run_dir)

    for suffix in ("_open.root", "_selected.root", ".root"):
        path = os.path.join(base, f"rec{run_number:06d}{suffix}")
        if os.path.exists(path):
            return path
    return None


def find_first_n_files(runs, n):
    """Find the first n runs (in order) that have rec files on disk.

    Returns list of (run_number, path) tuples, length <= n.
    """
    found = []
    for run in runs:
        path = find_rec_file(run)
        if path:
            found.append((run, path))
            if len(found) >= n:
                break
    return found


def build_default_path(run_number):
    """Construct the default rec file path without checking existence."""
    run_dir = f"{run_number // 1000}xxx"
    return os.path.join(REC_DIR, run_dir, f"rec{run_number:06d}_open.root")


def sample_train_val(by_date, no_file_check=False, max_train_per_day=200):
    """
    Build train and val runlists from grouped runs.

    - val:   1st rec file of each day where day_index % 3 == 0
    - train: runs 2 through max_train_per_day+1 (up to 200 runs per day)

    Returns (train_list, val_list) where each is [(run, path, date), ...].
    """
    train = []
    val = []
    n_train_files = 0
    n_val_files = 0

    for day_idx, date in enumerate(sorted(by_date.keys())):
        runs = by_date[date]
        is_val_day = (day_idx % 3 == 0)

        # Need: 1st for val (on val days), runs 2..201 for train
        need = max_train_per_day + 1  # +1 because 1st is reserved for val
        if no_file_check:
            entries = [(runs[i], build_default_path(runs[i]))
                       for i in range(min(need, len(runs)))]
        else:
            entries = find_first_n_files(runs, need)

        if not entries:
            continue

        # Val: 1st file on val days
        if is_val_day:
            val.append((entries[0][0], entries[0][1], date))
            n_val_files += 1

        # Train: 2nd file onward (skip the 1st)
        for entry in entries[1:]:
            train.append((entry[0], entry[1], date))
            n_train_files += 1

    print(f"[INFO] Train: {n_train_files} runs, Val: {n_val_files} runs")
    return train, val


def format_runlist(lines, label, min_run, max_run):
    """Format a runlist as a string."""
    output = []
    output.append(f"# Real data runlist ({label}): one run per day")
    output.append(f"# Run range: {min_run or 'start'} - {max_run or 'end'}")
    output.append(f"# Filters: Physics=1, Junk=0, StartTime >= 2021")
    output.append(f"# Total: {len(lines)} runs")
    output.append(f"# Format: <run_number> <rec_file_path>")
    output.append("")

    prev_date = None
    for run_number, path, date in lines:
        if date != prev_date:
            output.append(f"# {date}")
            prev_date = date
        output.append(f"{run_number} {path}")

    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/val real-data run lists from MEG2 database"
    )
    parser.add_argument("--min-run", type=int, default=None,
                        help="Minimum run number (default: no limit)")
    parser.add_argument("--max-run", type=int, default=None,
                        help="Maximum run number (default: no limit)")
    parser.add_argument("--output-dir", "-o", type=str, default="data/real_data",
                        help="Output directory for runlist files (default: data/real_data)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print runlists to stdout without writing files")
    parser.add_argument("--check", action="store_true",
                        help="Check database connection and exit")
    parser.add_argument("--no-file-check", action="store_true",
                        help="Skip checking if rec files exist on disk")
    args = parser.parse_args()

    if args.check:
        check_connection()
        sys.exit(0)

    # Query database
    rows = get_runs_with_dates(args.min_run, args.max_run)
    if not rows:
        print("[ERROR] No runs found. Check run range and database connection.")
        sys.exit(1)

    # Group by date
    by_date = group_by_date(rows)

    # Build train/val split
    train, val = sample_train_val(by_date, no_file_check=args.no_file_check)

    if not train and not val:
        print("[ERROR] No valid runs found.")
        sys.exit(1)

    train_content = format_runlist(train, "train", args.min_run, args.max_run)
    val_content = format_runlist(val, "val", args.min_run, args.max_run)

    if args.dry_run:
        print("\n=== TRAIN ===")
        print(train_content)
        print("=== VAL ===")
        print(val_content)
    else:
        os.makedirs(args.output_dir, exist_ok=True)
        train_path = os.path.join(args.output_dir, "runlist_train.txt")
        val_path = os.path.join(args.output_dir, "runlist_val.txt")
        with open(train_path, 'w') as f:
            f.write(train_content)
        with open(val_path, 'w') as f:
            f.write(val_content)
        print(f"[INFO] Written train runlist to: {train_path}")
        print(f"[INFO] Written val runlist to: {val_path}")


if __name__ == "__main__":
    main()
