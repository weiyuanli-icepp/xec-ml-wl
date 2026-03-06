#!/usr/bin/env python3
"""
Generate a run list for PrepareRealData by sampling one run per day from the MEG2 database.

Queries RunCatalog for physics runs (Physics=1, Junk=0, after 2021),
groups by date, picks one run per day, checks that the rec file exists
on disk, and writes a runlist compatible with macro/submit_prepare_realdata.sh.

Two-pass file search:
  1. For each sampled day, try up to 5 runs to find a rec file on disk.
  2. If pass 1 finds nothing for a day, try all remaining runs for that day.

Usage:
    # Sample one run per day from runs 430000-560000
    python macro/generate_realdata_runlist.py --min-run 430000 --max-run 560000

    # Dry run (print to stdout, don't write file)
    python macro/generate_realdata_runlist.py --min-run 430000 --max-run 560000 --dry-run

    # Skip checking if rec files exist on disk
    python macro/generate_realdata_runlist.py --min-run 430000 --max-run 560000 --no-file-check

    # Check database connection first
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
    query = (
        f"SELECT id, DATE(StartTime) AS run_date "
        f"FROM RunCatalog "
        f"WHERE id >= {min_run} AND id <= {max_run} "
        f"AND Junk = 0 AND Physics = 1 "
        f"AND StartTime >= '2021-01-01' "
        f"ORDER BY id"
    )

    rows = _run_mysql_query(query, database=DEFAULT_DATABASE)
    print(f"[INFO] Found {len(rows)} physics runs in range [{min_run}, {max_run}]")
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

    Tries rec{run}_open.root first, then rec{run}.root.
    Returns the path if found, None otherwise.
    """
    run_dir = f"{run_number // 1000}xxx"
    base = os.path.join(REC_DIR, run_dir)

    for suffix in ("_open.root", ".root"):
        path = os.path.join(base, f"rec{run_number:06d}{suffix}")
        if os.path.exists(path):
            return path
    return None


def sample_runs_with_file_check(by_date):
    """
    For each day, find one run whose rec file exists on disk.

    Pass 1: Try up to 5 evenly-spaced runs per day.
    Pass 2: For days that failed pass 1, try all runs for that day.
    """
    results = []  # (run_number, path, date)
    days_missing = []

    for date in sorted(by_date.keys()):
        runs = by_date[date]

        # Pass 1: try up to 5 evenly-spaced runs
        n_try = min(5, len(runs))
        indices = [i * len(runs) // n_try + len(runs) // (2 * n_try) for i in range(n_try)]
        candidates = [runs[i] for i in indices]

        found = False
        for run in candidates:
            path = find_rec_file(run)
            if path:
                results.append((run, path, date))
                found = True
                break

        if not found:
            days_missing.append(date)

    # Pass 2: for missing days, try all runs
    n_pass2_found = 0
    for date in days_missing:
        runs = by_date[date]
        # Skip the ones already tried in pass 1
        n_try = min(5, len(runs))
        indices_tried = set(i * len(runs) // n_try + len(runs) // (2 * n_try) for i in range(n_try))
        remaining = [runs[i] for i in range(len(runs)) if i not in indices_tried]

        found = False
        for run in remaining:
            path = find_rec_file(run)
            if path:
                results.append((run, path, date))
                found = True
                n_pass2_found += 1
                break

        if not found:
            pass  # No rec file for any run on this day

    # Sort by date
    results.sort(key=lambda x: x[2])

    n_total_days = len(by_date)
    n_found = len(results)
    n_missing = n_total_days - n_found
    print(f"[INFO] File search: {n_found}/{n_total_days} days have rec files "
          f"(pass1: {n_found - n_pass2_found}, pass2: {n_pass2_found}, missing: {n_missing})")
    return results


def sample_runs_no_file_check(by_date):
    """Pick the middle run of each day without checking file existence."""
    results = []
    for date in sorted(by_date.keys()):
        runs = by_date[date]
        mid = len(runs) // 2
        run = runs[mid]
        run_dir = f"{run // 1000}xxx"
        path = os.path.join(REC_DIR, run_dir, f"rec{run:06d}_open.root")
        results.append((run, path, date))
    print(f"[INFO] Sampled {len(results)} runs (1 per day, no file check)")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate a real-data run list (one run per day) from MEG2 database"
    )
    parser.add_argument("--min-run", type=int, default=430000,
                        help="Minimum run number (default: 430000)")
    parser.add_argument("--max-run", type=int, default=560000,
                        help="Maximum run number (default: 560000)")
    parser.add_argument("--output", "-o", type=str, default="data/real_data/runlist.txt",
                        help="Output runlist file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print runlist to stdout without writing file")
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

    # Sample one run per day
    if args.no_file_check:
        lines = sample_runs_no_file_check(by_date)
    else:
        lines = sample_runs_with_file_check(by_date)

    if not lines:
        print("[ERROR] No valid runs found.")
        sys.exit(1)

    # Format output
    output_lines = []
    output_lines.append(f"# Real data runlist: one run per day")
    output_lines.append(f"# Run range: {args.min_run} - {args.max_run}")
    output_lines.append(f"# Filters: Physics=1, Junk=0, StartTime >= 2021")
    output_lines.append(f"# Total: {len(lines)} runs")
    output_lines.append(f"# Format: <run_number> <rec_file_path>")
    output_lines.append("")

    prev_date = None
    for run_number, path, date in lines:
        if date != prev_date:
            output_lines.append(f"# {date}")
            prev_date = date
        output_lines.append(f"{run_number} {path}")

    content = "\n".join(output_lines) + "\n"

    if args.dry_run:
        print("\n" + content)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, 'w') as f:
            f.write(content)
        print(f"[INFO] Written runlist to: {args.output}")


if __name__ == "__main__":
    main()
