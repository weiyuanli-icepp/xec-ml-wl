#!/usr/bin/env python3
"""
Generate a run list for PrepareRealData by sampling one run per day from the MEG2 database.

Queries RunCatalog for physics runs in a given run range, groups by date,
picks one run per day, checks that the rec file exists on disk, and writes
a runlist compatible with macro/submit_prepare_realdata.sh.

Usage:
    # Sample one run per day from runs 430000-560000
    python macro/generate_realdata_runlist.py --min-run 430000 --max-run 560000

    # Custom output and rec file suffix
    python macro/generate_realdata_runlist.py --min-run 430000 --max-run 560000 \
        --output data/real_data/runlist.txt --suffix "_open.root"

    # Dry run (print to stdout, don't write file)
    python macro/generate_realdata_runlist.py --min-run 430000 --max-run 560000 --dry-run

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
    Query RunCatalog for runs in the given range, returning (run_number, date).

    First discovers the table schema to find the date column, then queries.
    """
    # Discover RunCatalog columns to find the date/time column
    schema_rows = _run_mysql_query("DESCRIBE RunCatalog", database=DEFAULT_DATABASE)
    columns = [row[0].lower() for row in schema_rows]
    print(f"[INFO] RunCatalog columns: {[row[0] for row in schema_rows]}")

    # Find the best date column
    date_col = None
    for candidate in ["start_time", "starttime", "start", "date", "timestamp",
                       "stop_time", "stoptime", "time"]:
        if candidate in columns:
            # Get the original-case name
            date_col = schema_rows[columns.index(candidate)][0]
            break

    if date_col is None:
        print("[ERROR] No date/time column found in RunCatalog.")
        print(f"  Available columns: {[row[0] for row in schema_rows]}")
        sys.exit(1)

    print(f"[INFO] Using date column: {date_col}")

    query = (
        f"SELECT id, DATE({date_col}) AS run_date "
        f"FROM RunCatalog "
        f"WHERE id >= {min_run} AND id <= {max_run} "
        f"AND {date_col} IS NOT NULL "
        f"ORDER BY id"
    )

    rows = _run_mysql_query(query, database=DEFAULT_DATABASE)
    print(f"[INFO] Found {len(rows)} runs in range [{min_run}, {max_run}]")
    return rows


def sample_one_per_day(rows):
    """Group runs by date and pick one per day (middle of the day's runs)."""
    by_date = defaultdict(list)
    for run_id, run_date in rows:
        if run_date and run_date not in ('NULL', ''):
            by_date[run_date].append(int(run_id))

    sampled = []
    for date in sorted(by_date.keys()):
        runs = sorted(by_date[date])
        # Pick the middle run of each day
        mid = len(runs) // 2
        sampled.append((runs[mid], date))

    print(f"[INFO] Sampled {len(sampled)} runs (1 per day) from {len(by_date)} days")
    return sampled


def find_rec_file(run_number, suffix="_open.root"):
    """Construct the expected rec file path and check if it exists."""
    run_dir = f"{run_number // 1000}xxx"
    filename = f"rec{run_number:06d}{suffix}"
    path = os.path.join(REC_DIR, run_dir, filename)

    # Also try without _open suffix
    if not os.path.exists(path) and suffix == "_open.root":
        alt = os.path.join(REC_DIR, run_dir, f"rec{run_number:06d}.root")
        if os.path.exists(alt):
            return alt

    if os.path.exists(path):
        return path
    return None


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
    parser.add_argument("--suffix", type=str, default="_open.root",
                        help="Rec file suffix (default: _open.root)")
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

    # Sample one run per day
    sampled = sample_one_per_day(rows)

    # Find rec files on disk
    lines = []
    n_found = 0
    n_missing = 0
    for run_number, date in sampled:
        if args.no_file_check:
            # Construct path without checking existence
            run_dir = f"{run_number // 1000}xxx"
            path = os.path.join(REC_DIR, run_dir, f"rec{run_number:06d}{args.suffix}")
            lines.append((run_number, path, date))
            n_found += 1
        else:
            path = find_rec_file(run_number, args.suffix)
            if path:
                lines.append((run_number, path, date))
                n_found += 1
            else:
                n_missing += 1

    print(f"[INFO] Rec files found: {n_found}, missing: {n_missing}")

    if not lines:
        print("[ERROR] No valid runs with rec files found.")
        sys.exit(1)

    # Format output
    output_lines = []
    output_lines.append(f"# Real data runlist: one run per day")
    output_lines.append(f"# Run range: {args.min_run} - {args.max_run}")
    output_lines.append(f"# Total: {len(lines)} runs")
    output_lines.append(f"# Format: <run_number> <rec_file_path>")
    output_lines.append("")

    prev_date = None
    for run_number, path, date in lines:
        # Add date comment when day changes
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
