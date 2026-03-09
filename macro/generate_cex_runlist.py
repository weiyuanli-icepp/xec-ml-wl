#!/usr/bin/env python3
"""
Generate CEX (charge exchange) run lists from the MEG2 database.

Queries RunCatalog for Pi0 CEX runs, extracts patch numbers from
RunDescription, and produces a runlist grouped by patch.

Known CEX run ranges:
  2022: 477126 (patch 9) - 480727 (patch 13)
  2023: 557628 (patch 9) - 560366 (patch 13)

Usage:
    python macro/generate_cex_runlist.py                    # Both years
    python macro/generate_cex_runlist.py --year 2023        # 2023 only
    python macro/generate_cex_runlist.py --dry-run          # Preview
    python macro/generate_cex_runlist.py --check            # Test DB

Requirements:
    pip install pymysql
"""

import os
import sys
import re
import argparse
from collections import defaultdict

# Add repo root to path
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

from lib.db_utils import _run_mysql_query, DEFAULT_DATABASE, check_connection

# Known CEX run ranges
CEX_RANGES = {
    2022: (477126, 480727),
    2023: (557628, 560366),
}

# Regex to extract patch number from RunDescription
PATCH_RE = re.compile(r"patch\s+number\s+(\d+)", re.IGNORECASE)


def get_cex_runs(min_run, max_run):
    """
    Query RunCatalog for CEX runs in the given range.

    Returns list of (run_number, RunDescription) tuples.
    """
    query = (
        f"SELECT id, RunDescription "
        f"FROM RunCatalog "
        f"WHERE id >= {min_run} AND id <= {max_run} "
        f"AND RunDescription LIKE '%CEX%' "
        f"ORDER BY id"
    )
    rows = _run_mysql_query(query, database=DEFAULT_DATABASE)
    print(f"[INFO] Found {len(rows)} CEX runs in range [{min_run}, {max_run}]")
    return rows


def parse_patch_number(description):
    """Extract patch number from RunDescription string."""
    if not description:
        return None
    m = PATCH_RE.search(str(description))
    return int(m.group(1)) if m else None


def group_by_patch(rows):
    """
    Group CEX runs by patch number.

    Returns {patch_number: [run_numbers]} sorted by patch and run.
    Runs with unparseable patch numbers are collected under patch=None.
    """
    by_patch = defaultdict(list)
    n_unparsed = 0
    for run_id, description in rows:
        patch = parse_patch_number(description)
        if patch is None:
            n_unparsed += 1
        by_patch[patch].append(int(run_id))

    # Sort runs within each patch
    for patch in by_patch:
        by_patch[patch].sort()

    n_patches = len([p for p in by_patch if p is not None])
    n_total = sum(len(v) for v in by_patch.values())
    print(f"[INFO] {n_patches} patches, {n_total} runs total"
          + (f" ({n_unparsed} with unparsed patch)" if n_unparsed else ""))
    return by_patch


def format_runlist(by_patch, year_label):
    """Format a CEX runlist grouped by patch."""
    output = []
    output.append(f"# CEX run list ({year_label})")
    n_total = sum(len(v) for v in by_patch.values())
    output.append(f"# Total: {n_total} runs across {len(by_patch)} patches")
    output.append(f"# Format: <run_number>")
    output.append("")

    for patch in sorted(by_patch.keys(), key=lambda p: (p is None, p)):
        runs = by_patch[patch]
        label = f"patch {patch}" if patch is not None else "unknown patch"
        output.append(f"# {label} ({len(runs)} runs)")
        for run in runs:
            output.append(str(run))
        output.append("")

    return "\n".join(output)


def format_summary(by_patch, year_label):
    """Format a summary table of patches."""
    output = []
    output.append(f"# CEX patch summary ({year_label})")
    output.append(f"# Format: <patch> <n_runs> <first_run> <last_run>")
    output.append("")

    for patch in sorted(by_patch.keys(), key=lambda p: (p is None, p)):
        runs = by_patch[patch]
        label = f"{patch:>4d}" if patch is not None else " ???"
        output.append(f"{label}  {len(runs):>4d}  {runs[0]:>6d}  {runs[-1]:>6d}")

    return "\n".join(output) + "\n"


def main():
    parser = argparse.ArgumentParser(
        description="Generate CEX run lists from MEG2 database"
    )
    parser.add_argument("--year", type=int, default=None, choices=[2022, 2023],
                        help="CEX year (default: both 2022 and 2023)")
    parser.add_argument("--min-run", type=int, default=None,
                        help="Override minimum run number")
    parser.add_argument("--max-run", type=int, default=None,
                        help="Override maximum run number")
    parser.add_argument("--output-dir", "-o", type=str, default="data/cex",
                        help="Output directory (default: data/cex)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout without writing files")
    parser.add_argument("--check", action="store_true",
                        help="Check database connection and exit")
    args = parser.parse_args()

    if args.check:
        check_connection()
        sys.exit(0)

    # Determine which years to query
    if args.year:
        years = [args.year]
    else:
        years = sorted(CEX_RANGES.keys())

    all_by_patch = defaultdict(list)

    for year in years:
        min_run = args.min_run or CEX_RANGES[year][0]
        max_run = args.max_run or CEX_RANGES[year][1]

        print(f"\n--- CEX {year} ---")
        rows = get_cex_runs(min_run, max_run)
        if not rows:
            print(f"[WARN] No CEX runs found for {year}")
            continue

        by_patch = group_by_patch(rows)

        # Print summary
        print(format_summary(by_patch, str(year)))

        if not args.dry_run:
            os.makedirs(args.output_dir, exist_ok=True)
            # Per-year runlist
            runlist_path = os.path.join(args.output_dir, f"cex{year}_runlist.txt")
            with open(runlist_path, 'w') as f:
                f.write(format_runlist(by_patch, str(year)))
            print(f"[INFO] Written: {runlist_path}")
        else:
            print(format_runlist(by_patch, str(year)))

        # Merge into combined
        for patch, runs in by_patch.items():
            all_by_patch[patch].extend(runs)

    # Combined output (if multiple years)
    if len(years) > 1 and all_by_patch:
        # Sort runs within each patch
        for patch in all_by_patch:
            all_by_patch[patch].sort()

        label = f"{years[0]}-{years[-1]}"
        print(f"\n--- Combined {label} ---")
        print(format_summary(all_by_patch, label))

        if not args.dry_run:
            combined_path = os.path.join(args.output_dir, "cex_runlist.txt")
            with open(combined_path, 'w') as f:
                f.write(format_runlist(all_by_patch, label))
            print(f"[INFO] Written: {combined_path}")


if __name__ == "__main__":
    main()
