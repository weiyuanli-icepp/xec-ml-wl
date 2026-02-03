#!/usr/bin/env python3
"""
Check ROOT files for invalid data values (NaN, inf, extreme values).

Usage:
    python macro/check_root_data_validity.py /path/to/data/directory
    python macro/check_root_data_validity.py /path/to/data/directory --tree tree --verbose
    python macro/check_root_data_validity.py /path/to/file.root  # single file
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

try:
    import uproot
except ImportError:
    print("Error: uproot not installed. Run: pip install uproot")
    sys.exit(1)


def check_single_file(
    filepath: str,
    tree_name: str = "tree",
    npho_branch: str = "npho",
    time_branch: str = "relative_time",
    verbose: bool = False,
) -> Dict:
    """
    Check a single ROOT file for invalid values.

    Returns:
        dict with file statistics and issues found
    """
    result = {
        "filepath": filepath,
        "valid": True,
        "issues": [],
        "stats": {},
    }

    try:
        with uproot.open(filepath) as root_file:
            # Find the tree
            if tree_name not in root_file:
                # Try to find any tree
                trees = [k for k in root_file.keys() if "tree" in k.lower()]
                if trees:
                    tree_name = trees[0].split(";")[0]
                else:
                    result["valid"] = False
                    result["issues"].append(f"Tree '{tree_name}' not found")
                    return result

            tree = root_file[tree_name]
            n_events = tree.num_entries
            result["stats"]["n_events"] = n_events

            if n_events == 0:
                result["issues"].append("Empty file (0 events)")
                return result

            # Check npho branch
            if npho_branch in tree:
                npho = tree[npho_branch].array(library="np")
                npho_stats = analyze_array(npho, "npho")
                result["stats"]["npho"] = npho_stats

                if npho_stats["n_nan"] > 0:
                    result["valid"] = False
                    result["issues"].append(f"npho has {npho_stats['n_nan']} NaN values")
                if npho_stats["n_inf"] > 0:
                    result["valid"] = False
                    result["issues"].append(f"npho has {npho_stats['n_inf']} inf values")
                if npho_stats["min"] < -1e-6:  # npho should be non-negative
                    result["issues"].append(f"npho has negative values (min={npho_stats['min']:.6f})")
                if npho_stats["max"] > 1e8:  # Sanity check for extreme values
                    result["issues"].append(f"npho has very large values (max={npho_stats['max']:.2e})")
            else:
                result["issues"].append(f"Branch '{npho_branch}' not found")

            # Check time branch
            if time_branch in tree:
                time = tree[time_branch].array(library="np")
                time_stats = analyze_array(time, "time")
                result["stats"]["time"] = time_stats

                if time_stats["n_nan"] > 0:
                    result["valid"] = False
                    result["issues"].append(f"time has {time_stats['n_nan']} NaN values")
                if time_stats["n_inf"] > 0:
                    result["valid"] = False
                    result["issues"].append(f"time has {time_stats['n_inf']} inf values")
            else:
                result["issues"].append(f"Branch '{time_branch}' not found")

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading file: {str(e)}")

    return result


def analyze_array(arr: np.ndarray, name: str) -> Dict:
    """Analyze a numpy array for statistics and invalid values."""
    # Flatten if multidimensional
    flat = arr.flatten() if arr.ndim > 1 else arr

    n_nan = np.isnan(flat).sum()
    n_inf = np.isinf(flat).sum()

    # Get stats excluding nan/inf
    valid_mask = np.isfinite(flat)
    valid_data = flat[valid_mask]

    if len(valid_data) > 0:
        stats = {
            "n_total": len(flat),
            "n_nan": int(n_nan),
            "n_inf": int(n_inf),
            "n_valid": len(valid_data),
            "min": float(np.min(valid_data)),
            "max": float(np.max(valid_data)),
            "mean": float(np.mean(valid_data)),
            "std": float(np.std(valid_data)),
            "percentile_1": float(np.percentile(valid_data, 1)),
            "percentile_99": float(np.percentile(valid_data, 99)),
        }
    else:
        stats = {
            "n_total": len(flat),
            "n_nan": int(n_nan),
            "n_inf": int(n_inf),
            "n_valid": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "std": float("nan"),
            "percentile_1": float("nan"),
            "percentile_99": float("nan"),
        }

    return stats


def find_root_files(path: str) -> List[str]:
    """Find all ROOT files in directory (or return single file)."""
    p = Path(path).expanduser()

    if p.is_file():
        return [str(p)]
    elif p.is_dir():
        files = sorted(p.glob("**/*.root"))
        return [str(f) for f in files]
    else:
        raise ValueError(f"Path does not exist: {path}")


def print_summary(results: List[Dict], verbose: bool = False):
    """Print summary of all results."""
    total = len(results)
    valid = sum(1 for r in results if r["valid"] and not r["issues"])
    with_issues = [r for r in results if r["issues"]]
    invalid = [r for r in results if not r["valid"]]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files checked: {total}")
    print(f"  Valid (no issues): {valid}")
    print(f"  With warnings:     {len(with_issues) - len(invalid)}")
    print(f"  Invalid (errors):  {len(invalid)}")

    # Aggregate statistics
    all_npho_stats = [r["stats"].get("npho", {}) for r in results if "npho" in r["stats"]]
    all_time_stats = [r["stats"].get("time", {}) for r in results if "time" in r["stats"]]

    if all_npho_stats:
        print("\n--- Aggregate npho statistics ---")
        total_nan = sum(s.get("n_nan", 0) for s in all_npho_stats)
        total_inf = sum(s.get("n_inf", 0) for s in all_npho_stats)
        all_min = min(s.get("min", float("inf")) for s in all_npho_stats if s.get("n_valid", 0) > 0)
        all_max = max(s.get("max", float("-inf")) for s in all_npho_stats if s.get("n_valid", 0) > 0)
        print(f"  Total NaN values: {total_nan}")
        print(f"  Total inf values: {total_inf}")
        print(f"  Global min: {all_min:.6f}")
        print(f"  Global max: {all_max:.2e}")

    if all_time_stats:
        print("\n--- Aggregate time statistics ---")
        total_nan = sum(s.get("n_nan", 0) for s in all_time_stats)
        total_inf = sum(s.get("n_inf", 0) for s in all_time_stats)
        all_min = min(s.get("min", float("inf")) for s in all_time_stats if s.get("n_valid", 0) > 0)
        all_max = max(s.get("max", float("-inf")) for s in all_time_stats if s.get("n_valid", 0) > 0)
        print(f"  Total NaN values: {total_nan}")
        print(f"  Total inf values: {total_inf}")
        print(f"  Global min: {all_min:.6e}")
        print(f"  Global max: {all_max:.6e}")

    # List problematic files
    if invalid:
        print("\n" + "=" * 70)
        print("INVALID FILES (training will likely fail with these)")
        print("=" * 70)
        for r in invalid:
            print(f"\n  {r['filepath']}")
            for issue in r["issues"]:
                print(f"    - {issue}")

    if with_issues and verbose:
        print("\n" + "=" * 70)
        print("FILES WITH WARNINGS")
        print("=" * 70)
        for r in with_issues:
            if r["valid"]:  # Only warnings, not errors
                print(f"\n  {r['filepath']}")
                for issue in r["issues"]:
                    print(f"    - {issue}")


def main():
    parser = argparse.ArgumentParser(
        description="Check ROOT files for invalid data values (NaN, inf, etc.)"
    )
    parser.add_argument(
        "path",
        help="Path to ROOT file or directory containing ROOT files"
    )
    parser.add_argument(
        "--tree", "-t",
        default="tree",
        help="Tree name (default: tree)"
    )
    parser.add_argument(
        "--npho-branch",
        default="npho",
        help="Npho branch name (default: npho)"
    )
    parser.add_argument(
        "--time-branch",
        default="relative_time",
        help="Time branch name (default: relative_time)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed info for each file"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop checking after first invalid file"
    )

    args = parser.parse_args()

    # Find files
    try:
        files = find_root_files(args.path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not files:
        print(f"No ROOT files found in {args.path}")
        sys.exit(1)

    print(f"Found {len(files)} ROOT file(s) to check")
    print("=" * 70)

    results = []
    for i, filepath in enumerate(files):
        print(f"[{i+1}/{len(files)}] Checking: {Path(filepath).name}", end="")
        sys.stdout.flush()

        result = check_single_file(
            filepath,
            tree_name=args.tree,
            npho_branch=args.npho_branch,
            time_branch=args.time_branch,
            verbose=args.verbose,
        )
        results.append(result)

        # Print status
        if result["valid"] and not result["issues"]:
            print(" ✓")
        elif result["valid"]:
            print(f" ⚠ ({len(result['issues'])} warning(s))")
        else:
            print(f" ✗ INVALID")
            for issue in result["issues"]:
                print(f"      {issue}")

        if args.verbose and "npho" in result["stats"]:
            s = result["stats"]["npho"]
            print(f"      npho: [{s['min']:.2f}, {s['max']:.2e}], mean={s['mean']:.2f}")
        if args.verbose and "time" in result["stats"]:
            s = result["stats"]["time"]
            print(f"      time: [{s['min']:.2e}, {s['max']:.2e}], mean={s['mean']:.2e}")

        if args.stop_on_error and not result["valid"]:
            print("\nStopping due to --stop-on-error flag")
            break

    # Print summary
    print_summary(results, verbose=args.verbose)

    # Exit with error code if any invalid files
    invalid_count = sum(1 for r in results if not r["valid"])
    sys.exit(1 if invalid_count > 0 else 0)


if __name__ == "__main__":
    main()
