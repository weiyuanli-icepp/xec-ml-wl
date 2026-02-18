#!/usr/bin/env python3
"""
Check ROOT files for data issues using the same preprocessing as the model.

This script applies the exact same normalization pipeline as XECStreamingDataset
and checks for NaN, inf, or extreme values that could cause training instability.

Usage:
    python macro/check_root_data_validity.py /path/to/data/directory
    python macro/check_root_data_validity.py /path/to/data/directory --verbose
    python macro/check_root_data_validity.py /path/to/file.root
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

try:
    import uproot
except ImportError:
    print("Error: uproot not installed. Run: pip install uproot")
    sys.exit(1)

# Import normalization defaults from lib
sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.geom_defs import (
    DEFAULT_NPHO_SCALE,
    DEFAULT_NPHO_SCALE2,
    DEFAULT_TIME_SCALE,
    DEFAULT_TIME_SHIFT,
    DEFAULT_SENTINEL_TIME,
    DEFAULT_NPHO_THRESHOLD,
)


def preprocess_chunk(
    raw_npho: np.ndarray,
    raw_time: np.ndarray,
    npho_scale: float,
    npho_scale2: float,
    time_scale: float,
    time_shift: float,
    sentinel_time: float,
    npho_threshold: float,
    sentinel_npho: float = -1.0,
) -> Dict:
    """
    Apply the same preprocessing as XECStreamingDataset._process_sub_chunk.

    Returns dict with preprocessed data and diagnostic info.
    """
    # Identify truly invalid npho values (same logic as dataset.py)
    mask_npho_invalid = (raw_npho > 9e9) | np.isnan(raw_npho)
    # Domain-breaking values for log1p
    domain_min = -npho_scale * 0.999
    mask_domain_break = (~mask_npho_invalid) & (raw_npho < domain_min)

    # Identify invalid time values
    mask_time_invalid = (
        mask_npho_invalid |
        (raw_npho < npho_threshold) |
        (np.abs(raw_time) > 9e9) |
        np.isnan(raw_time)
    )

    # Normalize npho: log1p transform (allow negatives through)
    raw_npho_safe = np.where(mask_npho_invalid | mask_domain_break, 0.0, raw_npho)
    npho_norm = np.log1p(raw_npho_safe / npho_scale) / npho_scale2
    npho_norm[mask_npho_invalid] = sentinel_npho  # dead channel → npho sentinel
    npho_norm[mask_domain_break] = 0.0                  # domain break → zero signal

    # Normalize time: linear transform
    time_norm = (raw_time / time_scale) - time_shift
    time_norm[mask_time_invalid] = sentinel_time

    return {
        "npho_norm": npho_norm,
        "time_norm": time_norm,
        "mask_npho_invalid": mask_npho_invalid,
        "mask_time_invalid": mask_time_invalid,
        "raw_npho": raw_npho,
        "raw_time": raw_time,
    }


def check_single_file(
    filepath: str,
    tree_name: str = "tree",
    npho_branch: str = "npho",
    time_branch: str = "relative_time",
    npho_scale: float = DEFAULT_NPHO_SCALE,
    npho_scale2: float = DEFAULT_NPHO_SCALE2,
    time_scale: float = DEFAULT_TIME_SCALE,
    time_shift: float = DEFAULT_TIME_SHIFT,
    sentinel_time: float = DEFAULT_SENTINEL_TIME,
    sentinel_npho: float = -1.0,
    npho_threshold: float = DEFAULT_NPHO_THRESHOLD,
    verbose: bool = False,
    max_events: int = None,
) -> Dict:
    """
    Check a single ROOT file using the model's preprocessing pipeline.
    """
    result = {
        "filepath": filepath,
        "valid": True,
        "issues": [],
        "warnings": [],
        "stats": {},
    }

    try:
        with uproot.open(filepath) as root_file:
            # Find the tree
            if tree_name not in root_file:
                trees = [k.split(";")[0] for k in root_file.keys() if not k.startswith("_")]
                if trees:
                    # Try common tree names
                    for candidate in ["tree", "Tree", "events", "Events"]:
                        if candidate in trees:
                            tree_name = candidate
                            break
                    else:
                        tree_name = trees[0]
                else:
                    result["valid"] = False
                    result["issues"].append(f"No trees found in file")
                    return result

            tree = root_file[tree_name]
            n_events = tree.num_entries
            result["stats"]["n_events"] = n_events
            result["stats"]["tree_name"] = tree_name

            if n_events == 0:
                result["warnings"].append("Empty file (0 events)")
                return result

            # Check branch availability
            available_branches = tree.keys()
            if npho_branch not in available_branches:
                result["valid"] = False
                result["issues"].append(f"Branch '{npho_branch}' not found. Available: {list(available_branches)[:10]}")
                return result
            if time_branch not in available_branches:
                result["valid"] = False
                result["issues"].append(f"Branch '{time_branch}' not found")
                return result

            # Load data (limit for large files)
            if max_events and n_events > max_events:
                raw_npho = tree[npho_branch].array(library="np", entry_stop=max_events)
                raw_time = tree[time_branch].array(library="np", entry_stop=max_events)
                result["stats"]["events_checked"] = max_events
            else:
                raw_npho = tree[npho_branch].array(library="np")
                raw_time = tree[time_branch].array(library="np")
                result["stats"]["events_checked"] = n_events

            # Apply preprocessing
            proc = preprocess_chunk(
                raw_npho, raw_time,
                npho_scale, npho_scale2, time_scale, time_shift,
                sentinel_time, npho_threshold,
                sentinel_npho=sentinel_npho,
            )

            # === Check for issues AFTER preprocessing ===

            # 1. Check for NaN in normalized values (excluding sentinel)
            npho_norm = proc["npho_norm"]
            time_norm = proc["time_norm"]

            # Valid (non-sentinel) positions
            valid_npho_mask = npho_norm != sentinel_npho
            valid_time_mask = time_norm != sentinel_time

            npho_valid = npho_norm[valid_npho_mask]
            time_valid = time_norm[valid_time_mask]

            # NaN after normalization (critical - causes loss NaN)
            n_nan_npho = np.isnan(npho_valid).sum()
            n_nan_time = np.isnan(time_valid).sum()

            if n_nan_npho > 0:
                result["valid"] = False
                result["issues"].append(f"NaN in normalized npho: {n_nan_npho} values")
            if n_nan_time > 0:
                result["valid"] = False
                result["issues"].append(f"NaN in normalized time: {n_nan_time} values")

            # Inf after normalization
            n_inf_npho = np.isinf(npho_valid).sum()
            n_inf_time = np.isinf(time_valid).sum()

            if n_inf_npho > 0:
                result["valid"] = False
                result["issues"].append(f"Inf in normalized npho: {n_inf_npho} values")
            if n_inf_time > 0:
                result["valid"] = False
                result["issues"].append(f"Inf in normalized time: {n_inf_time} values")

            # 2. Check for extreme values that could cause numerical issues
            if len(npho_valid) > 0:
                npho_max = np.max(npho_valid)
                npho_min = np.min(npho_valid)
                # Expected range for normalized npho: roughly [0, 3] with new scheme
                if npho_max > 10:
                    result["warnings"].append(f"Very large normalized npho: max={npho_max:.2f}")
                if npho_min < -0.1:
                    result["warnings"].append(f"Negative normalized npho: min={npho_min:.4f}")

                result["stats"]["npho_norm_min"] = float(npho_min)
                result["stats"]["npho_norm_max"] = float(npho_max)
                result["stats"]["npho_norm_mean"] = float(np.mean(npho_valid))

            if len(time_valid) > 0:
                time_max = np.max(time_valid)
                time_min = np.min(time_valid)
                # Expected range for normalized time: roughly [-2, 2]
                if time_max > 10 or time_min < -10:
                    result["warnings"].append(f"Extreme normalized time: [{time_min:.2f}, {time_max:.2f}]")

                result["stats"]["time_norm_min"] = float(time_min)
                result["stats"]["time_norm_max"] = float(time_max)
                result["stats"]["time_norm_mean"] = float(np.mean(time_valid))

            # 3. Statistics on invalid sensors
            n_invalid_npho = proc["mask_npho_invalid"].sum()
            n_invalid_time = proc["mask_time_invalid"].sum()
            total_sensors = raw_npho.size

            result["stats"]["n_invalid_npho"] = int(n_invalid_npho)
            result["stats"]["n_invalid_time"] = int(n_invalid_time)
            result["stats"]["pct_invalid_npho"] = 100 * n_invalid_npho / total_sensors
            result["stats"]["pct_invalid_time"] = 100 * n_invalid_time / total_sensors

            # High invalid rate warning
            if result["stats"]["pct_invalid_npho"] > 50:
                result["warnings"].append(f"High npho invalid rate: {result['stats']['pct_invalid_npho']:.1f}%")

            # 4. Check raw data ranges
            raw_npho_flat = proc["raw_npho"].flatten()
            valid_raw = raw_npho_flat[(raw_npho_flat < 9e9) & (raw_npho_flat > -npho_scale) & ~np.isnan(raw_npho_flat)]
            if len(valid_raw) > 0:
                result["stats"]["raw_npho_min"] = float(np.min(valid_raw))
                result["stats"]["raw_npho_max"] = float(np.max(valid_raw))
                result["stats"]["raw_npho_mean"] = float(np.mean(valid_raw))

            raw_time_flat = proc["raw_time"].flatten()
            valid_raw_t = raw_time_flat[(np.abs(raw_time_flat) < 9e9) & ~np.isnan(raw_time_flat)]
            if len(valid_raw_t) > 0:
                result["stats"]["raw_time_min"] = float(np.min(valid_raw_t))
                result["stats"]["raw_time_max"] = float(np.max(valid_raw_t))

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"Error reading file: {str(e)}")

    return result


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
    with_warnings = [r for r in results if r["warnings"]]
    invalid = [r for r in results if not r["valid"]]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files checked: {total}")
    print(f"  Valid (no issues): {valid}")
    print(f"  With warnings:     {len(with_warnings)}")
    print(f"  Invalid (errors):  {len(invalid)}")

    # Aggregate statistics
    all_stats = [r["stats"] for r in results if "npho_norm_mean" in r["stats"]]

    if all_stats:
        print("\n--- Aggregate Statistics (after preprocessing) ---")

        # Normalized npho
        all_npho_min = min(s.get("npho_norm_min", float("inf")) for s in all_stats)
        all_npho_max = max(s.get("npho_norm_max", float("-inf")) for s in all_stats)
        all_npho_mean = np.mean([s.get("npho_norm_mean", 0) for s in all_stats])
        print(f"  Normalized npho: [{all_npho_min:.3f}, {all_npho_max:.3f}], mean={all_npho_mean:.3f}")

        # Normalized time
        time_stats = [s for s in all_stats if "time_norm_mean" in s]
        if time_stats:
            all_time_min = min(s.get("time_norm_min", float("inf")) for s in time_stats)
            all_time_max = max(s.get("time_norm_max", float("-inf")) for s in time_stats)
            all_time_mean = np.mean([s.get("time_norm_mean", 0) for s in time_stats])
            print(f"  Normalized time: [{all_time_min:.3f}, {all_time_max:.3f}], mean={all_time_mean:.3f}")

        # Invalid rates
        total_invalid_npho = sum(s.get("n_invalid_npho", 0) for s in all_stats)
        total_invalid_time = sum(s.get("n_invalid_time", 0) for s in all_stats)
        total_events = sum(s.get("events_checked", 0) for s in all_stats)
        total_sensors = total_events * 4760
        if total_sensors > 0:
            print(f"  Invalid npho: {total_invalid_npho:,} ({100*total_invalid_npho/total_sensors:.2f}%)")
            print(f"  Invalid time: {total_invalid_time:,} ({100*total_invalid_time/total_sensors:.2f}%)")

    # List problematic files
    if invalid:
        print("\n" + "=" * 70)
        print("INVALID FILES (will likely cause training NaN)")
        print("=" * 70)
        for r in invalid:
            print(f"\n  {r['filepath']}")
            for issue in r["issues"]:
                print(f"    ✗ {issue}")

    if with_warnings and verbose:
        print("\n" + "=" * 70)
        print("FILES WITH WARNINGS")
        print("=" * 70)
        for r in with_warnings:
            if r["warnings"]:
                print(f"\n  {r['filepath']}")
                for warn in r["warnings"]:
                    print(f"    ⚠ {warn}")


def main():
    parser = argparse.ArgumentParser(
        description="Check ROOT files for data issues using the model's preprocessing pipeline"
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
        "--npho-scale",
        type=float,
        default=DEFAULT_NPHO_SCALE,
        help=f"Npho scale (default: {DEFAULT_NPHO_SCALE})"
    )
    parser.add_argument(
        "--npho-scale2",
        type=float,
        default=DEFAULT_NPHO_SCALE2,
        help=f"Npho scale2 (default: {DEFAULT_NPHO_SCALE2})"
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=DEFAULT_TIME_SCALE,
        help=f"Time scale (default: {DEFAULT_TIME_SCALE})"
    )
    parser.add_argument(
        "--time-shift",
        type=float,
        default=DEFAULT_TIME_SHIFT,
        help=f"Time shift (default: {DEFAULT_TIME_SHIFT})"
    )
    parser.add_argument(
        "--sentinel",
        type=float,
        default=DEFAULT_SENTINEL_TIME,
        help=f"Sentinel value (default: {DEFAULT_SENTINEL_TIME})"
    )
    parser.add_argument(
        "--npho-threshold",
        type=float,
        default=DEFAULT_NPHO_THRESHOLD,
        help=f"Npho threshold for valid timing (default: {DEFAULT_NPHO_THRESHOLD})"
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=10000,
        help="Max events to check per file (default: 10000, 0=all)"
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
    print(f"Using normalization: npho_scale={args.npho_scale}, npho_scale2={args.npho_scale2}")
    print(f"                     time_scale={args.time_scale:.2e}, time_shift={args.time_shift}")
    print(f"                     sentinel={args.sentinel}, npho_threshold={args.npho_threshold}")
    if args.max_events:
        print(f"Checking up to {args.max_events} events per file")
    print("=" * 70)

    results = []
    for i, filepath in enumerate(files):
        print(f"[{i+1}/{len(files)}] {Path(filepath).name}", end="")
        sys.stdout.flush()

        result = check_single_file(
            filepath,
            tree_name=args.tree,
            npho_branch=args.npho_branch,
            time_branch=args.time_branch,
            npho_scale=args.npho_scale,
            npho_scale2=args.npho_scale2,
            time_scale=args.time_scale,
            time_shift=args.time_shift,
            sentinel_time=args.sentinel,
            npho_threshold=args.npho_threshold,
            verbose=args.verbose,
            max_events=args.max_events if args.max_events > 0 else None,
        )
        results.append(result)

        # Print status
        if result["valid"] and not result["warnings"]:
            print(" ✓")
        elif result["valid"]:
            print(f" ⚠ ({len(result['warnings'])} warning(s))")
        else:
            print(f" ✗ INVALID")
            for issue in result["issues"]:
                print(f"      {issue}")

        if args.verbose and "npho_norm_mean" in result["stats"]:
            s = result["stats"]
            print(f"      npho_norm: [{s['npho_norm_min']:.3f}, {s['npho_norm_max']:.3f}], mean={s['npho_norm_mean']:.3f}")
            if "time_norm_mean" in s:
                print(f"      time_norm: [{s['time_norm_min']:.3f}, {s['time_norm_max']:.3f}], mean={s['time_norm_mean']:.3f}")
            print(f"      invalid: npho={s['pct_invalid_npho']:.1f}%, time={s['pct_invalid_time']:.1f}%")

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
