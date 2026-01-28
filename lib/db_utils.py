#!/usr/bin/env python3
"""
Database utilities for MEG II XEC ML.

Provides functions to query the MEG2 database for dead channel information.
Uses subprocess to call mysql CLI with --login-path for authentication.

Usage:
    from lib.db_utils import get_dead_channels

    dead_channels = get_dead_channels(run_number=430000)
    print(f"Dead channels: {dead_channels}")
"""

import subprocess
import numpy as np
from typing import List, Optional, Tuple
from functools import lru_cache


# Database connection settings
DEFAULT_LOGIN_PATH = "meg_ro"
DEFAULT_DATABASE = "MEG2"


def _run_mysql_query(query: str, login_path: str = DEFAULT_LOGIN_PATH,
                     database: str = DEFAULT_DATABASE) -> List[Tuple]:
    """
    Execute a MySQL query using the mysql CLI with --login-path.

    Args:
        query: SQL query string
        login_path: MySQL login path name (configured via mysql_config_editor)
        database: Database name

    Returns:
        List of tuples, each tuple is a row from the result

    Raises:
        RuntimeError: If mysql command fails
    """
    cmd = [
        "mysql",
        f"--login-path={login_path}",
        database,
        "-N",  # No column names
        "-B",  # Batch mode (tab-separated)
        "-e", query
    ]

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,  # text=True equivalent for Python 3.6
            check=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            "MySQL query failed:\n"
            "  Command: {}\n"
            "  Error: {}".format(' '.join(cmd), e.stderr)
        )
    except FileNotFoundError:
        raise RuntimeError(
            "mysql command not found. Please ensure MySQL client is installed and in PATH."
        )

    # Parse output
    rows = []
    for line in result.stdout.strip().split('\n'):
        if line:
            rows.append(tuple(line.split('\t')))

    return rows


def get_xec_pm_status_id(run_number: int, login_path: str = DEFAULT_LOGIN_PATH,
                         database: str = DEFAULT_DATABASE) -> Optional[int]:
    """
    Get XECPMStatus_id for a given run number by traversing the database hierarchy.

    Database hierarchy:
        RunCatalog (run id) → XECConf_id
        XECConf (id) → XECPMStatusDB_id
        XECPMStatusDB (id) → XECPMStatus_id

    Args:
        run_number: Run number to look up
        login_path: MySQL login path
        database: Database name

    Returns:
        XECPMStatus_id or None if not found
    """
    # Step 1: RunCatalog → XECConf_id
    query = f"SELECT XECConf_id FROM RunCatalog WHERE id = {run_number}"
    rows = _run_mysql_query(query, login_path, database)

    if not rows or rows[0][0] == 'NULL' or rows[0][0] == '':
        print(f"[WARNING] No XECConf_id found for run {run_number}")
        return None

    xec_conf_id = int(rows[0][0])

    # Step 2: XECConf → XECPMStatusDB_id
    query = f"SELECT XECPMStatusDB_id FROM XECConf WHERE id = {xec_conf_id}"
    rows = _run_mysql_query(query, login_path, database)

    if not rows or rows[0][0] == 'NULL' or rows[0][0] == '':
        print(f"[WARNING] No XECPMStatusDB_id found for XECConf_id {xec_conf_id}")
        return None

    xec_pm_status_db_id = int(rows[0][0])

    # Step 3: XECPMStatusDB → XECPMStatus_id
    query = f"SELECT XECPMStatus_id FROM XECPMStatusDB WHERE id = {xec_pm_status_db_id}"
    rows = _run_mysql_query(query, login_path, database)

    if not rows or rows[0][0] == 'NULL' or rows[0][0] == '':
        print(f"[WARNING] No XECPMStatus_id found for XECPMStatusDB_id {xec_pm_status_db_id}")
        return None

    return int(rows[0][0])


@lru_cache(maxsize=128)
def get_dead_channels(run_number: int, login_path: str = DEFAULT_LOGIN_PATH,
                      database: str = DEFAULT_DATABASE) -> np.ndarray:
    """
    Get list of dead channel indices for a given run number.

    Args:
        run_number: Run number to look up
        login_path: MySQL login path
        database: Database name

    Returns:
        NumPy array of dead channel indices (0-4759)
    """
    xec_pm_status_id = get_xec_pm_status_id(run_number, login_path, database)

    if xec_pm_status_id is None:
        print(f"[WARNING] Could not find PM status for run {run_number}, returning empty array")
        return np.array([], dtype=np.int32)

    # Query dead channels (IsBad = 1)
    query = f"""
        SELECT idx FROM XECPMStatus
        WHERE id = {xec_pm_status_id} AND IsBad = 1
        ORDER BY idx
    """
    rows = _run_mysql_query(query, login_path, database)

    dead_channels = np.array([int(row[0]) for row in rows], dtype=np.int32)

    return dead_channels


def get_dead_channel_mask(run_number: int, n_channels: int = 4760,
                          login_path: str = DEFAULT_LOGIN_PATH,
                          database: str = DEFAULT_DATABASE) -> np.ndarray:
    """
    Get boolean mask indicating dead channels.

    Args:
        run_number: Run number to look up
        n_channels: Total number of channels (default: 4760)
        login_path: MySQL login path
        database: Database name

    Returns:
        Boolean NumPy array of shape (n_channels,), True for dead channels
    """
    dead_channels = get_dead_channels(run_number, login_path, database)
    mask = np.zeros(n_channels, dtype=bool)
    mask[dead_channels] = True
    return mask


def get_dead_channel_info(run_number: int, login_path: str = DEFAULT_LOGIN_PATH,
                          database: str = DEFAULT_DATABASE) -> dict:
    """
    Get detailed dead channel information for a run.

    Args:
        run_number: Run number to look up
        login_path: MySQL login path
        database: Database name

    Returns:
        Dictionary with:
            - 'run_number': int
            - 'xec_pm_status_id': int or None
            - 'dead_channels': np.ndarray of indices
            - 'n_dead': int
            - 'dead_fraction': float (n_dead / 4760)
            - 'dead_by_face': dict mapping face name to count
    """
    from .geom_defs import (
        INNER_INDEX_MAP, US_INDEX_MAP, DS_INDEX_MAP,
        OUTER_COARSE_FULL_INDEX_MAP, TOP_HEX_ROWS, BOTTOM_HEX_ROWS,
        flatten_hex_rows
    )

    xec_pm_status_id = get_xec_pm_status_id(run_number, login_path, database)
    dead_channels = get_dead_channels(run_number, login_path, database)

    # Count dead by face
    dead_set = set(dead_channels)

    def count_dead_in_face(index_map):
        if isinstance(index_map, np.ndarray):
            valid_indices = index_map[index_map >= 0].flatten()
        else:
            valid_indices = np.array(index_map)
        return sum(1 for idx in valid_indices if idx in dead_set)

    # Flatten hex rows to get sensor indices
    top_hex_flat = flatten_hex_rows(TOP_HEX_ROWS)
    bot_hex_flat = flatten_hex_rows(BOTTOM_HEX_ROWS)

    dead_by_face = {
        'inner': count_dead_in_face(INNER_INDEX_MAP),
        'us': count_dead_in_face(US_INDEX_MAP),
        'ds': count_dead_in_face(DS_INDEX_MAP),
        'outer': count_dead_in_face(OUTER_COARSE_FULL_INDEX_MAP),
        'top': count_dead_in_face(top_hex_flat),
        'bot': count_dead_in_face(bot_hex_flat),
    }

    return {
        'run_number': run_number,
        'xec_pm_status_id': xec_pm_status_id,
        'dead_channels': dead_channels,
        'n_dead': len(dead_channels),
        'dead_fraction': len(dead_channels) / 4760,
        'dead_by_face': dead_by_face,
    }


def save_dead_channel_list(run_number: int, output_path: str,
                           login_path: str = DEFAULT_LOGIN_PATH,
                           database: str = DEFAULT_DATABASE):
    """
    Save dead channel list to a text file.

    Args:
        run_number: Run number
        output_path: Output file path
        login_path: MySQL login path
        database: Database name
    """
    dead_channels = get_dead_channels(run_number, login_path, database)

    with open(output_path, 'w') as f:
        f.write(f"# Dead channels for run {run_number}\n")
        f.write(f"# Total: {len(dead_channels)} / 4760\n")
        for ch in dead_channels:
            f.write(f"{ch}\n")

    print(f"[INFO] Saved {len(dead_channels)} dead channels to {output_path}")


def load_dead_channel_list(input_path: str) -> np.ndarray:
    """
    Load dead channel list from a text file.

    Args:
        input_path: Input file path

    Returns:
        NumPy array of dead channel indices
    """
    channels = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                channels.append(int(line))
    return np.array(channels, dtype=np.int32)


def print_dead_channel_summary(run_number: int, login_path: str = DEFAULT_LOGIN_PATH,
                               database: str = DEFAULT_DATABASE):
    """
    Print a summary of dead channels for a run.

    Args:
        run_number: Run number
        login_path: MySQL login path
        database: Database name
    """
    info = get_dead_channel_info(run_number, login_path, database)

    print(f"\n{'='*50}")
    print(f"Dead Channel Summary for Run {run_number}")
    print(f"{'='*50}")
    print(f"XECPMStatus_id: {info['xec_pm_status_id']}")
    print(f"Total dead: {info['n_dead']} / 4760 ({info['dead_fraction']*100:.2f}%)")
    print(f"\nDead by face:")
    for face, count in info['dead_by_face'].items():
        print(f"  {face:>6}: {count}")
    print(f"{'='*50}\n")


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query dead channels from MEG2 database")
    parser.add_argument("run_number", type=int, help="Run number to query")
    parser.add_argument("--login-path", default=DEFAULT_LOGIN_PATH,
                        help=f"MySQL login path (default: {DEFAULT_LOGIN_PATH})")
    parser.add_argument("--database", default=DEFAULT_DATABASE,
                        help=f"Database name (default: {DEFAULT_DATABASE})")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file to save dead channel list")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all dead channel indices")

    args = parser.parse_args()

    # Print summary
    print_dead_channel_summary(args.run_number, args.login_path, args.database)

    # Optionally list all channels
    if args.list:
        dead = get_dead_channels(args.run_number, args.login_path, args.database)
        print("Dead channel indices:")
        print(dead.tolist())

    # Optionally save to file
    if args.output:
        save_dead_channel_list(args.run_number, args.output, args.login_path, args.database)
