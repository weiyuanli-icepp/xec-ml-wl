#!/usr/bin/env python3
"""
Database utilities for MEG II XEC ML.

Provides functions to query the MEG2 database for dead channel information.

Authentication methods (in order of priority):
1. Environment variables: MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD
2. MySQL CLI with --login-path (requires mysql client with login-path support)

Usage:
    # Set environment variables
    export MYSQL_HOST="your_host"
    export MYSQL_USER="your_user"
    export MYSQL_PASSWORD="your_password"

    # Then use in Python
    from lib.db_utils import get_dead_channels
    dead_channels = get_dead_channels(run_number=430000)
"""

import os
import subprocess
import numpy as np
from typing import List, Optional, Tuple
from functools import lru_cache


# Database connection settings
DEFAULT_LOGIN_PATH = "meg_ro"
DEFAULT_DATABASE = "MEG2"
DEFAULT_HOST = os.environ.get("MYSQL_HOST", "")
DEFAULT_USER = os.environ.get("MYSQL_USER", "")
DEFAULT_PASSWORD = os.environ.get("MYSQL_PASSWORD", "")

# Try to import pymysql
_HAS_PYMYSQL = False
try:
    import pymysql
    _HAS_PYMYSQL = True
except ImportError:
    pass


def _run_mysql_query_pymysql(query: str, host: str, user: str, password: str,
                              database: str) -> List[Tuple]:
    """Execute query using pymysql."""
    if not _HAS_PYMYSQL:
        raise RuntimeError("pymysql not installed. Install with: pip install pymysql")

    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            cursorclass=pymysql.cursors.Cursor
        )
        with conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        return [tuple(str(x) if x is not None else '' for x in row) for row in rows]
    except pymysql.Error as e:
        raise RuntimeError("MySQL query failed: {}".format(e))


def _run_mysql_query_subprocess(query: str, login_path: str,
                                 database: str) -> List[Tuple]:
    """Execute query using mysql CLI with --login-path."""
    cmd = [
        "mysql",
        "--login-path={}".format(login_path),
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
            universal_newlines=True,
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


def _run_mysql_query(query: str, login_path: str = DEFAULT_LOGIN_PATH,
                     database: str = DEFAULT_DATABASE,
                     host: str = None, user: str = None,
                     password: str = None) -> List[Tuple]:
    """
    Execute a MySQL query.

    Tries pymysql with environment variables first, then falls back to
    mysql CLI with --login-path.

    Args:
        query: SQL query string
        login_path: MySQL login path name (for CLI fallback)
        database: Database name
        host, user, password: Override credentials (default: from env vars)

    Returns:
        List of tuples, each tuple is a row from the result

    Raises:
        RuntimeError: If query fails
    """
    # Use provided credentials or fall back to environment variables
    host = host or DEFAULT_HOST
    user = user or DEFAULT_USER
    password = password or DEFAULT_PASSWORD

    # Try pymysql first if credentials are available
    if _HAS_PYMYSQL and host and user and password:
        return _run_mysql_query_pymysql(query, host, user, password, database)

    # Fall back to subprocess with --login-path
    return _run_mysql_query_subprocess(query, login_path, database)


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
    query = "SELECT XECConf_id FROM RunCatalog WHERE id = {}".format(run_number)
    rows = _run_mysql_query(query, login_path, database)

    if not rows or rows[0][0] == 'NULL' or rows[0][0] == '':
        print("[WARNING] No XECConf_id found for run {}".format(run_number))
        return None

    xec_conf_id = int(rows[0][0])

    # Step 2: XECConf → XECPMStatusDB_id
    query = "SELECT XECPMStatusDB_id FROM XECConf WHERE id = {}".format(xec_conf_id)
    rows = _run_mysql_query(query, login_path, database)

    if not rows or rows[0][0] == 'NULL' or rows[0][0] == '':
        print("[WARNING] No XECPMStatusDB_id found for XECConf_id {}".format(xec_conf_id))
        return None

    xec_pm_status_db_id = int(rows[0][0])

    # Step 3: XECPMStatusDB → XECPMStatus_id
    query = "SELECT XECPMStatus_id FROM XECPMStatusDB WHERE id = {}".format(xec_pm_status_db_id)
    rows = _run_mysql_query(query, login_path, database)

    if not rows or rows[0][0] == 'NULL' or rows[0][0] == '':
        print("[WARNING] No XECPMStatus_id found for XECPMStatusDB_id {}".format(xec_pm_status_db_id))
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
        print("[WARNING] Could not find PM status for run {}, returning empty array".format(run_number))
        return np.array([], dtype=np.int32)

    # Query dead channels (IsBad = 1)
    query = """
        SELECT idx FROM XECPMStatus
        WHERE id = {} AND IsBad = 1
        ORDER BY idx
    """.format(xec_pm_status_id)
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
        f.write("# Dead channels for run {}\n".format(run_number))
        f.write("# Total: {} / 4760\n".format(len(dead_channels)))
        for ch in dead_channels:
            f.write("{}\n".format(ch))

    print("[INFO] Saved {} dead channels to {}".format(len(dead_channels), output_path))


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

    print("\n" + "=" * 50)
    print("Dead Channel Summary for Run {}".format(run_number))
    print("=" * 50)
    print("XECPMStatus_id: {}".format(info['xec_pm_status_id']))
    print("Total dead: {} / 4760 ({:.2f}%)".format(info['n_dead'], info['dead_fraction'] * 100))
    print("\nDead by face:")
    for face, count in info['dead_by_face'].items():
        print("  {:>6}: {}".format(face, count))
    print("=" * 50 + "\n")


def check_connection():
    """Check database connection and print status."""
    print("\nDatabase Connection Check")
    print("=" * 50)

    # Check environment variables
    if DEFAULT_HOST and DEFAULT_USER and DEFAULT_PASSWORD:
        print("Environment variables: SET")
        print("  MYSQL_HOST: {}".format(DEFAULT_HOST))
        print("  MYSQL_USER: {}".format(DEFAULT_USER))
        print("  MYSQL_PASSWORD: ****")
    else:
        print("Environment variables: NOT SET")
        missing = []
        if not DEFAULT_HOST:
            missing.append("MYSQL_HOST")
        if not DEFAULT_USER:
            missing.append("MYSQL_USER")
        if not DEFAULT_PASSWORD:
            missing.append("MYSQL_PASSWORD")
        print("  Missing: {}".format(", ".join(missing)))

    # Check pymysql
    if _HAS_PYMYSQL:
        print("pymysql: INSTALLED")
    else:
        print("pymysql: NOT INSTALLED (install with: pip install pymysql)")

    print("=" * 50 + "\n")

    if not _HAS_PYMYSQL and not (DEFAULT_HOST and DEFAULT_USER and DEFAULT_PASSWORD):
        print("To use this module, either:")
        print("1. Install pymysql and set environment variables:")
        print("   pip install pymysql")
        print("   export MYSQL_HOST=<host>")
        print("   export MYSQL_USER=<user>")
        print("   export MYSQL_PASSWORD=<password>")
        print("")
        print("2. Or use mysql CLI with --login-path (requires official MySQL client)")
        return False

    return True


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Query dead channels from MEG2 database")
    parser.add_argument("run_number", type=int, nargs='?', default=None,
                        help="Run number to query")
    parser.add_argument("--login-path", default=DEFAULT_LOGIN_PATH,
                        help="MySQL login path (default: {})".format(DEFAULT_LOGIN_PATH))
    parser.add_argument("--database", default=DEFAULT_DATABASE,
                        help="Database name (default: {})".format(DEFAULT_DATABASE))
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output file to save dead channel list")
    parser.add_argument("--list", "-l", action="store_true",
                        help="List all dead channel indices")
    parser.add_argument("--check", action="store_true",
                        help="Check database connection status")

    args = parser.parse_args()

    # Check connection status
    if args.check or args.run_number is None:
        check_connection()
        if args.run_number is None:
            exit(0)

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
