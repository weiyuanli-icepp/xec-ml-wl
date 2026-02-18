#!/bin/bash
# compare_dirs.sh - Compare files between two directories by size (fast version)
#
# Usage:
#   ./compare_dirs.sh DIR1 DIR2
#   ./compare_dirs.sh  # Uses default paths
#
# Options:
#   --checksum    Use MD5 checksum instead of size (much slower)

set -euo pipefail

# Default paths
DEFAULT_DIR1="/data/project/meg/shared/subprojects/spx"
DEFAULT_DIR2="/data/project/meg/data1/shared/subprojects/spx"

# Parse arguments
USE_CHECKSUM=false
DIR1=""
DIR2=""

for arg in "$@"; do
    if [[ "$arg" == "--checksum" ]]; then
        USE_CHECKSUM=true
    elif [[ -z "$DIR1" ]]; then
        DIR1="$arg"
    elif [[ -z "$DIR2" ]]; then
        DIR2="$arg"
    fi
done

DIR1="${DIR1:-$DEFAULT_DIR1}"
DIR2="${DIR2:-$DEFAULT_DIR2}"

# Validate directories
if [[ ! -d "$DIR1" ]]; then
    echo "[ERROR] Directory not found: $DIR1"
    exit 1
fi

if [[ ! -d "$DIR2" ]]; then
    echo "[ERROR] Directory not found: $DIR2"
    exit 1
fi

echo "Comparing: $DIR1"
echo "     with: $DIR2"
if [[ "$USE_CHECKSUM" == true ]]; then
    echo "    mode: checksum (MD5) - this will be slow"
else
    echo "    mode: size (fast)"
fi
echo ""

# Create temp files for comparison
TMPDIR="${TMPDIR:-/tmp}"
LIST1=$(mktemp "$TMPDIR/compare_dir1.XXXXXX")
LIST2=$(mktemp "$TMPDIR/compare_dir2.XXXXXX")
trap "rm -f '$LIST1' '$LIST2'" EXIT

if [[ "$USE_CHECKSUM" == true ]]; then
    echo "Building file list with checksums for DIR1... (this takes a while)"
    (cd "$DIR1" && find . -type f -exec md5sum {} \; | sort) > "$LIST1"
    echo "Building file list with checksums for DIR2..."
    (cd "$DIR2" && find . -type f -exec md5sum {} \; | sort) > "$LIST2"
else
    echo "Building file list for DIR1..."
    (cd "$DIR1" && find . -type f -printf '%p %s\n' | sort) > "$LIST1"
    echo "Building file list for DIR2..."
    (cd "$DIR2" && find . -type f -printf '%p %s\n' | sort) > "$LIST2"
fi

echo ""
echo "Files in DIR1: $(wc -l < "$LIST1")"
echo "Files in DIR2: $(wc -l < "$LIST2")"
echo ""

# Compare
echo "=== Differences ==="
DIFF_OUTPUT=$(diff "$LIST1" "$LIST2" || true)

if [[ -z "$DIFF_OUTPUT" ]]; then
    echo "(none)"
    echo ""
    echo "Directories are identical."
    exit 0
else
    # Parse diff output
    only_in_dir1=0
    only_in_dir2=0
    different=0

    while IFS= read -r line; do
        if [[ "$line" =~ ^\<\  ]]; then
            echo "[ONLY in DIR1] ${line#< }"
            ((only_in_dir1++))
        elif [[ "$line" =~ ^\>\  ]]; then
            echo "[ONLY in DIR2] ${line#> }"
            ((only_in_dir2++))
        fi
    done <<< "$DIFF_OUTPUT"

    echo ""
    echo "=== Summary ==="
    echo "  Only in DIR1: $only_in_dir1"
    echo "  Only in DIR2: $only_in_dir2"
    exit 1
fi
