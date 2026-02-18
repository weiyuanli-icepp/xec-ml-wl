#!/bin/bash
# compare_dirs.sh - Compare files between two directories by size
#
# Usage:
#   ./compare_dirs.sh DIR1 DIR2
#   ./compare_dirs.sh  # Uses default paths
#
# Options:
#   --checksum    Use MD5 checksum instead of size (slower but more accurate)

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
    echo "    mode: checksum (MD5)"
else
    echo "    mode: size"
fi
echo ""

# Count total files first
echo "Counting files..."
total_files=$(find "$DIR1" -type f | wc -l)
echo "Found $total_files files in DIR1"
echo ""

# Find all files in DIR1, compare with DIR2
diff_count=0
match_count=0
missing_count=0
processed=0

while IFS= read -r -d '' file1; do
    ((processed++))
    # Show progress every 1000 files
    if (( processed % 1000 == 0 )); then
        echo "[PROGRESS] $processed / $total_files files checked..."
    fi
    rel_path="${file1#$DIR1/}"
    file2="$DIR2/$rel_path"

    if [[ -f "$file2" ]]; then
        if [[ "$USE_CHECKSUM" == true ]]; then
            hash1=$(md5sum "$file1" | cut -d' ' -f1)
            hash2=$(md5sum "$file2" | cut -d' ' -f1)
            if [[ "$hash1" == "$hash2" ]]; then
                ((match_count++))
            else
                echo "[DIFF] $rel_path"
                echo "       DIR1: $hash1"
                echo "       DIR2: $hash2"
                ((diff_count++))
            fi
        else
            # Use stat -c for Linux, fallback to stat -f for macOS
            size1=$(stat -c%s "$file1" 2>/dev/null || stat -f%z "$file1")
            size2=$(stat -c%s "$file2" 2>/dev/null || stat -f%z "$file2")

            if [[ "$size1" -eq "$size2" ]]; then
                ((match_count++))
            else
                echo "[DIFF] $rel_path: $size1 vs $size2 bytes"
                ((diff_count++))
            fi
        fi
    else
        echo "[MISSING in DIR2] $rel_path"
        ((missing_count++))
    fi
done < <(find "$DIR1" -type f -print0)

echo "[PROGRESS] Checked all $processed files in DIR1"
echo "Checking for files only in DIR2..."

# Check for files only in DIR2
while IFS= read -r -d '' file2; do
    rel_path="${file2#$DIR2/}"
    file1="$DIR1/$rel_path"

    if [[ ! -f "$file1" ]]; then
        echo "[MISSING in DIR1] $rel_path"
        ((missing_count++))
    fi
done < <(find "$DIR2" -type f -print0)

echo ""
echo "=== Summary ==="
echo "  Matching:  $match_count"
echo "  Different: $diff_count"
echo "  Missing:   $missing_count"

if [[ "$diff_count" -eq 0 && "$missing_count" -eq 0 ]]; then
    echo ""
    echo "Directories are identical."
    exit 0
else
    exit 1
fi
