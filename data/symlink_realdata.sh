#!/bin/bash
set -euo pipefail

# Creates symlink directories for different real data training splits.
# Uses the already-processed DataGammaAngle_*.root files in train/.
#
# Usage: ./data/symlink_realdata.sh
#
# Splits (by sorted file order):
#   train_tiny:   first 10 files
#   train_small:  first 50 files
#   train_middle: first 150 files
#   train_large:  first 467 files (all currently processed)
#   train_max:    all available files

BASE_DIR="$HOME/meghome/xec-ml-wl/data/real_data"

if [[ ! -d "$BASE_DIR/train" ]]; then
    echo "Error: train directory not found in $BASE_DIR"
    exit 1
fi

cd "$BASE_DIR"

# Get sorted list of non-empty DataGammaAngle files
FILES=()
for f in train/DataGammaAngle_*.root; do
    if [[ -f "$f" ]] && [[ $(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null) -gt 1000 ]]; then
        FILES+=("$(basename "$f")")
    fi
done

N_TOTAL=${#FILES[@]}
echo "Found $N_TOTAL non-empty files in train/"

if [[ $N_TOTAL -eq 0 ]]; then
    echo "Error: no DataGammaAngle files found"
    exit 1
fi

# Create split directories
mkdir -p train_tiny train_small train_middle train_large train_max

# Helper: symlink first N files into a directory
link_first_n() {
    local dir="$1"
    local n="$2"
    local count=0
    for f in "${FILES[@]}"; do
        if [[ $count -ge $n ]]; then break; fi
        ln -sf "../train/$f" "$dir/$f"
        count=$((count + 1))
    done
    echo "  $dir: $count files"
}

link_first_n "train_tiny"   10
link_first_n "train_small"  50
link_first_n "train_middle" 150
link_first_n "train_large"  467
link_first_n "train_max"    "$N_TOTAL"

echo ""
echo "Done. Splits created in $BASE_DIR/"
echo "Val data is in: $BASE_DIR/val/ ($(ls val/DataGammaAngle_*.root 2>/dev/null | wc -l | tr -d ' ') files)"
