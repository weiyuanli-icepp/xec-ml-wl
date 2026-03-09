#!/bin/bash
#
# Submit SLURM jobs to validate the energy regressor on CEX real data.
# One job per patch (24 patches total), running in parallel.
#
# Usage:
#   bash macro/submit_validate_regressor_cex.sh                    # All patches (standard)
#   bash macro/submit_validate_regressor_cex.sh 13 12 21           # Only these patches
#   DRY_RUN=1 bash macro/submit_validate_regressor_cex.sh          # Preview
#
#   # Dead-channel recovery mode:
#   DEAD_CHANNEL=1 INPAINTER=inpainter.pt bash macro/submit_validate_regressor_cex.sh
#
# Environment variables:
#   CHECKPOINT  — path to model (.pth or .onnx, default: ene_50epoch_sqrt_DSmax ONNX)
#   CONFIG      — config YAML for normalization params (default: ene_50epoch_sqrt_DSmax)
#   CEX_DIR     — directory with CEX ROOT files (default: data/cex)
#   OUTPUT_BASE — output base directory (default: val_data/cex)
#   PARTITION   — SLURM partition (default: meg-long)
#   DRY_RUN     — set to 1 to preview without submitting
#   DEAD_CHANNEL — set to 1 to enable dead-channel recovery mode
#   INPAINTER   — path to inpainter TorchScript (.pt) for dead-channel mode
#   NEIGHBOR_K  — neighbor hops for averaging (default: 1)
#

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT="${CHECKPOINT:-~/meghome/xec-ml-wl/artifacts/scan_s7_fulldata/ene_s7_best.onnx}"
CONFIG="${CONFIG:-config/reg/scan/step7_fulldata.yaml}"
CEX_DIR="${CEX_DIR:-data/cex}"
OUTPUT_BASE="${OUTPUT_BASE:-val_data/cex}"
PARTITION="${PARTITION:-meg-long}"
DEAD_CHANNEL="${DEAD_CHANNEL:-0}"
INPAINTER="${INPAINTER:-~/meghome/xec-ml-wl/artifacts/inp_scan_s3_nphowt/inpainter.pt}"
NEIGHBOR_K="${NEIGHBOR_K:-1}"

case "$PARTITION" in
    meg-long|meg-short|mu3e) ACCOUNT_LINE="#SBATCH --account=meg" ;;
    *)                       ACCOUNT_LINE="" ;;
esac

TIME="${TIME:-06:00:00}"
MEM="${MEM:-16G}"
BATCH_SIZE="${BATCH_SIZE:-1024}"

# All 24 CEX23 patches
ALL_PATCHES=(13 12 21 20 5 4 22 14 6 19 11 3 1 2 7 8 9 10 15 16 17 18 23 24)

if [ $# -eq 0 ]; then
    PATCHES=("${ALL_PATCHES[@]}")
else
    PATCHES=("$@")
fi

echo "============================================"
echo "Energy Regressor CEX Validation"
echo "============================================"
echo "Model:       ${CHECKPOINT}"
echo "Config:      ${CONFIG}"
echo "CEX data:    ${CEX_DIR}"
echo "Output:      ${OUTPUT_BASE}"
echo "Patches:     ${PATCHES[*]}"
echo "Partition:   ${PARTITION}"
echo "Dead-channel: ${DEAD_CHANNEL}"
if [ "$DEAD_CHANNEL" = "1" ]; then
    echo "Inpainter:   ${INPAINTER:-<none>}"
    echo "Neighbor k:  ${NEIGHBOR_K}"
fi
echo "Dry run:     ${DRY_RUN}"
echo "============================================"
echo ""

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# PRE-FLIGHT CHECKLIST — verify before submitting:
#   1. CONFIG file exists and has correct npho_scheme / normalization
#      (step7_fulldata uses sqrt, scale=20000)
#   2. CEX_DIR points to correct location (CEX data may be stored in
#      different places depending on the server / shared storage)
#   3. ONNX model path is accessible from compute nodes
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
if [ "$DRY_RUN" != "1" ]; then
    echo ">>> PRE-FLIGHT: Please verify the following paths are correct <<<"
    echo "    Config:   ${CONFIG}"
    echo "    CEX data: ${CEX_DIR}"
    echo "    Model:    ${CHECKPOINT}"
    echo "    (CEX data may be in different locations — check before running!)"
    echo ""
fi

# Verify model and config exist
CHECKPOINT_EXPANDED="${CHECKPOINT/#\~/$HOME}"
if [ "$DRY_RUN" != "1" ]; then
    if [ ! -f "$CHECKPOINT_EXPANDED" ]; then
        echo "[ERROR] Model not found: $CHECKPOINT"
        exit 1
    fi
    if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
        echo "[ERROR] Config not found: $CONFIG"
        exit 1
    fi
fi

mkdir -p log "${OUTPUT_BASE}"

SUBMITTED=0
SKIPPED=0

# Build extra flags for dead-channel mode
EXTRA_FLAGS=""
if [ "$DEAD_CHANNEL" = "1" ]; then
    EXTRA_FLAGS="--dead-channel --neighbor-k ${NEIGHBOR_K}"
    if [ -n "$INPAINTER" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --inpainter-torchscript ${INPAINTER}"
    fi
fi

for PATCH in "${PATCHES[@]}"; do
    # Find CEX files for this patch (may be multiple from batched preprocessing)
    PATCH_FILES=( $(ls "${CEX_DIR}"/CEX23_patch${PATCH}_r*.root 2>/dev/null || true) )

    if [ ${#PATCH_FILES[@]} -eq 0 ]; then
        echo "--- Patch ${PATCH}: NO FILES FOUND, skipping ---"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    N_FILES=${#PATCH_FILES[@]}
    OUTDIR="${OUTPUT_BASE}/patch${PATCH}"

    echo "--- Patch ${PATCH}: ${N_FILES} file(s) ---"
    echo "  Output: ${OUTDIR}"

    if [ "$DRY_RUN" = "1" ]; then
        echo "  [DRY RUN] Skipping submission"
        for f in "${PATCH_FILES[@]}"; do
            echo "    $(basename "$f")"
        done
        echo ""
        SUBMITTED=$((SUBMITTED + 1))
        continue
    fi

    # Build space-separated list of input files for the Python script
    # If multiple files per patch, pass the glob pattern
    if [ ${N_FILES} -eq 1 ]; then
        VAL_PATH="${PATCH_FILES[0]}"
    else
        # Use glob pattern to match all files for this patch
        VAL_PATH="${CEX_DIR}/CEX23_patch${PATCH}_r*.root"
    fi

    mkdir -p "$HOME/.cache/xec-ml-wl"
    BATCH_SCRIPT=$(mktemp "$HOME/.cache/xec-ml-wl/val_cex_patch${PATCH}_XXXXXX.sh")

    cat > "${BATCH_SCRIPT}" << SLURM_EOF
#!/bin/bash
${ACCOUNT_LINE}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME}
#SBATCH --hint=nomultithread
#SBATCH --ntasks=1
#SBATCH --mem=${MEM}
#SBATCH --job-name=cex_val_p${PATCH}
#SBATCH --output=$HOME/meghome/xec-ml-wl/log/cex_val_patch${PATCH}_%j.log

echo "=== Energy Regressor CEX Validation: Patch ${PATCH} ==="
echo "Host: \$(hostname)"
echo "Date: \$(date)"
echo ""

# Conda setup
[[ -f /etc/profile.d/modules.sh ]] && source /etc/profile.d/modules.sh || true

ARM_CONDA="\$HOME/miniforge-arm/bin/conda"
X86_CONDA="/opt/psi/Programming/anaconda/2024.08/conda/bin/conda"

module load anaconda/2024.08 2>/dev/null || true

if [ -f "\$ARM_CONDA" ] && [ "\$(uname -m)" == "aarch64" ]; then
    eval "\$(\$ARM_CONDA shell.bash hook)"
elif command -v conda &> /dev/null; then
    eval "\$(conda shell.bash hook)"
elif [ -f "\$X86_CONDA" ]; then
    eval "\$(\$X86_CONDA shell.bash hook)"
else
    echo "CRITICAL ERROR: Could not find conda"
    exit 1
fi

conda activate xec-ml-wl

if [ -n "\$CONDA_PREFIX" ]; then
    export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH"
fi

cd \$HOME/meghome/xec-ml-wl
echo "[JOB] Directory: \$(pwd)"

python macro/validate_regressor.py \\
    ${CHECKPOINT} \\
    --val_path "${VAL_PATH}" \\
    --config ${CONFIG} \\
    --tasks energy \\
    --output_dir ${OUTDIR} \\
    --batch_size ${BATCH_SIZE} \\
    --device cpu ${EXTRA_FLAGS}

echo ""
echo "=== Done: \$(date) ==="
SLURM_EOF

    sbatch "${BATCH_SCRIPT}"
    SUBMITTED=$((SUBMITTED + 1))
    echo ""
    sleep 0.5
done

echo "============================================"
echo "[CEX] Submitted ${SUBMITTED} jobs, skipped ${SKIPPED}"
echo "Logs:   log/cex_val_patch*_*.log"
echo "Output: ${OUTPUT_BASE}/patch*/"
echo "============================================"
