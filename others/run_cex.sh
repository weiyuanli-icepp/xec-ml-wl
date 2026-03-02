#!/bin/bash
# Run CEXPreprocess.C via meganalyzer (same compilation method as MCXECPreprocess.C)
#
# Usage:
#   ./others/run_cex.sh 557545 1 13
#   ./others/run_cex.sh 557545 100 13 /my/output/dir
#
# Arguments:
#   sRun        — starting run number
#   nfiles      — number of consecutive runs
#   patchnumber — CEX patch number
#   outputDir   — output directory (default: current directory)

set -e

sRun=${1:?Usage: $0 sRun nfiles patchnumber [outputDir]}
nfiles=${2:?Usage: $0 sRun nfiles patchnumber [outputDir]}
patchnumber=${3:?Usage: $0 sRun nfiles patchnumber [outputDir]}
outputDir=${4:-.}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MACRO_PATH="${SCRIPT_DIR}/CEXPreprocess.C"
ANALYZER_DIR="${MEG2SYS}/analyzer"
BUILD_DIR="/tmp/cex_build_$$"

if [ ! -d "$ANALYZER_DIR" ]; then
    echo "ERROR: \$MEG2SYS/analyzer not found at $ANALYZER_DIR"
    echo "Make sure MEG2SYS is set (e.g. source your MEG setup script)"
    exit 1
fi

# Create loader macro (compiles CEXPreprocess.C with ACLiC, then runs it)
LOADER=$(mktemp /tmp/loader_cex_XXXXXX.C)
cat > "$LOADER" <<CEOF
#include <TSystem.h>
#include <TROOT.h>
#include <TString.h>

void $(basename "$LOADER" .C)() {
    gSystem->SetBuildDir("${BUILD_DIR}", kTRUE);
    gROOT->ProcessLine(TString::Format(".L ${MACRO_PATH}+"));
    gROOT->ProcessLine("CEXPreprocess(${sRun}, ${nfiles}, ${patchnumber}, \"${outputDir}\")");
}
CEOF

echo "[INFO] Macro:    $MACRO_PATH"
echo "[INFO] Loader:   $LOADER"
echo "[INFO] Args:     sRun=$sRun nfiles=$nfiles patch=$patchnumber outputDir=$outputDir"

cd "$ANALYZER_DIR"
./meganalyzer -I "${LOADER}()" -b -q
RET=$?

# Cleanup
rm -f "$LOADER"
rm -rf "$BUILD_DIR"

exit $RET
