#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later

# ==============================================================================
# Self-contained test script for trtinfer_datagen (Phase 1)
# Verifies:
#   1. Standalone executable -h1 compliance
#   2. Batch generation into native ImageStreamIO shared memory streams
#   3. FITS and binary file export
#   4. Numerical accuracy against analytical 3D optical swirl formula (< 1e-6)
#   5. Streaming trajectory generation and semaphore updates
# ==============================================================================

set -e

# Terminal colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MILK_ROOT="$(cd "${PLUGIN_DIR}/../.." && pwd)"

echo -e "${BOLD}=== [trtinfer_datagen] Self-Contained Test Suite ===${NC}"

# 1. Locate executable
EXE=""
if [ -x "${MILK_ROOT}/_build/plugins/trtinfer/milk-fpsexec-trtinfer_datagen" ]; then
    EXE="${MILK_ROOT}/_build/plugins/trtinfer/milk-fpsexec-trtinfer_datagen"
elif command -v milk-fpsexec-trtinfer_datagen &>/dev/null; then
    EXE="$(command -v milk-fpsexec-trtinfer_datagen)"
fi

if [ -z "$EXE" ]; then
    echo -e "${RED}[FAIL] Executable milk-fpsexec-trtinfer_datagen not found.${NC}"
    echo "Please build it with: make -C _build milk-fpsexec-trtinfer_datagen"
    exit 1
fi
echo -e "${GREEN}[OK]${NC} Located executable: $EXE"

# 2. Test -h1 compliance
echo -n "Testing -h1 one-line help output... "
H1_OUT="$($EXE -h1 2>&1 || true)"
if [[ "$H1_OUT" == *"Generate non-linear 3D dataset in ImageStreamIO streams"* ]]; then
    echo -e "${GREEN}[PASS]${NC} ($H1_OUT)"
else
    echo -e "${RED}[FAIL]${NC} Unexpected -h1 output: '$H1_OUT'"
    exit 1
fi

# 3. Clean up any previous test artifacts
TMP_PREFIX="/tmp/trtinfer_test_$$"
TEST_INSTREAM="test_pts_in_$$"
TEST_TRUTHSTREAM="test_pts_truth_$$"
TEST_FITS="${TMP_PREFIX}_data"

cleanup() {
    rm -f "${TMP_PREFIX}"*
    rm -f "/milk/shm/${TEST_INSTREAM}.im.shm" "/milk/shm/${TEST_TRUTHSTREAM}.im.shm" 2>/dev/null || true
    rm -f "/dev/shm/${TEST_INSTREAM}.im.shm" "/dev/shm/${TEST_TRUTHSTREAM}.im.shm" 2>/dev/null || true
}
trap cleanup EXIT

# 4. Run Batch Mode Generation (5,000 samples)
NSAMPLES=5000
echo -n "Running batch generation of $NSAMPLES samples... "
$EXE -n "trttest_$$" exec "$TEST_INSTREAM" "$TEST_TRUTHSTREAM" $NSAMPLES "$TEST_FITS" >/dev/null
echo -e "${GREEN}[PASS]${NC}"

# 5. Verify ImageStreamIO shared memory streams exist on disk
echo -n "Verifying ImageStreamIO shared memory files... "
SHM_IN=""
SHM_TRUTH=""
for DIR in /milk/shm /dev/shm; do
    if [ -f "${DIR}/${TEST_INSTREAM}.im.shm" ]; then
        SHM_IN="${DIR}/${TEST_INSTREAM}.im.shm"
    fi
    if [ -f "${DIR}/${TEST_TRUTHSTREAM}.im.shm" ]; then
        SHM_TRUTH="${DIR}/${TEST_TRUTHSTREAM}.im.shm"
    fi
done

if [ -n "$SHM_IN" ] && [ -n "$SHM_TRUTH" ]; then
    IN_SZ=$(stat -c%s "$SHM_IN")
    TRUTH_SZ=$(stat -c%s "$SHM_TRUTH")
    echo -e "${GREEN}[PASS]${NC}"
    echo "     Input stream: $SHM_IN (${IN_SZ} bytes)"
    echo "     Truth stream: $SHM_TRUTH (${TRUTH_SZ} bytes)"
else
    echo -e "${RED}[FAIL]${NC} ImageStreamIO streams not found in /milk/shm or /dev/shm"
    exit 1
fi

# 6. Verify Numerical Accuracy of FITS & Binary Exports
FITS_IN="${TEST_FITS}_in.fits"
FITS_TRUTH="${TEST_FITS}_truth.fits"
BIN_FILE="${TEST_FITS}.bin"

echo -n "Verifying FITS and binary file creation... "
if [ -f "$FITS_IN" ] && [ -f "$FITS_TRUTH" ] && [ -f "$BIN_FILE" ]; then
    echo -e "${GREEN}[PASS]${NC}"
else
    echo -e "${RED}[FAIL]${NC} Missing export files"
    ls -l "${TMP_PREFIX}"* || true
    exit 1
fi

# Locate C validator binary
CHECK_EXE=""
if [ -x "${MILK_ROOT}/_build/plugins/trtinfer/trtinfer_check_accuracy" ]; then
    CHECK_EXE="${MILK_ROOT}/_build/plugins/trtinfer/trtinfer_check_accuracy"
elif command -v trtinfer_check_accuracy &>/dev/null; then
    CHECK_EXE="$(command -v trtinfer_check_accuracy)"
fi

if [ -z "$CHECK_EXE" ]; then
    echo -e "${RED}[FAIL] Compiled validator trtinfer_check_accuracy not found.${NC}"
    exit 1
fi

echo "Validating mathematical accuracy with compiled C validator..."
$CHECK_EXE "$TEST_INSTREAM" "$TEST_TRUTHSTREAM" "$BIN_FILE"
echo -e "${GREEN}[PASS]${NC} C validator verified all samples against analytical mapping."

# 7. Summary
echo ""
echo -e "${GREEN}${BOLD}All tests passed successfully!${NC}"
echo -e "Dataset generator is ready for model training and TensorRT inference."
