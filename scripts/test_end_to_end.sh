#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later

# ==============================================================================
# Comprehensive End-to-End Test Suite for trtinfer Plugin
#
# Pipeline tested (100% pure compiled C/C++, zero Python):
#   1. Standalone -h1 one-line help compliance on all executables
#   2. Generation of 3D non-linear optical swirl dataset in ImageStreamIO shared memory
#   3. Ground-truth mathematical verification against analytical equations (< 1e-5)
#   4. Direct neural network training from ImageStreamIO using mini-batch Adam in C
#   5. Real-time inference execution via milk-fpsexec-trtinfer on ImageStreamIO streams
#   6. Full inference numerical accuracy verification (< 0.05 RMSE)
# ==============================================================================

set -e

# Terminal formatting
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
MILK_ROOT="$(cd "${PLUGIN_DIR}/../.." && pwd)"

echo -e "${BOLD}======================================================================${NC}"
echo -e "${BOLD}=== [trtinfer] Comprehensive End-to-End Test Suite (Zero Python) ===${NC}"
echo -e "${BOLD}======================================================================${NC}"

# 1. Locate Executables
DATAGEN_EXE="${MILK_ROOT}/_build/plugins/trtinfer/milk-fpsexec-trtinfer_datagen"
INFER_EXE="${MILK_ROOT}/_build/plugins/trtinfer/milk-fpsexec-trtinfer"
TRAIN_EXE="${MILK_ROOT}/_build/plugins/trtinfer/trtinfer_train_mlp"
CHECK_EXE="${MILK_ROOT}/_build/plugins/trtinfer/trtinfer_check_accuracy"

for EXE in "$DATAGEN_EXE" "$INFER_EXE" "$TRAIN_EXE" "$CHECK_EXE"; do
    if [ ! -x "$EXE" ]; then
        echo -e "${RED}[FAIL] Required executable not found: $EXE${NC}"
        echo "Please build with: make -C _build milktrtinfer milk-fpsexec-trtinfer milk-fpsexec-trtinfer_datagen trtinfer_train_mlp trtinfer_check_accuracy"
        exit 1
    fi
done
echo -e "${GREEN}[OK]${NC} All 4 compiled binaries located."

# 2. Test -h1 compliance
echo -n "Checking -h1 output for milk-fpsexec-trtinfer_datagen... "
H1_DATAGEN="$($DATAGEN_EXE -h1 2>&1 || true)"
if [[ "$H1_DATAGEN" == *"Generate non-linear 3D dataset in ImageStreamIO streams"* ]]; then
    echo -e "${GREEN}[PASS]${NC} ($H1_DATAGEN)"
else
    echo -e "${RED}[FAIL]${NC} Got: '$H1_DATAGEN'"
    exit 1
fi

echo -n "Checking -h1 output for milk-fpsexec-trtinfer... "
H1_INFER="$($INFER_EXE -h1 2>&1 || true)"
if [[ "$H1_INFER" == *"Real-time neural network inference compute unit on ImageStreamIO"* ]]; then
    echo -e "${GREEN}[PASS]${NC} ($H1_INFER)"
else
    echo -e "${RED}[FAIL]${NC} Got: '$H1_INFER'"
    exit 1
fi

# 3. Setup temporary identifiers
TEST_ID="e2e_$$"
INSTREAM="e2e_pts_in_${TEST_ID}"
TRUTHSTREAM="e2e_pts_truth_${TEST_ID}"
OUTSTREAM="e2e_pts_out_${TEST_ID}"
TMP_PREFIX="/tmp/trtinfer_${TEST_ID}"
MODEL_FILE="${TMP_PREFIX}_model.bin"
DATA_PREFIX="${TMP_PREFIX}_data"

cleanup() {
    rm -f "${TMP_PREFIX}"*
    rm -f "/milk/shm/${INSTREAM}.im.shm" "/milk/shm/${TRUTHSTREAM}.im.shm" "/milk/shm/${OUTSTREAM}.im.shm" 2>/dev/null || true
    rm -f "/dev/shm/${INSTREAM}.im.shm" "/dev/shm/${TRUTHSTREAM}.im.shm" "/dev/shm/${OUTSTREAM}.im.shm" 2>/dev/null || true
}
trap cleanup EXIT

# 4. Generate 3D Benchmark Dataset (5,000 samples)
NSAMPLES=5000
echo ""
echo -e "${BOLD}[Step 1/5] Generating $NSAMPLES 3D samples into ImageStreamIO streams...${NC}"
$DATAGEN_EXE -n "gen_${TEST_ID}" exec "$INSTREAM" "$TRUTHSTREAM" $NSAMPLES "$DATA_PREFIX" >/dev/null
echo -e "${GREEN}[PASS]${NC} Streams created: '$INSTREAM' and '$TRUTHSTREAM'"

# Verify shared memory files on disk
FOUND_SHM=0
for DIR in /milk/shm /dev/shm; do
    if [ -f "${DIR}/${INSTREAM}.im.shm" ] && [ -f "${DIR}/${TRUTHSTREAM}.im.shm" ]; then
        FOUND_SHM=1
        echo "       Shared memory confirmed in: $DIR"
        break
    fi
done

if [ $FOUND_SHM -eq 0 ]; then
    echo -e "${RED}[FAIL] ImageStreamIO stream files not found on disk${NC}"
    exit 1
fi

# 5. Verify Dataset Accuracy with C validator
echo ""
echo -e "${BOLD}[Step 2/5] Validating dataset numerical accuracy against analytical equations...${NC}"
$CHECK_EXE "$INSTREAM" "$TRUTHSTREAM" "${DATA_PREFIX}.bin"
echo -e "${GREEN}[PASS]${NC} Dataset matched analytical formulas to floating-point precision."

# 6. Train Neural Network (Pure C on ImageStreamIO)
echo ""
echo -e "${BOLD}[Step 3/5] Training 3-layer MLP directly from ImageStreamIO in pure C...${NC}"
$TRAIN_EXE -i "$INSTREAM" -t "$TRUTHSTREAM" -o "$MODEL_FILE" -e 500 -b 64 -r 0.005
if [ ! -f "$MODEL_FILE" ]; then
    echo -e "${RED}[FAIL] Model weights file was not created: $MODEL_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}[PASS]${NC} Neural network converged and model saved to $MODEL_FILE"

# 7. Execute Inference via milk-fpsexec-trtinfer
echo ""
echo -e "${BOLD}[Step 4/5] Executing inference compute unit on ImageStreamIO streams...${NC}"
$INFER_EXE -n "infer_${TEST_ID}" exec "$INSTREAM" "$OUTSTREAM" "$MODEL_FILE"
echo -e "${GREEN}[PASS]${NC} Inference completed. Output stream '$OUTSTREAM' created."

# 8. Verify Inference Accuracy
echo ""
echo -e "${BOLD}[Step 5/5] Validating neural network inference accuracy vs ground truth...${NC}"
$CHECK_EXE "$INSTREAM" "$TRUTHSTREAM" none "$OUTSTREAM"
echo -e "${GREEN}[PASS]${NC} Neural network successfully approximated the optical swirl mapping."

echo ""
echo -e "${BOLD}======================================================================${NC}"
echo -e "${GREEN}${BOLD}=== ALL TESTS PASSED: Full C/C++ Inference Pipeline Verified! ===${NC}"
echo -e "${BOLD}======================================================================${NC}"
