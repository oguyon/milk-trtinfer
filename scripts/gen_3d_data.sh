#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Olivier Guyon et al
# SPDX-License-Identifier: LGPL-3.0-or-later

# Utility script to generate non-linear 3D training dataset
# Usage: ./gen_3d_data.sh [nsamples] [output_prefix]

NSAMPLES=${1:-10000}
OUTNAME=${2:-"data_3d.fits"}

DATAGEN_BIN=$(which milk-fpsexec-trtinfer_datagen 2>/dev/null || find . -name "milk-fpsexec-trtinfer_datagen" | head -n 1)

if [ -z "$DATAGEN_BIN" ]; then
    echo "Error: milk-fpsexec-trtinfer_datagen executable not found."
    exit 1
fi

echo "Running $DATAGEN_BIN to generate $NSAMPLES samples into $OUTNAME..."
$DATAGEN_BIN set pts_in pts_truth $NSAMPLES 0 "$OUTNAME" 0.0
$DATAGEN_BIN exec
echo "Done. Streams 'pts_in' and 'pts_truth' are now active in ImageStreamIO shared memory."
