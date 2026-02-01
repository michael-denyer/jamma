#!/bin/bash
# Generate LMM reference data using GEMMA Docker container.
#
# This script runs GEMMA's univariate LMM (-lmm 1) on the mouse_hs1940
# example dataset and saves the output to tests/fixtures/lmm/ for use
# in validation tests.
#
# Prerequisites:
# - Docker installed and running
# - tests/fixtures/kinship/mouse_hs1940.cXX.txt must exist
#
# Usage:
#   ./scripts/generate_lmm_reference.sh

set -euo pipefail

# Paths
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXAMPLE_DATA="$PROJECT_ROOT/legacy/example"
FIXTURES_DIR="$PROJECT_ROOT/tests/fixtures/lmm"
KINSHIP_FILE="$PROJECT_ROOT/tests/fixtures/kinship/mouse_hs1940.cXX.txt"

# Check prerequisites
if [ ! -f "$KINSHIP_FILE" ]; then
    echo "Error: Kinship file not found: $KINSHIP_FILE"
    echo "Run kinship generation first (Phase 2 reference data)."
    exit 1
fi

if [ ! -f "$EXAMPLE_DATA/mouse_hs1940.bed" ]; then
    echo "Error: PLINK files not found in $EXAMPLE_DATA"
    exit 1
fi

# Create output directory
mkdir -p "$FIXTURES_DIR"

echo "Generating LMM reference data..."
echo "  Input: $EXAMPLE_DATA/mouse_hs1940"
echo "  Kinship: $KINSHIP_FILE"
echo "  Output: $FIXTURES_DIR"

# Run GEMMA via Docker (using Biocontainers image)
# Mount both legacy/example (for PLINK files) and tests/fixtures (for kinship and output)
# Use -w to set working directory and -outdir to specify output location
docker run --rm \
    -v "$EXAMPLE_DATA:/data/input:ro" \
    -v "$FIXTURES_DIR:/data/output" \
    -v "$KINSHIP_FILE:/data/kinship.cXX.txt:ro" \
    -w /data/output \
    quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0 \
    gemma \
    -bfile /data/input/mouse_hs1940 \
    -k /data/kinship.cXX.txt \
    -lmm 1 \
    -outdir /data/output \
    -o mouse_hs1940

# Check for output in expected location
if [ -f "$FIXTURES_DIR/mouse_hs1940.assoc.txt" ]; then
    echo ""
    echo "Success! Reference file created:"
    echo "  $FIXTURES_DIR/mouse_hs1940.assoc.txt"
    echo ""
    echo "File preview (first 5 lines):"
    head -5 "$FIXTURES_DIR/mouse_hs1940.assoc.txt"
elif [ -f "$FIXTURES_DIR/output/mouse_hs1940.assoc.txt" ]; then
    # GEMMA created output/ subdirectory
    mv "$FIXTURES_DIR/output/mouse_hs1940.assoc.txt" "$FIXTURES_DIR/mouse_hs1940.assoc.txt"
    rm -rf "$FIXTURES_DIR/output"
    echo ""
    echo "Success! Reference file created:"
    echo "  $FIXTURES_DIR/mouse_hs1940.assoc.txt"
    echo ""
    echo "File preview (first 5 lines):"
    head -5 "$FIXTURES_DIR/mouse_hs1940.assoc.txt"
else
    echo "Error: Expected output file not found"
    ls -laR "$FIXTURES_DIR/"
    exit 1
fi

echo ""
echo "Reference data generation complete."
echo "Run pytest tests/test_lmm_validation.py to validate."
