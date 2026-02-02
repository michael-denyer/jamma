#!/bin/bash
# Generate GEMMA Score test (-lmm 3) reference output for validation.
#
# MANUAL EXECUTION ONLY - This script requires GEMMA via Docker or local binary.
# CI environments typically do not have GEMMA available.
#
# Prerequisites:
# - GEMMA binary available (via Docker or local installation)
# - Test data in tests/fixtures/gemma_synthetic/
#
# Output:
# - tests/fixtures/gemma_score/gemma_score.assoc.txt
#
# Usage:
#   ./scripts/generate_score_reference.sh
#
# After generating, commit the output file to the repository so that
# validation tests can run in CI without requiring GEMMA.

set -euo pipefail

# Paths
FIXTURE_DIR="tests/fixtures/gemma_synthetic"
OUTPUT_DIR="tests/fixtures/gemma_score"
BFILE="${FIXTURE_DIR}/test"
KINSHIP="${FIXTURE_DIR}/gemma_kinship.cXX.txt"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run GEMMA Score test
# Try Docker first, fall back to local binary
if command -v docker &> /dev/null && docker image inspect gemma &> /dev/null; then
    echo "Running GEMMA via Docker..."
    docker run --rm -v "$(pwd):/data" \
        gemma \
        -bfile /data/${BFILE} \
        -k /data/${KINSHIP} \
        -lmm 3 \
        -o gemma_score \
        -outdir /data/${OUTPUT_DIR}
elif command -v gemma &> /dev/null; then
    echo "Running local GEMMA binary..."
    gemma \
        -bfile ${BFILE} \
        -k ${KINSHIP} \
        -lmm 3 \
        -o gemma_score \
        -outdir ${OUTPUT_DIR}
else
    echo "Error: Neither Docker (with gemma image) nor GEMMA binary available"
    echo ""
    echo "To install GEMMA:"
    echo "  Docker: docker pull genetics-statistics/gemma"
    echo "  Local:  See https://github.com/genetics-statistics/GEMMA"
    exit 1
fi

echo "Generated: ${OUTPUT_DIR}/gemma_score.assoc.txt"

# Verify output has expected format (p_score column)
if grep -q "p_score" "${OUTPUT_DIR}/gemma_score.assoc.txt"; then
    echo "Validation: Output has p_score column"
    echo ""
    echo "Next step: Commit this file to enable CI validation tests:"
    echo "  git add ${OUTPUT_DIR}/gemma_score.assoc.txt"
else
    echo "Warning: Output may not have expected format"
fi
