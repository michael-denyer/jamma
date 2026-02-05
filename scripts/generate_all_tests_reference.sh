#!/bin/bash
# Generate GEMMA All-Tests (-lmm 4) reference output for validation.
#
# MANUAL EXECUTION ONLY - This script requires GEMMA via Docker or local binary.
# CI environments typically do not have GEMMA available.
#
# Prerequisites:
# - GEMMA binary available (via Docker or local installation)
# - Test data in tests/fixtures/gemma_synthetic/
# - Covariate data in tests/fixtures/gemma_covariate/
#
# Output:
# - tests/fixtures/gemma_all_tests/gemma_all.assoc.txt (intercept-only)
# - tests/fixtures/gemma_all_tests/gemma_all_covar.assoc.txt (with covariates)
#
# Usage:
#   ./scripts/generate_all_tests_reference.sh
#
# After generating, commit the output files to the repository so that
# validation tests can run in CI without requiring GEMMA.

set -euo pipefail

# Paths
FIXTURE_DIR="tests/fixtures/gemma_synthetic"
COVAR_DIR="tests/fixtures/gemma_covariate"
OUTPUT_DIR="tests/fixtures/gemma_all_tests"
BFILE="${FIXTURE_DIR}/test"
KINSHIP="${FIXTURE_DIR}/gemma_kinship.cXX.txt"
COVARIATES="${COVAR_DIR}/covariates.txt"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Detect GEMMA execution method
if command -v docker &> /dev/null && docker image inspect gemma &> /dev/null; then
    GEMMA_CMD="docker"
    echo "Running GEMMA via Docker..."
elif command -v gemma &> /dev/null; then
    GEMMA_CMD="local"
    echo "Running local GEMMA binary..."
else
    echo "Error: Neither Docker (with gemma image) nor GEMMA binary available"
    echo ""
    echo "To install GEMMA:"
    echo "  Docker: docker pull genetics-statistics/gemma"
    echo "  Local:  See https://github.com/genetics-statistics/GEMMA"
    exit 1
fi

# Run GEMMA -lmm 4 (intercept-only)
echo ""
echo "=== Generating intercept-only reference ==="
if [ "$GEMMA_CMD" = "docker" ]; then
    docker run --rm -v "$(pwd):/data" \
        gemma \
        -bfile /data/${BFILE} \
        -k /data/${KINSHIP} \
        -lmm 4 \
        -o gemma_all \
        -outdir /data/${OUTPUT_DIR}
else
    gemma \
        -bfile ${BFILE} \
        -k ${KINSHIP} \
        -lmm 4 \
        -o gemma_all \
        -outdir ${OUTPUT_DIR}
fi

# Run GEMMA -lmm 4 (with covariates)
echo ""
echo "=== Generating with-covariates reference ==="
if [ "$GEMMA_CMD" = "docker" ]; then
    docker run --rm -v "$(pwd):/data" \
        gemma \
        -bfile /data/${BFILE} \
        -k /data/${KINSHIP} \
        -c /data/${COVARIATES} \
        -lmm 4 \
        -o gemma_all_covar \
        -outdir /data/${OUTPUT_DIR}
else
    gemma \
        -bfile ${BFILE} \
        -k ${KINSHIP} \
        -c ${COVARIATES} \
        -lmm 4 \
        -o gemma_all_covar \
        -outdir ${OUTPUT_DIR}
fi

# Verify output format (check for p_score in header)
echo ""
echo "=== Verifying output format ==="
PASS=true

if grep -q "p_score" "${OUTPUT_DIR}/gemma_all.assoc.txt"; then
    echo "OK: intercept-only has p_score column"
else
    echo "FAIL: intercept-only missing p_score column"
    PASS=false
fi

if grep -q "p_score" "${OUTPUT_DIR}/gemma_all_covar.assoc.txt"; then
    echo "OK: with-covariates has p_score column"
else
    echo "FAIL: with-covariates missing p_score column"
    PASS=false
fi

if [ "$PASS" = true ]; then
    echo ""
    echo "Generated:"
    echo "  ${OUTPUT_DIR}/gemma_all.assoc.txt"
    echo "  ${OUTPUT_DIR}/gemma_all_covar.assoc.txt"
    echo ""
    echo "Next step: Commit these files to enable CI validation tests:"
    echo "  git add ${OUTPUT_DIR}/gemma_all.assoc.txt ${OUTPUT_DIR}/gemma_all_covar.assoc.txt"
else
    echo ""
    echo "Warning: Output files may not have expected format"
    exit 1
fi
