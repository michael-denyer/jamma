#!/bin/bash
# Generate ALL missing GEMMA reference fixtures for the complete 16-cell test matrix.
#
# Test matrix: 4 modes x 2 covariate configs x 2 datasets = 16 cells
#
# MANUAL EXECUTION ONLY - This script requires GEMMA via Docker or local binary.
# CI environments typically do not have GEMMA available.
#
# Prerequisites:
# - GEMMA binary available (via Docker or local installation)
# - Synthetic test data in tests/fixtures/gemma_synthetic/
# - Mouse HS1940 data in tests/fixtures/mouse_hs1940/
# - Covariate files in tests/fixtures/gemma_covariate/ and mouse_hs1940/
#
# Output (9 NEW fixture files):
#   Synthetic with covariates:
#   - tests/fixtures/gemma_covariate/gemma_covariate_lrt.assoc.txt
#   - tests/fixtures/gemma_covariate/gemma_covariate_score.assoc.txt
#   Mouse HS1940 without covariates:
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_lrt.assoc.txt
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_score.assoc.txt
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_all.assoc.txt
#   Mouse HS1940 with covariates:
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_covar_wald.assoc.txt
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_covar_lrt.assoc.txt
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_covar_score.assoc.txt
#   - tests/fixtures/mouse_hs1940/mouse_hs1940_covar_all.assoc.txt
#
# Usage:
#   ./scripts/generate_all_gemma_fixtures.sh
#
# After generating, commit the output files to the repository so that
# validation tests can run in CI without requiring GEMMA.

set -euo pipefail

# ─── Detect GEMMA execution method ───────────────────────────────────────────

run_gemma() {
    if [ "$GEMMA_CMD" = "docker" ]; then
        docker run --rm --platform linux/amd64 -v "$(pwd):/data" gemma gemma "$@"
    else
        gemma "$@"
    fi
}

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
    echo "  Docker: docker pull quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0"
    echo "          docker tag quay.io/biocontainers/gemma:0.98.5--ha36d3ea_0 gemma"
    echo "  Local:  See https://github.com/genetics-statistics/GEMMA"
    exit 1
fi

# ─── Paths (Docker-mapped) ──────────────────────────────────────────────────

# Synthetic dataset
SYNTH_BFILE="/data/tests/fixtures/gemma_synthetic/test"
SYNTH_K="/data/tests/fixtures/gemma_synthetic/gemma_kinship.cXX.txt"
SYNTH_C="/data/tests/fixtures/gemma_covariate/covariates.txt"
SYNTH_COVAR_OUT="/data/tests/fixtures/gemma_covariate"

# Mouse HS1940 dataset
MOUSE_BFILE="/data/tests/fixtures/mouse_hs1940/mouse_hs1940"
MOUSE_C="/data/tests/fixtures/mouse_hs1940/covariates.txt"
MOUSE_OUT="/data/tests/fixtures/mouse_hs1940"

# ─── Step 1: Generate mouse_hs1940 kinship ──────────────────────────────────

echo ""
echo "=== Step 1: Generate mouse_hs1940 kinship matrix ==="

if [ -f "tests/fixtures/mouse_hs1940/mouse_hs1940_kinship.cXX.txt" ]; then
    echo "Kinship already exists, skipping."
else
    run_gemma \
        -bfile "${MOUSE_BFILE}" \
        -gk 1 \
        -o mouse_hs1940_kinship \
        -outdir "${MOUSE_OUT}"
fi

MOUSE_K="${MOUSE_OUT}/mouse_hs1940_kinship.cXX.txt"

# ─── Step 2: Generate mouse_hs1940 fixtures (no covariates) ─────────────────

echo ""
echo "=== Step 2: Mouse HS1940 — no covariates ==="

echo "--- Mode 2 (LRT) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -lmm 2 \
    -o mouse_hs1940_lrt \
    -outdir "${MOUSE_OUT}"

echo "--- Mode 3 (Score) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -lmm 3 \
    -o mouse_hs1940_score \
    -outdir "${MOUSE_OUT}"

echo "--- Mode 4 (All) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -lmm 4 \
    -o mouse_hs1940_all \
    -outdir "${MOUSE_OUT}"

# ─── Step 3: Generate mouse_hs1940 fixtures (with covariates) ───────────────

echo ""
echo "=== Step 3: Mouse HS1940 — with covariates ==="

echo "--- Mode 1 (Wald) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -c "${MOUSE_C}" \
    -lmm 1 \
    -o mouse_hs1940_covar_wald \
    -outdir "${MOUSE_OUT}"

echo "--- Mode 2 (LRT) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -c "${MOUSE_C}" \
    -lmm 2 \
    -o mouse_hs1940_covar_lrt \
    -outdir "${MOUSE_OUT}"

echo "--- Mode 3 (Score) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -c "${MOUSE_C}" \
    -lmm 3 \
    -o mouse_hs1940_covar_score \
    -outdir "${MOUSE_OUT}"

echo "--- Mode 4 (All) ---"
run_gemma \
    -bfile "${MOUSE_BFILE}" \
    -k "${MOUSE_K}" \
    -c "${MOUSE_C}" \
    -lmm 4 \
    -o mouse_hs1940_covar_all \
    -outdir "${MOUSE_OUT}"

# ─── Step 4: Generate synthetic fixtures (covariates, missing modes) ────────

echo ""
echo "=== Step 4: Synthetic — with covariates (missing modes) ==="

echo "--- Mode 2 (LRT) ---"
run_gemma \
    -bfile "${SYNTH_BFILE}" \
    -k "${SYNTH_K}" \
    -c "${SYNTH_C}" \
    -lmm 2 \
    -o gemma_covariate_lrt \
    -outdir "${SYNTH_COVAR_OUT}"

echo "--- Mode 3 (Score) ---"
run_gemma \
    -bfile "${SYNTH_BFILE}" \
    -k "${SYNTH_K}" \
    -c "${SYNTH_C}" \
    -lmm 3 \
    -o gemma_covariate_score \
    -outdir "${SYNTH_COVAR_OUT}"

# ─── Step 5: Verify all fixtures ────────────────────────────────────────────

echo ""
echo "=== Step 5: Verify all fixture files ==="

PASS=true

verify_columns() {
    local file="$1"
    shift
    local columns=("$@")

    if [ ! -f "$file" ]; then
        echo "FAIL: $file does not exist"
        PASS=false
        return
    fi

    local header
    header=$(head -1 "$file")
    for col in "${columns[@]}"; do
        if echo "$header" | grep -qw "$col"; then
            :
        else
            echo "FAIL: $file missing column '$col'"
            PASS=false
            return
        fi
    done

    local lines
    lines=$(wc -l < "$file")
    echo "OK: $(basename "$file") — ${lines} lines, columns: ${columns[*]}"
}

echo ""
echo "--- Synthetic with covariates ---"
verify_columns "tests/fixtures/gemma_covariate/gemma_covariate_lrt.assoc.txt" l_mle p_lrt
verify_columns "tests/fixtures/gemma_covariate/gemma_covariate_score.assoc.txt" p_score

echo ""
echo "--- Mouse HS1940 without covariates ---"
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_lrt.assoc.txt" l_mle p_lrt
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_score.assoc.txt" p_score
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_all.assoc.txt" p_wald p_lrt p_score

echo ""
echo "--- Mouse HS1940 with covariates ---"
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_covar_wald.assoc.txt" beta se p_wald
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_covar_lrt.assoc.txt" l_mle p_lrt
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_covar_score.assoc.txt" p_score
verify_columns "tests/fixtures/mouse_hs1940/mouse_hs1940_covar_all.assoc.txt" p_wald p_lrt p_score

if [ "$PASS" = true ]; then
    echo ""
    echo "All 9 new fixtures verified successfully."
    echo ""
    echo "Next step: Commit these files:"
    echo "  git add tests/fixtures/gemma_covariate/gemma_covariate_lrt.assoc.txt"
    echo "  git add tests/fixtures/gemma_covariate/gemma_covariate_score.assoc.txt"
    echo "  git add tests/fixtures/mouse_hs1940/*.assoc.txt"
    echo "  git add tests/fixtures/mouse_hs1940/mouse_hs1940_kinship.cXX.txt"
else
    echo ""
    echo "WARNING: Some fixtures failed verification"
    exit 1
fi
