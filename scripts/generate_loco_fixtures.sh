#!/bin/bash
# Generate GEMMA LOCO reference fixtures for JAMMA validation.
#
# Runs GEMMA git HEAD (-loco support) via Docker to produce per-chromosome
# .assoc.txt files for tests/fixtures/gemma_loco/. These reference files
# are used by test_gemma_loco_integration.py.
#
# GEMMA -loco requires:
#   1. A pre-computed kinship matrix (-k): GEMMA computes LOCO kinship internally
#      by subtracting the tested chromosome's contribution from the full kinship.
#   2. An annotation file (-a): 3 tab-separated columns (SNP_ID, bp_position, chr)
#
# Prerequisites:
#   - Docker running with gemma-loco image built:
#       docker build --platform linux/amd64 -t gemma-loco -f docker/Dockerfile.gemma docker/
#   - PLINK files and annotation file in tests/fixtures/gemma_loco/:
#       uv run python scripts/generate_loco_synthetic.py
#
# Usage:
#   bash scripts/generate_loco_fixtures.sh

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE_DIR="$PROJECT_ROOT/tests/fixtures/gemma_loco"

# ─── Prerequisite checks ──────────────────────────────────────────────────────

if [ ! -f "$FIXTURE_DIR/test.bed" ]; then
    echo "Error: PLINK files not found in $FIXTURE_DIR"
    echo "Run: uv run python scripts/generate_loco_synthetic.py"
    exit 1
fi

if [ ! -f "$FIXTURE_DIR/test_snps.txt" ]; then
    echo "Error: Annotation file not found: $FIXTURE_DIR/test_snps.txt"
    echo "Run: uv run python scripts/generate_loco_synthetic.py"
    exit 1
fi

if ! docker image inspect gemma-loco &>/dev/null; then
    echo "Error: Docker image 'gemma-loco' not found."
    echo "Build it: docker build --platform linux/amd64 -t gemma-loco -f docker/Dockerfile.gemma docker/"
    exit 1
fi

echo "Generating GEMMA LOCO reference fixtures..."
echo "  PLINK files : $FIXTURE_DIR/test"
echo "  Annotation  : $FIXTURE_DIR/test_snps.txt"
echo "  Output dir  : $FIXTURE_DIR"
echo ""

# Helper: run GEMMA in Docker with project root mounted to /data
run_gemma() {
    docker run --rm --platform linux/amd64 \
        -v "$PROJECT_ROOT:/data" \
        gemma-loco \
        "$@"
}

# ─── Step 1: Compute full kinship matrix ─────────────────────────────────────

echo "=== Step 1: Compute full kinship matrix ==="

# Clean up any previous kinship
rm -f "$FIXTURE_DIR/gemma_loco_kinship.cXX.txt"
rm -f "$FIXTURE_DIR/gemma_loco_kinship.log.txt"
rm -rf "$FIXTURE_DIR/output"

run_gemma \
    -bfile /data/tests/fixtures/gemma_loco/test \
    -gk 1 \
    -o gemma_loco_kinship \
    -outdir /data/tests/fixtures/gemma_loco

# Handle GEMMA's output/ subdirectory quirk (check both locations)
if [ -f "$FIXTURE_DIR/output/gemma_loco_kinship.cXX.txt" ]; then
    mv "$FIXTURE_DIR/output/gemma_loco_kinship.cXX.txt" "$FIXTURE_DIR/"
    rm -rf "$FIXTURE_DIR/output"
fi

if [ ! -f "$FIXTURE_DIR/gemma_loco_kinship.cXX.txt" ]; then
    echo "Error: Kinship file not created."
    ls -la "$FIXTURE_DIR/"
    exit 1
fi
echo "Kinship matrix computed: $FIXTURE_DIR/gemma_loco_kinship.cXX.txt"

# ─── Step 2: Smoke test GEMMA PLINK + LOCO with kinship ──────────────────────

echo ""
echo "=== Step 2: Smoke test GEMMA PLINK + LOCO ==="

SMOKE_OUT="$FIXTURE_DIR/smoke_test"
mkdir -p "$SMOKE_OUT"

if run_gemma \
    -bfile /data/tests/fixtures/gemma_loco/test \
    -k /data/tests/fixtures/gemma_loco/gemma_loco_kinship.cXX.txt \
    -a /data/tests/fixtures/gemma_loco/test_snps.txt \
    -loco 1 \
    -lmm 1 \
    -o gemma_smoke \
    -outdir /data/tests/fixtures/gemma_loco/smoke_test 2>&1; then
    echo "Smoke test: PLINK + LOCO + kinship works"
    rm -rf "$SMOKE_OUT"
else
    echo ""
    echo "ERROR: GEMMA PLINK + LOCO smoke test failed."
    rm -rf "$SMOKE_OUT"
    exit 1
fi

# ─── Step 3: Run GEMMA -loco for each chromosome ─────────────────────────────

echo ""
echo "=== Step 3: Run GEMMA -loco per chromosome ==="

for CHR in 1 2 3; do
    echo ""
    echo "--- Chromosome ${CHR} ---"

    # Clean up any previous output
    rm -f "$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt"
    rm -f "$FIXTURE_DIR/gemma_loco_chr${CHR}.log.txt"
    rm -rf "$FIXTURE_DIR/output"

    run_gemma \
        -bfile /data/tests/fixtures/gemma_loco/test \
        -k /data/tests/fixtures/gemma_loco/gemma_loco_kinship.cXX.txt \
        -a /data/tests/fixtures/gemma_loco/test_snps.txt \
        -loco "${CHR}" \
        -lmm 1 \
        -o "gemma_loco_chr${CHR}" \
        -outdir /data/tests/fixtures/gemma_loco

    # Handle GEMMA's output/ subdirectory quirk
    if [ -f "$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt" ]; then
        echo "Output: $FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt"
    elif [ -f "$FIXTURE_DIR/output/gemma_loco_chr${CHR}.assoc.txt" ]; then
        mv "$FIXTURE_DIR/output/gemma_loco_chr${CHR}.assoc.txt" "$FIXTURE_DIR/"
        rm -rf "$FIXTURE_DIR/output"
        echo "Output: $FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt (moved from output/)"
    else
        echo "Error: Expected output file not found for chr${CHR}"
        ls -laR "$FIXTURE_DIR/"
        exit 1
    fi
done

# ─── Step 4: Verify all 3 .assoc.txt files ───────────────────────────────────

echo ""
echo "=== Step 4: Verify fixture files ==="

PASS=true

for CHR in 1 2 3; do
    FILE="$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt"

    if [ ! -f "$FILE" ]; then
        echo "FAIL: $FILE does not exist"
        PASS=false
        continue
    fi

    HEADER=$(head -1 "$FILE")
    LINES=$(wc -l < "$FILE")

    # Check required columns
    for COL in chr rs ps beta se p_wald l_remle; do
        if ! echo "$HEADER" | grep -qw "$COL"; then
            echo "FAIL: $FILE missing column '$COL'"
            PASS=false
        fi
    done

    echo "OK: gemma_loco_chr${CHR}.assoc.txt — ${LINES} lines"
    echo "  Header: $HEADER"
    echo "  First 3 data rows:"
    # Read first 3 data rows (skip header) without SIGPIPE from head-within-pipe
    awk 'NR>=2 && NR<=4' "$FILE"
done

# ─── Step 5: Causal SNP significance check ───────────────────────────────────

echo ""
echo "=== Step 5: Causal SNP rs0000 significance ==="
echo "Expected: p_wald < 0.01 (causal effect size 0.5, 100 samples)"
if grep "rs0000" "$FIXTURE_DIR/gemma_loco_chr1.assoc.txt"; then
    :
else
    echo "WARNING: rs0000 not found in chr1 output"
fi

echo ""
echo "=== Row counts ==="
for CHR in 1 2 3; do
    wc -l "$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt"
done
echo "(Each file has all 500 SNPs + 1 header = 501 lines. GEMMA -loco with -k outputs"
echo " all SNPs for each run; each run uses a different LOCO-adjusted kinship.)"

# ─── Step 6: Clean up intermediate files ─────────────────────────────────────

echo ""
echo "=== Step 6: Clean up ==="
rm -f "$FIXTURE_DIR/gemma_loco_kinship.cXX.txt"
rm -f "$FIXTURE_DIR/gemma_loco_kinship.log.txt"
rm -f "$FIXTURE_DIR/gemma_loco_chr"*.log.txt
rm -rf "$FIXTURE_DIR/output"
echo "Intermediate files removed."
echo "Kept: test.{bed,bim,fam}, test_snps.txt, gemma_loco_chr{1,2,3}.assoc.txt"

# ─── Final summary ────────────────────────────────────────────────────────────

if [ "$PASS" = true ]; then
    echo ""
    echo "LOCO fixture generation complete."
    echo ""
    ls -la "$FIXTURE_DIR/"
else
    echo ""
    echo "WARNING: Fixture verification failed. Check output above."
    exit 1
fi
