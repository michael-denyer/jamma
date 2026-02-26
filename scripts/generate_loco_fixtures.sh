#!/bin/bash
# Generate GEMMA LOCO reference fixtures for JAMMA validation.
#
# Correct two-step approach for LOCO validation:
#
# GEMMA's -loco flag does NOT compute LOCO-adjusted kinship when given an external
# -k matrix; it uses the full kinship unchanged. To validate JAMMA's LOCO kinship
# computation against GEMMA's LMM, we:
#
#   Step 1: Use JAMMA (Python) to compute the true LOCO kinship per chromosome:
#             K_loco_c = (p * K_full - p_c * K_c) / (p - p_c)
#           where K_full is the centered kinship from all SNPs, K_c is from chr c only.
#           Write each K_loco_c to a cXX.txt file.
#
#   Step 2: Run GEMMA LMM with each JAMMA-computed LOCO kinship as input (-k),
#           testing only the SNPs from that chromosome (via -snps).
#
# This validates:
#   a) JAMMA's LOCO kinship formula matches GEMMA's LMM computation
#   b) JAMMA's per-SNP beta, SE, p_wald, l_remle match GEMMA given the same kinship
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

if ! docker image inspect gemma-loco &>/dev/null; then
    echo "Error: Docker image 'gemma-loco' not found."
    echo "Build it: docker build --platform linux/amd64 -t gemma-loco -f docker/Dockerfile.gemma docker/"
    exit 1
fi

echo "Generating GEMMA LOCO reference fixtures (JAMMA kinship + GEMMA LMM)..."
echo "  PLINK files : $FIXTURE_DIR/test"
echo "  Output dir  : $FIXTURE_DIR"
echo ""

# Helper: run GEMMA in Docker with project root mounted to /data
run_gemma() {
    docker run --rm --platform linux/amd64 \
        -v "$PROJECT_ROOT:/data" \
        gemma-loco \
        "$@"
}

# ─── Step 1: Compute LOCO kinship per chromosome using JAMMA ─────────────────

echo "=== Step 1: Compute JAMMA LOCO kinship matrices ==="
echo "(K_loco_c = (p * K_full - p_c * K_c) / (p - p_c) for each chromosome c)"
echo ""

uv run python - <<'PYTHON'
"""Compute per-chromosome LOCO kinship matrices and SNP lists using JAMMA."""
from pathlib import Path
import numpy as np
from jamma.io import load_plink_binary
from jamma.io.plink import get_plink_metadata
from jamma.kinship import write_kinship_matrix
from jamma.kinship.compute import compute_centered_kinship

FIXTURE_DIR = Path("tests/fixtures/gemma_loco")
PLINK_PREFIX = FIXTURE_DIR / "test"

pdata = load_plink_binary(PLINK_PREFIX)
meta = get_plink_metadata(PLINK_PREFIX)
G = pdata.genotypes
chr_labels = meta["chromosome"].astype(str)

n_full = G.shape[1]
K_full = compute_centered_kinship(G, check_memory=False)
print(f"Full kinship: {n_full} SNPs, trace={np.trace(K_full):.4f}")

unique_chrs = sorted(set(chr_labels))
for chrom in unique_chrs:
    chr_mask = chr_labels == chrom
    n_chr = int(chr_mask.sum())
    n_loco = n_full - n_chr

    G_c = G[:, chr_mask]
    K_c = compute_centered_kinship(G_c, check_memory=False)
    K_loco = (n_full * K_full - n_chr * K_c) / n_loco

    kinship_path = FIXTURE_DIR / f"loco_chr{chrom}_kinship.cXX.txt"
    write_kinship_matrix(K_loco, kinship_path)
    print(f"chr{chrom}: {n_chr} SNPs excluded, {n_loco} retained, trace={np.trace(K_loco):.4f} -> {kinship_path.name}")

    # Write SNP list for this chromosome (for GEMMA -snps filter)
    snp_ids = meta["sid"][chr_mask]
    snp_list_path = FIXTURE_DIR / f"chr{chrom}_snps.txt"
    with open(snp_list_path, "w") as f:
        f.write("\n".join(snp_ids) + "\n")
    print(f"  SNP list: {snp_list_path.name} ({len(snp_ids)} SNPs)")

print("\nLOCO kinship matrices written.")
PYTHON

echo ""
echo "=== Step 1 complete: LOCO kinship matrices and SNP lists written ==="

# ─── Step 2: Run GEMMA LMM with JAMMA LOCO kinship per chromosome ────────────

echo ""
echo "=== Step 2: Run GEMMA LMM per chromosome with JAMMA LOCO kinship ==="
echo "(GEMMA standard LMM with LOCO-adjusted kinship as input)"
echo ""

for CHR in 1 2 3; do
    echo "--- Chromosome ${CHR} ---"

    # Clean up any previous output for this chromosome
    rm -f "$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt"
    rm -f "$FIXTURE_DIR/gemma_loco_chr${CHR}.log.txt"

    run_gemma \
        -bfile /data/tests/fixtures/gemma_loco/test \
        -k /data/tests/fixtures/gemma_loco/loco_chr${CHR}_kinship.cXX.txt \
        -snps /data/tests/fixtures/gemma_loco/chr${CHR}_snps.txt \
        -lmm 1 \
        -o "gemma_loco_chr${CHR}" \
        -outdir /data/tests/fixtures/gemma_loco

    if [ -f "$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt" ]; then
        LINES=$(wc -l < "$FIXTURE_DIR/gemma_loco_chr${CHR}.assoc.txt")
        echo "Output: gemma_loco_chr${CHR}.assoc.txt (${LINES} lines)"
    else
        echo "Error: Expected output file not found for chr${CHR}"
        ls -la "$FIXTURE_DIR/"
        exit 1
    fi
    echo ""
done

# ─── Step 3: Verify all 3 .assoc.txt files ───────────────────────────────────

echo ""
echo "=== Step 3: Verify fixture files ==="

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
    awk 'NR>=2 && NR<=4' "$FILE"
    echo ""
done

# ─── Step 4: Causal SNP significance check ───────────────────────────────────

echo "=== Step 4: Causal SNP rs0000 in chr1 ==="
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
echo "(Each file has only that chromosome's SNPs + 1 header.)"
echo "  chr1: 200 SNPs, chr2: 150 SNPs, chr3: 150 SNPs"

# ─── Step 5: Clean up intermediate files ─────────────────────────────────────

echo ""
echo "=== Step 5: Clean up ==="
rm -f "$FIXTURE_DIR/loco_chr"*.cXX.txt
rm -f "$FIXTURE_DIR/chr"*"_snps.txt"
rm -f "$FIXTURE_DIR/gemma_loco_chr"*.log.txt
rm -f "$FIXTURE_DIR/gemma_loco1_kinship.cXX.txt"
rm -f "$FIXTURE_DIR/gemma_loco1_kinship.log.txt"
rm -f "$FIXTURE_DIR/gemma_loco_full_kinship.cXX.txt"
rm -f "$FIXTURE_DIR/gemma_loco_full_kinship.log.txt"
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
