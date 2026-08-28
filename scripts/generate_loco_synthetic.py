"""Generate multi-chromosome synthetic PLINK dataset for GEMMA LOCO validation.

Creates tests/fixtures/gemma_loco/ with:
  - test.bed / test.bim / test.fam  (100 samples, 500 SNPs, 3 chromosomes)
  - test_snps.txt                   (GEMMA annotation file: SNP_ID, bp_pos, chr)

The --loco-kinship subcommand writes the per-chromosome LOCO kinship matrices
and SNP lists that scripts/generate_gemma_fixtures.sh feeds to GEMMA as -k and
-snps inputs.

The annotation file (test_snps.txt) uses GEMMA's annotation format.
Not used by the current fixture generation pipeline (which uses -snps
per-chromosome filtering), but kept for potential GEMMA -loco -a usage.
Format: 3 tab-separated columns, no header: SNP_ID  bp_position  chromosome

Design:
  - 200 SNPs on chr 1, 150 on chr 2, 150 on chr 3
  - MAF range 0.1-0.5 (HW equilibrium, avoids GEMMA MAF filter)
  - Causal SNP rs0000 (index 0, chr 1) with effect size 0.5
  - Phenotype: 0.5 * genotype[rs0000] + N(0, 1)
  - Seed: np.random.default_rng(42)

Usage:
    uv run python scripts/generate_loco_synthetic.py
    uv run python scripts/generate_loco_synthetic.py --loco-kinship <bfile> <outdir>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SAMPLES = 100
N_SNPS = 500

# Chromosome distribution
CHR_COUNTS = {"1": 200, "2": 150, "3": 150}

EFFECT_SIZE = 0.5
SEED = 42

OUTPUT_DIR = Path(__file__).parent.parent / "tests" / "fixtures" / "gemma_loco"

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_synthetic_plink() -> None:
    """Generate multi-chromosome PLINK files with GEMMA annotation file."""
    from bed_reader import to_bed

    rng = np.random.default_rng(SEED)

    # ---- SNP metadata -------------------------------------------------------
    # Build per-SNP metadata in chromosome order
    chromosomes: list[str] = []
    for chrom, count in CHR_COUNTS.items():
        chromosomes.extend([chrom] * count)

    snp_ids = [f"rs{j:04d}" for j in range(N_SNPS)]

    # Unique bp positions across all SNPs (1000-step spacing)
    bp_positions = list(range(1000, 1000 + N_SNPS * 1000, 1000))

    # ---- Genotype simulation ------------------------------------------------
    # Sample MAF uniformly in [0.1, 0.5] per SNP
    mafs = rng.uniform(0.1, 0.5, size=N_SNPS)

    # Compute HW equilibrium allele probabilities for dosage 0, 1, 2
    # P(g=0) = (1-maf)^2, P(g=1) = 2*maf*(1-maf), P(g=2) = maf^2
    p0 = (1 - mafs) ** 2
    p1 = 2 * mafs * (1 - mafs)
    # p2 = mafs**2  (implicitly: 1 - p0 - p1)

    # Sample genotypes for each sample/SNP via multinomial
    genotypes = np.zeros((N_SAMPLES, N_SNPS), dtype=np.int8)
    for j in range(N_SNPS):
        probs = [p0[j], p1[j], 1.0 - p0[j] - p1[j]]
        dosages = rng.choice([0, 1, 2], size=N_SAMPLES, p=probs)
        genotypes[:, j] = dosages

    # ---- Phenotype ----------------------------------------------------------
    # Causal SNP is rs0000 (index 0 on chr 1)
    phenotypes = EFFECT_SIZE * genotypes[:, 0].astype(np.float64) + rng.normal(
        0, 1, N_SAMPLES
    )

    # ---- Sample IDs ---------------------------------------------------------
    fam_ids = [f"FAM{i:03d}" for i in range(N_SAMPLES)]
    ind_ids = [f"IND{i:03d}" for i in range(N_SAMPLES)]

    # ---- Write PLINK files --------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bed_path = OUTPUT_DIR / "test.bed"

    to_bed(
        str(bed_path),
        genotypes,
        properties={
            "fid": fam_ids,
            "iid": ind_ids,
            "father": ["0"] * N_SAMPLES,
            "mother": ["0"] * N_SAMPLES,
            "sex": [0] * N_SAMPLES,
            "pheno": phenotypes.tolist(),
            "chromosome": chromosomes,
            "sid": snp_ids,
            "cm_position": [0] * N_SNPS,
            "bp_position": bp_positions,
            "allele_1": ["A"] * N_SNPS,
            "allele_2": ["G"] * N_SNPS,
        },
    )

    # ---- Write GEMMA annotation file ----------------------------------------
    # Format: SNP_ID <tab> bp_position <tab> chromosome (no header)
    annotation_path = OUTPUT_DIR / "test_snps.txt"
    with open(annotation_path, "w") as f:
        for snp_id, bp_pos, chrom in zip(  # noqa: B905 -- all same length
            snp_ids, bp_positions, chromosomes
        ):
            f.write(f"{snp_id}\t{bp_pos}\t{chrom}\n")

    # ---- Summary ------------------------------------------------------------
    print("Generated LOCO synthetic dataset:")
    print(f"  Samples : {N_SAMPLES}")
    print(f"  SNPs    : {N_SNPS}")
    for chrom, count in CHR_COUNTS.items():
        print(f"    chr {chrom}: {count} SNPs")
    print(f"  Causal SNP: rs0000 (chr 1, effect size {EFFECT_SIZE})")
    print("")
    print("Output files:")
    print(f"  {OUTPUT_DIR / 'test.bed'}")
    print(f"  {OUTPUT_DIR / 'test.bim'}")
    print(f"  {OUTPUT_DIR / 'test.fam'}")
    print(f"  {annotation_path}")


def write_loco_kinship_fixtures(bfile: Path, outdir: Path) -> None:
    """Write per-chromosome LOCO kinship matrices and SNP lists for GEMMA.

    The LOCO kinship is derived by subtraction rather than by recomputing from
    the retained SNPs:

        K_loco_c = (p * K_full - p_c * K_c) / (p - p_c)

    Recomputing from ``G[:, chr != c]`` is a different operation and changes the
    committed fixture bytes, so the subtraction stands.

    Kinship is written with ``legacy_text=True`` because GEMMA reads only the
    ``.cXX.txt`` format; the JAMMA default would emit ``.npy``.

    Args:
        bfile: PLINK binary prefix (no extension).
        outdir: Directory receiving the ``.cXX.txt`` and SNP-list files.
    """
    from jamma.io import load_plink_binary
    from jamma.io.plink import get_plink_metadata
    from jamma.kinship import write_kinship_matrix
    from jamma.kinship.compute import compute_centered_kinship

    outdir.mkdir(parents=True, exist_ok=True)

    pdata = load_plink_binary(bfile)
    meta = get_plink_metadata(bfile)
    genotypes = pdata.genotypes
    chr_labels = meta.chromosome.astype(str)

    n_full = genotypes.shape[1]
    k_full = compute_centered_kinship(genotypes, check_memory=False)
    print(f"Full kinship: {n_full} SNPs, trace={np.trace(k_full):.4f}")

    for chrom in sorted(set(chr_labels)):
        chr_mask = chr_labels == chrom
        n_chr = int(chr_mask.sum())
        n_loco = n_full - n_chr

        k_chr = compute_centered_kinship(genotypes[:, chr_mask], check_memory=False)
        k_loco = (n_full * k_full - n_chr * k_chr) / n_loco

        kinship_path = outdir / f"loco_chr{chrom}_kinship.cXX.txt"
        write_kinship_matrix(k_loco, kinship_path, legacy_text=True)
        print(
            f"chr{chrom}: {n_chr} SNPs excluded, {n_loco} retained, "
            f"trace={np.trace(k_loco):.4f} -> {kinship_path.name}"
        )

        snp_ids = meta.sid[chr_mask]
        snp_list_path = outdir / f"chr{chrom}_snps.txt"
        snp_list_path.write_text("\n".join(snp_ids) + "\n")
        print(f"  SNP list: {snp_list_path.name} ({len(snp_ids)} SNPs)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--loco-kinship",
        nargs=2,
        metavar=("BFILE", "OUTDIR"),
        help="Write per-chromosome LOCO kinship matrices and SNP lists.",
    )
    args = parser.parse_args(argv)

    if args.loco_kinship:
        bfile, outdir = args.loco_kinship
        write_loco_kinship_fixtures(Path(bfile), Path(outdir))
    else:
        generate_synthetic_plink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
