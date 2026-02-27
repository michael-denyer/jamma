"""Integration tests comparing JAMMA LOCO to GEMMA LMM reference outputs.

These tests validate that JAMMA's LOCO (Leave-One-Chromosome-Out) implementation
produces results numerically equivalent to GEMMA LMM on the same LOCO-adjusted
kinship matrices, using a multi-chromosome synthetic dataset.

Dataset:
- 100 samples, 500 SNPs across 3 chromosomes (chr1: 200, chr2: 150, chr3: 150)
- Causal SNP rs0000 on chr1 with effect size 0.5
- Phenotype: 0.5 * genotype[rs0000] + N(0, 1)

Validation approach:
- For each chromosome c, JAMMA computes K_loco_c = (p*K_full - p_c*K_c) / (p - p_c)
- Each K_loco_c is written to disk and used as input to GEMMA standard LMM
- GEMMA LMM with K_loco_c as external kinship is the reference
- JAMMA LOCO (run_lmm_loco) is validated against this reference per chromosome

This validates both:
  a) JAMMA's LOCO kinship formula matches GEMMA's LMM expectations
  b) JAMMA's per-SNP beta, SE, p_wald, l_remle match GEMMA given the same kinship

GEMMA's -loco flag does NOT compute LOCO-adjusted kinship when given an external -k
matrix (it uses the full kinship unchanged). The fixtures were generated via:
  scripts/generate_loco_fixtures.sh
which uses JAMMA to compute LOCO kinship and GEMMA for LMM association testing.

Fixture data:
- tests/fixtures/gemma_loco/test.{bed,bim,fam} - PLINK files
- tests/fixtures/gemma_loco/test_snps.txt - GEMMA annotation file
- tests/fixtures/gemma_loco/gemma_loco_chr{1,2,3}.assoc.txt - GEMMA LMM results
  with JAMMA LOCO kinship (chr-specific SNPs only, 200/150/150 rows respectively)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from tests.conftest import load_phenotypes_from_fam

pytest.importorskip("jax")

from jamma.lmm.loco import run_lmm_loco

# LOCO requires JAX backend (eigendecomp + batch Uab)
pytestmark = pytest.mark.requires_jax

# Fixture paths
_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
FIXTURE_DIR = _FIXTURE_ROOT / "gemma_loco"
PLINK_PREFIX = FIXTURE_DIR / "test"

# GEMMA reference fixture files (one per chromosome, containing only that chr's SNPs)
GEMMA_CHR_ASSOC = {
    "1": FIXTURE_DIR / "gemma_loco_chr1.assoc.txt",
    "2": FIXTURE_DIR / "gemma_loco_chr2.assoc.txt",
    "3": FIXTURE_DIR / "gemma_loco_chr3.assoc.txt",
}

# Expected SNP counts per chromosome
EXPECTED_SNP_COUNTS = {"1": 200, "2": 150, "3": 150}

_CHROMOSOMES = ["1", "2", "3"]


@pytest.mark.tier1
class TestGemmaLocoFixtureProperties:
    """Verify fixture data integrity before running expensive LOCO comparisons."""

    def test_fixture_files_exist(self):
        """All 7 fixture files exist: 3 PLINK + annotation + 3 assoc.txt."""
        assert PLINK_PREFIX.with_suffix(".bed").exists(), "Missing test.bed"
        assert PLINK_PREFIX.with_suffix(".bim").exists(), "Missing test.bim"
        assert PLINK_PREFIX.with_suffix(".fam").exists(), "Missing test.fam"
        assert (FIXTURE_DIR / "test_snps.txt").exists(), "Missing test_snps.txt"
        for chrom, path in GEMMA_CHR_ASSOC.items():
            assert path.exists(), f"Missing gemma_loco_chr{chrom}.assoc.txt"

    def test_plink_dimensions(self):
        """FAM file has 100 samples; BIM file has 500 SNPs."""
        with open(PLINK_PREFIX.with_suffix(".fam")) as f:
            n_samples = sum(1 for _ in f)
        with open(PLINK_PREFIX.with_suffix(".bim")) as f:
            n_snps = sum(1 for _ in f)
        assert n_samples == 100, f"Expected 100 samples, got {n_samples}"
        assert n_snps == 500, f"Expected 500 SNPs, got {n_snps}"

    def test_multi_chromosome_distribution(self):
        """BIM file has expected SNP counts: chr1=200, chr2=150, chr3=150."""
        chr_counts: dict[str, int] = {}
        with open(PLINK_PREFIX.with_suffix(".bim")) as f:
            for line in f:
                fields = line.strip().split()
                chrom = fields[0]
                chr_counts[chrom] = chr_counts.get(chrom, 0) + 1

        assert chr_counts.get("1", 0) == 200, (
            f"Expected 200 SNPs on chr1, got {chr_counts.get('1', 0)}"
        )
        assert chr_counts.get("2", 0) == 150, (
            f"Expected 150 SNPs on chr2, got {chr_counts.get('2', 0)}"
        )
        assert chr_counts.get("3", 0) == 150, (
            f"Expected 150 SNPs on chr3, got {chr_counts.get('3', 0)}"
        )

    def test_annotation_file_format(self):
        """test_snps.txt has 500 lines with 3 tab-separated columns.

        This file is not consumed by the LOCO validation tests but is checked
        for format consistency (generated for potential GEMMA -loco -a usage).
        """
        with open(FIXTURE_DIR / "test_snps.txt") as f:
            lines = f.readlines()
        assert len(lines) == 500, f"Expected 500 annotation lines, got {len(lines)}"
        for i, line in enumerate(lines[:5]):  # spot-check first 5
            fields = line.strip().split("\t")
            assert len(fields) == 3, (
                f"Line {i + 1}: expected 3 tab-sep cols, got {len(fields)}: {line!r}"
            )

    def test_fixture_snp_counts(self):
        """Each fixture file has the expected number of SNPs (chr-specific)."""
        for chrom, path in GEMMA_CHR_ASSOC.items():
            df = pd.read_csv(path, sep="\t")
            expected = EXPECTED_SNP_COUNTS[chrom]
            assert len(df) == expected, (
                f"chr{chrom} fixture: expected {expected} rows, got {len(df)}"
            )

    def test_fixture_chr_homogeneity(self):
        """Each fixture file contains only SNPs from its own chromosome."""
        for chrom, path in GEMMA_CHR_ASSOC.items():
            df = pd.read_csv(path, sep="\t")
            unique_chrs = df["chr"].astype(str).unique()
            assert len(unique_chrs) == 1 and unique_chrs[0] == chrom, (
                f"chr{chrom} fixture has unexpected chromosomes: {unique_chrs}"
            )

    def test_causal_snp_in_chr1_reference(self):
        """rs0000 exists in gemma_loco_chr1.assoc.txt with p_wald < 0.01.

        The chr1 fixture uses K_loco_chr1 (chr2+chr3 kinship, excluding chr1).
        rs0000 is the causal SNP on chr1; it should still be significant because
        the association is in the genotype (not in the kinship).
        """
        df = pd.read_csv(GEMMA_CHR_ASSOC["1"], sep="\t")
        rs0000_rows = df[df["rs"] == "rs0000"]
        assert len(rs0000_rows) == 1, (
            f"Expected 1 row for rs0000, got {len(rs0000_rows)}"
        )
        p_wald = rs0000_rows["p_wald"].iloc[0]
        assert p_wald < 0.01, (
            f"Causal SNP rs0000 p_wald = {p_wald:.4e}, expected < 0.01"
        )


@pytest.mark.tier1
class TestGemmaLocoValidation:
    """Validate JAMMA LOCO against GEMMA LMM reference with LOCO-adjusted kinship.

    Each chromosome's JAMMA LOCO results are compared to the corresponding
    gemma_loco_chrN.assoc.txt fixture, which was generated by running GEMMA LMM
    with JAMMA's LOCO-adjusted kinship (K_loco_N) as input.
    """

    @pytest.fixture(scope="class")
    def gemma_loco_assoc(self) -> dict[str, pd.DataFrame]:
        """Load per-chromosome GEMMA reference .assoc.txt files.

        Returns dict keyed by chromosome label ("1", "2", "3"). Each fixture
        contains only that chromosome's SNPs, tested with the LOCO-adjusted kinship
        (K_loco_c computed by JAMMA).
        """
        return {
            chrom: pd.read_csv(path, sep="\t")
            for chrom, path in GEMMA_CHR_ASSOC.items()
        }

    @pytest.fixture(scope="class")
    def jamma_loco_results(self) -> dict[str, pd.DataFrame]:
        """Run JAMMA LOCO and return results grouped by chromosome.

        Returns dict keyed by chromosome label, values are DataFrames with
        columns: rs, beta, se, p_wald, l_remle, logl_H1.
        """
        phenotypes = load_phenotypes_from_fam(PLINK_PREFIX.with_suffix(".fam"))
        results, _n_tested = run_lmm_loco(
            bed_path=PLINK_PREFIX,
            phenotypes=phenotypes,
            lmm_mode=1,
            maf_threshold=0.0,
            miss_threshold=1.0,
            show_progress=False,
            check_memory=False,
        )

        # Group by chromosome
        by_chr: dict[str, list] = {}
        for r in results:
            by_chr.setdefault(r.chr, []).append(r)

        # Convert each chromosome's results to DataFrame
        return {
            chrom: pd.DataFrame(
                [
                    {
                        "rs": r.rs,
                        "beta": r.beta,
                        "se": r.se,
                        "p_wald": r.p_wald,
                        "l_remle": r.l_remle,
                        "logl_H1": r.logl_H1,
                    }
                    for r in assoc_list
                ]
            )
            for chrom, assoc_list in by_chr.items()
        }

    @pytest.fixture(scope="class")
    def merged_per_chr(
        self,
        gemma_loco_assoc: dict[str, pd.DataFrame],
        jamma_loco_results: dict[str, pd.DataFrame],
    ) -> dict[str, pd.DataFrame]:
        """Merge GEMMA and JAMMA DataFrames on 'rs' for each chromosome.

        Returns dict[str, pd.DataFrame] with suffixes ("_gemma", "_jamma").
        Asserts merge is lossless (no dropped rows from ID mismatch).
        """
        result = {}
        for chrom in _CHROMOSOMES:
            merged = pd.merge(
                gemma_loco_assoc[chrom],
                jamma_loco_results[chrom],
                on="rs",
                suffixes=("_gemma", "_jamma"),
                how="inner",
            )
            expected = len(gemma_loco_assoc[chrom])
            assert len(merged) == expected, (
                f"chr{chrom}: inner join dropped rows — "
                f"GEMMA has {expected} SNPs but merge produced {len(merged)}. "
                f"Check SNP ID alignment between GEMMA and JAMMA results."
            )
            result[chrom] = merged
        return result

    # ------------------------------------------------------------------
    # Structural tests
    # ------------------------------------------------------------------

    def test_all_chromosomes_present(self, jamma_loco_results: dict[str, pd.DataFrame]):
        """JAMMA returns results for all 3 chromosomes."""
        assert set(jamma_loco_results.keys()) == {"1", "2", "3"}, (
            f"Expected chromosomes {{1, 2, 3}}, got {set(jamma_loco_results.keys())}"
        )

    def test_loco_snp_count_per_chromosome(
        self,
        gemma_loco_assoc: dict[str, pd.DataFrame],
        jamma_loco_results: dict[str, pd.DataFrame],
    ):
        """JAMMA tests the same number of SNPs per chromosome as GEMMA."""
        for chrom in _CHROMOSOMES:
            n_gemma = len(gemma_loco_assoc[chrom])
            n_jamma = len(jamma_loco_results[chrom])
            assert n_jamma == n_gemma, (
                f"chr{chrom}: JAMMA={n_jamma} SNPs, GEMMA={n_gemma} SNPs"
            )

    # ------------------------------------------------------------------
    # LOCO-03: Per-chromosome numerical parity
    # ------------------------------------------------------------------

    def test_loco_beta_per_chromosome(self, merged_per_chr: dict[str, pd.DataFrame]):
        """LOCO-03: Per-chromosome beta matches GEMMA within 1e-5 absolute tolerance.

        Tighter than the general 1e-2 relative tolerance (EQUIVALENCE.md) because
        the synthetic dataset has clean signal and avoids flat-optima edge cases.
        """
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            max_diff = np.max(np.abs(merged["beta_gemma"] - merged["beta_jamma"]))
            assert max_diff < 1e-5, (
                f"chr{chrom}: beta max abs diff {max_diff:.2e} >= 1e-5"
            )

    def test_loco_se_per_chromosome(self, merged_per_chr: dict[str, pd.DataFrame]):
        """LOCO-03: Per-chromosome SE matches GEMMA within 1e-6 absolute tolerance."""
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            max_diff = np.max(np.abs(merged["se_gemma"] - merged["se_jamma"]))
            assert max_diff < 1e-6, (
                f"chr{chrom}: SE max abs diff {max_diff:.2e} >= 1e-6"
            )

    def test_loco_pwald_per_chromosome(self, merged_per_chr: dict[str, pd.DataFrame]):
        """LOCO-03: Per-chromosome p_wald matches GEMMA within 1e-5 tolerance."""
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            max_diff = np.max(np.abs(merged["p_wald_gemma"] - merged["p_wald_jamma"]))
            assert max_diff < 1e-5, (
                f"chr{chrom}: p_wald max abs diff {max_diff:.2e} >= 1e-5"
            )

    # ------------------------------------------------------------------
    # LOCO-04: Lambda (variance ratio) parity
    # ------------------------------------------------------------------

    def test_loco_lambda_per_chromosome(self, merged_per_chr: dict[str, pd.DataFrame]):
        """LOCO-04: Per-chromosome l_remle matches GEMMA within 2e-4 tolerance.

        Note: On this synthetic dataset (100 samples, weak signal), l_remle
        converges to the lower bound (1e-5) for all SNPs on all chromosomes.
        This test is a structural check only — it cannot distinguish a correct
        optimizer from one that always returns l_min. Real-data validation with
        non-boundary lambda values is in test_gemma_integration.py.
        """
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            max_diff = np.max(np.abs(merged["l_remle_gemma"] - merged["l_remle_jamma"]))
            assert max_diff < 2e-4, (
                f"chr{chrom}: l_remle max abs diff {max_diff:.2e} >= 2e-4"
            )

    # ------------------------------------------------------------------
    # LOCO-04b: Log-likelihood parity
    # ------------------------------------------------------------------

    def test_loco_logl_h1_per_chromosome(self, merged_per_chr: dict[str, pd.DataFrame]):
        """LOCO-04b: Per-chromosome logl_H1 matches GEMMA within 1e-4 tolerance."""
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            max_diff = np.max(np.abs(merged["logl_H1_gemma"] - merged["logl_H1_jamma"]))
            assert max_diff < 1e-4, (
                f"chr{chrom}: logl_H1 max abs diff {max_diff:.2e} >= 1e-4"
            )

    # ------------------------------------------------------------------
    # LOCO-05: Rank correlation
    # ------------------------------------------------------------------

    def test_loco_pvalue_rank_correlation(
        self, merged_per_chr: dict[str, pd.DataFrame]
    ):
        """LOCO-05: Spearman rank correlation of p-values across all chromosomes >= 0.9999."""  # noqa: E501
        all_gemma_pvals = []
        all_jamma_pvals = []
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            all_gemma_pvals.extend(merged["p_wald_gemma"].tolist())
            all_jamma_pvals.extend(merged["p_wald_jamma"].tolist())

        rho, _ = spearmanr(all_gemma_pvals, all_jamma_pvals)
        assert rho > 0.9999, f"P-value Spearman rank correlation {rho:.6f} < 0.9999"

    # ------------------------------------------------------------------
    # LOCO-06: Top hits match
    # ------------------------------------------------------------------

    def test_loco_top_hits_per_chromosome(
        self, merged_per_chr: dict[str, pd.DataFrame]
    ):
        """LOCO-06: Top-5 hits per chromosome match exactly between GEMMA and JAMMA."""
        for chrom in _CHROMOSOMES:
            merged = merged_per_chr[chrom]
            gemma_top5 = set(merged.nsmallest(5, "p_wald_gemma")["rs"])
            jamma_top5 = set(merged.nsmallest(5, "p_wald_jamma")["rs"])
            assert gemma_top5 == jamma_top5, (
                f"chr{chrom}: top-5 mismatch: GEMMA={gemma_top5}, JAMMA={jamma_top5}"
            )

    # ------------------------------------------------------------------
    # LOCO-07: Causal SNP detection
    # ------------------------------------------------------------------

    def test_causal_snp_top_hit_in_loco(self, merged_per_chr: dict[str, pd.DataFrame]):
        """LOCO-07: Causal SNP rs0000 is in top 5 chr1 hits and significant.

        The chr1 fixture uses K_loco_chr1 (chr2+chr3 kinship, excludes chr1).
        With LOCO-adjusted kinship, other chr1 SNPs in LD with rs0000 may rank
        higher, but rs0000 must still be in the top 5 and significant (p < 0.01)
        in both JAMMA and GEMMA results.
        """
        merged_chr1 = merged_per_chr["1"]

        # rs0000 must appear in top 5 hits by p_wald (not necessarily #1)
        gemma_top5 = set(merged_chr1.nsmallest(5, "p_wald_gemma")["rs"])
        jamma_top5 = set(merged_chr1.nsmallest(5, "p_wald_jamma")["rs"])

        assert "rs0000" in gemma_top5, (
            f"GEMMA chr1: rs0000 not in top-5 hits: {gemma_top5}"
        )
        assert "rs0000" in jamma_top5, (
            f"JAMMA chr1: rs0000 not in top-5 hits: {jamma_top5}"
        )

        # rs0000 must be significant (p_wald < 0.01)
        rs0000_row = merged_chr1[merged_chr1["rs"] == "rs0000"]
        assert len(rs0000_row) == 1, "rs0000 not found in merged chr1 results"

        gemma_pval = rs0000_row["p_wald_gemma"].iloc[0]
        jamma_pval = rs0000_row["p_wald_jamma"].iloc[0]

        assert gemma_pval < 0.01, (
            f"GEMMA chr1 rs0000 p_wald = {gemma_pval:.4e}, expected < 0.01"
        )
        assert jamma_pval < 0.01, (
            f"JAMMA chr1 rs0000 p_wald = {jamma_pval:.4e}, expected < 0.01"
        )
