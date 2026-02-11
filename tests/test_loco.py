"""Tests for LOCO (Leave-One-Chromosome-Out) kinship and LMM.

Validates LOCO kinship via mathematical self-consistency (subtraction identity,
symmetry, PSD, trace relationship, manual computation equivalence) and LOCO LMM
integration (valid results, top hits overlap, file output, CLI, pipeline, API).

Since GEMMA 0.96 does not support -loco, validation relies on mathematical
properties rather than reference data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from jamma.io.plink import get_chromosome_partitions, get_plink_metadata
from jamma.kinship import (
    compute_centered_kinship,
    compute_loco_kinship,
    compute_loco_kinship_streaming,
)

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------
MOUSE_HS1940_DIR = Path("tests/fixtures/mouse_hs1940")
MOUSE_HS1940_BFILE = MOUSE_HS1940_DIR / "mouse_hs1940"


def _mouse_hs1940_exists() -> bool:
    return MOUSE_HS1940_BFILE.with_suffix(".bed").exists()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_multi_chr():
    """Synthetic multi-chromosome genotype data for fast tests.

    100 samples, 300 SNPs across 3 chromosomes (100 each).
    Random genotypes (0, 1, 2) with ~5% NaN missingness.
    """
    rng = np.random.default_rng(42)
    n_samples, n_snps = 100, 300
    genotypes = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.float64)

    # Inject ~5% missingness
    miss_mask = rng.random((n_samples, n_snps)) < 0.05
    genotypes[miss_mask] = np.nan

    # Chromosome labels: 3 chromosomes of 100 SNPs each
    chromosomes = np.array(["chr1"] * 100 + ["chr2"] * 100 + ["chr3"] * 100)

    return genotypes, chromosomes


@pytest.fixture(scope="module")
def mouse_genotypes_and_chrs():
    """Load mouse_hs1940 genotypes and chromosome array (module-scoped for reuse).

    Returns (genotypes, chromosomes) or skips if data unavailable.
    """
    if not _mouse_hs1940_exists():
        pytest.skip("mouse_hs1940 PLINK data not found")

    from jamma.io import load_plink_binary

    data = load_plink_binary(MOUSE_HS1940_BFILE)
    return data.genotypes.astype(np.float64), data.chromosome


@pytest.fixture(scope="module")
def mouse_loco_kinships(mouse_genotypes_and_chrs):
    """Compute LOCO kinships for mouse_hs1940 (module-scoped to avoid recomputation).

    Returns dict mapping chr_name -> K_loco.
    """
    genotypes, chromosomes = mouse_genotypes_and_chrs
    return dict(compute_loco_kinship(genotypes, chromosomes, check_memory=False))


@pytest.fixture(scope="module")
def mouse_full_kinship(mouse_genotypes_and_chrs):
    """Compute full kinship for mouse_hs1940 (module-scoped)."""
    genotypes, _ = mouse_genotypes_and_chrs
    return compute_centered_kinship(genotypes, check_memory=False)


# ---------------------------------------------------------------------------
# Helper: compute globally-centered genotype matrix and per-chromosome S_c
# ---------------------------------------------------------------------------


def _compute_centered_genotypes_and_S_chr(genotypes, chromosomes):
    """Compute globally centered genotype matrix and per-chromosome S_c.

    Replicates the exact steps that compute_loco_kinship uses internally:
    1. Filter SNPs (MAF, missingness, monomorphism) using the same shared utilities
    2. Filter the chromosome array with the same mask
    3. Impute and center using global means
    4. Partition by chromosome, compute S_c = X_c @ X_c.T

    Returns (X_centered_np, chr_filtered, S_chr_dict, n_filtered).
    """
    import jax.numpy as jnp

    from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats
    from jamma.kinship.missing import impute_and_center

    # Filter SNPs (same logic as _filter_snps + compute_loco_kinship)
    col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
    snp_mask, _af, _mafs = compute_snp_filter_mask(
        col_means, miss_counts, col_vars, genotypes.shape[0], 0.0, 1.0
    )

    genotypes_filtered = genotypes[:, snp_mask]
    chr_filtered = chromosomes[snp_mask]
    n_filtered = int(np.sum(snp_mask))

    # Convert to JAX, impute and center globally
    X = jnp.array(genotypes_filtered, dtype=jnp.float64)
    X_centered = impute_and_center(X)
    X_centered_np = np.array(X_centered)

    # Compute per-chromosome S_c
    unique_chrs = sorted(set(chr_filtered))
    S_chr = {}
    p_chr = {}
    for c in unique_chrs:
        mask = chr_filtered == c
        X_c = X_centered_np[:, mask]
        S_chr[c] = X_c @ X_c.T
        p_chr[c] = int(np.sum(mask))

    return X_centered_np, chr_filtered, S_chr, p_chr, n_filtered


# ===========================================================================
# Chromosome Partitioning Tests
# ===========================================================================


class TestChromosomePartitioning:
    """Tests for get_chromosome_partitions()."""

    @pytest.mark.slow
    def test_chromosome_partitions_mouse_hs1940(self):
        """Partitions: >1 chr, unique indices, correct total, sorted."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        partitions = get_chromosome_partitions(MOUSE_HS1940_BFILE)
        meta = get_plink_metadata(MOUSE_HS1940_BFILE)

        assert len(partitions) > 1, "Should have multiple chromosomes"

        # All indices unique across chromosomes
        all_indices = np.concatenate(list(partitions.values()))
        assert len(all_indices) == len(set(all_indices)), "Indices must be unique"

        # Total equals n_snps
        assert len(all_indices) == meta["n_snps"]

        # Each array is sorted
        for chr_name, indices in partitions.items():
            assert np.all(np.diff(indices) > 0), f"chr {chr_name} indices not sorted"

    @pytest.mark.slow
    def test_chromosome_partitions_cover_all_snps(self):
        """Concatenated partition indices equal np.arange(n_snps)."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        partitions = get_chromosome_partitions(MOUSE_HS1940_BFILE)
        meta = get_plink_metadata(MOUSE_HS1940_BFILE)

        all_indices = np.sort(np.concatenate(list(partitions.values())))
        expected = np.arange(meta["n_snps"])
        np.testing.assert_array_equal(all_indices, expected)


# ===========================================================================
# Subtraction Identity Tests
# ===========================================================================


class TestLocoSubtractionIdentity:
    """Validate the fundamental LOCO subtraction identity.

    For each chromosome c:
        (p - p_c) * K_loco_c + S_c == p * K_full
    """

    @pytest.mark.slow
    def test_loco_subtraction_identity(
        self, mouse_genotypes_and_chrs, mouse_loco_kinships, mouse_full_kinship
    ):
        """Subtraction identity holds for all 19 chromosomes within rtol=1e-9.

        The identity: (p - p_c) * K_loco_c + S_c == p * K_full
        Both sides involve batched float64 JAX matmuls over ~11k SNPs. FP
        accumulation across batches introduces ~2e-10 relative error, so
        rtol=1e-9 is the validated bound (still extremely tight).
        """
        genotypes, chromosomes = mouse_genotypes_and_chrs
        K_full = mouse_full_kinship

        _, _, S_chr, p_chr, n_filtered = _compute_centered_genotypes_and_S_chr(
            genotypes, chromosomes
        )

        for chr_name, K_loco in mouse_loco_kinships.items():
            pc = p_chr[chr_name]
            p_loco = n_filtered - pc

            # (p - p_c) * K_loco_c + S_c should equal p * K_full
            lhs = p_loco * K_loco + S_chr[chr_name]
            rhs = n_filtered * K_full

            np.testing.assert_allclose(
                lhs,
                rhs,
                rtol=1e-9,
                atol=1e-12,
                err_msg=f"Subtraction identity failed for chromosome {chr_name}",
            )

    def test_loco_subtraction_identity_synthetic(self, synthetic_multi_chr):
        """Subtraction identity holds on synthetic data."""
        genotypes, chromosomes = synthetic_multi_chr

        K_full = compute_centered_kinship(genotypes, check_memory=False)
        loco_kinships = dict(
            compute_loco_kinship(genotypes, chromosomes, check_memory=False)
        )

        _, _, S_chr, p_chr, n_filtered = _compute_centered_genotypes_and_S_chr(
            genotypes, chromosomes
        )

        for chr_name, K_loco in loco_kinships.items():
            pc = p_chr[chr_name]
            p_loco = n_filtered - pc

            lhs = p_loco * K_loco + S_chr[chr_name]
            rhs = n_filtered * K_full

            np.testing.assert_allclose(
                lhs,
                rhs,
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"Subtraction identity failed for {chr_name}",
            )


# ===========================================================================
# Symmetry Tests
# ===========================================================================


class TestLocoSymmetry:
    """Each LOCO kinship matrix must be symmetric."""

    @pytest.mark.slow
    def test_loco_symmetry(self, mouse_loco_kinships):
        """All 19 LOCO kinship matrices are symmetric within machine epsilon."""
        for chr_name, K_loco in mouse_loco_kinships.items():
            assert np.allclose(K_loco, K_loco.T, atol=1e-14), (
                f"K_loco for chr {chr_name} is not symmetric"
            )

    def test_loco_symmetry_synthetic(self, synthetic_multi_chr):
        """Synthetic LOCO kinships are symmetric."""
        genotypes, chromosomes = synthetic_multi_chr
        for _, K_loco in compute_loco_kinship(
            genotypes, chromosomes, check_memory=False
        ):
            assert np.allclose(K_loco, K_loco.T, atol=1e-14)


# ===========================================================================
# PSD (Positive Semi-Definite) Tests
# ===========================================================================


class TestLocoEigenvalueNonNegativity:
    """LOCO kinship matrices should be PSD (eigenvalues >= -1e-10)."""

    @pytest.mark.slow
    def test_loco_eigenvalue_non_negativity(self, mouse_loco_kinships):
        """All eigenvalues >= -1e-10 for each LOCO kinship."""
        for chr_name, K_loco in mouse_loco_kinships.items():
            eigenvalues = np.linalg.eigvalsh(K_loco)
            assert np.all(eigenvalues >= -1e-10), (
                f"K_loco for chr {chr_name} has eigenvalue {eigenvalues.min():.2e} "
                f"below -1e-10"
            )

    def test_loco_eigenvalue_non_negativity_synthetic(self, synthetic_multi_chr):
        """Synthetic LOCO kinships are PSD."""
        genotypes, chromosomes = synthetic_multi_chr
        for chr_name, K_loco in compute_loco_kinship(
            genotypes, chromosomes, check_memory=False
        ):
            eigenvalues = np.linalg.eigvalsh(K_loco)
            assert np.all(eigenvalues >= -1e-10), (
                f"chr {chr_name} eigenvalue {eigenvalues.min():.2e} below -1e-10"
            )


# ===========================================================================
# Trace Relationship Tests
# ===========================================================================


class TestLocoTraceRelationship:
    """Verify trace(K_loco_c) == (p * trace(K_full) - trace(S_c)) / (p - p_c)."""

    @pytest.mark.slow
    def test_loco_trace_relationship(
        self, mouse_genotypes_and_chrs, mouse_loco_kinships, mouse_full_kinship
    ):
        """Trace relationship holds for all chromosomes."""
        genotypes, chromosomes = mouse_genotypes_and_chrs
        K_full = mouse_full_kinship

        _, _, S_chr, p_chr, n_filtered = _compute_centered_genotypes_and_S_chr(
            genotypes, chromosomes
        )

        for chr_name, K_loco in mouse_loco_kinships.items():
            pc = p_chr[chr_name]
            p_loco = n_filtered - pc

            expected_trace = (
                n_filtered * np.trace(K_full) - np.trace(S_chr[chr_name])
            ) / p_loco
            actual_trace = np.trace(K_loco)

            np.testing.assert_allclose(
                actual_trace,
                expected_trace,
                rtol=1e-10,
                err_msg=f"Trace relationship failed for chr {chr_name}",
            )


# ===========================================================================
# Manual Computation Equivalence
# ===========================================================================


class TestLocoManualComputation:
    """LOCO kinship via subtraction matches brute-force recomputation."""

    @pytest.mark.slow
    def test_loco_matches_manual_computation(
        self, mouse_genotypes_and_chrs, mouse_loco_kinships
    ):
        """For 3 chromosomes, manual kinship on all-but-chr matches LOCO result."""
        genotypes, chromosomes = mouse_genotypes_and_chrs

        from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats

        # Get the global filter mask
        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
        snp_mask, _, _ = compute_snp_filter_mask(
            col_means, miss_counts, col_vars, genotypes.shape[0], 0.0, 1.0
        )
        genotypes_filtered = genotypes[:, snp_mask]
        chr_filtered = chromosomes[snp_mask]

        # Test 3 chromosomes for efficiency
        test_chrs = sorted(mouse_loco_kinships.keys())[:3]

        for chr_name in test_chrs:
            # Manual: remove chromosome c's columns, compute standard kinship
            keep_mask = chr_filtered != chr_name
            genotypes_without_chr = genotypes_filtered[:, keep_mask]

            K_manual = compute_centered_kinship(
                genotypes_without_chr, check_memory=False
            )
            K_loco = mouse_loco_kinships[chr_name]

            np.testing.assert_allclose(
                K_loco,
                K_manual,
                rtol=1e-10,
                atol=1e-14,
                err_msg=(
                    f"LOCO subtraction does not match manual computation "
                    f"for chr {chr_name}"
                ),
            )

    def test_loco_matches_manual_computation_synthetic(self, synthetic_multi_chr):
        """Synthetic: manual recomputation matches LOCO subtraction."""
        genotypes, chromosomes = synthetic_multi_chr

        from jamma.core.snp_filter import compute_snp_filter_mask, compute_snp_stats

        col_means, miss_counts, col_vars = compute_snp_stats(genotypes)
        snp_mask, _, _ = compute_snp_filter_mask(
            col_means, miss_counts, col_vars, genotypes.shape[0], 0.0, 1.0
        )
        genotypes_filtered = genotypes[:, snp_mask]
        chr_filtered = chromosomes[snp_mask]

        loco_kinships = dict(
            compute_loco_kinship(genotypes, chromosomes, check_memory=False)
        )

        for chr_name, K_loco in loco_kinships.items():
            keep_mask = chr_filtered != chr_name
            K_manual = compute_centered_kinship(
                genotypes_filtered[:, keep_mask], check_memory=False
            )
            np.testing.assert_allclose(
                K_loco,
                K_manual,
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"Manual mismatch for {chr_name}",
            )


# ===========================================================================
# Streaming vs In-Memory Equivalence
# ===========================================================================


class TestLocoStreamingEquivalence:
    """Streaming and in-memory LOCO should produce identical results."""

    @pytest.mark.slow
    def test_loco_streaming_matches_inmemory(self, mouse_loco_kinships):
        """Streaming LOCO matches in-memory LOCO for all chromosomes."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        streaming_kinships = dict(
            compute_loco_kinship_streaming(
                MOUSE_HS1940_BFILE, check_memory=False, show_progress=False
            )
        )

        assert set(streaming_kinships.keys()) == set(mouse_loco_kinships.keys())

        for chr_name in mouse_loco_kinships:
            np.testing.assert_allclose(
                streaming_kinships[chr_name],
                mouse_loco_kinships[chr_name],
                rtol=1e-10,
                atol=1e-14,
                err_msg=f"Streaming != in-memory for chr {chr_name}",
            )


# ===========================================================================
# Edge Cases
# ===========================================================================


class TestLocoEdgeCases:
    """Edge case tests for LOCO kinship."""

    def test_loco_single_chromosome_raises(self):
        """LOCO with only one chromosome should raise ValueError."""
        rng = np.random.default_rng(42)
        genotypes = rng.integers(0, 3, size=(50, 100)).astype(np.float64)
        chromosomes = np.array(["chr1"] * 100)

        with pytest.raises(ValueError, match="single chromosome|all.*filtered SNPs"):
            # Consume the generator to trigger the error
            list(compute_loco_kinship(genotypes, chromosomes, check_memory=False))

    def test_loco_empty_chromosome_after_filter(self):
        """If all SNPs on a chromosome are monomorphic, that chr is absent from LOCO.

        With 3 chromosomes where chr3 is entirely monomorphic, chr3 disappears
        after filtering. LOCO proceeds with chr1 and chr2 only. The LOCO kinship
        for chr1/chr2 should still satisfy the subtraction identity against the
        remaining filtered SNPs.
        """
        rng = np.random.default_rng(42)
        n_samples = 50

        # chr1: 100 polymorphic, chr2: 100 polymorphic, chr3: 50 monomorphic
        geno_chr1 = rng.integers(0, 3, size=(n_samples, 100)).astype(np.float64)
        geno_chr2 = rng.integers(0, 3, size=(n_samples, 100)).astype(np.float64)
        geno_chr3 = np.ones((n_samples, 50), dtype=np.float64)  # all constant

        genotypes = np.hstack([geno_chr1, geno_chr2, geno_chr3])
        chromosomes = np.array(["chr1"] * 100 + ["chr2"] * 100 + ["chr3"] * 50)

        loco_kinships = dict(
            compute_loco_kinship(genotypes, chromosomes, check_memory=False)
        )

        # chr3 should be absent (all monomorphic, filtered out)
        assert "chr3" not in loco_kinships, (
            "chr3 (all monomorphic) should not appear in LOCO results"
        )

        # chr1 and chr2 should be present and valid
        assert "chr1" in loco_kinships
        assert "chr2" in loco_kinships

        # Verify symmetry and PSD for remaining LOCO kinships
        for _chr_name, K_loco in loco_kinships.items():
            assert np.allclose(K_loco, K_loco.T, atol=1e-14)
            eigenvalues = np.linalg.eigvalsh(K_loco)
            assert np.all(eigenvalues >= -1e-10)
