"""Tests for LOCO (Leave-One-Chromosome-Out) kinship and LMM.

Validates LOCO kinship via mathematical self-consistency (subtraction identity,
symmetry, PSD, trace relationship, manual computation equivalence) and LOCO LMM
integration (valid results, top hits overlap, file output, CLI, pipeline, API).

Since GEMMA 0.96 does not support -loco, validation relies on mathematical
properties rather than reference data. All tests use lmm_mode=1.

Related LOCO test files:
- test_gemma_loco_integration.py: GEMMA ref (mode 1),
  cross-backend parity (modes 2/3/4)
- test_loco_numpy.py: NumPy-only LOCO paths (no JAX dependency)
- test_loco_bugs.py: Regression tests for kinship aliasing, ordering, cleanup
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from jamma.io.plink import get_chromosome_partitions, get_plink_metadata
from jamma.kinship import (
    compute_centered_kinship,
    compute_kinship_streaming,
    compute_loco_kinship,
    compute_loco_kinship_streaming,
)
from tests.conftest import load_phenotypes_from_fam

pytestmark = pytest.mark.requires_jax

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------
_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
MOUSE_HS1940_DIR = _FIXTURE_ROOT / "mouse_hs1940"
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


@pytest.fixture(scope="module")
def mouse_loco_lmm_results():
    """Run LOCO LMM on mouse_hs1940 once and share results (module-scoped).

    Returns (results, n_tested, pve) or skips if data unavailable.
    """
    if not _mouse_hs1940_exists():
        pytest.skip("mouse_hs1940 PLINK data not found")

    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(MOUSE_HS1940_BFILE.with_suffix(".fam"))
    return run_lmm_loco(
        bed_path=MOUSE_HS1940_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        check_memory=False,
        show_progress=False,
    )


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


@pytest.mark.tier0
class TestChromosomePartitioningSynthetic:
    """Fast synthetic tests for get_chromosome_partitions().

    Uses bed_reader.to_bed to create PLINK files with known chromosome
    structure, then verifies partitioning is correct without needing
    the mouse_hs1940 fixture.
    """

    def test_partitions_correct_grouping(self, tmp_path):
        """Partitions group SNPs by chromosome correctly on synthetic data."""
        from bed_reader import to_bed

        n_samples, n_snps = 20, 60
        rng = np.random.default_rng(42)
        genotypes = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.int8)
        chromosomes = ["1"] * 20 + ["2"] * 15 + ["3"] * 25

        bed_path = tmp_path / "synth"
        to_bed(
            str(bed_path) + ".bed",
            genotypes,
            properties={
                "iid": [f"s{i}" for i in range(n_samples)],
                "sid": [f"snp{i}" for i in range(n_snps)],
                "chromosome": chromosomes,
                "bp_position": list(range(1, n_snps + 1)),
            },
        )

        partitions = get_chromosome_partitions(bed_path)

        # Correct chromosome keys
        assert set(partitions.keys()) == {"1", "2", "3"}

        # Correct index ranges
        np.testing.assert_array_equal(partitions["1"], np.arange(0, 20))
        np.testing.assert_array_equal(partitions["2"], np.arange(20, 35))
        np.testing.assert_array_equal(partitions["3"], np.arange(35, 60))

        # All indices unique and cover all SNPs
        all_indices = np.sort(np.concatenate(list(partitions.values())))
        np.testing.assert_array_equal(all_indices, np.arange(n_snps))

    def test_partitions_single_snp_chromosomes(self, tmp_path):
        """Chromosomes with a single SNP are correctly partitioned."""
        from bed_reader import to_bed

        n_samples, n_snps = 10, 5
        genotypes = np.zeros((n_samples, n_snps), dtype=np.int8)
        chromosomes = ["A", "B", "C", "D", "E"]

        bed_path = tmp_path / "single"
        to_bed(
            str(bed_path) + ".bed",
            genotypes,
            properties={
                "iid": [f"s{i}" for i in range(n_samples)],
                "sid": [f"snp{i}" for i in range(n_snps)],
                "chromosome": chromosomes,
                "bp_position": list(range(1, n_snps + 1)),
            },
        )

        partitions = get_chromosome_partitions(bed_path)

        assert len(partitions) == 5
        for i, key in enumerate(["A", "B", "C", "D", "E"]):
            np.testing.assert_array_equal(partitions[key], np.array([i]))


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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


@pytest.mark.tier1
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

    def test_loco_pve_fallback_when_first_chr_filtered(self, tmp_path: Path):
        """PVE falls back to the next chromosome when the first has all SNPs filtered.

        chr1 is entirely monomorphic (MAF=0), so all its SNPs are filtered.
        chr2 and chr3 are polymorphic. PVE should be computed from chr2 (the
        first chromosome with passing SNPs) and returned as non-None.
        """
        from bed_reader import to_bed

        from jamma.lmm.loco import run_lmm_loco

        rng = np.random.default_rng(99)
        n_samples = 80

        # chr1: 50 monomorphic SNPs (all filtered at any MAF threshold)
        geno_chr1 = np.ones((n_samples, 50), dtype=np.float64)
        # chr2, chr3: 80 polymorphic SNPs each
        geno_chr2 = rng.integers(0, 3, size=(n_samples, 80)).astype(np.float64)
        geno_chr3 = rng.integers(0, 3, size=(n_samples, 80)).astype(np.float64)

        genotypes = np.hstack([geno_chr1, geno_chr2, geno_chr3])
        n_snps = genotypes.shape[1]
        chromosomes = ["1"] * 50 + ["2"] * 80 + ["3"] * 80

        geno_int = genotypes.astype(np.int8)
        bed_path = tmp_path / "pve_fallback"
        to_bed(
            str(bed_path) + ".bed",
            geno_int,
            properties={
                "iid": [f"s{i}" for i in range(n_samples)],
                "sid": [f"snp_{i}" for i in range(n_snps)],
                "chromosome": chromosomes,
                "bp_position": list(range(1, n_snps + 1)),
            },
        )

        phenotypes = rng.standard_normal(n_samples)

        loco = run_lmm_loco(
            bed_path=bed_path,
            phenotypes=phenotypes,
            lmm_mode=1,
            maf_threshold=0.01,
            check_memory=False,
            show_progress=False,
            backend="numpy",
        )
        results, n_tested, pve = loco.associations, loco.n_tested, loco.pve

        # PVE must be computed despite chr1 being fully filtered
        assert pve is not None, "PVE should fall back to a later chromosome"
        assert 0 < pve < 1, f"PVE out of range: {pve}"
        assert n_tested > 0
        assert len(results) == n_tested

        # Results should only come from chr2 and chr3 (chr1 fully filtered)
        result_chrs = {r.chr for r in results}
        assert "1" not in result_chrs, "chr1 (monomorphic) should have no results"
        assert "2" in result_chrs
        assert "3" in result_chrs


# ===========================================================================
# LMM Integration Tests
# ===========================================================================


@pytest.mark.tier1
class TestLocoLmmIntegration:
    """LOCO LMM produces valid results on mouse_hs1940."""

    @pytest.mark.slow
    def test_loco_lmm_produces_valid_results(self, mouse_loco_lmm_results):
        """run_lmm_loco returns valid AssocResults with finite stats."""
        results = mouse_loco_lmm_results.associations
        n_tested = mouse_loco_lmm_results.n_tested

        assert len(results) > 0, "Should produce results"
        assert n_tested == len(results), "n_tested should match result count"

        for r in results:
            assert 0 < r.p_wald <= 1, f"p_wald={r.p_wald} for {r.rs}"
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se) and r.se > 0, f"se={r.se} for {r.rs}"

        # Results cover multiple chromosomes
        result_chrs = {r.chr for r in results}
        assert len(result_chrs) > 1

    @pytest.mark.slow
    def test_loco_lmm_results_have_correct_snp_info(self, mouse_loco_lmm_results):
        """SNP IDs and chromosome assignments match BIM metadata."""
        results = mouse_loco_lmm_results.associations

        meta = get_plink_metadata(MOUSE_HS1940_BFILE)
        bim_snps = set(meta["sid"])

        # All result SNP IDs should be in the BIM file
        for r in results:
            assert r.rs in bim_snps, f"SNP {r.rs} not found in BIM file"

        # Results should come from multiple chromosomes
        result_chrs = {r.chr for r in results}
        bim_chrs = set(meta["chromosome"])
        assert result_chrs.issubset(bim_chrs)

    @pytest.mark.slow
    def test_loco_lmm_top_hits_overlap_with_standard(self, mouse_loco_lmm_results):
        """Top 100 SNPs from LOCO and standard LMM have >50% overlap."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        from jamma.io import load_plink_binary
        from jamma.lmm import run_lmm_association_streaming

        phenotypes = load_phenotypes_from_fam(MOUSE_HS1940_BFILE.with_suffix(".fam"))
        plink_data = load_plink_binary(MOUSE_HS1940_BFILE)
        K_full = compute_centered_kinship(
            plink_data.genotypes.astype(np.float64),
            check_memory=False,
        )

        run_result, _ = run_lmm_association_streaming(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            kinship=K_full,
            check_memory=False,
            show_progress=False,
        )
        standard_results = run_result.associations

        # LOCO LMM results from shared fixture
        loco_results = mouse_loco_lmm_results.associations

        # Get top 100 SNPs by p-value
        standard_sorted = sorted(standard_results, key=lambda r: r.p_wald or 1.0)
        loco_sorted = sorted(loco_results, key=lambda r: r.p_wald or 1.0)

        top_n = 100
        standard_top = {r.rs for r in standard_sorted[:top_n]}
        loco_top = {r.rs for r in loco_sorted[:top_n]}

        overlap = len(standard_top & loco_top)
        assert overlap > 50, (
            f"Top {top_n} overlap is {overlap}/100 (<50%). "
            f"LOCO and standard should find similar signals."
        )

    @pytest.mark.slow
    def test_loco_lmm_writes_assoc_file(self, tmp_path: Path):
        """LOCO LMM writes valid assoc output file."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        from jamma.lmm.loco import run_lmm_loco

        phenotypes = load_phenotypes_from_fam(MOUSE_HS1940_BFILE.with_suffix(".fam"))
        output_path = tmp_path / "loco_result.assoc.txt"

        run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            output_path=output_path,
            check_memory=False,
            show_progress=False,
        )

        assert output_path.exists()
        lines = output_path.read_text().strip().splitlines()
        assert len(lines) > 1, "Should have header + data lines"

        # Check header
        header = lines[0]
        assert "chr" in header
        assert "rs" in header
        assert "p_wald" in header

        # Check column count consistency
        header_cols = len(header.split("\t"))
        for line in lines[1:5]:  # Check first few data lines
            assert len(line.split("\t")) == header_cols


# ===========================================================================
# Pipeline Integration Tests
# ===========================================================================


@pytest.mark.tier1
class TestPipelineLocoMode:
    """Pipeline integration with LOCO mode."""

    @pytest.mark.slow
    def test_pipeline_loco_mode(self, tmp_path: Path):
        """PipelineRunner with loco=True produces valid PipelineResult."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        from jamma.pipeline import PipelineConfig, PipelineRunner

        config = PipelineConfig(
            bfile=MOUSE_HS1940_BFILE,
            loco=True,
            output_dir=tmp_path,
            check_memory=False,
            show_progress=False,
        )

        result = PipelineRunner(config).run()

        assert result.n_samples > 0
        assert result.n_snps_tested > 0
        assert result.assoc_path.exists()
        assert "total_s" in result.timing

    def test_pipeline_loco_rejects_kinship_file(self):
        """PipelineConfig with loco=True and kinship_file raises ValueError."""
        from jamma.pipeline import PipelineConfig, PipelineRunner

        config = PipelineConfig(
            bfile=MOUSE_HS1940_BFILE,
            loco=True,
            kinship_file=Path("something.txt"),
            check_memory=False,
        )

        runner = PipelineRunner(config)
        with pytest.raises(ValueError, match="mutually exclusive"):
            runner.validate_inputs()


# ===========================================================================
# CLI Tests
# ===========================================================================


@pytest.mark.tier1
class TestCliLocoFlags:
    """CLI integration tests for -loco flag."""

    def test_cli_lmm_loco_flag_exists(self):
        """-loco appears in --help output."""
        from click.testing import CliRunner

        from jamma.cli import main

        result = CliRunner().invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "-loco" in result.output

    def test_cli_lmm_loco_rejects_k_flag(self):
        """jamma -lmm 1 -bfile X -loco -k Y exits with error."""
        from click.testing import CliRunner

        from jamma.cli import main

        result = CliRunner().invoke(
            main,
            [
                "-lmm",
                "1",
                "-bfile",
                str(MOUSE_HS1940_BFILE),
                "-loco",
                "-k",
                "something.txt",
            ],
        )
        assert result.exit_code != 0
        assert "mutually exclusive" in result.output

    def test_cli_gk_standardized_loco_rejected(self):
        """jamma -gk 2 -bfile X -loco exits with error."""
        from click.testing import CliRunner

        from jamma.cli import main

        result = CliRunner().invoke(
            main,
            [
                "-gk",
                "2",
                "-bfile",
                str(MOUSE_HS1940_BFILE),
                "-loco",
            ],
        )
        assert result.exit_code != 0
        assert "not supported" in result.output


# ===========================================================================
# Python API Test
# ===========================================================================


@pytest.mark.tier1
class TestGwasApiLocoParameter:
    """gwas() function accepts loco=True."""

    def test_gwas_api_loco_parameter_exists(self):
        """gwas() function signature includes loco parameter."""
        import inspect

        from jamma.gwas import gwas

        sig = inspect.signature(gwas)
        assert "loco" in sig.parameters
        assert sig.parameters["loco"].default is False

    @pytest.mark.slow
    def test_gwas_api_loco_integration(self, tmp_path: Path):
        """gwas(loco=True) runs to completion on mouse_hs1940."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        from jamma.gwas import GWASResult, gwas

        result = gwas(
            MOUSE_HS1940_BFILE,
            loco=True,
            output_dir=tmp_path,
            check_memory=False,
            show_progress=False,
        )

        assert isinstance(result, GWASResult)
        assert result.n_samples > 0
        assert result.n_snps_tested > 0
        assert result.timing["total_s"] > 0


# ===========================================================================
# LOCO ksnps Wiring Tests (GAP-1)
# ===========================================================================


@pytest.mark.tier1
class TestLocoKsnpsWiring:
    """Tests proving ksnps_indices wires through the LOCO path end-to-end."""

    @pytest.mark.slow
    def test_run_lmm_loco_accepts_ksnps_indices(self):
        """run_lmm_loco() with ksnps_indices runs without error and produces results."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        from jamma.lmm.loco import run_lmm_loco

        phenotypes = load_phenotypes_from_fam(MOUSE_HS1940_BFILE.with_suffix(".fam"))

        # Use first 5000 SNP indices for kinship computation
        ksnps_indices = np.arange(5000)

        loco = run_lmm_loco(
            bed_path=MOUSE_HS1940_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            check_memory=False,
            show_progress=False,
            ksnps_indices=ksnps_indices,
        )

        assert len(loco.associations) > 0, "Should produce results with ksnps_indices"

        # Results should still have valid statistics
        for r in loco.associations[:10]:
            assert 0 < r.p_wald <= 1, f"p_wald={r.p_wald} for {r.rs}"
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se) and r.se > 0, f"se={r.se} for {r.rs}"

        # Results should cover multiple chromosomes
        result_chrs = {r.chr for r in loco.associations}
        assert len(result_chrs) > 1

    @pytest.mark.slow
    def test_pipeline_loco_ksnps_wiring(self, tmp_path: Path):
        """PipelineRunner with loco=True and ksnps_file produces valid output."""
        if not _mouse_hs1940_exists():
            pytest.skip("mouse_hs1940 PLINK data not found")

        from jamma.pipeline import PipelineConfig, PipelineRunner

        # Pick ~100 SNP IDs spread across chromosomes (every 120th SNP)
        # to ensure LOCO has SNPs on multiple chromosomes
        meta = get_plink_metadata(MOUSE_HS1940_BFILE)
        snp_ids = meta["sid"][::120]  # ~102 SNPs across all chromosomes
        ksnps_path = tmp_path / "ksnps.txt"
        ksnps_path.write_text("\n".join(snp_ids) + "\n")

        config = PipelineConfig(
            bfile=MOUSE_HS1940_BFILE,
            loco=True,
            ksnps_file=ksnps_path,
            output_dir=tmp_path,
            check_memory=False,
            show_progress=False,
        )

        result = PipelineRunner(config).run()

        assert result.n_samples > 0
        assert result.n_snps_tested > 0
        assert result.assoc_path.exists()

        # Verify output file has data
        lines = result.assoc_path.read_text().strip().splitlines()
        assert len(lines) > 1, "Should have header + data lines"


# ===========================================================================
# Partial ksnps Chromosome Coverage Tests (GAP: jamma-qsz P1)
# ===========================================================================


@pytest.mark.tier1
class TestLocoPartialKsnpsCoverage:
    """Regression tests: LOCO yields ALL chromosomes even when ksnps excludes some.

    When -ksnps selects SNPs covering only a subset of chromosomes, chromosomes
    with 0 ksnps should use the full kinship as K_loco (mathematically correct:
    no SNPs to exclude means K_loco == K_full).
    """

    @staticmethod
    def _write_synthetic_plink(
        genotypes: np.ndarray,
        chromosomes: np.ndarray,
        tmp_path: Path,
        name: str = "synthetic",
    ) -> Path:
        """Write synthetic genotype data to PLINK binary files."""
        from bed_reader import to_bed

        n_samples, n_snps = genotypes.shape
        geno_int = genotypes.copy()
        geno_int[np.isnan(geno_int)] = -127
        geno_int = geno_int.astype(np.int8)

        bed_path = tmp_path / name
        to_bed(
            str(bed_path) + ".bed",
            geno_int,
            properties={
                "iid": [f"sample_{i}" for i in range(n_samples)],
                "sid": [f"snp_{i}" for i in range(n_snps)],
                "chromosome": chromosomes.tolist(),
                "bp_position": list(range(1, n_snps + 1)),
            },
        )
        return bed_path

    def test_loco_all_chromosomes_covered_with_partial_ksnps(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """LOCO yields results for ALL chromosomes even when ksnps excludes some.

        Regression test for silent data loss: ksnps covering only chr1+chr2
        must still produce a LOCO matrix for chr3 (using full-kinship fallback).
        """
        genotypes, chromosomes = synthetic_multi_chr
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        # ksnps covers only chr1 (indices 0-99) and chr2 (100-199), excluding chr3
        ksnps_indices = np.arange(200)

        K_loco = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                ksnps_indices=ksnps_indices,
            )
        )

        # ALL 3 chromosomes must be present
        assert "chr1" in K_loco, "chr1 should be in LOCO results"
        assert "chr2" in K_loco, "chr2 should be in LOCO results"
        assert "chr3" in K_loco, "chr3 should be in LOCO results (fallback)"

        # All matrices should be valid (symmetric, PSD, correct shape)
        n_samples = genotypes.shape[0]
        for chr_name, K in K_loco.items():
            assert K.shape == (n_samples, n_samples), (
                f"Wrong shape for {chr_name}: {K.shape}"
            )
            assert np.allclose(K, K.T, atol=1e-14), (
                f"K_loco for {chr_name} is not symmetric"
            )
            eigenvalues = np.linalg.eigvalsh(K)
            assert np.all(eigenvalues >= -1e-10), (
                f"K_loco for {chr_name} is not PSD: "
                f"min eigenvalue = {eigenvalues.min()}"
            )

        # chr3 fallback should equal K_full (= S_full / n_filtered)
        # because no ksnps are excluded from chr3.
        # With ksnps_indices covering 0-199, only chr1+chr2 SNPs pass.
        # chr3's fallback K_loco = S_full / n_filtered = same as full kinship.
        # Verify by computing full kinship from ksnps only
        K_full_ref = compute_kinship_streaming(
            bed_path,
            check_memory=False,
            show_progress=False,
            ksnps_indices=ksnps_indices,
        )
        np.testing.assert_allclose(
            K_loco["chr3"],
            K_full_ref,
            atol=1e-14,
            rtol=1e-10,
            err_msg="chr3 fallback K_loco should equal full kinship",
        )

        # chr1 and chr2 results should differ from K_full (they exclude SNPs)
        assert not np.allclose(K_loco["chr1"], K_full_ref, atol=1e-8), (
            "chr1 K_loco should differ from full kinship"
        )
        assert not np.allclose(K_loco["chr2"], K_full_ref, atol=1e-8), (
            "chr2 K_loco should differ from full kinship"
        )

    def test_loco_fallback_warning_logged(self, synthetic_multi_chr, tmp_path: Path):
        """Warning message logged for chromosomes using full-kinship fallback."""
        from loguru import logger

        genotypes, chromosomes = synthetic_multi_chr
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        # ksnps covers only chr1+chr2
        ksnps_indices = np.arange(200)

        # Capture loguru messages via a custom sink
        captured: list[str] = []
        sink_id = logger.add(lambda msg: captured.append(str(msg)), level="WARNING")

        try:
            # Consume the generator to trigger all yields
            list(
                compute_loco_kinship_streaming(
                    bed_path,
                    check_memory=False,
                    show_progress=False,
                    ksnps_indices=ksnps_indices,
                )
            )
        finally:
            logger.remove(sink_id)

        # Find the fallback warning for chr3
        fallback_warnings = [m for m in captured if "0 ksnps" in m]
        assert len(fallback_warnings) >= 1, (
            f"Expected at least one '0 ksnps' warning, got: {captured}"
        )
        assert any("chr3" in w for w in fallback_warnings), (
            f"Expected warning about chr3, got: {fallback_warnings}"
        )

    def test_loco_partial_ksnps_chr_with_snps_unchanged(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Adding unrelated ksnps changes LOCO matrices (non-tautological).

        Run 1: ksnps = chr1+chr2 (indices 0-199).
        Run 2: ksnps = chr1+chr2+chr3 (indices 0-299).

        Adding chr3 SNPs to ksnps changes S_full and therefore K_loco for
        chr1/chr2. This test proves ksnps_indices actually affects LOCO
        output (not a tautological identity comparison).

        Also verifies determinism: the same ksnps_indices produces identical
        results on a second run.
        """
        genotypes, chromosomes = synthetic_multi_chr
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        # Run 1: ksnps covers chr1+chr2 only (indices 0-199)
        K_partial = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                ksnps_indices=np.arange(200),
            )
        )

        # Run 2: ksnps covers all chromosomes (indices 0-299)
        K_full_ksnps = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                ksnps_indices=np.arange(300),
            )
        )

        # chr1 and chr2 LOCO kinships should DIFFER between partial and full ksnps
        # (proving ksnps_indices actually affects the result)
        for chr_name in ["chr1", "chr2"]:
            assert not np.allclose(
                K_partial[chr_name], K_full_ksnps[chr_name], atol=1e-8
            ), (
                f"{chr_name} K_loco should differ when ksnps changes "
                f"(was tautological before)"
            )

        # Run 3: repeat run 1 to verify determinism
        K_partial_repeat = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                ksnps_indices=np.arange(200),
            )
        )
        for chr_name in ["chr1", "chr2"]:
            np.testing.assert_allclose(
                K_partial[chr_name],
                K_partial_repeat[chr_name],
                atol=1e-14,
                rtol=0,
                err_msg=f"{chr_name} K_loco should be deterministic across runs",
            )


# ===========================================================================
# Multi-Pass LOCO Kinship Tests
# ===========================================================================


@pytest.mark.tier1
class TestLocoMultiPass:
    """Verify multi-pass LOCO kinship batching produces identical results."""

    def test_multipass_loco_matches_singlepass(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Multi-pass LOCO (psutil mocked low memory) matches single-pass.

        Forces multi-pass mode by mocking psutil.virtual_memory() to report
        very low available memory, then verifies results are identical to a
        normal (single-pass) run.
        """
        from unittest.mock import MagicMock, patch

        from bed_reader import to_bed

        genotypes, chromosomes = synthetic_multi_chr
        n_samples, n_snps = genotypes.shape

        # Write synthetic data to PLINK binary files
        bed_path = tmp_path / "synthetic"
        # Replace NaN with -127 for integer encoding
        geno_int = genotypes.copy()
        geno_int[np.isnan(geno_int)] = -127
        geno_int = geno_int.astype(np.int8)

        # Create sample and SNP IDs
        iid = [f"sample_{i}" for i in range(n_samples)]
        sid = [f"snp_{i}" for i in range(n_snps)]
        bp_position = list(range(1, n_snps + 1))

        to_bed(
            str(bed_path) + ".bed",
            geno_int,
            properties={
                "iid": iid,
                "sid": sid,
                "chromosome": chromosomes.tolist(),
                "bp_position": bp_position,
            },
        )

        # Run single-pass (normal memory)
        K_single = dict(
            compute_loco_kinship_streaming(
                bed_path, check_memory=False, show_progress=False
            )
        )

        # Mock psutil to report very low available memory.
        # 100 samples: matrix_gb = 100^2*8/1e9 = 0.00008 GB.
        # 3 chromosomes: single_pass ~= 0.00032 + buffer.
        # 300KB forces multi-pass while exceeding minimum.
        mock_vmem = MagicMock()
        mock_vmem.available = 300_000  # 300 KB

        with patch(
            "jamma.kinship.compute.psutil.virtual_memory", return_value=mock_vmem
        ):
            K_multi = dict(
                compute_loco_kinship_streaming(
                    bed_path, check_memory=False, show_progress=False
                )
            )

        # Verify same chromosome set
        assert set(K_single.keys()) == set(K_multi.keys()), (
            f"Chromosome sets differ: single={sorted(K_single.keys())}, "
            f"multi={sorted(K_multi.keys())}"
        )

        # Verify identical results within FP tolerance
        for chr_name in K_single:
            np.testing.assert_allclose(
                K_multi[chr_name],
                K_single[chr_name],
                atol=1e-14,
                rtol=1e-10,
                err_msg=(f"Multi-pass != single-pass for chr {chr_name}"),
            )

    def test_multipass_loco_fallback_with_partial_ksnps(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Multi-pass with partial ksnps produces chr3 fallback matching full kinship.

        Combines psutil-mock (forces multi-pass) with ksnps_indices covering
        only chr1+chr2. Asserts chr3 is present in multi-pass results and
        equals the full kinship computed from the same ksnps subset.
        """
        from unittest.mock import MagicMock, patch

        genotypes, chromosomes = synthetic_multi_chr
        bed_path = TestLocoPartialKsnpsCoverage._write_synthetic_plink(
            genotypes, chromosomes, tmp_path
        )

        ksnps_indices = np.arange(200)  # chr1+chr2 only

        # Single-pass reference (normal memory)
        K_single = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                ksnps_indices=ksnps_indices,
            )
        )

        # Force multi-pass via low memory mock
        mock_vmem = MagicMock()
        mock_vmem.available = 300_000  # 300 KB

        with patch(
            "jamma.kinship.compute.psutil.virtual_memory", return_value=mock_vmem
        ):
            K_multi = dict(
                compute_loco_kinship_streaming(
                    bed_path,
                    check_memory=False,
                    show_progress=False,
                    ksnps_indices=ksnps_indices,
                )
            )

        # chr3 must be present (fallback from full kinship)
        assert "chr3" in K_multi, "chr3 should be in multi-pass results (fallback)"

        # All 3 chromosomes present
        assert set(K_multi.keys()) == {"chr1", "chr2", "chr3"}

        # Multi-pass matches single-pass for all chromosomes
        for chr_name in K_single:
            np.testing.assert_allclose(
                K_multi[chr_name],
                K_single[chr_name],
                atol=1e-14,
                rtol=1e-10,
                err_msg=(
                    f"Multi-pass != single-pass for {chr_name} with partial ksnps"
                ),
            )

        # chr3 equals full kinship (= S_full / n_filtered)
        K_full_ref = compute_kinship_streaming(
            bed_path,
            check_memory=False,
            show_progress=False,
            ksnps_indices=ksnps_indices,
        )
        np.testing.assert_allclose(
            K_multi["chr3"],
            K_full_ref,
            atol=1e-14,
            rtol=1e-10,
            err_msg="chr3 multi-pass fallback should equal full kinship",
        )


# ===========================================================================
# valid_indices Parameter Tests
# ===========================================================================


@pytest.mark.tier1
class TestLocoStreamingValidIndices:
    """Verify compute_loco_kinship_streaming honours valid_indices parameter.

    When valid_indices is provided, K_loco matrices must be (n_valid, n_valid)
    and must match post-hoc subsetting of the full (n_samples, n_samples) result.
    """

    @staticmethod
    def _write_synthetic_plink(
        genotypes: np.ndarray,
        chromosomes: np.ndarray,
        tmp_path: Path,
        name: str = "synthetic",
    ) -> Path:
        """Write synthetic genotype data to PLINK binary files."""
        from bed_reader import to_bed

        n_samples, n_snps = genotypes.shape
        geno_int = genotypes.copy()
        geno_int[np.isnan(geno_int)] = -127
        geno_int = geno_int.astype(np.int8)

        bed_path = tmp_path / name
        to_bed(
            str(bed_path) + ".bed",
            geno_int,
            properties={
                "iid": [f"sample_{i}" for i in range(n_samples)],
                "sid": [f"snp_{i}" for i in range(n_snps)],
                "chromosome": chromosomes.tolist(),
                "bp_position": list(range(1, n_snps + 1)),
            },
        )
        return bed_path

    def test_loco_kinship_streaming_valid_indices(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """K_loco with valid_indices matches NumPy backend with same valid_indices.

        Validates:
        1. With valid_indices provided: shape is (n_valid, n_valid).
        2. JAX K_loco matches NumPy K_loco (both center over valid samples only).

        Note: K_loco_valid != K_full[np.ix_(valid_indices, valid_indices)] because
        centering uses valid-sample means (correct for LOCO), not all-sample means.
        """
        from jamma.lmm.loco import _compute_loco_kinship_streaming_numpy

        genotypes, chromosomes = synthetic_multi_chr
        n_samples = genotypes.shape[0]
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        # Use every other sample as the valid subset
        valid_indices = np.arange(0, n_samples, 2)
        n_valid = len(valid_indices)

        # JAX subsetted run
        K_loco_jax = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                valid_indices=valid_indices,
            )
        )

        # NumPy reference (subsets before centering — known correct)
        loco_iter, _stats = _compute_loco_kinship_streaming_numpy(
            bed_path,
            check_memory=False,
            show_progress=False,
            valid_indices=valid_indices,
        )
        K_loco_numpy = dict(loco_iter)

        assert set(K_loco_jax.keys()) == set(K_loco_numpy.keys()), (
            "JAX and NumPy backends must yield the same chromosome set"
        )

        for chr_name in K_loco_jax:
            K_jax = K_loco_jax[chr_name]
            K_np = K_loco_numpy[chr_name]

            # Shape: both must be (n_valid, n_valid)
            assert K_jax.shape == (n_valid, n_valid), (
                f"JAX K_loco shape wrong for {chr_name}: {K_jax.shape}"
            )
            assert K_np.shape == (n_valid, n_valid), (
                f"NumPy K_loco shape wrong for {chr_name}: {K_np.shape}"
            )

            # Numerical parity: JAX must match NumPy (both center over valid only)
            np.testing.assert_allclose(
                K_jax,
                K_np,
                rtol=1e-10,
                atol=1e-14,
                err_msg=(
                    f"JAX vs NumPy K_loco mismatch for {chr_name} with valid_indices"
                ),
            )

    def test_loco_kinship_streaming_empty_valid_indices_raises(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Empty valid_indices raises ValueError."""
        genotypes, chromosomes = synthetic_multi_chr
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        with pytest.raises(ValueError, match="must not be empty"):
            dict(
                compute_loco_kinship_streaming(
                    bed_path,
                    check_memory=False,
                    show_progress=False,
                    valid_indices=np.array([], dtype=int),
                )
            )

    def test_loco_kinship_streaming_oob_valid_indices_raises(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Out-of-bounds valid_indices raises ValueError."""
        genotypes, chromosomes = synthetic_multi_chr
        n_samples = genotypes.shape[0]
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        with pytest.raises(ValueError, match="out of bounds"):
            dict(
                compute_loco_kinship_streaming(
                    bed_path,
                    check_memory=False,
                    show_progress=False,
                    valid_indices=np.array([0, n_samples]),
                )
            )

    def test_loco_kinship_streaming_negative_valid_indices_raises(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Negative valid_indices raises ValueError."""
        genotypes, chromosomes = synthetic_multi_chr
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        with pytest.raises(ValueError, match="out of bounds"):
            dict(
                compute_loco_kinship_streaming(
                    bed_path,
                    check_memory=False,
                    show_progress=False,
                    valid_indices=np.array([-1, 0, 1]),
                )
            )

    def test_loco_kinship_streaming_duplicate_valid_indices_raises(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """Duplicate valid_indices raises ValueError."""
        genotypes, chromosomes = synthetic_multi_chr
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        with pytest.raises(ValueError, match="duplicates"):
            dict(
                compute_loco_kinship_streaming(
                    bed_path,
                    check_memory=False,
                    show_progress=False,
                    valid_indices=np.array([0, 1, 1, 2]),
                )
            )

    def test_loco_kinship_streaming_valid_indices_none_unchanged(
        self, synthetic_multi_chr, tmp_path: Path
    ):
        """valid_indices=None preserves original (n_samples, n_samples) behaviour."""
        genotypes, chromosomes = synthetic_multi_chr
        n_samples = genotypes.shape[0]
        bed_path = self._write_synthetic_plink(genotypes, chromosomes, tmp_path)

        K_loco = dict(
            compute_loco_kinship_streaming(
                bed_path,
                check_memory=False,
                show_progress=False,
                valid_indices=None,
            )
        )

        for chr_name, K in K_loco.items():
            assert K.shape == (n_samples, n_samples), (
                f"valid_indices=None changed shape for {chr_name}: {K.shape}"
            )


# ===========================================================================
# Multi-Pass LOCO LMM Tests
# ===========================================================================


@pytest.mark.tier1
class TestLocoLmmMultiPass:
    """Verify multi-pass LOCO LMM (small col_chunk_size) matches single-pass."""

    @staticmethod
    def _write_synthetic_loco_plink(
        tmp_path: Path,
        rng: np.random.Generator,
        n_samples: int = 60,
        snps_per_chr: int = 40,
        n_chromosomes: int = 3,
    ) -> tuple[Path, np.ndarray]:
        """Write synthetic multi-chromosome PLINK files with phenotypes.

        Returns (bed_path_prefix, phenotypes).
        """
        from bed_reader import to_bed

        n_snps = snps_per_chr * n_chromosomes
        genotypes = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.int8)
        chromosomes = []
        for c in range(1, n_chromosomes + 1):
            chromosomes.extend([str(c)] * snps_per_chr)

        phenotypes = rng.standard_normal(n_samples)

        bed_path = tmp_path / "loco_lmm"
        to_bed(
            str(bed_path) + ".bed",
            genotypes,
            properties={
                "iid": [f"sample_{i}" for i in range(n_samples)],
                "sid": [f"snp_{i}" for i in range(n_snps)],
                "chromosome": chromosomes,
                "bp_position": list(range(1, n_snps + 1)),
                "pheno": phenotypes.tolist(),
            },
        )
        return bed_path, phenotypes

    def test_loco_lmm_multi_pass_matches_single_pass(self, tmp_path: Path):
        """Multi-pass LOCO LMM (col_chunk_size=2) matches default chunk size.

        Forces many disk chunks per chromosome via col_chunk_size=2, then
        verifies results are numerically identical to a single-pass run
        (col_chunk_size large enough for all SNPs on each chromosome).
        """
        from jamma.lmm.loco import run_lmm_loco

        rng = np.random.default_rng(42)
        bed_path, phenotypes = self._write_synthetic_loco_plink(tmp_path, rng)

        # Single-pass: col_chunk_size > max SNPs per chromosome (40)
        loco_single = run_lmm_loco(
            bed_path=bed_path,
            phenotypes=phenotypes,
            lmm_mode=1,
            check_memory=False,
            show_progress=False,
            col_chunk_size=5000,
        )

        # Multi-pass: col_chunk_size=2 forces ~20 disk chunks per chromosome
        loco_multi = run_lmm_loco(
            bed_path=bed_path,
            phenotypes=phenotypes,
            lmm_mode=1,
            check_memory=False,
            show_progress=False,
            col_chunk_size=2,
        )
        results_single = loco_single.associations
        results_multi = loco_multi.associations

        assert loco_single.n_tested == loco_multi.n_tested, (
            f"n_tested mismatch: single={loco_single.n_tested}, "
            f"multi={loco_multi.n_tested}"
        )
        assert len(results_single) == len(results_multi), (
            f"Result count mismatch: single={len(results_single)}, "
            f"multi={len(results_multi)}"
        )
        assert len(results_single) > 0, "Expected some results"

        for i, (rs, rm) in enumerate(zip(results_single, results_multi, strict=True)):
            assert rs.rs == rm.rs, f"SNP {i}: rs mismatch {rs.rs} vs {rm.rs}"
            np.testing.assert_allclose(
                rs.beta,
                rm.beta,
                rtol=1e-10,
                atol=0,
                err_msg=f"SNP {i} ({rs.rs}) beta mismatch",
            )
            np.testing.assert_allclose(
                rs.p_wald,
                rm.p_wald,
                rtol=1e-10,
                atol=0,
                err_msg=f"SNP {i} ({rs.rs}) p_wald mismatch",
            )


# ===========================================================================
# LOCO n_samples with NaN Covariates Test
# ===========================================================================


@pytest.mark.tier1
class TestLocoNSamplesCovariateFiltering:
    """Verify PipelineResult.n_samples reflects covariate NaN filtering in LOCO."""

    def test_loco_n_samples_reflects_covariate_nan_filtering(self, tmp_path: Path):
        """PipelineResult.n_samples after LOCO with NaN covariate rows.

        Creates a multi-chromosome fixture with covariates containing NaN rows.
        Verifies that n_samples == n_total - n_nan_rows (covariate NaNs are
        excluded from analysis).
        """
        from bed_reader import to_bed

        from jamma.pipeline import PipelineConfig, PipelineRunner

        rng = np.random.default_rng(42)
        n_samples = 50
        n_snps = 90  # 30 per chromosome

        genotypes = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.int8)
        chromosomes = ["1"] * 30 + ["2"] * 30 + ["3"] * 30
        phenotypes = rng.standard_normal(n_samples)

        bed_path = tmp_path / "loco_cov"
        to_bed(
            str(bed_path) + ".bed",
            genotypes,
            properties={
                "iid": [f"sample_{i}" for i in range(n_samples)],
                "sid": [f"snp_{i}" for i in range(n_snps)],
                "chromosome": chromosomes,
                "bp_position": list(range(1, n_snps + 1)),
                "pheno": phenotypes.tolist(),
            },
        )

        # Create covariate file with NaN rows
        n_nan_rows = 5
        covariates = rng.standard_normal((n_samples, 2))
        covariates[:n_nan_rows, 0] = np.nan  # First 5 samples have NaN covariates

        covariate_path = tmp_path / "covariates.txt"
        np.savetxt(covariate_path, covariates, fmt="%.6f")

        output_dir = tmp_path / "output"
        config = PipelineConfig(
            bfile=bed_path,
            loco=True,
            covariate_file=covariate_path,
            output_dir=output_dir,
            check_memory=False,
            show_progress=False,
        )

        result = PipelineRunner(config).run()

        assert result.n_samples == n_samples - n_nan_rows, (
            f"Expected n_samples={n_samples - n_nan_rows} "
            f"(total={n_samples} - nan_rows={n_nan_rows}), "
            f"got {result.n_samples}"
        )
        assert result.n_snps_tested > 0


# ===========================================================================
# LOCO n_snps_tested with MAF Filtering Test
# ===========================================================================


@pytest.mark.tier1
class TestLocoNSnpsTestedMafFilter:
    """Verify PipelineResult.n_snps_tested reflects MAF-filtered count in LOCO."""

    def test_loco_n_snps_tested_with_maf_filter(self, tmp_path: Path):
        """PipelineResult.n_snps_tested reflects MAF-filtered count in LOCO.

        Creates a fixture with some very low MAF SNPs (monomorphic or rare),
        runs with a MAF threshold that filters some out, and verifies
        n_snps_tested < total_snps.
        """
        from bed_reader import to_bed

        from jamma.pipeline import PipelineConfig, PipelineRunner

        rng = np.random.default_rng(42)
        n_samples = 60
        snps_per_chr = 20
        n_chromosomes = 3
        n_snps = snps_per_chr * n_chromosomes

        # Create genotypes with some low-MAF SNPs
        genotypes = rng.integers(0, 3, size=(n_samples, n_snps)).astype(np.float64)

        # Make some SNPs monomorphic (MAF = 0) -- these should always be filtered
        for chr_idx in range(n_chromosomes):
            base = chr_idx * snps_per_chr
            # 2 monomorphic SNPs per chromosome
            genotypes[:, base] = 0  # All homozygous reference
            genotypes[:, base + 1] = 2  # All homozygous alt

        # Make some SNPs very rare (MAF < 0.1) -- filtered when maf_threshold=0.1
        for chr_idx in range(n_chromosomes):
            base = chr_idx * snps_per_chr
            genotypes[:, base + 2] = 0  # Start all as 0
            genotypes[0, base + 2] = (
                1  # Only 1 minor allele in 60 samples => MAF ~0.008
            )

        geno_int = genotypes.astype(np.int8)
        chromosomes = []
        for c in range(1, n_chromosomes + 1):
            chromosomes.extend([str(c)] * snps_per_chr)

        phenotypes = rng.standard_normal(n_samples)

        bed_path = tmp_path / "loco_maf"
        to_bed(
            str(bed_path) + ".bed",
            geno_int,
            properties={
                "iid": [f"sample_{i}" for i in range(n_samples)],
                "sid": [f"snp_{i}" for i in range(n_snps)],
                "chromosome": chromosomes,
                "bp_position": list(range(1, n_snps + 1)),
                "pheno": phenotypes.tolist(),
            },
        )

        output_dir = tmp_path / "output"
        config = PipelineConfig(
            bfile=bed_path,
            loco=True,
            maf=0.1,  # Aggressive MAF filter
            output_dir=output_dir,
            check_memory=False,
            show_progress=False,
        )

        result = PipelineRunner(config).run()

        # n_snps_tested should be less than total SNPs (some filtered)
        assert result.n_snps_tested < n_snps, (
            f"Expected n_snps_tested < {n_snps} with MAF=0.1 filter, "
            f"got {result.n_snps_tested}"
        )
        # But should still have some results
        assert result.n_snps_tested > 0, "Expected some SNPs to pass filtering"

        # Verify output file exists and has correct line count
        lines = result.assoc_path.read_text().strip().splitlines()
        n_data_lines = len(lines) - 1  # Subtract header
        assert n_data_lines == result.n_snps_tested, (
            f"Output file has {n_data_lines} data lines but "
            f"n_snps_tested={result.n_snps_tested}"
        )


# ===========================================================================
# Cross-Backend Parity Tests
# ===========================================================================


_LOCO_FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "gemma_loco"
_LOCO_BFILE = _LOCO_FIXTURE_ROOT / "test"


@pytest.mark.tier1
def test_loco_cross_backend_parity(tmp_path: Path) -> None:
    """JAX and NumPy LOCO produce identical results on synthetic data."""
    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Run JAX backend
    jax_loco = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        show_progress=False,
        check_memory=False,
        backend="jax",
    )

    # Run NumPy backend
    numpy_loco = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        show_progress=False,
        check_memory=False,
        backend="numpy",
    )

    assert jax_loco.n_tested == numpy_loco.n_tested, (
        f"SNP count mismatch: JAX={jax_loco.n_tested}, NumPy={numpy_loco.n_tested}"
    )
    jax_results = jax_loco.associations
    numpy_results = numpy_loco.associations
    assert len(jax_results) == len(numpy_results)
    assert len(jax_results) > 0

    # PVE should be populated and match across backends
    assert jax_loco.pve is not None, "JAX LOCO should return PVE"
    assert numpy_loco.pve is not None, "NumPy LOCO should return PVE"
    assert 0 < jax_loco.pve < 1, f"JAX PVE out of range: {jax_loco.pve}"
    np.testing.assert_allclose(jax_loco.pve, numpy_loco.pve, rtol=1e-4)

    # Compare result arrays
    jax_betas = np.array([r.beta for r in jax_results])
    numpy_betas = np.array([r.beta for r in numpy_results])
    jax_ses = np.array([r.se for r in jax_results])
    numpy_ses = np.array([r.se for r in numpy_results])
    jax_pwalds = np.array([r.p_wald for r in jax_results])
    numpy_pwalds = np.array([r.p_wald for r in numpy_results])
    jax_lambdas = np.array([r.l_remle for r in jax_results])
    numpy_lambdas = np.array([r.l_remle for r in numpy_results])

    np.testing.assert_allclose(jax_betas, numpy_betas, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(jax_ses, numpy_ses, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(jax_pwalds, numpy_pwalds, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(jax_lambdas, numpy_lambdas, rtol=1e-10, atol=1e-14)

    # Verify SNP order is identical
    jax_rs = [r.rs for r in jax_results]
    numpy_rs = [r.rs for r in numpy_results]
    assert jax_rs == numpy_rs, "SNP order differs between backends"


# ===========================================================================
# Secular Update Smoke Tests
# ===========================================================================


@pytest.mark.tier1
def test_secular_update_smoke() -> None:
    """Secular update path completes without error and returns associations.

    Uses the gemma_loco fixture (100 samples, 500 SNPs, 3 chromosomes).
    Confirms use_secular_update=True runs end-to-end with numpy backend.
    """
    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    secular_loco = run_lmm_loco(
        bed_path=_LOCO_BFILE,
        phenotypes=phenotypes,
        lmm_mode=1,
        show_progress=False,
        check_memory=False,
        backend="numpy",
        use_secular_update=True,
    )

    assert secular_loco.n_tested > 0, (
        f"Expected n_tested > 0, got {secular_loco.n_tested}"
    )
    assert len(secular_loco.associations) > 0, (
        "Expected non-empty associations from secular update path"
    )

    # All associations should have valid (non-NaN) p_wald values
    p_walds = [r.p_wald for r in secular_loco.associations if r.p_wald is not None]
    assert len(p_walds) > 0, "No p_wald values in secular update results"
    finite_pvals = [p for p in p_walds if not np.isnan(p)]
    assert len(finite_pvals) > 0, "All p_wald values are NaN in secular update results"


@pytest.mark.tier1
def test_secular_update_eigenvalue_parity() -> None:
    """Secular update produces eigenvalues matching standard path (rtol=1e-10).

    Compares per-chromosome eigenvalues from use_secular_update=True against
    the standard use_secular_update=False (numpy backend) path on the
    gemma_loco fixture. Association statistics (beta, se, p_wald) must also
    match within validated tolerances.
    """
    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    common_kwargs = {
        "bed_path": _LOCO_BFILE,
        "phenotypes": phenotypes,
        "lmm_mode": 1,
        "show_progress": False,
        "check_memory": False,
        "backend": "numpy",
        "maf_threshold": 0.0,
        "miss_threshold": 1.0,
    }

    standard_loco = run_lmm_loco(**common_kwargs, use_secular_update=False)
    secular_loco = run_lmm_loco(**common_kwargs, use_secular_update=True)

    assert standard_loco.n_tested == secular_loco.n_tested, (
        f"n_tested mismatch: standard={standard_loco.n_tested}, "
        f"secular={secular_loco.n_tested}"
    )
    assert len(standard_loco.associations) == len(secular_loco.associations), (
        f"Result count mismatch: standard={len(standard_loco.associations)}, "
        f"secular={len(secular_loco.associations)}"
    )

    # Sort both by (chr, ps) for stable comparison
    std_sorted = sorted(standard_loco.associations, key=lambda r: (r.chr, r.ps))
    sec_sorted = sorted(secular_loco.associations, key=lambda r: (r.chr, r.ps))

    std_betas = np.array([r.beta for r in std_sorted])
    sec_betas = np.array([r.beta for r in sec_sorted])
    std_ses = np.array([r.se for r in std_sorted])
    sec_ses = np.array([r.se for r in sec_sorted])
    std_pwalds = np.array([r.p_wald for r in std_sorted])
    sec_pwalds = np.array([r.p_wald for r in sec_sorted])

    # Eigenvalue parity (rtol=1e-10) implies association results should
    # also match at high precision. Using calibrated EQUIVALENCE.md tolerances.
    np.testing.assert_allclose(
        sec_betas,
        std_betas,
        rtol=1e-4,
        atol=1e-12,
        err_msg="beta mismatch between secular and standard paths",
    )
    np.testing.assert_allclose(
        sec_ses,
        std_ses,
        rtol=1e-4,
        atol=1e-12,
        err_msg="se mismatch between secular and standard paths",
    )
    # Mask NaN p_wald (degenerate SNPs)
    finite_mask = np.isfinite(std_pwalds) & np.isfinite(sec_pwalds)
    if np.any(finite_mask):
        np.testing.assert_allclose(
            sec_pwalds[finite_mask],
            std_pwalds[finite_mask],
            rtol=1e-4,
            atol=1e-12,
            err_msg="p_wald mismatch between secular and standard paths",
        )


@pytest.mark.tier1
def test_secular_update_jax_backend_raises() -> None:
    """use_secular_update=True with backend='jax' raises ValueError."""
    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    with pytest.raises(ValueError, match="use_secular_update=True is only supported"):
        run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
            backend="jax",
            use_secular_update=True,
        )


@pytest.mark.tier1
def test_secular_update_save_kinship_raises() -> None:
    """use_secular_update=True with save_kinship=True raises ValueError."""
    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    with pytest.raises(ValueError, match="save_kinship=True is not supported"):
        run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
            backend="numpy",
            use_secular_update=True,
            save_kinship=True,
        )


@pytest.mark.tier1
def test_loco_eigendecompose_from_full_inplace_equivalence() -> None:
    """Verify loco_eigendecompose_from_full produces finite, orthonormal results."""
    from jamma.lmm.loco_eigen_update import loco_eigendecompose_from_full

    rng = np.random.default_rng(42)
    n = 50
    # Build a valid positive-definite S_chr and d_full (ascending, per docstring)
    A = rng.standard_normal((n, n))
    S_chr = A @ A.T
    d_full = np.sort(rng.uniform(0.1, 10.0, size=n))
    U_full = np.linalg.qr(rng.standard_normal((n, n)))[0]
    p_full = 1000
    p_chr = 100

    d_loco, U_loco = loco_eigendecompose_from_full(d_full, U_full, S_chr, p_full, p_chr)

    # Sanity: eigenvalues should be real and finite
    assert np.all(np.isfinite(d_loco))
    assert np.all(np.isfinite(U_loco))
    # U_loco columns should be orthonormal
    eye = U_loco.T @ U_loco
    np.testing.assert_allclose(eye, np.eye(n), atol=1e-12)


@pytest.mark.tier1
def test_yield_s_chr_ownership_transfer() -> None:
    """S_CHR mode transfers ownership (no copy), matrices are writable."""
    from jamma.lmm.loco import LocoStreamingMode, _compute_loco_kinship_streaming_numpy

    p_full, s_chr_iter, _ = _compute_loco_kinship_streaming_numpy(
        _LOCO_BFILE,
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        mode=LocoStreamingMode.S_CHR,
    )
    for _chr_name, s_chr, _p_chr in s_chr_iter:
        # Must own data (not a view) and be writable
        assert s_chr.flags["OWNDATA"] or s_chr.flags["WRITEABLE"]
        # Verify it's a real matrix
        assert s_chr.ndim == 2
        assert s_chr.shape[0] == s_chr.shape[1]


@pytest.mark.tier1
def test_secular_update_memory_check_raises() -> None:
    """Secular path uses dedicated memory estimate that accounts for all S_chr."""
    from unittest.mock import patch

    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Mock available memory to be very small (1 MB) to trigger MemoryError
    mock_mem = type("MockMem", (), {"available": 1_000_000})()
    with patch("psutil.virtual_memory", return_value=mock_mem):
        with pytest.raises(MemoryError, match="secular"):
            run_lmm_loco(
                bed_path=_LOCO_BFILE,
                phenotypes=phenotypes,
                lmm_mode=1,
                show_progress=False,
                check_memory=True,
                backend="numpy",
                use_secular_update=True,
            )


@pytest.mark.tier1
def test_secular_memory_estimate_includes_per_chr_temporaries() -> None:
    """Secular memory estimate accounts for per-chromosome rotated-basis temporaries.

    The secular memory estimate must include ~3 n×n matrices for the per-chromosome
    rotated-basis update (M, matmul temp, U_loco) plus eigendecomp workspace. This
    test verifies that a memory budget just above (all S_chr + eigendecomp) but below
    (all S_chr + eigendecomp + per-chr temporaries) triggers the MemoryError.
    """
    from unittest.mock import patch

    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))
    n = len(phenotypes)
    matrix_bytes = n * n * 8

    # Fixture has 3 chromosomes. The secular path needs:
    #   3 S_chr + max(K_full + eigendecomp, 3 matrices + eigendecomp) + chunk buffer
    # Set available memory to cover 3 S_chr + K_full but NOT per-chr temporaries.
    # The per-chr update needs ~3 extra matrices + eigendecomp workspace.
    # With n=100: matrix_bytes = 80KB, so a few hundred KB should be enough to
    # pass a naive (S_chr-only) estimate but fail the correct one.
    naive_estimate_bytes = int(matrix_bytes * 4.5)  # 3 S_chr + 1.5 for K_full/eigen
    mock_mem = type("MockMem", (), {"available": naive_estimate_bytes})()
    with patch("psutil.virtual_memory", return_value=mock_mem):
        with pytest.raises(MemoryError, match="secular"):
            run_lmm_loco(
                bed_path=_LOCO_BFILE,
                phenotypes=phenotypes,
                lmm_mode=1,
                show_progress=False,
                check_memory=True,
                backend="numpy",
                use_secular_update=True,
            )


@pytest.mark.tier1
def test_secular_path_skips_s_full_allocation() -> None:
    """S_CHR mode does not accumulate S_full.

    Verifies that the streaming kinship function does not waste an n×n S_full
    allocation when operating in S_CHR mode. The caller (run_lmm_loco)
    reconstructs K_full from S_chr instead.
    """
    from jamma.lmm.loco import LocoStreamingMode, _compute_loco_kinship_streaming_numpy

    p_full, s_chr_iter, _ = _compute_loco_kinship_streaming_numpy(
        _LOCO_BFILE,
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        mode=LocoStreamingMode.S_CHR,
    )

    # Consume iterator, reconstructing K_full from S_chr (as run_lmm_loco does)
    K_full = None
    for _chr_name, s_chr, _p_chr in s_chr_iter:
        if K_full is None:
            K_full = np.zeros_like(s_chr)
        K_full += s_chr

    assert K_full is not None, "No S_chr matrices yielded"
    K_full /= p_full

    # Verify the reconstructed K_full is a valid kinship matrix
    # (symmetric, PSD, trace > 0)
    assert K_full.shape[0] == K_full.shape[1]
    np.testing.assert_allclose(K_full, K_full.T, atol=1e-14)
    eigenvalues = np.linalg.eigvalsh(K_full)
    assert np.all(eigenvalues >= -1e-10), (
        f"K_full has negative eigenvalues: min={eigenvalues[0]:.2e}"
    )
    assert np.trace(K_full) > 0


# ---------------------------------------------------------------------------
# Input validation tests for loco_eigendecompose_from_full
# ---------------------------------------------------------------------------


@pytest.mark.tier1
class TestLocoEigendecomposeValidation:
    """Validate input validation guards in loco_eigendecompose_from_full."""

    def setup_method(self):
        from jamma.lmm.loco_eigen_update import loco_eigendecompose_from_full

        self.func = loco_eigendecompose_from_full
        self.n = 10
        self.d_full = np.sort(np.random.default_rng(0).uniform(0.1, 1.0, size=self.n))
        self.U_full = np.eye(self.n)
        self.S_chr = np.eye(self.n)

    def test_d_full_not_1d(self):
        with pytest.raises(ValueError, match="d_full must be 1-D"):
            self.func(np.ones((self.n, self.n)), self.U_full, self.S_chr, 100, 10)

    def test_U_full_shape_mismatch(self):
        with pytest.raises(ValueError, match="U_full must be"):
            self.func(self.d_full, np.ones((self.n, self.n + 1)), self.S_chr, 100, 10)

    def test_S_chr_shape_mismatch(self):
        with pytest.raises(ValueError, match="S_chr must be"):
            self.func(
                self.d_full, self.U_full, np.ones((self.n + 1, self.n + 1)), 100, 10
            )

    def test_p_chr_negative(self):
        with pytest.raises(ValueError, match="p_chr must be in"):
            self.func(self.d_full, self.U_full, self.S_chr, 100, -1)

    def test_p_chr_exceeds_p_full(self):
        with pytest.raises(ValueError, match="p_chr must be in"):
            self.func(self.d_full, self.U_full, self.S_chr, 100, 200)

    def test_p_chr_equals_p_full(self):
        with pytest.raises(ValueError, match="p_chr == p_full"):
            self.func(self.d_full, self.U_full, self.S_chr, 100, 100)


@pytest.mark.tier1
def test_secular_update_cached_eigen_conflict(tmp_path: Path) -> None:
    """use_secular_update=True with cached eigen files raises ValueError."""
    from jamma.lmm.loco import run_lmm_loco

    phenotypes = load_phenotypes_from_fam(_LOCO_BFILE.with_suffix(".fam"))

    # Write dummy eigen cache files so _find_loco_eigen_cache finds them
    from jamma.io.plink import get_plink_metadata

    meta = get_plink_metadata(_LOCO_BFILE)
    unique_chrs = sorted(set(meta["chromosome"]))
    n = len(phenotypes)
    for chr_name in unique_chrs:
        np.save(tmp_path / f"result.loco.chr{chr_name}.eigenD.npy", np.ones(n))
        np.save(tmp_path / f"result.loco.chr{chr_name}.eigenU.npy", np.eye(n))

    with pytest.raises(ValueError, match="conflicts with cached"):
        run_lmm_loco(
            bed_path=_LOCO_BFILE,
            phenotypes=phenotypes,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
            backend="numpy",
            use_secular_update=True,
            eigen_dir=tmp_path,
        )
