"""Tests for JAX-optimized LMM runner."""

from pathlib import Path

import numpy as np
import pytest

from jamma.kinship import compute_centered_kinship
from jamma.lmm.runner_jax import (
    MAX_SAFE_CHUNK,
    _compute_chunk_size,
    auto_tune_chunk_size,
    run_lmm_association_jax,
)
from jamma.validation import compare_assoc_results, load_gemma_assoc

# GEMMA covariate fixture paths (Score and LRT with covariates)
COVARIATE_FIXTURE_DIR = Path("tests/fixtures/gemma_covariate")
GEMMA_COVARIATE_SCORE = COVARIATE_FIXTURE_DIR / "gemma_covariate_score.assoc.txt"
GEMMA_COVARIATE_LRT = COVARIATE_FIXTURE_DIR / "gemma_covariate_lrt.assoc.txt"

# GEMMA synthetic fixture paths (used for covariate data)
FIXTURE_DIR = Path("tests/fixtures/gemma_synthetic")
COVARIATE_FILE = COVARIATE_FIXTURE_DIR / "covariates.txt"


def _make_synthetic_gwas_data(
    seed: int, n_samples: int = 200, n_snps: int = 500
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate synthetic GWAS data with simple genetic component.

    Args:
        seed: Random seed for reproducibility.
        n_samples: Number of individuals.
        n_snps: Number of SNPs.

    Returns:
        Tuple of (genotypes, phenotype, snp_info).
    """
    rng = np.random.default_rng(seed)

    mafs = rng.uniform(0.1, 0.4, n_snps)
    genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
    for j in range(n_snps):
        p = mafs[j]
        genotypes[:, j] = rng.choice(
            [0, 1, 2], size=n_samples, p=[(1 - p) ** 2, 2 * p * (1 - p), p**2]
        )

    causal_idx = rng.choice(n_snps, 10, replace=False)
    betas = rng.standard_normal(10)
    G = genotypes[:, causal_idx]
    G_std = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-8)
    genetic = G_std @ betas
    noise = rng.standard_normal(n_samples)
    phenotype = genetic + noise
    phenotype = (phenotype - phenotype.mean()) / phenotype.std()

    snp_info = [
        {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
        for j in range(n_snps)
    ]

    return genotypes, phenotype, snp_info


class TestChunkSizeComputation:
    """Tests for chunk size calculation to avoid int32 overflow."""

    def test_small_dataset_no_chunking(self):
        """Small datasets should not be chunked."""
        chunk = _compute_chunk_size(n_samples=1000, n_snps=10_000, n_grid=50)
        assert chunk == 10_000  # Full dataset

    def test_large_dataset_is_chunked(self):
        """Large datasets should be chunked to avoid int32 overflow."""
        # 100k samples × 95k SNPs would overflow without chunking
        chunk = _compute_chunk_size(n_samples=100_000, n_snps=95_000, n_grid=50)
        assert chunk < 95_000  # Should be chunked
        assert chunk >= 100  # Should be at least minimum

    def test_chunk_size_scales_with_samples(self):
        """Larger sample counts should result in smaller chunk sizes."""
        chunk_10k = _compute_chunk_size(n_samples=10_000, n_snps=100_000)
        chunk_50k = _compute_chunk_size(n_samples=50_000, n_snps=100_000)
        assert chunk_50k < chunk_10k

    def test_auto_tune_respects_max_chunk(self):
        """auto_tune_chunk_size should not exceed MAX_SAFE_CHUNK."""
        # Even with lots of memory budget, should cap at MAX_SAFE_CHUNK
        chunk = auto_tune_chunk_size(
            n_samples=1000, n_filtered=1_000_000, mem_budget_gb=100.0
        )
        assert chunk <= MAX_SAFE_CHUNK

    def test_auto_tune_respects_filtered_count(self):
        """auto_tune_chunk_size should not exceed n_filtered when above min_chunk."""
        # With n_filtered > min_chunk (1000 default), should respect n_filtered
        chunk = auto_tune_chunk_size(
            n_samples=1000, n_filtered=5000, mem_budget_gb=10.0
        )
        assert chunk <= 5000


class TestJaxRunnerBasic:
    """Basic tests for run_lmm_association_jax."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic GWAS data."""
        return _make_synthetic_gwas_data(seed=42)

    def test_returns_results(self, synthetic_data):
        """JAX runner should return list of AssocResult."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0
        assert all(hasattr(r, "beta") for r in results)
        assert all(hasattr(r, "p_wald") for r in results)

    def test_results_have_valid_values(self, synthetic_data):
        """Results should have finite, reasonable values."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )

        for r in results:
            if not np.isnan(r.beta):
                assert np.isfinite(r.beta)
                assert np.isfinite(r.se)
                assert 0 <= r.p_wald <= 1
                assert r.l_remle > 0

    def test_with_precomputed_eigen(self, synthetic_data):
        """Should accept pre-computed eigendecomposition."""
        from jamma.lmm.eigen import eigendecompose_kinship

        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)
        eigenvalues, eigenvectors = eigendecompose_kinship(kinship)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0

    def test_rejects_partial_eigendecomp(self, synthetic_data):
        """Should raise if only eigenvalues or eigenvectors provided."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)
        eigenvalues = np.ones(genotypes.shape[0])

        with pytest.raises(ValueError, match="Must provide both"):
            run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                eigenvalues=eigenvalues,
                eigenvectors=None,
                show_progress=False,
                check_memory=False,
            )


class TestJaxRunnerCleanup:
    """Tests for JAX runner cleanup to prevent SIGSEGV."""

    def test_multiple_runs_dont_accumulate_memory(self):
        """Multiple runs should not accumulate device memory."""
        import gc

        import psutil

        from jamma.kinship import compute_centered_kinship

        rng = np.random.default_rng(42)
        n_samples = 100
        n_snps = 200

        genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.float64)
        phenotype = rng.standard_normal(n_samples)
        kinship = compute_centered_kinship(genotypes)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        # Warmup and get baseline memory
        _ = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )
        gc.collect()
        baseline_mb = psutil.Process().memory_info().rss / 1e6

        # Run multiple times
        for _ in range(5):
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )
            gc.collect()

        final_mb = psutil.Process().memory_info().rss / 1e6
        delta_mb = final_mb - baseline_mb

        # Allow some variance, but should not grow significantly
        assert delta_mb < 100, f"Memory grew by {delta_mb:.0f}MB over 5 runs"


class TestJaxScoreMode:
    """Validation tests for JAX Score mode (lmm_mode=3) against NumPy runner."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic GWAS data for Score tests."""
        return _make_synthetic_gwas_data(seed=100)

    def test_score_returns_correct_fields(self, synthetic_data):
        """Score mode sets p_score and leaves p_wald/l_remle as None."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=3,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0

        for r in results:
            # Score-specific field must be set
            assert r.p_score is not None, f"p_score is None for {r.rs}"

            # Wald-specific fields must NOT be set
            assert r.p_wald is None, f"p_wald should be None in Score mode for {r.rs}"
            assert r.l_remle is None, f"l_remle should be None in Score mode for {r.rs}"

            # Beta/se are informational but should be finite
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se), f"se not finite for {r.rs}"

            # Metadata must be populated
            assert r.chr == "1"
            assert r.rs.startswith("rs")
            assert r.ps >= 0
            assert r.n_miss >= 0
            assert r.allele1 == "A"
            assert r.allele0 == "G"
            assert 0 <= r.af <= 1

    def test_score_values_valid(self, synthetic_data):
        """JAX Score p-values are finite and in valid range [0, 1]."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=3,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0, "No results returned"

        for r in results:
            assert np.isfinite(r.p_score), f"p_score not finite for {r.rs}"
            assert 0 <= r.p_score <= 1, f"p_score={r.p_score} out of [0,1] for {r.rs}"
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se), f"se not finite for {r.rs}"
            assert r.se > 0, f"se should be positive for {r.rs}"

    @pytest.mark.skipif(
        not GEMMA_COVARIATE_SCORE.exists(),
        reason="GEMMA Score+covariate fixture not available",
    )
    def test_score_with_covariates_matches_gemma(self):
        """Score mode with covariates matches GEMMA -lmm 3 -c reference."""
        from jamma.io import load_plink_binary
        from jamma.kinship.io import read_kinship_matrix

        # Load GEMMA synthetic test data (same as covariate fixture)
        plink = load_plink_binary(FIXTURE_DIR / "test")
        kinship = read_kinship_matrix(
            FIXTURE_DIR / "gemma_kinship.cXX.txt", n_samples=plink.n_samples
        )
        fam_data = np.loadtxt(FIXTURE_DIR / "test.fam", dtype=str)
        phenotypes = fam_data[:, 5].astype(float)
        covariates = np.loadtxt(COVARIATE_FILE)

        snp_info = [
            {
                "chr": str(plink.chromosome[i]),
                "rs": str(plink.sid[i]),
                "pos": int(plink.bp_position[i]),
                "a1": str(plink.allele_1[i]),
                "a0": str(plink.allele_2[i]),
            }
            for i in range(plink.n_snps)
        ]

        results_jax = run_lmm_association_jax(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=3,
            show_progress=False,
            check_memory=False,
        )

        reference = load_gemma_assoc(GEMMA_COVARIATE_SCORE)
        comparison = compare_assoc_results(results_jax, reference)
        assert comparison.passed, (
            f"JAX Score+covariates vs GEMMA failed:\n"
            f"  p_score: {comparison.p_score.message}\n"
            f"  beta: {comparison.beta.message}"
        )


class TestJaxLrtMode:
    """Validation tests for JAX LRT mode (lmm_mode=2) against NumPy runner."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic GWAS data with population structure for LRT.

        Creates 4 subpopulations with differentiated allele frequencies
        so kinship captures real structure. Phenotype has a strong
        polygenic component (h2 ~ 0.5) keeping null lambda in the
        interior of [l_min, l_max]. Boundary lambda causes Brent vs
        golden section to diverge on flat likelihood surfaces.
        """
        rng = np.random.default_rng(200)
        n_per_pop = 75
        n_pops = 4
        n_samples = n_per_pop * n_pops
        n_snps = 500

        # Generate allele freqs differentiated by population (Fst ~ 0.05)
        ancestral_freqs = rng.uniform(0.15, 0.45, n_snps)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
        for pop in range(n_pops):
            start = pop * n_per_pop
            end = (pop + 1) * n_per_pop
            drift = rng.normal(0, 0.05, n_snps)
            pop_freqs = np.clip(ancestral_freqs + drift, 0.05, 0.95)
            for j in range(n_snps):
                p = pop_freqs[j]
                genotypes[start:end, j] = rng.choice(
                    [0, 1, 2],
                    size=n_per_pop,
                    p=[(1 - p) ** 2, 2 * p * (1 - p), p**2],
                )

        # Polygenic phenotype: many causal SNPs + kinship-correlated noise
        K = compute_centered_kinship(genotypes)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(n_samples))

        # Polygenic signal through kinship (h2 ~ 0.5)
        genetic = L @ rng.standard_normal(n_samples)
        genetic = genetic / genetic.std()

        # Add fixed effects from a few causal SNPs
        causal_idx = rng.choice(n_snps, 5, replace=False)
        G_causal = genotypes[:, causal_idx]
        G_std = (G_causal - G_causal.mean(0)) / (G_causal.std(0) + 1e-8)
        fixed_effects = G_std @ rng.normal(0, 0.3, 5)

        noise = rng.standard_normal(n_samples)
        phenotype = genetic + fixed_effects + noise
        phenotype = (phenotype - phenotype.mean()) / phenotype.std()

        snp_info = [
            {
                "chr": "1",
                "rs": f"rs{j}",
                "pos": j * 1000,
                "a1": "A",
                "a0": "G",
            }
            for j in range(n_snps)
        ]

        return genotypes, phenotype, snp_info

    def test_lrt_returns_correct_fields(self, synthetic_data):
        """LRT mode sets p_lrt/l_mle, beta/se are NaN, p_wald is None."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=2,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0

        for r in results:
            # LRT-specific fields must be set
            assert r.p_lrt is not None, f"p_lrt is None for {r.rs}"
            assert r.l_mle is not None, f"l_mle is None for {r.rs}"

            # beta/se are NaN in pure LRT mode (matching GEMMA -lmm 2)
            assert np.isnan(r.beta), f"beta should be NaN in LRT mode for {r.rs}"
            assert np.isnan(r.se), f"se should be NaN in LRT mode for {r.rs}"

            # Wald-specific fields must NOT be set
            assert r.p_wald is None, f"p_wald should be None in LRT mode for {r.rs}"
            assert r.l_remle is None, f"l_remle should be None in LRT mode for {r.rs}"

    def test_lrt_values_valid(self, synthetic_data):
        """JAX LRT p-values are finite and in valid range [0, 1]."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=2,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0, "No results returned"

        for r in results:
            assert np.isfinite(r.p_lrt), f"p_lrt not finite for {r.rs}"
            assert 0 <= r.p_lrt <= 1, f"p_lrt={r.p_lrt} out of [0,1] for {r.rs}"
            assert r.l_mle is not None, f"l_mle is None for {r.rs}"
            assert np.isfinite(r.l_mle), f"l_mle not finite for {r.rs}"
            assert r.l_mle > 0, f"l_mle={r.l_mle} not positive for {r.rs}"

    @pytest.mark.skipif(
        not GEMMA_COVARIATE_LRT.exists(),
        reason="GEMMA LRT+covariate fixture not available",
    )
    def test_lrt_with_covariates_matches_gemma(self):
        """LRT mode with covariates matches GEMMA -lmm 2 -c reference."""
        from jamma.io import load_plink_binary
        from jamma.kinship.io import read_kinship_matrix
        from jamma.validation import ToleranceConfig

        # Load GEMMA synthetic test data
        plink = load_plink_binary(FIXTURE_DIR / "test")
        kinship = read_kinship_matrix(
            FIXTURE_DIR / "gemma_kinship.cXX.txt", n_samples=plink.n_samples
        )
        fam_data = np.loadtxt(FIXTURE_DIR / "test.fam", dtype=str)
        phenotypes = fam_data[:, 5].astype(float)
        covariates = np.loadtxt(COVARIATE_FILE)

        snp_info = [
            {
                "chr": str(plink.chromosome[i]),
                "rs": str(plink.sid[i]),
                "pos": int(plink.bp_position[i]),
                "a1": str(plink.allele_1[i]),
                "a0": str(plink.allele_2[i]),
            }
            for i in range(plink.n_snps)
        ]

        results_jax = run_lmm_association_jax(
            genotypes=plink.genotypes,
            phenotypes=phenotypes,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=2,
            show_progress=False,
            check_memory=False,
        )

        reference = load_gemma_assoc(GEMMA_COVARIATE_LRT)

        # Use relaxed pvalue_rtol for LRT (chi-squared amplifies differences)
        config = ToleranceConfig(pvalue_rtol=5e-3)
        comparison = compare_assoc_results(results_jax, reference, config=config)
        assert comparison.passed, (
            f"JAX LRT+covariates vs GEMMA failed:\n"
            f"  p_lrt: {comparison.p_lrt.message}\n"
            f"  l_mle: {comparison.l_mle.message}"
        )

    def test_lrt_pvalues_bounded(self, synthetic_data):
        """All LRT p-values in [0,1] and l_mle values positive."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=2,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0

        for r in results:
            assert 0 <= r.p_lrt <= 1, f"p_lrt={r.p_lrt} out of [0,1] for {r.rs}"
            assert r.l_mle > 0, f"l_mle={r.l_mle} not positive for {r.rs}"


class TestJaxAllTestsMode:
    """Validation tests for JAX all-tests mode (lmm_mode=4) against NumPy runner."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic GWAS data with population structure for all-tests.

        Uses the same 4-subpopulation design as TestJaxLrtMode: differentiated
        allele frequencies (Fst~0.05) and polygenic phenotype (h2~0.5) keep
        null lambda in the interior of [l_min, l_max], which is required for
        well-conditioned MLE optimization in both LRT and all-tests modes.
        """
        rng = np.random.default_rng(300)
        n_per_pop = 75
        n_pops = 4
        n_samples = n_per_pop * n_pops
        n_snps = 500

        # Generate allele freqs differentiated by population (Fst ~ 0.05)
        ancestral_freqs = rng.uniform(0.15, 0.45, n_snps)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
        for pop in range(n_pops):
            start = pop * n_per_pop
            end = (pop + 1) * n_per_pop
            drift = rng.normal(0, 0.05, n_snps)
            pop_freqs = np.clip(ancestral_freqs + drift, 0.05, 0.95)
            for j in range(n_snps):
                p = pop_freqs[j]
                genotypes[start:end, j] = rng.choice(
                    [0, 1, 2],
                    size=n_per_pop,
                    p=[(1 - p) ** 2, 2 * p * (1 - p), p**2],
                )

        # Polygenic phenotype: many causal SNPs + kinship-correlated noise
        K = compute_centered_kinship(genotypes)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(n_samples))

        # Polygenic signal through kinship (h2 ~ 0.5)
        genetic = L @ rng.standard_normal(n_samples)
        genetic = genetic / genetic.std()

        # Add fixed effects from a few causal SNPs
        causal_idx = rng.choice(n_snps, 5, replace=False)
        G_causal = genotypes[:, causal_idx]
        G_std = (G_causal - G_causal.mean(0)) / (G_causal.std(0) + 1e-8)
        fixed_effects = G_std @ rng.normal(0, 0.3, 5)

        noise = rng.standard_normal(n_samples)
        phenotype = genetic + fixed_effects + noise
        phenotype = (phenotype - phenotype.mean()) / phenotype.std()

        snp_info = [
            {
                "chr": "1",
                "rs": f"rs{j}",
                "pos": j * 1000,
                "a1": "A",
                "a0": "G",
            }
            for j in range(n_snps)
        ]

        return genotypes, phenotype, snp_info

    def test_all_tests_returns_correct_fields(self, synthetic_data):
        """Mode 4 populates ALL fields: p_wald, p_lrt, p_score, etc."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=4,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0

        for r in results:
            # All fields must be populated (not None)
            assert r.p_wald is not None, f"p_wald is None for {r.rs}"
            assert r.p_lrt is not None, f"p_lrt is None for {r.rs}"
            assert r.p_score is not None, f"p_score is None for {r.rs}"
            assert r.l_remle is not None, f"l_remle is None for {r.rs}"
            assert r.l_mle is not None, f"l_mle is None for {r.rs}"
            assert r.logl_H1 is not None, f"logl_H1 is None for {r.rs}"

            # Beta/se must be finite (Wald-derived, not NaN like pure LRT)
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se), f"se not finite for {r.rs}"

            # Metadata must be populated
            assert r.chr == "1"
            assert r.rs.startswith("rs")
            assert r.ps >= 0
            assert r.n_miss >= 0
            assert r.allele1 == "A"
            assert r.allele0 == "G"
            assert 0 <= r.af <= 1

            # Wald-specific bounds
            assert r.l_remle > 0, f"l_remle={r.l_remle} not positive for {r.rs}"
            assert 0 <= r.p_wald <= 1, f"p_wald={r.p_wald} out of [0,1] for {r.rs}"

            # LRT-specific bounds
            assert r.l_mle > 0, f"l_mle={r.l_mle} not positive for {r.rs}"
            assert 0 <= r.p_lrt <= 1, f"p_lrt={r.p_lrt} out of [0,1] for {r.rs}"

            # Score-specific bounds
            assert 0 <= r.p_score <= 1, f"p_score={r.p_score} out of [0,1] for {r.rs}"

    def test_all_tests_self_consistent(self, synthetic_data):
        """JAX mode 4 fields are internally consistent across test types."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=4,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0, "No results returned"

        for r in results:
            # All p-values finite and in [0, 1]
            assert np.isfinite(r.p_wald), f"p_wald not finite for {r.rs}"
            assert 0 <= r.p_wald <= 1, f"p_wald out of [0,1] for {r.rs}"
            assert np.isfinite(r.p_score), f"p_score not finite for {r.rs}"
            assert 0 <= r.p_score <= 1, f"p_score out of [0,1] for {r.rs}"
            assert np.isfinite(r.p_lrt), f"p_lrt not finite for {r.rs}"
            assert 0 <= r.p_lrt <= 1, f"p_lrt out of [0,1] for {r.rs}"

            # Beta and SE finite
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se), f"se not finite for {r.rs}"
            assert r.se > 0, f"se should be positive for {r.rs}"

            # Lambda values positive
            assert r.l_remle > 0, f"l_remle={r.l_remle} not positive for {r.rs}"
            assert r.l_mle > 0, f"l_mle={r.l_mle} not positive for {r.rs}"

    def test_all_tests_with_covariates_valid(self, synthetic_data):
        """Mode 4 with covariates produces valid results."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        rng = np.random.default_rng(301)
        n_samples = genotypes.shape[0]
        covariates = np.column_stack(
            [
                np.ones(n_samples),
                rng.standard_normal(n_samples),
            ]
        )

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=4,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0, "JAX mode 4 with covariates returned no results"

        for r in results:
            # All fields populated
            assert r.p_wald is not None, f"p_wald is None for {r.rs}"
            assert r.p_lrt is not None, f"p_lrt is None for {r.rs}"
            assert r.p_score is not None, f"p_score is None for {r.rs}"

            # All p-values in valid range
            assert 0 <= r.p_wald <= 1, f"p_wald out of [0,1] for {r.rs}"
            assert 0 <= r.p_lrt <= 1, f"p_lrt out of [0,1] for {r.rs}"
            assert 0 <= r.p_score <= 1, f"p_score out of [0,1] for {r.rs}"

            # Beta, SE finite
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se), f"se not finite for {r.rs}"

    def test_all_tests_pvalues_bounded(self, synthetic_data):
        """All mode 4 values in valid ranges: p in [0,1], lambdas > 0."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=4,
            show_progress=False,
            check_memory=False,
        )

        assert len(results) > 0

        for r in results:
            assert 0 <= r.p_wald <= 1, f"p_wald={r.p_wald} out of [0,1] for {r.rs}"
            assert 0 <= r.p_lrt <= 1, f"p_lrt={r.p_lrt} out of [0,1] for {r.rs}"
            assert 0 <= r.p_score <= 1, f"p_score={r.p_score} out of [0,1] for {r.rs}"
            assert r.l_remle > 0, f"l_remle={r.l_remle} not positive for {r.rs}"
            assert r.l_mle > 0, f"l_mle={r.l_mle} not positive for {r.rs}"
            assert np.isfinite(r.beta), f"beta not finite for {r.rs}"
            assert np.isfinite(r.se), f"se not finite for {r.rs}"
            assert np.isfinite(r.logl_H1), f"logl_H1 not finite for {r.rs}"

    def test_all_tests_wald_matches_mode1(self, synthetic_data):
        """Mode 4 Wald component is identical to mode 1 (same code path)."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results_mode4 = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=4,
            show_progress=False,
            check_memory=False,
        )

        results_mode1 = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=1,
            show_progress=False,
            check_memory=False,
        )

        m4_by_rs = {r.rs: r for r in results_mode4}
        m1_by_rs = {r.rs: r for r in results_mode1}
        common_rs = sorted(set(m4_by_rs) & set(m1_by_rs))

        assert len(common_rs) > 0, "No common SNPs between mode 4 and mode 1"

        for rs in common_rs:
            m4 = m4_by_rs[rs]
            m1 = m1_by_rs[rs]

            # Wald fields should be near-identical (same REML code path)
            np.testing.assert_allclose(
                m4.p_wald,
                m1.p_wald,
                rtol=1e-10,
                err_msg=f"p_wald mismatch for {rs}",
            )
            np.testing.assert_allclose(
                m4.beta,
                m1.beta,
                rtol=1e-10,
                err_msg=f"beta mismatch for {rs}",
            )
            np.testing.assert_allclose(
                m4.se,
                m1.se,
                rtol=1e-10,
                err_msg=f"se mismatch for {rs}",
            )
            np.testing.assert_allclose(
                m4.l_remle,
                m1.l_remle,
                rtol=1e-10,
                err_msg=f"l_remle mismatch for {rs}",
            )
            np.testing.assert_allclose(
                m4.logl_H1,
                m1.logl_H1,
                rtol=1e-10,
                err_msg=f"logl_H1 mismatch for {rs}",
            )
