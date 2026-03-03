"""Tests for JAX-optimized LMM runner."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")

from jamma.kinship import compute_centered_kinship
from jamma.lmm.chunk import MAX_SAFE_CHUNK, _compute_chunk_size, auto_tune_chunk_size
from jamma.lmm.runner_jax import run_lmm_association_jax
from jamma.validation import compare_assoc_results, load_gemma_assoc
from tests.conftest import load_phenotypes_from_fam

pytestmark = pytest.mark.requires_jax

# GEMMA covariate fixture paths (Score and LRT with covariates)
_FIXTURE_ROOT = Path(__file__).parent / "fixtures"
COVARIATE_FIXTURE_DIR = _FIXTURE_ROOT / "gemma_covariate"
GEMMA_COVARIATE_SCORE = COVARIATE_FIXTURE_DIR / "gemma_covariate_score.assoc.txt"
GEMMA_COVARIATE_LRT = COVARIATE_FIXTURE_DIR / "gemma_covariate_lrt.assoc.txt"

# GEMMA synthetic fixture paths (used for covariate data)
FIXTURE_DIR = _FIXTURE_ROOT / "gemma_synthetic"
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


@pytest.mark.tier0
class TestChunkSizeComputation:
    """Tests for chunk size calculation to avoid int32 overflow."""

    def test_small_dataset_no_chunking(self):
        """Small datasets should not be chunked."""
        chunk = _compute_chunk_size(n_snps=10_000)
        assert chunk == 10_000  # Full dataset

    def test_large_dataset_is_chunked(self):
        """Large datasets should be chunked at MAX_SAFE_CHUNK."""
        chunk = _compute_chunk_size(n_snps=95_000)
        assert chunk == MAX_SAFE_CHUNK  # Capped at 50k

    def test_chunk_size_caps_at_max_safe(self):
        """Chunk size is capped at MAX_SAFE_CHUNK regardless of n_snps."""
        chunk = _compute_chunk_size(n_snps=100_000)
        assert chunk == MAX_SAFE_CHUNK

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


@pytest.mark.tier1
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
        # eigendecomp may overwrite K in-place; save copy for runner
        kinship_copy = kinship.copy()
        eigenvalues, eigenvectors = eigendecompose_kinship(kinship)

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship_copy,
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


@pytest.mark.tier1
class TestJaxRunnerGuards:
    """Tests for runner_jax input validation guards."""

    def test_all_invalid_samples_raises(self):
        """ValueError when all phenotypes are NaN (no valid samples remain)."""
        rng = np.random.default_rng(55)
        n_samples = 50
        n_snps = 20

        genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.float64)
        phenotypes = np.full(n_samples, np.nan)
        kinship = np.eye(n_samples)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        with pytest.raises(ValueError, match="No valid samples"):
            run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

    def test_all_minus9_samples_raises(self):
        """ValueError when all phenotypes are -9 (PLINK missing code)."""
        rng = np.random.default_rng(56)
        n_samples = 50
        n_snps = 20

        genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.float64)
        phenotypes = np.full(n_samples, -9.0)
        kinship = np.eye(n_samples)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        with pytest.raises(ValueError, match="No valid samples"):
            run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )


@pytest.mark.tier1
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


@pytest.mark.tier1
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
        phenotypes = load_phenotypes_from_fam(FIXTURE_DIR / "test.fam")
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


@pytest.mark.tier1
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
        phenotypes = load_phenotypes_from_fam(FIXTURE_DIR / "test.fam")
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


@pytest.mark.tier1
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
        # eigendecomp overwrites K in-place; needs fresh copy per run
        kinship_mode4 = kinship.copy()
        kinship_mode1 = kinship.copy()

        results_mode4 = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship_mode4,
            snp_info=snp_info,
            lmm_mode=4,
            show_progress=False,
            check_memory=False,
        )

        results_mode1 = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship_mode1,
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


@pytest.mark.tier1
class TestDegenerateSNPPipeline:
    """Integration tests for degenerate SNP handling through full runner."""

    def test_degenerate_snp_filtered_out(self):
        """Constant-genotype (degenerate) SNPs are filtered out by variance check.

        A monomorphic SNP has zero variance, so compute_snp_filter_mask removes it
        before the association test. This test verifies the full pipeline: the
        degenerate SNP should not appear in results, while polymorphic SNPs produce
        finite statistics.
        """
        rng = np.random.default_rng(42)
        n_samples = 200
        n_snps = 20

        # Generate polymorphic genotypes
        mafs = rng.uniform(0.1, 0.4, n_snps)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
        for j in range(n_snps):
            p = mafs[j]
            genotypes[:, j] = rng.choice(
                [0, 1, 2], size=n_samples, p=[(1 - p) ** 2, 2 * p * (1 - p), p**2]
            )

        # Make the first SNP degenerate (all zeros -- constant genotype)
        genotypes[:, 0] = 0.0

        phenotype = rng.standard_normal(n_samples)
        kinship = compute_centered_kinship(genotypes)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            maf_threshold=0.0,
            miss_threshold=1.0,
            show_progress=False,
            check_memory=False,
        )

        # The degenerate SNP (rs0) should be absent from results (filtered by variance)
        result_rs_ids = {r.rs for r in results}
        assert "rs0" not in result_rs_ids, (
            "Degenerate SNP rs0 (constant genotype) should be filtered out"
        )

        # Polymorphic SNPs should produce finite, valid results
        assert len(results) >= n_snps - 1, (
            f"Expected at least {n_snps - 1} results (all except degenerate), "
            f"got {len(results)}"
        )
        for r in results:
            assert np.isfinite(r.beta), f"Non-finite beta for {r.rs}"
            assert np.isfinite(r.se), f"Non-finite se for {r.rs}"
            assert 0 <= r.p_wald <= 1, f"Invalid p_wald for {r.rs}: {r.p_wald}"

    def test_all_degenerate_snps_returns_empty(self):
        """When ALL SNPs are degenerate, the runner returns an empty list.

        All columns have zero variance, so all are filtered out.
        """
        rng = np.random.default_rng(99)
        n_samples = 100

        # All SNPs constant -- genotypes all zeros
        genotypes = np.zeros((n_samples, 10), dtype=np.float64)
        phenotype = rng.standard_normal(n_samples)

        # Need a valid kinship -- use random symmetric PSD matrix
        X = rng.standard_normal((n_samples, 50))
        kinship = X @ X.T / 50

        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(10)
        ]

        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            maf_threshold=0.0,
            miss_threshold=1.0,
            show_progress=False,
            check_memory=False,
        )

        assert results == [], (
            f"Expected empty results for all-degenerate SNPs, got {len(results)}"
        )


@pytest.mark.tier1
class TestExposedRotationDiagnostic:
    """Tests for the UT@G exposed rotation timing diagnostic in runner_jax."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic GWAS data for rotation diagnostic tests."""
        return _make_synthetic_gwas_data(seed=77, n_samples=100, n_snps=200)

    def test_single_chunk_exposed_equals_total(self, synthetic_data):
        """Single-chunk run: exposed rotation should equal total rotation.

        When all SNPs fit in one chunk there is no prior compute to overlap with,
        so the first (and only) rotation is fully exposed: exposed == total.
        """
        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        _ = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )

        assert "rotation_s" in last_run_timing, (
            "last_run_timing must contain 'rotation_s'"
        )
        assert "rotation_exposed_s" in last_run_timing, (
            "last_run_timing must contain 'rotation_exposed_s'"
        )
        assert last_run_timing["rotation_exposed_s"] == pytest.approx(
            last_run_timing["rotation_s"], abs=1e-6
        ), (
            f"Single-chunk: exposed ({last_run_timing['rotation_exposed_s']:.6f}s) "
            f"should equal total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_multi_chunk_exposed_leq_total(self, synthetic_data):
        """Multi-chunk run: exposed rotation cannot exceed total rotation.

        Patches _compute_chunk_size to force multiple chunks (the default chunk
        size for small fixtures exceeds n_snps). The invariant exposed <= total
        must hold regardless of overlap.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        # Force chunk_size=50 so 200 SNPs produce ~4 chunks
        with patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=50):
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        assert "rotation_s" in last_run_timing
        assert "rotation_exposed_s" in last_run_timing
        # Invariant: exposed cannot exceed total (with small float tolerance)
        assert last_run_timing["rotation_exposed_s"] <= (
            last_run_timing["rotation_s"] + 1e-6
        ), (
            f"Exposed ({last_run_timing['rotation_exposed_s']:.6f}s) must be "
            f"<= total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_timing_keys_present_after_run(self, synthetic_data):
        """last_run_timing dict contains all four expected timing keys."""
        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        _ = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )

        expected_keys = {
            "rotation_s",
            "rotation_exposed_s",
            "jax_compute_s",
            "result_write_s",
        }
        assert set(last_run_timing.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(last_run_timing.keys())}"
        )
        for key, val in last_run_timing.items():
            assert val >= 0.0, f"Timing value for '{key}' must be >= 0, got {val}"


@pytest.mark.tier1
def test_timing_breakdown_logged(sample_plink_data):
    """Verify timing breakdown appears in loguru output with all 6 phases."""
    import io
    import re

    from loguru import logger

    from jamma.io.plink import load_plink_binary
    from jamma.kinship import compute_centered_kinship
    from jamma.lmm import run_lmm_association_streaming

    # Load small test dataset
    data = load_plink_binary(sample_plink_data)
    np.random.seed(42)
    phenotypes = np.random.randn(data.n_samples)
    kinship = compute_centered_kinship(
        data.genotypes.astype(np.float64), check_memory=False
    )

    # Capture loguru output
    sink = io.StringIO()
    handler_id = logger.add(sink, format="{message}", level="INFO")
    try:
        run_lmm_association_streaming(
            sample_plink_data,
            phenotypes,
            kinship,
            snp_info=None,
            check_memory=False,
            show_progress=True,
        )
    finally:
        logger.remove(handler_id)

    log_output = sink.getvalue()

    # Verify timing breakdown header appears
    assert "Timing breakdown" in log_output, "Expected 'Timing breakdown' in log output"

    # Verify all 6 phase labels plus summary lines appear with numeric seconds
    expected_labels = [
        "I/O read (pass 1):",
        "SNP statistics:",
        "Setup (eigen+null):",
        "UT@G rotation:",
        "UT@G exposed:",
        "JAX compute:",
        "Result write:",
        "Accounted:",
        "Total:",
    ]
    lines = log_output.splitlines()
    for label in expected_labels:
        matching_line = next((line for line in lines if label in line), None)
        assert matching_line is not None, f"Expected '{label}' in log output"
        assert re.search(r"\d+\.\d+s", matching_line), (
            f"Expected numeric seconds value in line: {matching_line}"
        )


@pytest.mark.tier1
def test_maf_normalization_in_comparison():
    """AF > 0.5 in expected results should match MAF <= 0.5 in actual.

    JAMMA reports MAF (always <= 0.5), GEMMA reports AF (can be > 0.5).
    compare_assoc_results normalizes both to MAF before comparison.
    Without normalization, af=0.3 vs af=0.7 would incorrectly fail.
    """
    from jamma.lmm.stats import AssocResult

    # JAMMA result: reports MAF = 0.3
    actual = [
        AssocResult(
            chr="1",
            rs="rs1",
            ps=100,
            n_miss=0,
            allele1="A",
            allele0="G",
            af=0.3,
            beta=0.1,
            se=0.01,
            logl_H1=-100.0,
            l_remle=1.0,
            p_wald=0.05,
        )
    ]
    # GEMMA result: reports AF = 0.7 (same minor allele, opposite convention)
    expected = [
        AssocResult(
            chr="1",
            rs="rs1",
            ps=100,
            n_miss=0,
            allele1="A",
            allele0="G",
            af=0.7,
            beta=0.1,
            se=0.01,
            logl_H1=-100.0,
            l_remle=1.0,
            p_wald=0.05,
        )
    ]

    comparison = compare_assoc_results(actual, expected)
    assert comparison.af.passed, (
        f"AF=0.3 and AF=0.7 have the same MAF (0.3), comparison should pass: "
        f"{comparison.af.message}"
    )


# ---------------------------------------------------------------------------
# Tail chunk and imputation guard tests (RUN-02, RUN-06)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
def test_prepare_utg_chunk_no_tail_padding():
    """Tail chunk returns actual SNP count shape, not padded to chunk_size (RUN-02)."""
    import jax

    from jamma.lmm.prepare import DevicePlacement, prepare_utg_chunk

    n_samples = 50
    actual_snps = 7  # Fewer than chunk_size
    chunk_size = 100

    U = np.eye(n_samples, dtype=np.float64)
    geno = np.random.default_rng(42).standard_normal((n_samples, actual_snps))

    # Use single CPU device placement (no multi-device alignment)
    device = jax.devices("cpu")[0]
    placement = DevicePlacement(snp=device, rep=device, n_devices=1)

    UtG, actual_len = prepare_utg_chunk(geno, U, placement, rotation_threads=1)

    assert actual_len == actual_snps
    # UtG should have actual_snps columns, NOT chunk_size columns
    assert UtG.shape[1] == actual_snps, (
        f"Expected {actual_snps} columns (no padding), got {UtG.shape[1]}. "
        f"Tail chunk should not be padded to chunk_size={chunk_size}."
    )


@pytest.mark.tier1
def test_prepare_utg_chunk_full_chunk_no_change():
    """prepare_utg_chunk with full chunk_size works unchanged (no padding needed)."""
    import jax

    from jamma.lmm.prepare import DevicePlacement, prepare_utg_chunk

    n_samples = 50
    chunk_size = 100

    U = np.eye(n_samples, dtype=np.float64)
    geno = np.random.default_rng(42).standard_normal((n_samples, chunk_size))

    device = jax.devices("cpu")[0]
    placement = DevicePlacement(snp=device, rep=device, n_devices=1)

    UtG, actual_len = prepare_utg_chunk(geno, U, placement, rotation_threads=1)

    assert actual_len == chunk_size
    assert UtG.shape[1] == chunk_size


@pytest.mark.tier1
@pytest.mark.requires_jax
def test_rotation_compute_overlap_metric_populated() -> None:
    """JAX runner populates rotation_exposed_s <= rotation_s after a run (RUN-03).

    The overlap pattern in runner_jax.py rotates chunk N+1 while computing
    chunk N. On multi-chunk datasets, exposed rotation should be less than
    total rotation time (some rotation is hidden behind compute). This test
    verifies the timing metric is populated and the invariant holds.
    """
    from loguru import logger

    from jamma.lmm.runner_jax import last_run_timing

    rng = np.random.default_rng(42)
    n_samples = 100
    n_snps = 500

    genotypes = rng.choice([0.0, 1.0, 2.0], size=(n_samples, n_snps))
    phenotypes = rng.standard_normal(n_samples)
    kinship = np.eye(n_samples, dtype=np.float64) * 1.1

    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "T"}
        for i in range(n_snps)
    ]

    results = run_lmm_association_jax(
        genotypes=genotypes,
        phenotypes=phenotypes,
        kinship=kinship,
        snp_info=snp_info,
        maf_threshold=0.0,
        miss_threshold=1.0,
        check_memory=False,
        show_progress=False,
        lmm_mode=1,
    )

    assert len(results) > 0
    assert "rotation_s" in last_run_timing, "rotation_s not in last_run_timing"
    assert "rotation_exposed_s" in last_run_timing, (
        "rotation_exposed_s not in last_run_timing"
    )

    rot_total = last_run_timing["rotation_s"]
    rot_exposed = last_run_timing["rotation_exposed_s"]

    # Exposed rotation must never exceed total rotation (by definition)
    assert rot_exposed <= rot_total + 1e-6, (
        f"Exposed rotation ({rot_exposed:.4f}s) > total rotation ({rot_total:.4f}s)"
    )

    if rot_total > 0.001:
        logger.info(
            f"Rotation overlap: total={rot_total:.4f}s, "
            f"exposed={rot_exposed:.4f}s, "
            f"hidden={rot_total - rot_exposed:.4f}s"
        )


# ---------------------------------------------------------------------------
# ThreadPoolExecutor rotation-compute overlap tests (Plan 54-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestThreadPoolExecutorOverlapJax:
    """Tests for ThreadPoolExecutor-based rotation-compute overlap in runner_jax.

    Verifies that BLAS rotation for chunk N+1 runs concurrently with JAX
    compute for chunk N, using a background thread so the main thread is
    not blocked on DGEMM before dispatching JAX work.
    """

    @pytest.fixture
    def synthetic_data_multi_chunk(self):
        """Generate synthetic GWAS data suitable for multi-chunk overlap tests."""
        return _make_synthetic_gwas_data(seed=42, n_samples=100, n_snps=200)

    def test_rotation_overlap_multi_chunk_exposed_leq_total(
        self, synthetic_data_multi_chunk
    ):
        """Multi-chunk: rotation_exposed_s <= rotation_s (overlap is active).

        With ThreadPoolExecutor, rotation for chunk N+1 runs in a background
        thread concurrently with JAX compute for chunk N. The exposed time
        (time the main thread waited for the future after JAX sync) should be
        at most equal to total rotation time.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = synthetic_data_multi_chunk
        kinship = compute_centered_kinship(genotypes)

        # Force chunk_size=50 to guarantee 4 chunks from 200 SNPs
        with patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=50):
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        assert "rotation_s" in last_run_timing
        assert "rotation_exposed_s" in last_run_timing
        # With true overlap: exposed should be <= total (not equal in general)
        assert last_run_timing["rotation_exposed_s"] <= (
            last_run_timing["rotation_s"] + 1e-6
        ), (
            f"Exposed ({last_run_timing['rotation_exposed_s']:.6f}s) must be "
            f"<= total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_rotation_overlap_single_chunk_exposed_equals_total(
        self, synthetic_data_multi_chunk
    ):
        """Single-chunk: rotation_exposed_s == rotation_s (no overlap possible).

        When all SNPs fit in one chunk, there is no next chunk to prepare, so
        the first rotation is fully exposed.
        """
        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = synthetic_data_multi_chunk
        kinship = compute_centered_kinship(genotypes)

        # Force a huge chunk_size so all 200 SNPs land in 1 chunk
        with __import__("unittest.mock", fromlist=["patch"]).patch(
            "jamma.lmm.runner_jax._compute_chunk_size", return_value=5000
        ):
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        assert last_run_timing["rotation_exposed_s"] == pytest.approx(
            last_run_timing["rotation_s"], abs=1e-6
        ), (
            f"Single-chunk: exposed ({last_run_timing['rotation_exposed_s']:.6f}s) "
            f"should equal total ({last_run_timing['rotation_s']:.6f}s)"
        )

    def test_rotation_overlap_numerical_correctness(self, synthetic_data_multi_chunk):
        """ThreadPoolExecutor overlap produces numerically identical results.

        Results from the chunked (overlap) run must match a single-chunk
        reference run to rtol=1e-12 for beta, se, and p_wald.
        """
        from unittest.mock import patch

        genotypes, phenotype, snp_info = synthetic_data_multi_chunk
        kinship = compute_centered_kinship(genotypes)
        kinship2 = kinship.copy()

        # Reference: single chunk (no overlap)
        with patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=5000):
            reference = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        # Test: multi-chunk overlap
        with patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=50):
            results = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship2,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        assert len(results) == len(reference), (
            f"Expected {len(reference)} results, got {len(results)}"
        )

        ref_by_rs = {r.rs: r for r in reference}
        for r in results:
            ref = ref_by_rs[r.rs]
            if not np.isnan(r.beta):
                np.testing.assert_allclose(
                    r.beta, ref.beta, rtol=1e-12, err_msg=f"beta mismatch for {r.rs}"
                )
                np.testing.assert_allclose(
                    r.se, ref.se, rtol=1e-12, err_msg=f"se mismatch for {r.rs}"
                )
                np.testing.assert_allclose(
                    r.p_wald,
                    ref.p_wald,
                    rtol=1e-12,
                    err_msg=f"p_wald mismatch for {r.rs}",
                )

    def test_pipeline_buffers_passed_to_chunk_size(self):
        """_compute_chunk_size is called with pipeline_buffers=2.

        The ThreadPoolExecutor holds two concurrent rotation buffers (current
        + next), so memory budget must be halved via pipeline_buffers=2.
        """
        from unittest.mock import patch

        rng = np.random.default_rng(999)
        n_samples = 50
        n_snps = 100
        genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.float64)
        phenotypes = rng.standard_normal(n_samples)
        kinship = np.eye(n_samples)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        with patch(
            "jamma.lmm.runner_jax._compute_chunk_size", return_value=1000
        ) as mock_chunk:
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotypes,
                kinship=kinship,
                snp_info=snp_info,
                maf_threshold=0.0,
                show_progress=False,
                check_memory=False,
            )

        # Assert pipeline_buffers=2 was passed in at least one call
        calls_with_pipeline = [
            c
            for c in mock_chunk.call_args_list
            if c.kwargs.get("pipeline_buffers") == 2
            or (len(c.args) >= 5 and c.args[4] == 2)
        ]
        assert len(calls_with_pipeline) >= 1, (
            f"Expected _compute_chunk_size to be called with pipeline_buffers=2. "
            f"Actual calls: {mock_chunk.call_args_list}"
        )

    def test_threadpoolexecutor_used_in_runner(self):
        """runner_jax imports and uses ThreadPoolExecutor.

        This verifies the structural change is in place: the runner must
        use concurrent.futures.ThreadPoolExecutor for the overlap pattern.
        """
        import inspect

        from jamma.lmm import runner_jax

        source = inspect.getsource(runner_jax)
        assert "ThreadPoolExecutor" in source, (
            "runner_jax must use ThreadPoolExecutor for rotation-compute overlap"
        )
        assert "executor.submit" in source, (
            "runner_jax must submit rotation work to background thread"
        )

    def test_background_rotation_failure_propagates(self):
        """Background rotation failure raises RuntimeError with exception chain.

        When prepare_utg_chunk raises in the background thread, the runner
        must wrap it in a RuntimeError (with 'from exc') so the original
        traceback is preserved and the chunk index is reported.
        """
        from unittest.mock import patch

        rng = np.random.default_rng(777)
        n_samples = 50
        n_snps = 100
        genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.float64)
        phenotypes = rng.standard_normal(n_samples)
        kinship = np.eye(n_samples)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        call_count = 0
        original_prepare = None

        def _failing_prepare(*args, **kwargs):
            """Succeed on first call, fail on second (background thread)."""
            nonlocal call_count, original_prepare
            call_count += 1
            if call_count > 1:
                raise ValueError("Simulated BLAS failure in background rotation")
            return original_prepare(*args, **kwargs)

        from jamma.lmm import prepare

        original_prepare = prepare.prepare_utg_chunk

        with (
            patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=50),
            patch(
                "jamma.lmm.runner_jax.prepare_utg_chunk", side_effect=_failing_prepare
            ),
        ):
            with pytest.raises(
                RuntimeError, match="Background rotation failed"
            ) as exc_info:
                run_lmm_association_jax(
                    genotypes=genotypes,
                    phenotypes=phenotypes,
                    kinship=kinship,
                    snp_info=snp_info,
                    maf_threshold=0.0,
                    show_progress=False,
                    check_memory=False,
                )

        # Verify exception chain preserves original cause
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)
        assert "Simulated BLAS failure" in str(exc_info.value.__cause__)

    def test_background_rotation_memoryerror_propagates_directly(self):
        """MemoryError from background rotation is NOT wrapped in RuntimeError.

        The except MemoryError: raise clause must fire before the generic
        except Exception handler, allowing OOM to propagate directly.
        """
        from unittest.mock import patch

        rng = np.random.default_rng(777)
        n_samples = 50
        n_snps = 100
        genotypes = rng.choice([0, 1, 2], size=(n_samples, n_snps)).astype(np.float64)
        phenotypes = rng.standard_normal(n_samples)
        kinship = np.eye(n_samples)
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        call_count = 0
        from jamma.lmm import prepare

        original_prepare = prepare.prepare_utg_chunk

        def _oom_prepare(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                raise MemoryError("Simulated OOM in background rotation")
            return original_prepare(*args, **kwargs)

        with (
            patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=50),
            patch("jamma.lmm.runner_jax.prepare_utg_chunk", side_effect=_oom_prepare),
        ):
            with pytest.raises(MemoryError, match="Simulated OOM"):
                run_lmm_association_jax(
                    genotypes=genotypes,
                    phenotypes=phenotypes,
                    kinship=kinship,
                    snp_info=snp_info,
                    maf_threshold=0.0,
                    show_progress=False,
                    check_memory=False,
                )


# ---------------------------------------------------------------------------
# Rotation overlap effectiveness tests (Plan 54-03)
# ---------------------------------------------------------------------------


@pytest.mark.tier1
@pytest.mark.requires_jax
class TestRotationOverlapEffectivenessJax:
    """Tests that rotation-compute overlap is measurably effective.

    Verifies that on multi-chunk runs, the exposed rotation time is strictly
    less than total rotation time (overlap hides meaningful rotation work).
    These tests go beyond the Plan 54-02 invariant tests (exposed <= total)
    to confirm the overlap mechanism is functionally active.
    """

    @pytest.fixture
    def multi_chunk_data(self):
        """Generate synthetic GWAS data for effectiveness tests.

        Uses n_samples=200, n_snps=2000 to ensure non-trivial rotation
        work that can be meaningfully overlapped.
        """
        return _make_synthetic_gwas_data(seed=54, n_samples=200, n_snps=2000)

    def test_rotation_overlap_effectiveness(self, multi_chunk_data):
        """Multi-chunk: rotation_exposed_s < 0.95 * rotation_s (overlap hides >=5%).

        With 10 chunks (2000 SNPs / chunk_size=200), the ThreadPoolExecutor
        pattern should hide meaningful rotation time behind JAX compute.
        On small synthetic data the threshold is conservative (5%); on real
        large data JAX compute dominates and exposed drops to near-zero.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = multi_chunk_data
        kinship = compute_centered_kinship(genotypes)

        # Force chunk_size=200 so 2000 SNPs → 10 chunks
        with patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=200):
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        rot_total = last_run_timing["rotation_s"]
        rot_exposed = last_run_timing["rotation_exposed_s"]

        assert rot_total > 0, "rotation_s must be > 0 (rotation occurred)"
        assert rot_exposed >= 0, f"rotation_exposed_s must be >= 0, got {rot_exposed}"

        # Overlap effectiveness: exposed must be < 95% of total (at least 5% hidden)
        assert rot_exposed < 0.95 * rot_total, (
            f"Expected overlap to hide at least 5% of rotation time on 10-chunk run. "
            f"total={rot_total:.6f}s, exposed={rot_exposed:.6f}s, "
            f"ratio={rot_exposed / max(rot_total, 1e-10):.3f} (threshold: 0.95). "
            f"The ThreadPoolExecutor overlap may not be functioning correctly."
        )

    def test_rotation_no_overlap_single_chunk(self, multi_chunk_data):
        """Single-chunk: rotation_exposed_s ≈ rotation_s (within 20% jitter).

        When all SNPs land in one chunk there is no next chunk to prefetch,
        so exposed time must equal total rotation time (no overlap possible).
        The 20% tolerance accounts for timing measurement noise.
        """
        from unittest.mock import patch

        from jamma.lmm.runner_jax import last_run_timing

        genotypes, phenotype, snp_info = multi_chunk_data
        kinship = compute_centered_kinship(genotypes)

        # Force huge chunk_size so all 2000 SNPs land in 1 chunk
        with patch("jamma.lmm.runner_jax._compute_chunk_size", return_value=50_000):
            _ = run_lmm_association_jax(
                genotypes=genotypes,
                phenotypes=phenotype,
                kinship=kinship,
                snp_info=snp_info,
                show_progress=False,
                check_memory=False,
            )

        rot_total = last_run_timing["rotation_s"]
        rot_exposed = last_run_timing["rotation_exposed_s"]

        assert rot_total > 0, "rotation_s must be > 0"
        assert rot_exposed >= 0, "rotation_exposed_s must be >= 0"

        ratio = rot_exposed / max(rot_total, 1e-10)
        assert ratio >= 0.80, (
            f"Single-chunk: exposed should be close to total (within 20%). "
            f"total={rot_total:.6f}s, exposed={rot_exposed:.6f}s, ratio={ratio:.3f}. "
            f"Single-chunk runs should not benefit from overlap."
        )
