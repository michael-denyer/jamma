"""Tests for JAX-optimized LMM runner."""

import numpy as np
import pytest

from jamma.kinship import compute_centered_kinship
from jamma.lmm.runner_jax import (
    MAX_SAFE_CHUNK,
    _compute_chunk_size,
    auto_tune_chunk_size,
    run_lmm_association_jax,
)


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
        rng = np.random.default_rng(42)
        n_samples = 200
        n_snps = 500

        # Generate genotypes
        mafs = rng.uniform(0.1, 0.4, n_snps)
        genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)
        for j in range(n_snps):
            p = mafs[j]
            genotypes[:, j] = rng.choice(
                [0, 1, 2], size=n_samples, p=[(1 - p) ** 2, 2 * p * (1 - p), p**2]
            )

        # Generate phenotype with genetic component
        causal_idx = rng.choice(n_snps, 10, replace=False)
        betas = rng.standard_normal(10)
        G = genotypes[:, causal_idx]
        G_std = (G - G.mean(axis=0)) / (G.std(axis=0) + 1e-8)
        genetic = G_std @ betas
        noise = rng.standard_normal(n_samples)
        phenotype = genetic + noise
        phenotype = (phenotype - phenotype.mean()) / phenotype.std()

        # SNP info
        snp_info = [
            {"chr": "1", "rs": f"rs{j}", "pos": j * 1000, "a1": "A", "a0": "G"}
            for j in range(n_snps)
        ]

        return genotypes, phenotype, snp_info

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
