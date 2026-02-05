"""Tests for JAX-optimized LMM runner."""

import numpy as np
import pytest

from jamma.kinship import compute_centered_kinship
from jamma.lmm import run_lmm_association
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


class TestJaxScoreMode:
    """Validation tests for JAX Score mode (lmm_mode=3) against NumPy runner."""

    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic GWAS data for Score tests."""
        rng = np.random.default_rng(100)
        n_samples = 200
        n_snps = 500

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

    def test_score_matches_numpy_runner(self, synthetic_data):
        """JAX Score p-values match NumPy Score p-values within 1e-8."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results_jax = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=3,
            show_progress=False,
            check_memory=False,
        )

        results_np = run_lmm_association(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=3,
        )

        # Build lookup by rs name
        np_by_rs = {r.rs: r for r in results_np}
        jax_by_rs = {r.rs: r for r in results_jax}
        common_rs = sorted(set(np_by_rs) & set(jax_by_rs))

        assert len(common_rs) > 0, "No common SNPs between JAX and NumPy results"

        max_p_diff = 0.0
        max_beta_reldiff = 0.0
        for rs in common_rs:
            j = jax_by_rs[rs]
            n = np_by_rs[rs]

            p_diff = abs(j.p_score - n.p_score)
            max_p_diff = max(max_p_diff, p_diff)
            assert p_diff < 1e-8, (
                f"p_score diff {p_diff:.2e} > 1e-8 for {rs} "
                f"(JAX={j.p_score:.10e}, NumPy={n.p_score:.10e})"
            )

            beta_reldiff = abs(j.beta - n.beta) / max(abs(n.beta), 1e-10)
            max_beta_reldiff = max(max_beta_reldiff, beta_reldiff)
            assert (
                beta_reldiff < 1e-4
            ), f"beta relative diff {beta_reldiff:.2e} > 1e-4 for {rs}"

        print(f"\nScore parity: {len(common_rs)} SNPs compared")
        print(f"  max |p_score diff|: {max_p_diff:.2e}")
        print(f"  max |beta rel diff|: {max_beta_reldiff:.2e}")

    def test_score_with_covariates(self, synthetic_data):
        """Score mode with n_cvt=2 covariates matches NumPy runner."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        rng = np.random.default_rng(101)
        n_samples = genotypes.shape[0]
        covariates = np.column_stack(
            [
                np.ones(n_samples),
                rng.standard_normal(n_samples),
            ]
        )

        results_jax = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=3,
            show_progress=False,
            check_memory=False,
        )

        results_np = run_lmm_association(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=3,
        )

        assert len(results_jax) > 0, "JAX Score with covariates returned no results"

        np_by_rs = {r.rs: r for r in results_np}
        jax_by_rs = {r.rs: r for r in results_jax}
        common_rs = sorted(set(np_by_rs) & set(jax_by_rs))

        assert (
            len(common_rs) > 0
        ), "No common SNPs between JAX and NumPy covariate results"

        max_p_diff = 0.0
        for rs in common_rs:
            j = jax_by_rs[rs]
            n = np_by_rs[rs]

            p_diff = abs(j.p_score - n.p_score)
            max_p_diff = max(max_p_diff, p_diff)
            assert (
                p_diff < 1e-8
            ), f"p_score diff {p_diff:.2e} > 1e-8 for {rs} with covariates"

        print(
            f"\nScore+covariates parity: {len(common_rs)} SNPs, "
            f"max p_score diff: {max_p_diff:.2e}"
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

    def test_lrt_matches_numpy_runner(self, synthetic_data):
        """JAX LRT p-values match NumPy LRT p-values within 2e-3."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        results_jax = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=2,
            show_progress=False,
            check_memory=False,
        )

        results_np = run_lmm_association(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            lmm_mode=2,
        )

        np_by_rs = {r.rs: r for r in results_np}
        jax_by_rs = {r.rs: r for r in results_jax}
        common_rs = sorted(set(np_by_rs) & set(jax_by_rs))

        assert len(common_rs) > 0

        max_p_diff = 0.0
        median_diffs = []
        outliers = []
        for rs in common_rs:
            j = jax_by_rs[rs]
            n = np_by_rs[rs]

            p_diff = abs(j.p_lrt - n.p_lrt)
            max_p_diff = max(max_p_diff, p_diff)
            median_diffs.append(p_diff)

            if p_diff > 1e-3:
                outliers.append((rs, p_diff, j.p_lrt, n.p_lrt))

            assert p_diff < 2e-3, (
                f"p_lrt diff {p_diff:.2e} > 2e-3 for {rs} "
                f"(JAX={j.p_lrt:.10e}, NumPy={n.p_lrt:.10e})"
            )

        median_diff = float(np.median(median_diffs))
        print(f"\nLRT parity: {len(common_rs)} SNPs compared")
        print(f"  max |p_lrt diff|: {max_p_diff:.2e}")
        print(f"  median |p_lrt diff|: {median_diff:.2e}")
        if outliers:
            print(f"  outliers > 1e-3: {len(outliers)}")
            for rs, d, jv, nv in outliers[:5]:
                print(f"    {rs}: diff={d:.2e} JAX={jv:.6e} NP={nv:.6e}")

    def test_lrt_with_covariates(self, synthetic_data):
        """LRT mode with n_cvt=2 covariates matches NumPy runner."""
        genotypes, phenotype, snp_info = synthetic_data
        kinship = compute_centered_kinship(genotypes)

        rng = np.random.default_rng(201)
        n_samples = genotypes.shape[0]
        covariates = np.column_stack(
            [
                np.ones(n_samples),
                rng.standard_normal(n_samples),
            ]
        )

        results_jax = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=2,
            show_progress=False,
            check_memory=False,
        )

        results_np = run_lmm_association(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=kinship,
            snp_info=snp_info,
            covariates=covariates,
            lmm_mode=2,
        )

        assert len(results_jax) > 0, "JAX LRT with covariates returned no results"

        np_by_rs = {r.rs: r for r in results_np}
        jax_by_rs = {r.rs: r for r in results_jax}
        common_rs = sorted(set(np_by_rs) & set(jax_by_rs))

        assert len(common_rs) > 0

        max_p_diff = 0.0
        for rs in common_rs:
            j = jax_by_rs[rs]
            n = np_by_rs[rs]

            p_diff = abs(j.p_lrt - n.p_lrt)
            max_p_diff = max(max_p_diff, p_diff)
            assert p_diff < 2e-3, (
                f"p_lrt diff {p_diff:.2e} > 2e-3 for {rs} " f"with covariates"
            )

        print(
            f"\nLRT+covariates parity: {len(common_rs)} SNPs, "
            f"max p_lrt diff: {max_p_diff:.2e}"
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
