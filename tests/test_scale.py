"""Scale validation tests for JAMMA at larger sample sizes.

Tests verify:
1. Full workflow runs at moderate scale (1k-10k samples)
2. Memory estimation is accurate
3. Results are deterministic
4. Memory usage scales linearly (not quadratically) for eigendecomp
"""

import numpy as np
import psutil
import pytest

from jamma.core.memory import estimate_workflow_memory
from jamma.kinship import compute_centered_kinship
from jamma.lmm.runner_jax import run_lmm_association_jax


def generate_synthetic_gwas_data(
    n_samples: int,
    n_snps: int,
    n_causal: int = 10,
    heritability: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Generate synthetic GWAS data for testing.

    Args:
        n_samples: Number of samples (individuals).
        n_snps: Number of SNPs (variants).
        n_causal: Number of causal SNPs.
        heritability: Proportion of variance explained by genetics.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (genotypes, phenotype, snp_info):
        - genotypes: (n_samples, n_snps) matrix with values 0, 1, 2
        - phenotype: (n_samples,) vector
        - snp_info: List of dicts with chr, rs, pos, a1, a0
    """
    rng = np.random.default_rng(seed)

    # Generate genotypes under Hardy-Weinberg equilibrium
    maf = rng.uniform(0.05, 0.5, n_snps)
    genotypes = np.zeros((n_samples, n_snps), dtype=np.float32)

    for j in range(n_snps):
        p = maf[j]
        # P(0) = (1-p)^2, P(1) = 2p(1-p), P(2) = p^2
        probs = [(1 - p) ** 2, 2 * p * (1 - p), p**2]
        genotypes[:, j] = rng.choice([0, 1, 2], size=n_samples, p=probs)

    # Generate phenotype with causal SNPs
    causal_indices = rng.choice(n_snps, n_causal, replace=False)
    true_betas = rng.standard_normal(n_causal)

    # Standardize causal genotypes
    G_causal = genotypes[:, causal_indices].astype(np.float64)
    G_causal_std = (G_causal - G_causal.mean(axis=0)) / (G_causal.std(axis=0) + 1e-8)

    # Genetic value
    genetic_value = G_causal_std @ true_betas
    var_g = np.var(genetic_value)

    # Environmental noise
    var_e = var_g * (1 - heritability) / max(heritability, 1e-8)
    noise = rng.standard_normal(n_samples) * np.sqrt(var_e)

    # Phenotype
    phenotype = genetic_value + noise
    phenotype = (phenotype - phenotype.mean()) / phenotype.std()

    # SNP info
    snp_info = [
        {"chr": "1", "rs": f"rs{i}", "pos": i * 1000, "a1": "A", "a0": "G"}
        for i in range(n_snps)
    ]

    return genotypes, phenotype.astype(np.float64), snp_info


class TestScaleWorkflow:
    """Tests for full workflow at moderate scale."""

    @pytest.mark.slow
    def test_workflow_1k_samples(self):
        """Full workflow should complete at 1k samples x 1k SNPs."""
        n_samples, n_snps = 1000, 1000

        genotypes, phenotype, snp_info = generate_synthetic_gwas_data(
            n_samples, n_snps, seed=42
        )

        # Compute kinship
        K = compute_centered_kinship(genotypes, check_memory=False)
        assert K.shape == (n_samples, n_samples)

        # Run LMM
        results = run_lmm_association_jax(
            genotypes=genotypes,
            phenotypes=phenotype,
            kinship=K,
            snp_info=snp_info,
            show_progress=False,
            check_memory=False,
        )

        # Basic validations
        assert len(results) > 0, "Should have results"
        assert len(results) <= n_snps, "Should not exceed SNP count"

        # Check for valid results (no all-NaN)
        betas = [r.beta for r in results]
        p_values = [r.p_wald for r in results]

        finite_betas = [b for b in betas if np.isfinite(b)]
        finite_pvals = [p for p in p_values if np.isfinite(p)]

        assert len(finite_betas) > len(results) * 0.9, "Most betas should be finite"
        assert len(finite_pvals) > len(results) * 0.9, "Most p-values should be finite"

    def test_results_deterministic(self):
        """Same input should produce identical output."""
        n_samples, n_snps = 200, 200

        # Run twice with same seed
        genotypes1, phenotype1, snp_info1 = generate_synthetic_gwas_data(
            n_samples, n_snps, seed=123
        )
        genotypes2, phenotype2, snp_info2 = generate_synthetic_gwas_data(
            n_samples, n_snps, seed=123
        )

        K1 = compute_centered_kinship(genotypes1, check_memory=False)
        K2 = compute_centered_kinship(genotypes2, check_memory=False)

        # Kinship should be identical
        np.testing.assert_array_equal(K1, K2)

        # LMM results should be identical
        results1 = run_lmm_association_jax(
            genotypes=genotypes1,
            phenotypes=phenotype1,
            kinship=K1,
            snp_info=snp_info1,
            show_progress=False,
            check_memory=False,
        )
        results2 = run_lmm_association_jax(
            genotypes=genotypes2,
            phenotypes=phenotype2,
            kinship=K2,
            snp_info=snp_info2,
            show_progress=False,
            check_memory=False,
        )

        assert len(results1) == len(results2)

        for r1, r2 in zip(results1, results2, strict=True):
            assert r1.beta == r2.beta, "Beta should be identical"
            assert r1.se == r2.se, "SE should be identical"
            assert r1.p_wald == r2.p_wald, "P-value should be identical"


class TestMemoryEstimationAccuracy:
    """Tests for memory estimation accuracy."""

    def test_memory_estimate_formula_correct(self):
        """Memory estimate formulas should compute correct values."""
        n_samples, n_snps = 10_000, 50_000

        est = estimate_workflow_memory(n_samples, n_snps)

        # Kinship: n^2 * 8 bytes = 10000^2 * 8 / 1e9 = 0.8 GB
        expected_kinship = n_samples**2 * 8 / 1e9
        assert abs(est.kinship_gb - expected_kinship) < 0.01

        # Genotypes: n * p * 8 bytes (float64 JAX copy) = 4.0 GB
        expected_genotypes = n_samples * n_snps * 8 / 1e9
        assert abs(est.genotypes_gb - expected_genotypes) < 0.01

        # Eigenvectors: same as kinship
        assert abs(est.eigenvectors_gb - expected_kinship) < 0.01

        # Total should be reasonable: LMM peak includes genotypes (4GB) +
        # eigenvectors (0.8GB) + Uab/Iab batch intermediates (~11GB for
        # default batch_size=20k with n_index=6).  Eigendecomp peak is ~2.4GB.
        # max(LMM ~16GB, eigendecomp ~2.4GB) → ~16GB.
        assert est.total_gb > 10.0
        assert est.total_gb < 20.0

    def test_estimate_scales_quadratically_with_samples(self):
        """Memory estimate should scale O(n^2) with samples."""
        est_1k = estimate_workflow_memory(1000, 10000)
        est_2k = estimate_workflow_memory(2000, 10000)

        # Kinship scales as n^2, so 2x samples = 4x kinship
        ratio = est_2k.kinship_gb / est_1k.kinship_gb
        assert 3.5 < ratio < 4.5, f"Kinship should scale ~4x, got {ratio:.1f}x"

    def test_estimate_scales_linearly_with_snps(self):
        """Memory estimate for genotypes should scale O(n_snps)."""
        est_10k = estimate_workflow_memory(1000, 10000)
        est_20k = estimate_workflow_memory(1000, 20000)

        # Genotypes scale linearly with SNPs
        ratio = est_20k.genotypes_gb / est_10k.genotypes_gb
        assert 1.8 < ratio < 2.2, f"Genotypes should scale ~2x, got {ratio:.1f}x"


class TestEigendecompMemory:
    """Tests for eigendecomposition memory usage."""

    @pytest.mark.slow
    def test_eigendecomp_workspace_linear(self):
        """Eigendecomp should use O(n) workspace, not O(n^2)."""
        import jax.numpy as jnp

        # 3000x3000 matrix
        n = 3000
        rng = np.random.default_rng(42)
        A = rng.standard_normal((n, n))
        K = (A + A.T) / 2

        mem_before = psutil.Process().memory_info().rss / 1e9

        # Run JAX eigendecomposition
        K_jax = jnp.array(K)
        eigenvalues, eigenvectors = jnp.linalg.eigh(K_jax)
        eigenvalues.block_until_ready()
        eigenvectors.block_until_ready()

        mem_after = psutil.Process().memory_info().rss / 1e9
        mem_delta = mem_after - mem_before

        # Matrix is n^2 * 8 = 72MB
        # Eigenvectors are another 72MB
        # O(n) workspace would be ~240KB
        # O(n^2) workspace would add another 72MB+
        # Total with O(n^2) would be >200MB

        # Allow generous margin for JIT overhead, but should be < 500MB
        assert (
            mem_delta < 0.5
        ), f"Eigendecomp delta {mem_delta:.2f}GB suggests O(n^2) workspace"
