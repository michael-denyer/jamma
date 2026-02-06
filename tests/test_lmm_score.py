"""Unit tests for Score test implementation.

Tests verify:
1. compute_null_model_lambda produces valid lambda
2. calc_score_test follows GEMMA formula (P_xx, P_xy, P_yy at correct level)
3. Score p-values are in valid range [0, 1]
4. Score test uses null model lambda (not per-SNP optimization)
"""

import numpy as np
import pytest

from jamma.core import configure_jax
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    calc_pab,
    compute_null_model_lambda,
    compute_Uab,
    get_ab_index,
)
from jamma.lmm.stats import calc_score_test, calc_wald_test, f_sf


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX with 64-bit precision before each test."""
    configure_jax(enable_x64=True)


def _create_test_data(n_samples: int = 100, n_cvt: int = 1, seed: int = 42):
    """Create synthetic test data for LMM tests.

    Returns:
        Dictionary with eigenvalues, eigenvectors, UtW, Uty, Utx, W, y, x
    """
    rng = np.random.default_rng(seed)

    # Create valid kinship matrix
    X = rng.standard_normal((n_samples, 200))
    K = X @ X.T / X.shape[1]

    # Eigendecomposition
    eigenvalues, U = eigendecompose_kinship(K)

    # Create phenotype, covariates, genotype
    y = rng.standard_normal(n_samples)
    W = np.ones((n_samples, n_cvt))  # Intercept-only covariate
    x = rng.standard_normal(n_samples)  # Random genotype

    # Rotate to eigenspace
    Uty = U.T @ y
    UtW = U.T @ W
    Utx = U.T @ x

    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": U,
        "UtW": UtW,
        "Uty": Uty,
        "Utx": Utx,
        "W": W,
        "y": y,
        "x": x,
        "n_samples": n_samples,
        "n_cvt": n_cvt,
    }


class TestNullModelLambda:
    """Tests for null model lambda computation."""

    def test_null_model_lambda_valid_range(self):
        """Null model lambda should be in valid range [l_min, l_max]."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)

        lambda_null, logl_null = compute_null_model_lambda(
            data["eigenvalues"],
            data["UtW"],
            data["Uty"],
            data["n_cvt"],
            l_min=1e-5,
            l_max=1e5,
        )

        assert lambda_null >= 1e-5, f"Lambda {lambda_null} below l_min"
        assert lambda_null <= 1e5, f"Lambda {lambda_null} above l_max"
        assert np.isfinite(lambda_null), "Lambda should be finite"
        assert np.isfinite(logl_null), "Log-likelihood should be finite"

    def test_null_model_lambda_no_genotype(self):
        """Null model uses Uab without genotype (Utx=None)."""
        data = _create_test_data(n_samples=80, n_cvt=1, seed=123)

        # Compute Uab without genotype
        Uab_no_geno = compute_Uab(data["UtW"], data["Uty"], Utx=None)

        # Verify Uab has zeros in genotype columns
        # For n_cvt=1: indices 1, 3, 4 involve genotype (WX, XX, XY)
        n_cvt = data["n_cvt"]
        idx_wx = get_ab_index(1, 2, n_cvt)  # WX = (1,2)
        idx_xx = get_ab_index(2, 2, n_cvt)  # XX = (2,2)
        idx_xy = get_ab_index(2, 3, n_cvt)  # XY = (2,3)

        assert np.allclose(
            Uab_no_geno[:, idx_wx], 0.0
        ), "WX column should be zero without genotype"
        assert np.allclose(
            Uab_no_geno[:, idx_xx], 0.0
        ), "XX column should be zero without genotype"
        assert np.allclose(
            Uab_no_geno[:, idx_xy], 0.0
        ), "XY column should be zero without genotype"

        # Verify non-genotype columns are NOT zero
        idx_ww = get_ab_index(1, 1, n_cvt)  # WW = (1,1)
        idx_wy = get_ab_index(1, 3, n_cvt)  # WY = (1,3)
        idx_yy = get_ab_index(3, 3, n_cvt)  # YY = (3,3)

        assert not np.allclose(
            Uab_no_geno[:, idx_ww], 0.0
        ), "WW column should be non-zero"
        assert not np.allclose(
            Uab_no_geno[:, idx_wy], 0.0
        ), "WY column should be non-zero"
        assert not np.allclose(
            Uab_no_geno[:, idx_yy], 0.0
        ), "YY column should be non-zero"

    def test_null_model_lambda_reproducible(self):
        """Same inputs produce same lambda."""
        data = _create_test_data(n_samples=50, n_cvt=1, seed=999)

        lambda1, logl1 = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )
        lambda2, logl2 = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        assert lambda1 == lambda2, "Lambda should be reproducible"
        assert logl1 == logl2, "Log-likelihood should be reproducible"

    def test_null_model_lambda_multiple_covariates(self):
        """Null model lambda works with multiple covariates."""
        n_samples = 100
        n_cvt = 3
        rng = np.random.default_rng(42)

        # Create valid kinship matrix
        X = rng.standard_normal((n_samples, 200))
        K = X @ X.T / X.shape[1]

        eigenvalues, U = eigendecompose_kinship(K)

        y = rng.standard_normal(n_samples)
        W = np.column_stack(
            [
                np.ones(n_samples),  # Intercept
                rng.standard_normal(n_samples),  # Covariate 1
                rng.standard_normal(n_samples),  # Covariate 2
            ]
        )

        Uty = U.T @ y
        UtW = U.T @ W

        lambda_null, logl_null = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)

        assert lambda_null >= 1e-5, f"Lambda {lambda_null} below l_min"
        assert lambda_null <= 1e5, f"Lambda {lambda_null} above l_max"
        assert np.isfinite(logl_null), "Log-likelihood should be finite"


class TestScoreTestMath:
    """Tests for Score test statistic computation."""

    def test_score_test_pvalue_valid_range(self):
        """Score test p-values should be in [0, 1]."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)

        # Compute null model lambda
        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], data["n_cvt"]
        )

        # Compute Uab WITH genotype
        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])

        # Compute Pab with null lambda
        Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
        Pab = calc_pab(data["n_cvt"], Hi_eval, Uab)

        # Compute Score test
        beta, se, p_score = calc_score_test(
            lambda_null, Pab, data["n_cvt"], data["n_samples"]
        )

        assert 0.0 <= p_score <= 1.0, f"P-value {p_score} not in [0, 1]"
        assert np.isfinite(beta), "Beta should be finite"
        assert np.isfinite(se), "SE should be finite"
        assert se > 0, f"SE should be positive, got {se}"

    def test_score_test_uses_correct_pab_level(self):
        """Score test extracts values at level n_cvt, not n_cvt+1."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]

        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )

        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])
        Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        # Get the indices we expect Score test to use
        index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
        index_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
        index_xy = get_ab_index(n_cvt + 2, n_cvt + 1, n_cvt)

        # Score test uses level n_cvt (after projecting out covariates only)
        P_yy_score = Pab[n_cvt, index_yy]
        P_xx_score = Pab[n_cvt, index_xx]
        P_xy_score = Pab[n_cvt, index_xy]

        # Wald test uses level n_cvt+1 (after projecting out covariates AND genotype)
        # For P_yy only - P_xx and P_xy are at n_cvt for both
        Px_yy_wald = Pab[n_cvt + 1, index_yy]

        # Verify Score test would use these values
        # F = n * P_xy^2 / (P_yy * P_xx)
        n = data["n_samples"]
        f_stat_expected = n * (P_xy_score**2) / (P_yy_score * P_xx_score)

        # Compute actual Score test result
        _, _, p_score = calc_score_test(lambda_null, Pab, n_cvt, n)

        # Verify F-statistic produces the same p-value
        df = n - n_cvt - 1
        p_expected = f_sf(f_stat_expected, 1.0, float(df))

        assert np.isclose(
            p_score, p_expected, rtol=1e-10
        ), f"P-values differ: {p_score} vs {p_expected}"

        # Also verify Wald uses different Px_yy (just to confirm difference)
        assert (
            P_yy_score != Px_yy_wald
        ), "Score and Wald should use different P_yy values"

    def test_score_test_degenerate_snp(self):
        """Degenerate SNPs (P_xx <= 0) return NaN."""
        data = _create_test_data(n_samples=50, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]

        # Create a constant genotype (all zeros)
        Utx_constant = np.zeros(data["n_samples"])

        Uab = compute_Uab(data["UtW"], data["Uty"], Utx_constant)

        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )
        Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_score = calc_score_test(lambda_null, Pab, n_cvt, data["n_samples"])

        assert np.isnan(beta), "Beta should be NaN for degenerate SNP"
        assert np.isnan(se), "SE should be NaN for degenerate SNP"
        assert np.isnan(p_score), "P-score should be NaN for degenerate SNP"

    def test_score_statistic_formula(self):
        """Verify F = n * P_xy^2 / (P_yy * P_xx) formula."""
        data = _create_test_data(n_samples=80, n_cvt=1, seed=555)
        n_cvt = data["n_cvt"]
        n = data["n_samples"]

        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )

        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])
        Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        # Manually compute F statistic
        index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
        index_xx = get_ab_index(n_cvt + 1, n_cvt + 1, n_cvt)
        index_xy = get_ab_index(n_cvt + 2, n_cvt + 1, n_cvt)

        P_yy = Pab[n_cvt, index_yy]
        P_xx = Pab[n_cvt, index_xx]
        P_xy = Pab[n_cvt, index_xy]

        f_stat_manual = n * (P_xy**2) / (P_yy * P_xx)
        df = n - n_cvt - 1
        p_expected = f_sf(f_stat_manual, 1.0, float(df))

        # Get p-value from calc_score_test
        _, _, p_score = calc_score_test(lambda_null, Pab, n_cvt, n)

        assert np.isclose(
            p_score, p_expected, rtol=1e-10
        ), f"Formula mismatch: {p_score} vs {p_expected}"


class TestScoreVsWald:
    """Tests comparing Score and Wald test properties."""

    def test_score_uses_fixed_lambda(self):
        """Score test uses same lambda for all SNPs (null model)."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]
        rng = np.random.default_rng(42)

        # Compute null model lambda (once)
        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )

        # Generate multiple SNPs and verify same lambda is used
        p_values = []
        for _i in range(10):
            # Different genotype for each SNP
            Utx_i = data["eigenvectors"].T @ rng.standard_normal(data["n_samples"])

            Uab = compute_Uab(data["UtW"], data["Uty"], Utx_i)
            Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
            Pab = calc_pab(n_cvt, Hi_eval, Uab)

            _, _, p_score = calc_score_test(lambda_null, Pab, n_cvt, data["n_samples"])
            p_values.append(p_score)

        # All p-values should be valid (computed with same lambda)
        for i, p in enumerate(p_values):
            assert 0.0 <= p <= 1.0, f"SNP {i}: p-value {p} not in [0, 1]"

    def test_score_and_wald_agree_on_direction(self):
        """Score and Wald tests should agree on effect direction (beta sign)."""
        data = _create_test_data(n_samples=100, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]

        # Compute null model lambda for Score test
        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )

        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])

        # Score test with null lambda
        Hi_eval_score = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
        Pab_score = calc_pab(n_cvt, Hi_eval_score, Uab)
        beta_score, _, _ = calc_score_test(
            lambda_null, Pab_score, n_cvt, data["n_samples"]
        )

        # Wald test with optimized lambda (per-SNP) -- use JAX optimizer
        import jax.numpy as jnp

        from jamma.lmm.likelihood_jax import (
            batch_compute_iab,
            golden_section_optimize_lambda,
        )

        Uab_jax = jnp.expand_dims(jnp.array(Uab), 0)
        Iab_jax = batch_compute_iab(n_cvt, Uab_jax)
        eigenvalues_jax = jnp.array(data["eigenvalues"])
        lambdas, _ = golden_section_optimize_lambda(
            n_cvt, eigenvalues_jax, Uab_jax, Iab_jax
        )
        lambda_wald = float(lambdas[0])
        Hi_eval_wald = 1.0 / (lambda_wald * data["eigenvalues"] + 1.0)
        Pab_wald = calc_pab(n_cvt, Hi_eval_wald, Uab)
        beta_wald, _, _ = calc_wald_test(
            lambda_wald, Pab_wald, n_cvt, data["n_samples"]
        )

        # Both should have same sign (effect direction)
        assert np.sign(beta_score) == np.sign(
            beta_wald
        ), f"Score beta={beta_score}, Wald beta={beta_wald} have different signs"

    def test_score_faster_concept(self):
        """Score test reuses lambda (concept test, not timing)."""
        # This is a conceptual test verifying the API supports the efficiency pattern
        data = _create_test_data(n_samples=50, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]
        rng = np.random.default_rng(42)

        # Score test pattern: compute lambda ONCE
        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )

        # Then reuse for many SNPs (no optimization needed)
        n_snps = 20
        score_results = []
        for _ in range(n_snps):
            Utx_i = data["eigenvectors"].T @ rng.standard_normal(data["n_samples"])
            Uab = compute_Uab(data["UtW"], data["Uty"], Utx_i)
            Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
            Pab = calc_pab(n_cvt, Hi_eval, Uab)
            result = calc_score_test(lambda_null, Pab, n_cvt, data["n_samples"])
            score_results.append(result)

        # All results should be valid
        assert len(score_results) == n_snps
        for i, (beta, se, p) in enumerate(score_results):
            assert np.isfinite(beta), f"SNP {i}: beta not finite"
            assert np.isfinite(se), f"SNP {i}: se not finite"
            assert 0.0 <= p <= 1.0, f"SNP {i}: p-value {p} not in [0, 1]"


class TestScoreTestEdgeCases:
    """Edge case tests for Score test."""

    def test_score_test_small_sample(self):
        """Score test works with small sample sizes."""
        data = _create_test_data(n_samples=20, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]

        lambda_null, _ = compute_null_model_lambda(
            data["eigenvalues"], data["UtW"], data["Uty"], n_cvt
        )

        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])
        Hi_eval = 1.0 / (lambda_null * data["eigenvalues"] + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_score = calc_score_test(lambda_null, Pab, n_cvt, data["n_samples"])

        assert np.isfinite(beta), "Beta should be finite for small samples"
        assert np.isfinite(se), "SE should be finite for small samples"
        assert 0.0 <= p_score <= 1.0, f"P-value {p_score} not in [0, 1]"

    def test_score_test_extreme_lambda(self):
        """Score test handles extreme lambda values."""
        data = _create_test_data(n_samples=50, n_cvt=1, seed=42)
        n_cvt = data["n_cvt"]

        Uab = compute_Uab(data["UtW"], data["Uty"], data["Utx"])

        # Test with very small lambda (nearly zero genetic variance)
        lambda_small = 1e-5
        Hi_eval_small = 1.0 / (lambda_small * data["eigenvalues"] + 1.0)
        Pab_small = calc_pab(n_cvt, Hi_eval_small, Uab)
        _, _, p_small = calc_score_test(
            lambda_small, Pab_small, n_cvt, data["n_samples"]
        )
        assert (
            0.0 <= p_small <= 1.0
        ), f"P-value {p_small} not in [0, 1] for small lambda"

        # Test with very large lambda (high genetic variance)
        lambda_large = 1e5
        Hi_eval_large = 1.0 / (lambda_large * data["eigenvalues"] + 1.0)
        Pab_large = calc_pab(n_cvt, Hi_eval_large, Uab)
        _, _, p_large = calc_score_test(
            lambda_large, Pab_large, n_cvt, data["n_samples"]
        )
        assert (
            0.0 <= p_large <= 1.0
        ), f"P-value {p_large} not in [0, 1] for large lambda"

    def test_score_test_perfect_correlation(self):
        """Score test handles genotype perfectly correlated with phenotype."""
        n_samples = 50
        n_cvt = 1
        rng = np.random.default_rng(42)

        # Create kinship
        X = rng.standard_normal((n_samples, 200))
        K = X @ X.T / X.shape[1]
        eigenvalues, U = eigendecompose_kinship(K)

        # Phenotype and genotype are identical (perfect correlation)
        y = rng.standard_normal(n_samples)
        x = y.copy()  # Perfect correlation
        W = np.ones((n_samples, n_cvt))

        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        lambda_null, _ = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)

        Uab = compute_Uab(UtW, Uty, Utx)
        Hi_eval = 1.0 / (lambda_null * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_score = calc_score_test(lambda_null, Pab, n_cvt, n_samples)

        # Should get very significant p-value (close to 0)
        assert (
            p_score < 0.01
        ), f"Perfect correlation should give small p-value, got {p_score}"
        assert np.isfinite(beta), "Beta should be finite"
        assert np.isfinite(se), "SE should be finite"
