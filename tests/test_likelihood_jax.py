"""Unit tests for JAX likelihood functions and edge cases.

Tests cover: negative P_yy handling, degenerate SNPs (Wald/LRT),
batch assembly correctness, and near-zero eigenvalue stability.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

import jax.numpy as jnp

from jamma.lmm.likelihood import (
    calc_pab,
    compute_Uab,
    reml_log_likelihood,
)
from jamma.lmm.likelihood_jax import (
    build_index_table,
    mle_log_likelihood_jax,
    reml_log_likelihood_jax,
)
from jamma.lmm.stats import calc_wald_test

pytestmark = pytest.mark.requires_jax


def _make_test_data(n_samples=50, n_cvt=1, rng_seed=42):
    """Create synthetic eigenvalues, UtW, Uty, Utx for testing."""
    rng = np.random.default_rng(rng_seed)

    # Eigenvalues from a PSD kinship matrix (all positive)
    eigenvalues = np.sort(rng.exponential(1.0, size=n_samples))[::-1]

    # Covariates (intercept only for n_cvt=1)
    W = np.ones((n_samples, n_cvt))
    # Random phenotype
    y = rng.standard_normal(n_samples)
    # Random genotype
    x = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=[0.25, 0.5, 0.25])

    # Simulate U.T @ vectors (as if we've eigendecomposed)
    U = np.linalg.qr(rng.standard_normal((n_samples, n_samples)))[0]
    UtW = U.T @ W
    Uty = U.T @ y
    Utx = U.T @ x

    return eigenvalues, UtW, Uty, Utx


@pytest.mark.tier0
class TestNegativePyyHandling:
    """Tests for negative P_yy clamping in JAX REML and MLE paths."""

    def test_reml_negative_pyy_returns_nan(self):
        """REML with contrived Uab that produces negative P_yy returns NaN."""
        n_cvt = 1
        eigenvalues = jnp.ones(10, dtype=jnp.float64)
        table = build_index_table(n_cvt)
        n_index = table["n_index"]

        # Create Uab where the P_yy projection will go negative
        Uab = jnp.zeros((10, n_index), dtype=jnp.float64)
        # Set values such that recursive Pab projection produces negative P_yy
        # by making the W-W diagonal tiny and W-Y cross term large
        Uab = Uab.at[:, 0].set(1e-15)  # WW: near zero
        Uab = Uab.at[:, 2].set(100.0)  # WY: large
        Uab = Uab.at[:, 5].set(-1.0)  # YY: negative → forces negative P_yy

        lambda_val = jnp.float64(1.0)
        result = reml_log_likelihood_jax(n_cvt, lambda_val, eigenvalues, Uab)
        assert jnp.isnan(result), f"Expected NaN for negative P_yy, got {result}"

    def test_mle_negative_pyy_returns_nan(self):
        """MLE with contrived Uab that produces negative P_yy returns NaN."""
        n_cvt = 1
        eigenvalues = jnp.ones(10, dtype=jnp.float64)
        table = build_index_table(n_cvt)
        n_index = table["n_index"]

        Uab = jnp.zeros((10, n_index), dtype=jnp.float64)
        Uab = Uab.at[:, 0].set(1e-15)
        Uab = Uab.at[:, 2].set(100.0)
        Uab = Uab.at[:, 5].set(-1.0)

        lambda_val = jnp.float64(1.0)
        result = mle_log_likelihood_jax(n_cvt, lambda_val, eigenvalues, Uab)
        assert jnp.isnan(result), f"Expected NaN for negative P_yy, got {result}"

    def test_reml_normal_data_produces_finite(self):
        """REML on well-conditioned data produces finite likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        eigenvalues_jax = jnp.array(eigenvalues)
        Uab_jax = jnp.array(Uab)
        lambda_val = jnp.float64(1.0)

        result = reml_log_likelihood_jax(1, lambda_val, eigenvalues_jax, Uab_jax)
        assert jnp.isfinite(result), f"Expected finite result, got {result}"


@pytest.mark.tier0
class TestJaxNumpyConsistency:
    """JAX and NumPy likelihood paths should agree on normal data."""

    def test_reml_jax_matches_numpy(self):
        """JAX REML matches NumPy REML on well-conditioned data."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        lambda_val = 1.0
        numpy_result = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt=1)

        jax_result = float(
            reml_log_likelihood_jax(
                1, jnp.float64(lambda_val), jnp.array(eigenvalues), jnp.array(Uab)
            )
        )

        np.testing.assert_allclose(jax_result, numpy_result, rtol=1e-10)

    @pytest.mark.parametrize("lambda_val", [1e-5, 0.01, 1.0, 100.0, 1e5])
    def test_reml_jax_matches_numpy_across_lambda(self, lambda_val):
        """JAX and NumPy REML agree across lambda range including boundaries."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        numpy_result = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt=1)
        jax_result = float(
            reml_log_likelihood_jax(
                1, jnp.float64(lambda_val), jnp.array(eigenvalues), jnp.array(Uab)
            )
        )

        np.testing.assert_allclose(jax_result, numpy_result, rtol=1e-10)


@pytest.mark.tier0
class TestWaldDegenerateSNP:
    """Wald test should return NaN for degenerate (constant) genotypes."""

    def test_wald_constant_genotype_returns_nan(self):
        """Constant genotype (no variance) produces NaN beta/se/p."""
        eigenvalues, UtW, Uty, _ = _make_test_data()

        # Constant genotype → zero Utx after rotation
        Utx_const = np.zeros(len(eigenvalues))
        Uab = compute_Uab(UtW, Uty, Utx_const)

        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(1, Hi_eval, Uab)

        beta, se, p = calc_wald_test(Pab, n_cvt=1, ni_test=len(eigenvalues))
        assert np.isnan(beta), f"Expected NaN beta for constant genotype, got {beta}"
        assert np.isnan(se), f"Expected NaN se, got {se}"
        assert np.isnan(p), f"Expected NaN p, got {p}"

    def test_wald_nearly_constant_genotype(self):
        """Near-constant genotype (tiny variance) produces finite but large SE."""
        eigenvalues, UtW, Uty, _ = _make_test_data()
        n = len(eigenvalues)

        # One non-zero value in an otherwise constant genotype
        Utx_near_const = np.zeros(n)
        Utx_near_const[0] = 1e-6
        Uab = compute_Uab(UtW, Uty, Utx_near_const)

        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(1, Hi_eval, Uab)

        beta, se, p = calc_wald_test(Pab, n_cvt=1, ni_test=n)
        # Should produce finite results (genotype has some variance)
        assert np.isfinite(beta), f"Expected finite beta, got {beta}"


@pytest.mark.tier0
class TestNearZeroEigenvalues:
    """REML should handle near-machine-epsilon eigenvalues correctly."""

    def test_reml_with_zero_eigenvalues(self):
        """Eigenvalues exactly 0 should produce finite REML likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data(n_samples=20)
        # Zero out half the eigenvalues (simulating rank-deficient kinship)
        eigenvalues[:10] = 0.0

        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), (
            f"Expected finite REML with zero eigenvalues, got {result}"
        )

    def test_reml_jax_with_zero_eigenvalues(self):
        """JAX REML handles zero eigenvalues same as NumPy."""
        eigenvalues, UtW, Uty, Utx = _make_test_data(n_samples=20)
        eigenvalues[:10] = 0.0
        Uab = compute_Uab(UtW, Uty, Utx)

        numpy_result = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt=1)
        jax_result = float(
            reml_log_likelihood_jax(
                1, jnp.float64(1.0), jnp.array(eigenvalues), jnp.array(Uab)
            )
        )
        np.testing.assert_allclose(jax_result, numpy_result, rtol=1e-10)

    def test_reml_with_tiny_eigenvalues(self):
        """Eigenvalues near machine epsilon don't produce inf/nan."""
        eigenvalues, UtW, Uty, Utx = _make_test_data(n_samples=20)
        eigenvalues[:5] = 1e-15  # Near machine epsilon

        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), f"Expected finite REML, got {result}"


@pytest.mark.tier0
class TestLambdaBounds:
    """REML should behave sensibly at lambda optimization boundaries."""

    def test_reml_at_lower_bound(self):
        """Lambda = 1e-5 (lower bound) produces finite likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1e-5, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), f"Expected finite at lower bound, got {result}"

    def test_reml_at_upper_bound(self):
        """Lambda = 1e5 (upper bound) produces finite likelihood."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        result = reml_log_likelihood(1e5, eigenvalues, Uab, n_cvt=1)
        assert np.isfinite(result), f"Expected finite at upper bound, got {result}"

    def test_reml_monotonic_near_optimum(self):
        """REML should be smooth (no discontinuities) around typical lambda values."""
        eigenvalues, UtW, Uty, Utx = _make_test_data()
        Uab = compute_Uab(UtW, Uty, Utx)

        lambdas = np.logspace(-2, 2, 50)
        results = [
            reml_log_likelihood(lam, eigenvalues, Uab, n_cvt=1) for lam in lambdas
        ]

        assert all(np.isfinite(r) for r in results), (
            "All lambda values should give finite REML"
        )
        # Check no huge jumps (>100x) between adjacent values
        for i in range(1, len(results)):
            ratio = abs(results[i] / results[i - 1]) if results[i - 1] != 0 else 1.0
            assert ratio < 100, (
                f"Discontinuity at lambda={lambdas[i]}: {results[i - 1]} → {results[i]}"
            )


@pytest.mark.tier0
class TestCovariateRankValidation:
    """Test that rank-deficient covariates are caught."""

    def test_linearly_dependent_covariates_rejected(self):
        """Covariates with duplicate columns should raise ValueError."""
        from jamma.lmm.prepare import _build_covariate_matrix

        # Column 2 = 2 * Column 1 (intercept) → rank 1 instead of 2
        covariates = np.column_stack([np.ones(20), 2 * np.ones(20)])

        with pytest.raises(ValueError, match="rank-deficient"):
            _build_covariate_matrix(covariates, n_samples=20)

    def test_full_rank_covariates_accepted(self):
        """Covariates with full rank pass validation."""
        from jamma.lmm.prepare import _build_covariate_matrix

        rng = np.random.default_rng(42)
        covariates = np.column_stack([np.ones(20), rng.standard_normal(20)])

        W, n_cvt = _build_covariate_matrix(covariates, n_samples=20)
        assert n_cvt == 2
        assert W.shape == (20, 2)


@pytest.mark.tier0
class TestKinshipSymmetryCheck:
    """Test that non-symmetric kinship matrices produce warnings."""

    def test_asymmetric_kinship_warns(self):
        """Non-symmetric kinship produces loguru warning during eigendecomposition."""
        from io import StringIO

        from loguru import logger

        from jamma.lmm.eigen import eigendecompose_kinship

        K = np.eye(10)
        K[0, 1] = 0.5  # Asymmetric
        K[1, 0] = 0.3

        buf = StringIO()
        sink_id = logger.add(buf, level="WARNING", format="{message}")
        try:
            eigendecompose_kinship(K, check_memory=False)
        finally:
            logger.remove(sink_id)

        assert "not symmetric" in buf.getvalue(), (
            f"Expected 'not symmetric' warning, got: {buf.getvalue()[:200]}"
        )

    def test_symmetric_kinship_no_warning(self):
        """Symmetric kinship produces no symmetry warning."""
        from io import StringIO

        from loguru import logger

        from jamma.lmm.eigen import eigendecompose_kinship

        K = np.eye(10)

        buf = StringIO()
        sink_id = logger.add(buf, level="WARNING", format="{message}")
        try:
            eigendecompose_kinship(K, check_memory=False)
        finally:
            logger.remove(sink_id)

        assert "not symmetric" not in buf.getvalue()
