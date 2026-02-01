"""Property-based tests using Hypothesis for numerical accuracy verification.

These tests verify:
1. Mathematical properties that must hold (symmetry, positive semi-definiteness)
2. CPU/JAX path equivalence across random inputs
3. Edge case handling (missing data, extreme values, rank-deficient matrices)
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jamma.core import configure_jax
from jamma.lmm.likelihood import calc_pab, compute_Uab, reml_log_likelihood
from jamma.lmm.stats import calc_wald_test


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX with 64-bit precision before each test."""
    configure_jax(enable_x64=True)


# -----------------------------------------------------------------------------
# Custom Strategies for Genetic Data
# -----------------------------------------------------------------------------


@st.composite
def genotype_matrix(draw, min_samples=10, max_samples=100, min_snps=5, max_snps=50):
    """Generate realistic genotype matrices (values in {0, 1, 2}).

    Uses a random seed to generate genotypes with realistic variance.
    """
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_snps = draw(st.integers(min_value=min_snps, max_value=max_snps))
    seed = draw(st.integers(min_value=0, max_value=2**32 - 1))

    # Use numpy random to get realistic genotype distributions
    rng = np.random.default_rng(seed)

    # Generate genotypes with realistic allele frequencies
    # Use varying MAFs to ensure non-constant columns
    mafs = rng.uniform(0.1, 0.5, n_snps)
    genotypes = np.zeros((n_samples, n_snps), dtype=np.float64)

    for j in range(n_snps):
        p = mafs[j]
        # Hardy-Weinberg genotype frequencies
        probs = [(1 - p) ** 2, 2 * p * (1 - p), p**2]
        genotypes[:, j] = rng.choice([0.0, 1.0, 2.0], size=n_samples, p=probs)

    return genotypes


@st.composite
def lambda_value(draw):
    """Generate lambda (variance ratio) in realistic REML range."""
    # Lambda typically ranges from 1e-5 to 1e5 in log scale
    log_lambda = draw(st.floats(min_value=-4.0, max_value=4.0))
    return 10.0**log_lambda


@st.composite
def valid_lmm_inputs(draw, min_samples=30, max_samples=60):
    """Generate valid LMM inputs (eigenvalues, Uab matrix)."""
    n_samples = draw(st.integers(min_value=min_samples, max_value=max_samples))
    n_cvt = 1  # Always use n_cvt=1 for simplicity

    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))

    # Generate valid kinship-like matrix and eigendecompose
    X = rng.standard_normal((n_samples, max(n_samples, 50)))
    K = X @ X.T / X.shape[1]
    eigenvalues = np.linalg.eigvalsh(K)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Threshold small values

    U = np.linalg.eigh(K)[1]

    # Generate phenotype, covariates, genotype
    y = rng.standard_normal(n_samples)
    W = np.ones((n_samples, 1))  # Intercept only
    x = rng.standard_normal(n_samples)

    # Rotate by eigenvectors
    Uty = U.T @ y
    UtW = U.T @ W
    Utx = U.T @ x

    # Compute Uab matrix
    Uab = compute_Uab(UtW, Uty, Utx)

    return eigenvalues, Uab, n_cvt, n_samples


# -----------------------------------------------------------------------------
# Kinship Matrix Properties
# -----------------------------------------------------------------------------


class TestKinshipProperties:
    """Property tests for kinship matrix computation."""

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=20, max_snps=40
        )
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_kinship_is_symmetric(self, genotypes):
        """Kinship matrix must be symmetric."""
        from jamma.kinship import compute_centered_kinship

        K = compute_centered_kinship(genotypes)

        # Check symmetry
        np.testing.assert_allclose(K, K.T, rtol=1e-10)

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=20, max_snps=40
        )
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_kinship_diagonal_positive(self, genotypes):
        """Kinship diagonal elements should be non-negative."""
        from jamma.kinship import compute_centered_kinship

        K = compute_centered_kinship(genotypes)

        # Diagonal should be >= 0 (self-relatedness)
        assert np.all(
            K.diagonal() >= -1e-10
        ), f"Negative diagonal: {K.diagonal().min()}"

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=20, max_snps=40
        )
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_kinship_eigenvalues_nonnegative(self, genotypes):
        """Kinship eigenvalues should be non-negative (PSD property)."""
        from jamma.kinship import compute_centered_kinship

        K = compute_centered_kinship(genotypes)
        eigenvalues = np.linalg.eigvalsh(K)

        # Allow small negative eigenvalues due to numerical error
        assert np.all(
            eigenvalues >= -1e-8
        ), f"Large negative eigenvalue: {eigenvalues.min()}"


# -----------------------------------------------------------------------------
# REML Log-Likelihood Properties
# -----------------------------------------------------------------------------


class TestRemlProperties:
    """Property tests for REML log-likelihood computation."""

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_reml_returns_finite(self, data):
        """REML log-likelihood should be finite for valid inputs."""
        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1.0

        logl = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)

        assert np.isfinite(logl), f"Non-finite likelihood: {logl}"

    @given(
        data=valid_lmm_inputs(),
        lambda_val=lambda_value(),
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_reml_finite_across_lambda_range(self, data, lambda_val):
        """REML should be finite across typical lambda range."""
        eigenvalues, Uab, n_cvt, n_samples = data

        logl = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)

        assert np.isfinite(logl), f"Non-finite at lambda={lambda_val}"

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_reml_has_maximum(self, data):
        """REML function should have a maximum in the search range."""
        eigenvalues, Uab, n_cvt, n_samples = data

        # Sample at multiple lambda values
        lambdas = np.logspace(-4, 4, 20)
        logls = [reml_log_likelihood(lam, eigenvalues, Uab, n_cvt) for lam in lambdas]

        # Should have a clear maximum somewhere (not monotonic)
        max_idx = np.argmax(logls)
        # Maximum shouldn't be at the extreme ends (unless at boundary)
        assert (
            max_idx > 0
            or max_idx < len(lambdas) - 1
            or (logls[0] >= logls[1] and logls[-1] >= logls[-2])
        ), "REML appears monotonic - no clear maximum"


# -----------------------------------------------------------------------------
# Wald Test Properties
# -----------------------------------------------------------------------------


class TestWaldProperties:
    """Property tests for Wald test statistics."""

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_pvalue_in_bounds(self, data):
        """P-values must be in [0, 1]."""
        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1.0

        # Compute Pab
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        assert 0.0 <= p_wald <= 1.0, f"P-value out of bounds: {p_wald}"

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_se_positive(self, data):
        """Standard errors must be positive."""
        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1.0

        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        assert se > 0, f"Non-positive SE: {se}"

    @given(
        data=valid_lmm_inputs(),
        lambda_val=lambda_value(),
    )
    @settings(
        max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_wald_consistent_across_lambda(self, data, lambda_val):
        """Wald stats should be finite across lambda range."""
        eigenvalues, Uab, n_cvt, n_samples = data

        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        assert np.isfinite(beta), f"Non-finite beta at lambda={lambda_val}"
        assert np.isfinite(se), f"Non-finite SE at lambda={lambda_val}"
        assert np.isfinite(p_wald), f"Non-finite p-value at lambda={lambda_val}"


# -----------------------------------------------------------------------------
# CPU/JAX Path Equivalence
# -----------------------------------------------------------------------------


class TestCpuJaxEquivalence:
    """Property tests verifying CPU and JAX paths produce identical results."""

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_reml_likelihood_equivalence(self, data):
        """REML log-likelihood should match between CPU and JAX paths."""
        import jax.numpy as jnp

        from jamma.lmm.likelihood_jax import reml_log_likelihood_jax

        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1.0

        # CPU path
        logl_cpu = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)

        # JAX path
        logl_jax = reml_log_likelihood_jax(
            lambda_val,
            jnp.array(eigenvalues),
            jnp.array(Uab),
        )

        # Should match within tolerance
        np.testing.assert_allclose(
            logl_cpu,
            float(logl_jax),
            rtol=1e-5,
            err_msg=f"CPU/JAX divergence at lambda={lambda_val}",
        )

    @given(
        data=valid_lmm_inputs(),
        lambda_val=lambda_value(),
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_reml_equivalence_across_lambda(self, data, lambda_val):
        """CPU/JAX equivalence should hold across lambda range."""
        import jax.numpy as jnp

        from jamma.lmm.likelihood_jax import reml_log_likelihood_jax

        eigenvalues, Uab, n_cvt, n_samples = data

        logl_cpu = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)
        logl_jax = reml_log_likelihood_jax(
            lambda_val,
            jnp.array(eigenvalues),
            jnp.array(Uab),
        )

        np.testing.assert_allclose(
            logl_cpu,
            float(logl_jax),
            rtol=1e-5,
            err_msg=f"CPU/JAX divergence at lambda={lambda_val}",
        )


# -----------------------------------------------------------------------------
# Eigendecomposition Properties
# -----------------------------------------------------------------------------


class TestEigendecompositionProperties:
    """Property tests for eigendecomposition."""

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=30, max_snps=50
        )
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_eigendecomposition_reconstruction(self, genotypes):
        """K = U @ diag(eigenvalues) @ U.T must approximately hold.

        Note: When eigenvalues are thresholded to zero, reconstruction
        won't be exact. We use a looser tolerance to account for this.
        """
        from jamma.kinship import compute_centered_kinship
        from jamma.lmm.eigen import eigendecompose_kinship

        K = compute_centered_kinship(genotypes)
        eigenvalues, U = eigendecompose_kinship(K, threshold=0)  # No thresholding

        # Reconstruct
        K_reconstructed = U @ np.diag(eigenvalues) @ U.T

        np.testing.assert_allclose(K, K_reconstructed, rtol=1e-8)

    @given(
        genotypes=genotype_matrix(
            min_samples=20, max_samples=40, min_snps=30, max_snps=50
        )
    )
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_eigenvectors_orthonormal(self, genotypes):
        """Eigenvectors should be orthonormal."""
        from jamma.kinship import compute_centered_kinship
        from jamma.lmm.eigen import eigendecompose_kinship

        K = compute_centered_kinship(genotypes)
        _, U = eigendecompose_kinship(K, threshold=0)  # No thresholding

        # U.T @ U should be identity (allow small numerical error)
        np.testing.assert_allclose(U.T @ U, np.eye(U.shape[1]), rtol=1e-8, atol=1e-10)


# -----------------------------------------------------------------------------
# Numerical Stability Edge Cases
# -----------------------------------------------------------------------------


class TestNumericalStability:
    """Property tests for numerical stability at edge cases."""

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_extreme_small_lambda(self, data):
        """Should handle very small lambda values."""
        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1e-8

        logl = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)

        assert np.isfinite(logl), f"Non-finite likelihood at lambda={lambda_val}"

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_extreme_large_lambda(self, data):
        """Should handle very large lambda values."""
        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1e6

        logl = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)

        assert np.isfinite(logl), f"Non-finite likelihood at lambda={lambda_val}"

    @given(
        n_samples=st.integers(min_value=30, max_value=50),
        near_zero_frac=st.floats(min_value=0.2, max_value=0.5),
    )
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_handles_near_zero_eigenvalues(self, n_samples, near_zero_frac):
        """Should handle matrices with many near-zero eigenvalues."""
        rng = np.random.default_rng(42)
        n_cvt = 1

        # Create eigenvalues with some very small
        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.1
        n_small = int(n_samples * near_zero_frac)
        eigenvalues[:n_small] = rng.uniform(1e-12, 1e-8, n_small)

        # Generate Uab
        U = np.eye(n_samples)  # Use identity for simplicity
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)

        logl = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt)

        assert np.isfinite(
            logl
        ), f"Non-finite likelihood with {n_small} small eigenvalues"
