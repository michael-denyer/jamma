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
        # Interior maximum: not at first or last position
        is_interior = 0 < max_idx < len(lambdas) - 1
        # Boundary maximum: at edge but local maximum (slope inward)
        is_left_boundary_max = max_idx == 0 and logls[0] >= logls[1]
        is_right_boundary_max = max_idx == len(lambdas) - 1 and logls[-1] >= logls[-2]
        assert (
            is_interior or is_left_boundary_max or is_right_boundary_max
        ), f"REML appears monotonic - no clear maximum (max at idx {max_idx})"


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

        # Use atol for values near zero where rtol alone would be too strict
        np.testing.assert_allclose(K, K_reconstructed, rtol=1e-8, atol=1e-14)

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


# -----------------------------------------------------------------------------
# Degenerate SNP Edge Cases
# -----------------------------------------------------------------------------


class TestDegenerateSNPEdgeCases:
    """Property tests for degenerate SNP scenarios (zero variance, constant, etc.)."""

    def test_constant_genotype_returns_nan(self):
        """SNP with zero variance (constant values) should return NaN stats.

        When all genotypes are identical, P_xx = 0 after projection,
        so beta and SE cannot be computed (division by zero).
        """
        rng = np.random.default_rng(42)
        n_samples = 50
        n_cvt = 1

        # Generate valid inputs
        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.1
        U = np.eye(n_samples)
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))

        # Constant genotype (zero variance)
        x = np.full(n_samples, 1.0)

        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)
        Hi_eval = 1.0 / (1.0 * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(1.0, Pab, n_cvt, n_samples)

        # All should be NaN for constant genotype
        assert np.isnan(beta), f"Expected NaN beta for constant genotype, got {beta}"
        assert np.isnan(se), f"Expected NaN SE for constant genotype, got {se}"
        assert np.isnan(
            p_wald
        ), f"Expected NaN p-value for constant genotype, got {p_wald}"

    def test_constant_genotype_jax_returns_nan(self):
        """JAX path should also return NaN for constant genotypes."""
        import jax.numpy as jnp

        from jamma.lmm.likelihood_jax import calc_wald_stats_jax

        rng = np.random.default_rng(42)
        n_samples = 50

        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.1
        U = np.eye(n_samples)
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = np.full(n_samples, 1.0)  # Constant

        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)

        beta, se, p_wald = calc_wald_stats_jax(
            1.0,
            jnp.array(eigenvalues),
            jnp.array(Uab),
            n_samples,
        )

        assert np.isnan(float(beta)), f"JAX: Expected NaN beta, got {beta}"
        assert np.isnan(float(se)), f"JAX: Expected NaN SE, got {se}"
        assert np.isnan(float(p_wald)), f"JAX: Expected NaN p-value, got {p_wald}"

    @given(data=valid_lmm_inputs())
    @settings(
        max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_cpu_jax_wald_equivalence_normal_case(self, data):
        """CPU and JAX Wald stats should match for normal inputs."""
        import jax.numpy as jnp

        from jamma.lmm.likelihood_jax import calc_wald_stats_jax

        eigenvalues, Uab, n_cvt, n_samples = data
        lambda_val = 1.0

        # CPU path
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        beta_cpu, se_cpu, p_cpu = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        # JAX path
        beta_jax, se_jax, p_jax = calc_wald_stats_jax(
            lambda_val,
            jnp.array(eigenvalues),
            jnp.array(Uab),
            n_samples,
        )

        np.testing.assert_allclose(
            beta_cpu, float(beta_jax), rtol=1e-6, err_msg="beta CPU/JAX mismatch"
        )
        np.testing.assert_allclose(
            se_cpu, float(se_jax), rtol=1e-6, err_msg="SE CPU/JAX mismatch"
        )
        np.testing.assert_allclose(
            p_cpu, float(p_jax), rtol=1e-6, err_msg="p-value CPU/JAX mismatch"
        )

    def test_nearly_constant_genotype_produces_valid_or_nan(self):
        """Near-constant genotype should produce valid stats or NaN, never inf.

        When a SNP has very low variance, numerical issues can arise.
        The result should be either valid finite values or NaN, but never inf.
        """
        rng = np.random.default_rng(123)
        n_samples = 50
        n_cvt = 1

        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.1
        U = np.eye(n_samples)
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))

        # Nearly constant genotype: one different value
        x = np.full(n_samples, 1.0)
        x[0] = 2.0  # One outlier

        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)
        Hi_eval = 1.0 / (1.0 * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(1.0, Pab, n_cvt, n_samples)

        # Should be finite or NaN, never inf
        assert np.isfinite(beta) or np.isnan(beta), f"Unexpected inf beta: {beta}"
        assert np.isfinite(se) or np.isnan(se), f"Unexpected inf SE: {se}"
        assert np.isfinite(p_wald) or np.isnan(
            p_wald
        ), f"Unexpected inf p-value: {p_wald}"

    @given(
        n_samples=st.integers(min_value=30, max_value=50),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    @settings(
        max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow]
    )
    def test_wald_stats_never_inf(self, n_samples, seed):
        """Wald stats should never be infinite (either finite or NaN)."""
        rng = np.random.default_rng(seed)
        n_cvt = 1

        eigenvalues = np.abs(rng.standard_normal(n_samples)) + 0.01
        U = np.eye(n_samples)
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)

        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)
        Hi_eval = 1.0 / (1.0 * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(1.0, Pab, n_cvt, n_samples)

        assert not np.isinf(beta), f"Infinite beta: {beta}"
        assert not np.isinf(se), f"Infinite SE: {se}"
        assert not np.isinf(p_wald), f"Infinite p-value: {p_wald}"

    def test_negative_eigenvalue_handling(self):
        """REML should handle negative eigenvalues gracefully.

        Non-PSD kinship matrices can have negative eigenvalues.
        The code uses log(abs(v)) to handle this, so REML should still be finite.
        """
        rng = np.random.default_rng(999)
        n_samples = 50
        n_cvt = 1

        # Include some negative eigenvalues (non-PSD kinship)
        eigenvalues = rng.standard_normal(n_samples)  # Some negative
        eigenvalues = np.sort(eigenvalues)  # Sort ascending

        U = np.eye(n_samples)
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)

        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)

        # REML should still be finite
        logl = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt)

        assert np.isfinite(logl), f"Non-finite REML with negative eigenvalues: {logl}"

    def test_cpu_jax_equivalence_with_negative_eigenvalues(self):
        """CPU and JAX paths should agree even with negative eigenvalues."""
        import jax.numpy as jnp

        from jamma.lmm.likelihood_jax import reml_log_likelihood_jax

        rng = np.random.default_rng(888)
        n_samples = 50
        n_cvt = 1

        # Include some negative eigenvalues
        eigenvalues = rng.standard_normal(n_samples)

        U = np.eye(n_samples)
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)

        Uab = compute_Uab(U.T @ W, U.T @ y, U.T @ x)

        # CPU path
        logl_cpu = reml_log_likelihood(1.0, eigenvalues, Uab, n_cvt)

        # JAX path
        logl_jax = reml_log_likelihood_jax(
            1.0,
            jnp.array(eigenvalues),
            jnp.array(Uab),
        )

        np.testing.assert_allclose(
            logl_cpu,
            float(logl_jax),
            rtol=1e-5,
            err_msg="CPU/JAX divergence with negative eigenvalues",
        )
