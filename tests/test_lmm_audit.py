"""LMM numerical-guard regressions: safe_sqrt, P_yy clamp, covariate
over-parameterization, golden-section optimizer.

Covers boundary behavior at the edge of the LMM numerical envelope. Each
section corresponds to a specific guard in the LMM code; if you remove or
change a guard, the matching section here must change too.
"""

import numpy as np
import pytest

from jamma.lmm.likelihood import (
    _clamp_p_yy,
    _golden_section_minimize,
    compute_Uab,
    mle_log_likelihood,
    reml_log_likelihood,
)
from jamma.lmm.prepare_common import _build_covariate_matrix
from tests.reference.stats import safe_sqrt


@pytest.mark.tier0
class TestSafeSqrt:
    """T1: safe_sqrt boundary behavior."""

    def test_positive_value(self):
        assert safe_sqrt(4.0) == pytest.approx(2.0)

    def test_zero(self):
        assert safe_sqrt(0.0) == 0.0

    def test_small_negative_tolerated(self):
        """Values with |d| < 0.001 use abs(d) — tolerates FP rounding."""
        result = safe_sqrt(-1e-6)
        assert np.isfinite(result)
        assert result == pytest.approx(np.sqrt(1e-6))

    def test_large_negative_returns_nan(self):
        """d = -1.0, |d| >= 0.001, d < 0 => NaN."""
        assert np.isnan(safe_sqrt(-1.0))

    def test_boundary_at_threshold(self):
        """|d| == 0.001 is NOT < 0.001, so d < 0 => NaN."""
        assert np.isnan(safe_sqrt(-0.001))
        # Just inside threshold: |d| < 0.001 => abs used
        assert np.isfinite(safe_sqrt(-0.0009))

    def test_very_small_positive(self):
        result = safe_sqrt(1e-15)
        assert result == pytest.approx(np.sqrt(1e-15))


@pytest.mark.tier0
class TestClampPyy:
    """T2: P_yy clamping helper."""

    def test_positive_above_min_unchanged(self):
        assert _clamp_p_yy(1.0, 0.1) == 1.0

    def test_near_zero_clamped(self):
        assert _clamp_p_yy(1e-12, 0.1) == 1e-8

    def test_zero_clamped(self):
        assert _clamp_p_yy(0.0, 0.1) == 1e-8

    def test_negative_returns_nan(self):
        result = _clamp_p_yy(-0.01, 0.1)
        assert np.isnan(result)

    def test_exactly_at_min_unchanged(self):
        result = _clamp_p_yy(1e-8, 0.1)
        assert result == 1e-8


@pytest.mark.tier0
class TestPyyInLogLikelihood:
    """T2 continued: P_yy clamping produces finite log-likelihood."""

    @pytest.fixture
    def synthetic_eigen(self):
        """Small synthetic eigendecomposition."""
        rng = np.random.default_rng(42)
        n = 20
        X = rng.standard_normal((n, 50))
        K = X @ X.T / 50
        eigenvalues, U = np.linalg.eigh(K)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return eigenvalues, U

    @pytest.fixture
    def synthetic_uab(self, synthetic_eigen):
        """Small synthetic Uab for testing (null model, no genotype)."""
        eigenvalues, U = synthetic_eigen
        n = len(eigenvalues)
        rng = np.random.default_rng(99)
        y = rng.standard_normal(n)
        W = np.ones((n, 1))
        Uty = U.T @ y
        UtW = U.T @ W
        Uab = compute_Uab(UtW, Uty)
        return Uab

    def test_reml_returns_finite(self, synthetic_eigen, synthetic_uab):
        eigenvalues, _ = synthetic_eigen
        result = reml_log_likelihood(0.5, eigenvalues, synthetic_uab, 1, nc_total=2)
        assert np.isfinite(result)

    def test_reml_null_returns_finite(self, synthetic_eigen, synthetic_uab):
        eigenvalues, _ = synthetic_eigen
        result = reml_log_likelihood(0.5, eigenvalues, synthetic_uab, 1, nc_total=1)
        assert np.isfinite(result)

    def test_mle_returns_finite(self, synthetic_eigen, synthetic_uab):
        eigenvalues, _ = synthetic_eigen
        result = mle_log_likelihood(0.5, eigenvalues, synthetic_uab, 1, nc_total=2)
        assert np.isfinite(result)

    def test_mle_null_returns_finite(self, synthetic_eigen, synthetic_uab):
        eigenvalues, _ = synthetic_eigen
        result = mle_log_likelihood(0.5, eigenvalues, synthetic_uab, 1, nc_total=1)
        assert np.isfinite(result)


@pytest.mark.tier0
class TestBuildCovariateMatrixOverparameterized:
    """T3: Over-parameterization guard."""

    def test_raises_when_overparameterized(self):
        """n_samples <= n_cvt + 1 should raise ValueError."""
        covariates = np.ones((3, 3))  # 3 samples, 3 covariates
        with pytest.raises(ValueError, match="Over-parameterized"):
            _build_covariate_matrix(covariates, n_samples=3)

    def test_raises_exactly_at_boundary(self):
        """n_samples == n_cvt + 1 leaves df=0."""
        covariates = np.column_stack([np.ones(2), np.array([0.5, 1.5])])
        with pytest.raises(ValueError, match="Over-parameterized"):
            _build_covariate_matrix(covariates, n_samples=2)

    def test_valid_when_sufficient_samples(self):
        """n_samples > n_cvt + 1 should succeed."""
        covariates = np.column_stack(
            [np.ones(10), np.random.default_rng(0).standard_normal(10)]
        )
        W, n_cvt = _build_covariate_matrix(covariates, n_samples=10)
        assert n_cvt == 2
        assert W.shape == (10, 2)


@pytest.mark.tier0
class TestGoldenSectionMinimize:
    """T4: Golden section optimizer standalone behavior."""

    def test_finds_minimum_of_quadratic(self):
        """Optimizer finds minimum of (lambda - 1)^2."""
        opt_lambda, opt_val = _golden_section_minimize(
            lambda x: (x - 1.0) ** 2, l_min=1e-5, l_max=1e5
        )
        assert abs(opt_lambda - 1.0) < 0.1
        # opt_val is -func(opt_lambda), so near 0
        assert opt_val > -0.01

    def test_minimum_at_lower_boundary(self):
        """Optimizer handles minimum at l_min."""
        opt_lambda, _ = _golden_section_minimize(lambda x: x, l_min=1e-5, l_max=1e5)
        assert opt_lambda < 0.01

    def test_minimum_at_upper_boundary(self):
        """Optimizer handles minimum at l_max."""
        opt_lambda, _ = _golden_section_minimize(lambda x: -x, l_min=1e-5, l_max=1e5)
        assert opt_lambda > 1e4

    def test_returns_negative_of_function_value(self):
        """Return convention: (lambda, -func(lambda))."""

        def f(x):
            return (x - 10.0) ** 2 + 5.0

        opt_lambda, opt_val = _golden_section_minimize(f, l_min=1e-1, l_max=1e3)
        assert opt_val == pytest.approx(-f(opt_lambda), abs=1e-6)

    def test_convergence_with_more_iterations(self):
        """More iterations -> tighter convergence."""

        def f(x):
            return (x - 1.0) ** 2

        lam_5, _ = _golden_section_minimize(f, n_iter=5)
        lam_50, _ = _golden_section_minimize(f, n_iter=50)
        # 50 iterations should get closer to 1.0 than 5
        assert abs(lam_50 - 1.0) < abs(lam_5 - 1.0)
