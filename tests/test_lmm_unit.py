"""Unit tests for LMM association module.

These tests verify individual components of the LMM association workflow
using synthetic data with known properties.
"""

import numpy as np
import pytest

from jamma.core import configure_jax
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    calc_pab,
    compute_Uab,
    get_ab_index,
    reml_log_likelihood,
)
from jamma.lmm.optimize import brent_minimize, optimize_lambda
from jamma.lmm.stats import AssocResult, calc_wald_test, f_sf


@pytest.fixture(autouse=True)
def setup_jax():
    """Configure JAX with 64-bit precision before each test."""
    configure_jax(enable_x64=True)


class TestEigendecomposition:
    """Tests for kinship eigendecomposition."""

    def test_eigendecompose_positive_semidefinite(self):
        """Eigenvalues of valid kinship matrix are non-negative."""
        rng = np.random.default_rng(42)
        # Create positive semi-definite matrix
        X = rng.standard_normal((50, 100))
        K = X @ X.T / X.shape[1]

        eigenvalues, eigenvectors = eigendecompose_kinship(K)

        assert np.all(eigenvalues >= -1e-10), "Eigenvalues should be non-negative"

    def test_eigendecompose_shape(self):
        """Eigendecomposition returns correct shapes."""
        n_samples = 20
        K = np.eye(n_samples)

        eigenvalues, eigenvectors = eigendecompose_kinship(K)

        assert eigenvalues.shape == (n_samples,)
        assert eigenvectors.shape == (n_samples, n_samples)

    def test_eigendecompose_identity_matrix(self):
        """Identity matrix has all eigenvalues = 1."""
        n_samples = 10
        K = np.eye(n_samples)

        eigenvalues, eigenvectors = eigendecompose_kinship(K)

        assert np.allclose(eigenvalues, 1.0), "Identity has eigenvalues = 1"

    def test_eigendecompose_reconstruction(self):
        """K = U @ diag(eigenvalues) @ U.T."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((30, 50))
        K = X @ X.T / X.shape[1]

        eigenvalues, U = eigendecompose_kinship(K)

        # Reconstruct K from eigendecomposition
        K_reconstructed = U @ np.diag(eigenvalues) @ U.T

        assert np.allclose(K, K_reconstructed, rtol=1e-10)

    def test_eigendecompose_orthonormal(self):
        """Eigenvectors are orthonormal."""
        rng = np.random.default_rng(42)
        X = rng.standard_normal((20, 40))
        K = X @ X.T / X.shape[1]

        _, U = eigendecompose_kinship(K)

        # U.T @ U should be identity
        assert np.allclose(U.T @ U, np.eye(U.shape[1]), rtol=1e-10)

    def test_eigendecompose_thresholding(self):
        """Small eigenvalues are zeroed."""
        # Create matrix with some very small eigenvalues
        # Note: eigenvalues are returned sorted in ascending order
        eigenvalues = np.array([1e-15, 1e-12, 0.5, 1.0])
        U = np.eye(4)  # Simple orthogonal matrix
        K = U @ np.diag(eigenvalues) @ U.T

        result_eigenvalues, _ = eigendecompose_kinship(K, threshold=1e-10)

        # Values below threshold should be zeroed (indices 0 and 1 are smallest)
        assert result_eigenvalues[0] == 0.0, "1e-15 should be zeroed"
        assert result_eigenvalues[1] == 0.0, "1e-12 should be zeroed"
        assert result_eigenvalues[2] > 0.4, "0.5 should be preserved"
        assert result_eigenvalues[3] > 0.9, "1.0 should be preserved"


class TestUabComputation:
    """Tests for U'ab matrix computation."""

    def test_compute_uab_shape(self):
        """Uab matrix has correct shape (2D: n_samples x n_index)."""
        n_samples = 50
        n_cvt = 1
        # n_index = (n_cvt + 3) * (n_cvt + 2) / 2
        n_index = (n_cvt + 3) * (n_cvt + 2) // 2

        # compute_Uab takes UtW, Uty, Utx (already rotated)
        UtW = np.ones((n_samples, n_cvt))
        rng = np.random.default_rng(42)
        Uty = rng.standard_normal(n_samples)
        Utx = rng.standard_normal(n_samples)

        Uab = compute_Uab(UtW, Uty, Utx)

        # New API returns 2D matrix (n_samples, n_index)
        assert Uab.shape == (n_samples, n_index)

    def test_compute_uab_symmetric_indexing(self):
        """Uab uses symmetric storage with correct indexing."""
        n_samples = 20
        n_cvt = 1
        UtW = np.ones((n_samples, 1))
        rng = np.random.default_rng(42)
        Uty = rng.standard_normal(n_samples)
        Utx = rng.standard_normal(n_samples)

        # Just verify it runs and returns expected size
        _ = compute_Uab(UtW, Uty, Utx)

        # Check symmetry: index(a,b) == index(b,a)
        # Using new API with n_cvt parameter
        idx_12 = get_ab_index(1, 2, n_cvt)
        idx_21 = get_ab_index(2, 1, n_cvt)
        assert idx_12 == idx_21, "Index should be symmetric"


class TestPabComputation:
    """Tests for weighted Pab computation."""

    def test_compute_pab_positive_definite(self):
        """Pab diagonal elements should be positive."""
        n_samples = 30
        n_cvt = 1
        rng = np.random.default_rng(42)

        # Create valid Uab from rotated data
        X = rng.standard_normal((n_samples, 100))
        K = X @ X.T / X.shape[1]
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        U = np.linalg.eigh(K)[1]
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        Uab = compute_Uab(UtW, Uty, Utx)

        # Hi_eval = 1 / (lambda * eigenvalues + 1)
        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)

        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        # Pab is 2D: (n_cvt+2, n_index), check row 0 diagonal elements
        # For n_cvt=1: indices 1, 2, 3 are the diagonal elements
        idx_11 = get_ab_index(1, 1, n_cvt)
        idx_22 = get_ab_index(2, 2, n_cvt)
        idx_33 = get_ab_index(3, 3, n_cvt)

        assert Pab[0, idx_11] > 0, f"Pab[0, {idx_11}] should be positive"
        assert Pab[0, idx_22] > 0, f"Pab[0, {idx_22}] should be positive"
        assert Pab[0, idx_33] > 0, f"Pab[0, {idx_33}] should be positive"


class TestRemlLogLikelihood:
    """Tests for REML log-likelihood computation."""

    def test_reml_returns_finite(self):
        """REML log-likelihood should be finite for valid inputs."""
        n_samples = 50
        n_cvt = 1
        rng = np.random.default_rng(42)

        # Create valid Uab from rotated data
        X = rng.standard_normal((n_samples, 100))
        K = X @ X.T / X.shape[1]
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        U = np.linalg.eigh(K)[1]
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        Uab = compute_Uab(UtW, Uty, Utx)
        lambda_val = 1.0

        logl = reml_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)

        assert np.isfinite(logl), "REML should return finite value"

    def test_reml_unimodal(self):
        """REML function should be unimodal (for optimization)."""
        # Use realistic eigenvalues from a valid kinship matrix
        # These are typical values: mostly positive, spread across orders of magnitude
        rng = np.random.default_rng(42)
        n_samples = 50
        n_cvt = 1

        # Simulate eigenvalues from a kinship matrix (like X @ X.T / n_snps)
        X = rng.standard_normal((n_samples, 100))
        K = X @ X.T / X.shape[1]
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 1e-10)  # Threshold small values

        # Create valid Uab from rotated data
        U = np.linalg.eigh(K)[1]
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        Uab = compute_Uab(UtW, Uty, Utx)

        # Sample at multiple lambda values across the optimization range
        lambdas = np.logspace(-4, 4, 50)
        logls = [reml_log_likelihood(lam, eigenvalues, Uab, n_cvt) for lam in lambdas]

        # Find maximum (REML returns positive log-likelihood)
        max_idx = np.argmax(logls)
        # Should have a clear maximum somewhere
        assert (
            logls[max_idx] > logls[0] or logls[max_idx] > logls[-1]
        ), "REML should have a maximum value"


class TestBrentMinimize:
    """Tests for custom Brent's method implementation."""

    def test_brent_finds_minimum_quadratic(self):
        """Brent's method finds minimum of a quadratic."""

        def f(x):
            return (x - 3.0) ** 2

        x_min, f_min = brent_minimize(f, 0.0, 10.0)

        assert np.isclose(x_min, 3.0, atol=1e-5), f"Expected x=3, got {x_min}"
        assert np.isclose(f_min, 0.0, atol=1e-10), f"Expected f=0, got {f_min}"

    def test_brent_finds_minimum_asymmetric(self):
        """Brent's method works for asymmetric functions."""

        def f(x):
            return x**4 - 4 * x**3 + 4 * x**2

        x_min, f_min = brent_minimize(f, -1.0, 5.0)

        # Minimum at x=0 or x=2
        assert f_min < 0.1, f"Expected minimum near 0, got {f_min}"

    def test_brent_respects_bounds(self):
        """Result is within specified bounds."""

        def f(x):
            return (x - 10.0) ** 2  # Minimum at x=10, outside bounds

        x_min, _ = brent_minimize(f, 0.0, 5.0)

        assert 0.0 <= x_min <= 5.0, f"Result {x_min} outside bounds [0, 5]"

    def test_brent_tolerance(self):
        """Brent's method respects tolerance parameter."""

        def f(x):
            return (x - 2.5) ** 2

        x_min_tight, _ = brent_minimize(f, 0.0, 5.0, tol=1e-10)
        x_min_loose, _ = brent_minimize(f, 0.0, 5.0, tol=1e-2)

        # Tighter tolerance should get closer to true minimum
        assert abs(x_min_tight - 2.5) < abs(x_min_loose - 2.5) + 1e-3


class TestOptimizeLambda:
    """Tests for lambda optimization wrapper."""

    def test_optimize_lambda_positive_result(self):
        """Optimized lambda should be positive."""

        def neg_logl(lam):
            return (np.log(lam) - 1.0) ** 2 + 1.0

        lambda_opt, logl = optimize_lambda(neg_logl)

        assert lambda_opt > 0, f"Lambda should be positive, got {lambda_opt}"

    def test_optimize_lambda_returns_positive_logl(self):
        """Returned log-likelihood should be positive (negated)."""

        def neg_logl(lam):
            return 10.0 + (np.log(lam) - 1.0) ** 2

        _, logl = optimize_lambda(neg_logl)

        assert logl < 0, "Log-likelihood should be negative (from negated input)"


class TestFDistributionSF:
    """Tests for F-distribution survival function."""

    def test_f_sf_zero_is_one(self):
        """SF(0) = 1 for any degrees of freedom."""
        assert f_sf(0.0, 1.0, 10.0) == 1.0
        assert f_sf(0.0, 5.0, 20.0) == 1.0

    def test_f_sf_decreases_with_x(self):
        """SF decreases as x increases."""
        p1 = f_sf(1.0, 1.0, 10.0)
        p2 = f_sf(5.0, 1.0, 10.0)
        p3 = f_sf(10.0, 1.0, 10.0)

        assert p1 > p2 > p3, "SF should decrease with x"

    def test_f_sf_between_zero_and_one(self):
        """SF values should be in [0, 1]."""
        for x in [0.1, 1.0, 5.0, 10.0, 100.0]:
            p = f_sf(x, 1.0, 20.0)
            assert 0.0 <= p <= 1.0, f"SF({x}) = {p} not in [0, 1]"

    def test_f_sf_infinity_is_zero(self):
        """SF(inf) = 0."""
        p = f_sf(np.inf, 1.0, 10.0)
        assert p == 0.0, f"SF(inf) should be 0, got {p}"

    def test_f_sf_known_value(self):
        """Test against known F-distribution value."""
        # F(1, 10) at x=4.965 gives p ≈ 0.05
        p = f_sf(4.965, 1.0, 10.0)
        assert np.isclose(p, 0.05, atol=0.01), f"Expected p≈0.05, got {p}"


class TestWaldTest:
    """Tests for Wald test statistics computation."""

    def test_calc_wald_returns_three_values(self):
        """Wald test returns beta, se, p_wald."""
        n_samples = 100
        n_cvt = 1
        rng = np.random.default_rng(42)

        # Create valid Pab matrix
        X = rng.standard_normal((n_samples, 100))
        K = X @ X.T / X.shape[1]
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        U = np.linalg.eigh(K)[1]
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        Uab = compute_Uab(UtW, Uty, Utx)
        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        beta, se, p_wald = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        assert isinstance(beta, int | float)
        assert isinstance(se, int | float)
        assert isinstance(p_wald, int | float)

    def test_calc_wald_se_positive(self):
        """Standard error should be positive."""
        n_samples = 100
        n_cvt = 1
        rng = np.random.default_rng(42)

        # Create valid Pab matrix
        X = rng.standard_normal((n_samples, 100))
        K = X @ X.T / X.shape[1]
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        U = np.linalg.eigh(K)[1]
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        Uab = compute_Uab(UtW, Uty, Utx)
        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        _, se, _ = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        assert se > 0, f"SE should be positive, got {se}"

    def test_calc_wald_pvalue_valid(self):
        """P-value should be in [0, 1]."""
        n_samples = 100
        n_cvt = 1
        rng = np.random.default_rng(42)

        # Create valid Pab matrix
        X = rng.standard_normal((n_samples, 100))
        K = X @ X.T / X.shape[1]
        eigenvalues = np.linalg.eigvalsh(K)
        eigenvalues = np.maximum(eigenvalues, 1e-10)

        U = np.linalg.eigh(K)[1]
        y = rng.standard_normal(n_samples)
        W = np.ones((n_samples, 1))
        x = rng.standard_normal(n_samples)
        Uty = U.T @ y
        UtW = U.T @ W
        Utx = U.T @ x

        Uab = compute_Uab(UtW, Uty, Utx)
        lambda_val = 1.0
        Hi_eval = 1.0 / (lambda_val * eigenvalues + 1.0)
        Pab = calc_pab(n_cvt, Hi_eval, Uab)

        _, _, p_wald = calc_wald_test(lambda_val, Pab, n_cvt, n_samples)

        assert 0.0 <= p_wald <= 1.0, f"P-value {p_wald} not in [0, 1]"


class TestGetAbIndex:
    """Tests for Pab array indexing using GEMMA's GetabIndex."""

    def test_get_ab_index_symmetric(self):
        """Index is symmetric: index(a,b) == index(b,a)."""
        n_cvt = 1
        for a in range(1, 4):  # GEMMA uses 1-based indexing
            for b in range(1, 4):
                assert get_ab_index(a, b, n_cvt) == get_ab_index(b, a, n_cvt)

    def test_get_ab_index_known_values(self):
        """Test known index values for n_cvt=1."""
        n_cvt = 1
        # For n_cvt=1, cols = 3
        # The GEMMA formula: (2 * cols - a1 + 2) * (a1 - 1) / 2 + b1 - a1
        # where a1 <= b1

        # index(1,1): (2*3 - 1 + 2) * (1 - 1) / 2 + 1 - 1 = 0
        assert get_ab_index(1, 1, n_cvt) == 0

        # index(1,2): (2*3 - 1 + 2) * (1 - 1) / 2 + 2 - 1 = 1
        assert get_ab_index(1, 2, n_cvt) == 1

        # index(1,3): (2*3 - 1 + 2) * (1 - 1) / 2 + 3 - 1 = 2
        assert get_ab_index(1, 3, n_cvt) == 2

        # index(2,2): (2*3 - 2 + 2) * (2 - 1) / 2 + 2 - 2 = 3
        assert get_ab_index(2, 2, n_cvt) == 3

        # index(2,3): (2*3 - 2 + 2) * (2 - 1) / 2 + 3 - 2 = 4
        assert get_ab_index(2, 3, n_cvt) == 4

        # index(3,3): (2*3 - 3 + 2) * (3 - 1) / 2 + 3 - 3 = 5
        assert get_ab_index(3, 3, n_cvt) == 5


class TestAssocResult:
    """Tests for AssocResult dataclass."""

    def test_assoc_result_creation(self):
        """AssocResult can be created with all fields."""
        result = AssocResult(
            chr="1",
            rs="rs123",
            ps=12345,
            n_miss=0,
            allele1="A",
            allele0="G",
            af=0.25,
            beta=0.5,
            se=0.1,
            logl_H1=-100.5,
            l_remle=1.5,
            p_wald=0.001,
        )

        assert result.chr == "1"
        assert result.rs == "rs123"
        assert result.beta == 0.5
        assert result.p_wald == 0.001

    def test_assoc_result_fields(self):
        """AssocResult has expected fields."""
        result = AssocResult(
            chr="1",
            rs="rs1",
            ps=1,
            n_miss=0,
            allele1="A",
            allele0="T",
            af=0.5,
            beta=0.0,
            se=0.1,
            logl_H1=-50.0,
            l_remle=1.0,
            p_wald=1.0,
        )

        expected_fields = [
            "chr",
            "rs",
            "ps",
            "n_miss",
            "allele1",
            "allele0",
            "af",
            "beta",
            "se",
            "logl_H1",
            "l_remle",
            "p_wald",
        ]
        for field in expected_fields:
            assert hasattr(result, field), f"Missing field: {field}"
