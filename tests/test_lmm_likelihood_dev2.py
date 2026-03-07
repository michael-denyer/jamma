"""Tests for REML second derivative functions (CalcPPab, CalcPPPab, LogRL_dev2).

Validates calc_ppab, calc_pppab, and reml_log_likelihood_dev2 against
GEMMA's exact algorithm. These functions are the mathematical core for
computing se(pve) via the delta method.

GEMMA Reference (mouse_hs1940.log.txt):
- pve = 0.609795, se(pve) = 0.032753 (intercept-only, n_cvt=1)
"""

from pathlib import Path

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    calc_pab,
    calc_ppab,
    calc_pppab,
    compute_null_model_lambda,
    compute_Uab,
    finite_difference_dev2,
    reml_log_likelihood_dev2,
    reml_log_likelihood_null,
)
from jamma.lmm.prepare_common import compute_and_log_pve
from tests.conftest import load_phenotypes_from_fam

# GEMMA reference values
GEMMA_SE_PVE = 0.032753  # from mouse_hs1940.log.txt (intercept-only)

# Fixture paths
MOUSE_FIXTURE = Path(__file__).parent / "fixtures" / "lmm"
SYNTHETIC_FIXTURE = Path(__file__).parent / "fixtures" / "gemma_synthetic"


@pytest.fixture
def synthetic_null_model():
    """Load synthetic test data and compute null model quantities."""
    if not (SYNTHETIC_FIXTURE / "test.bed").exists():
        pytest.skip("GEMMA synthetic fixture not available")

    plink = load_plink_binary(SYNTHETIC_FIXTURE / "test")
    kinship = read_kinship_matrix(
        SYNTHETIC_FIXTURE / "gemma_kinship.cXX.txt", n_samples=plink.n_samples
    )
    phenotypes = load_phenotypes_from_fam(SYNTHETIC_FIXTURE / "test.fam")
    eigenvalues, U = eigendecompose_kinship(kinship)

    W = np.ones((plink.n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ phenotypes
    n_cvt = 1

    lambda_remle, _logl = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    return {
        "eigenvalues": eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "Uab": Uab,
        "n_cvt": n_cvt,
        "lambda_remle": lambda_remle,
    }


@pytest.fixture
def mouse_null_model():
    """Load mouse_hs1940 data and compute null model quantities."""
    if not (MOUSE_FIXTURE / "mouse_hs1940.kinship.cXX.txt").exists():
        pytest.skip("Mouse HS1940 fixture not available")

    plink = load_plink_binary(MOUSE_FIXTURE / "mouse_hs1940")
    kinship = read_kinship_matrix(
        MOUSE_FIXTURE / "mouse_hs1940.kinship.cXX.txt", n_samples=plink.n_samples
    )
    phenotypes = load_phenotypes_from_fam(MOUSE_FIXTURE / "mouse_hs1940.fam")
    eigenvalues, U = eigendecompose_kinship(kinship)

    W = np.ones((plink.n_samples, 1))
    UtW = U.T @ W
    Uty = U.T @ phenotypes
    n_cvt = 1

    lambda_remle, _logl = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    return {
        "eigenvalues": eigenvalues,
        "UtW": UtW,
        "Uty": Uty,
        "Uab": Uab,
        "n_cvt": n_cvt,
        "lambda_remle": lambda_remle,
        "n_samples": plink.n_samples,
    }


# --- calc_ppab tests ---


class TestCalcPPab:
    """Tests for calc_ppab (second-order projected Pab)."""

    def test_row0_equals_hihieval_dot_uab(self, synthetic_null_model):
        """Row 0 of PPab is HiHi_eval @ Uab (base case, no recursion)."""
        data = synthetic_null_model
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = data["n_cvt"]
        lam = data["lambda_remle"]

        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval

        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        PPab = calc_ppab(n_cvt, HiHi_eval, Uab, Pab)

        # Row 0 should be direct weighted dot products
        expected_row0 = HiHi_eval @ Uab
        np.testing.assert_allclose(PPab[0, :], expected_row0, rtol=1e-12)

    def test_shape_ncvt1(self, synthetic_null_model):
        """For n_cvt=1: PPab shape is (3, 6)."""
        data = synthetic_null_model
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        lam = data["lambda_remle"]

        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval

        Pab = calc_pab(1, Hi_eval, Uab)
        PPab = calc_ppab(1, HiHi_eval, Uab, Pab)

        assert PPab.shape == (3, 6)

    def test_shape_ncvt2(self):
        """For n_cvt=2: PPab shape is (4, 10)."""
        n = 50
        n_cvt = 2
        rng = np.random.default_rng(42)
        eigenvalues = rng.uniform(0.1, 2.0, size=n)
        lam = 1.5

        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval

        # Build Uab for n_cvt=2 with 2 covariates
        UtW = rng.standard_normal((n, 2))
        Uty = rng.standard_normal(n)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        PPab = calc_ppab(n_cvt, HiHi_eval, Uab, Pab)

        assert PPab.shape == (4, 10)

    def test_degenerate_psww_zero(self):
        """When Pab[p-1, ww] == 0, recursion uses ps2_ab unchanged."""
        n = 10
        n_cvt = 1
        # Create degenerate data where covariate is zero -> Pab[0, ww] = 0
        UtW = np.zeros((n, 1))
        Uty = np.ones(n)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        eigenvalues = np.ones(n)
        lam = 1.0
        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval

        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        PPab = calc_ppab(n_cvt, HiHi_eval, Uab, Pab)

        # Should not raise; PPab row 1 falls back to ps2_ab
        assert PPab.shape == (3, 6)
        assert np.all(np.isfinite(PPab))


# --- calc_pppab tests ---


class TestCalcPPPab:
    """Tests for calc_pppab (third-order projected Pab)."""

    def test_row0_equals_hihihieval_dot_uab(self, synthetic_null_model):
        """Row 0 of PPPab is HiHiHi_eval @ Uab (base case)."""
        data = synthetic_null_model
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = data["n_cvt"]
        lam = data["lambda_remle"]

        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval
        HiHiHi_eval = HiHi_eval * Hi_eval

        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        PPab = calc_ppab(n_cvt, HiHi_eval, Uab, Pab)
        PPPab = calc_pppab(n_cvt, HiHiHi_eval, Uab, Pab, PPab)

        expected_row0 = HiHiHi_eval @ Uab
        np.testing.assert_allclose(PPPab[0, :], expected_row0, rtol=1e-12)

    def test_shape_ncvt1(self, synthetic_null_model):
        """For n_cvt=1: PPPab shape is (3, 6)."""
        data = synthetic_null_model
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        lam = data["lambda_remle"]

        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval
        HiHiHi_eval = HiHi_eval * Hi_eval

        Pab = calc_pab(1, Hi_eval, Uab)
        PPab = calc_ppab(1, HiHi_eval, Uab, Pab)
        PPPab = calc_pppab(1, HiHiHi_eval, Uab, Pab, PPab)

        assert PPPab.shape == (3, 6)

    def test_accepts_pab_and_ppab(self, synthetic_null_model):
        """calc_pppab requires both Pab and PPab as inputs (dependency chain)."""
        data = synthetic_null_model
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = data["n_cvt"]
        lam = data["lambda_remle"]

        Hi_eval = 1.0 / (lam * eigenvalues + 1.0)
        HiHi_eval = Hi_eval * Hi_eval
        HiHiHi_eval = HiHi_eval * Hi_eval

        Pab = calc_pab(n_cvt, Hi_eval, Uab)
        PPab = calc_ppab(n_cvt, HiHi_eval, Uab, Pab)
        # This should work — PPPab depends on both Pab and PPab
        PPPab = calc_pppab(n_cvt, HiHiHi_eval, Uab, Pab, PPab)

        assert PPPab is not None
        assert np.all(np.isfinite(PPPab))


# --- reml_log_likelihood_dev2 tests ---


class TestRemlLogLikelihoodDev2:
    """Tests for reml_log_likelihood_dev2 (REML second derivative)."""

    def test_returns_negative_at_optimum(self, synthetic_null_model):
        """Second derivative at REML optimum should be negative (maximum)."""
        data = synthetic_null_model
        dev2 = reml_log_likelihood_dev2(
            data["lambda_remle"], data["eigenvalues"], data["Uab"], data["n_cvt"]
        )
        assert dev2 < 0, f"Expected negative dev2 at REML optimum, got {dev2}"

    def test_se_pve_matches_gemma_mouse(self, mouse_null_model):
        """se(pve) derived from dev2 matches GEMMA's 0.032753 within rtol=1e-3."""
        data = mouse_null_model
        lam = data["lambda_remle"]
        eigenvalues = data["eigenvalues"]
        n = data["n_samples"]

        dev2 = reml_log_likelihood_dev2(lam, eigenvalues, data["Uab"], data["n_cvt"])

        # dev2 should be negative at maximum
        assert dev2 < 0, f"Expected negative dev2, got {dev2}"

        # Delta method: se(lambda) then se(pve)
        se_lambda = np.sqrt(-1.0 / dev2)
        trace_G = np.sum(eigenvalues) / n
        pve_se = trace_G / (trace_G * lam + 1.0) ** 2 * se_lambda

        np.testing.assert_allclose(
            pve_se,
            GEMMA_SE_PVE,
            rtol=1e-3,
            err_msg=f"se(pve)={pve_se:.6f} vs GEMMA={GEMMA_SE_PVE}",
        )

    def test_ncvt2_delegates_to_finite_diff(self):
        """For n_cvt>1, reml_log_likelihood_dev2 delegates to finite_difference_dev2."""
        n = 200
        n_cvt = 4
        rng = np.random.default_rng(200)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = np.sqrt(eigenvalues) * rng.standard_normal(n) + rng.standard_normal(n)

        lam, _ = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        dev2_wrapper = reml_log_likelihood_dev2(lam, eigenvalues, Uab, n_cvt)
        dev2_direct = finite_difference_dev2(lam, eigenvalues, Uab, n_cvt)

        assert dev2_wrapper == dev2_direct, (
            f"n_cvt>1 should delegate: wrapper={dev2_wrapper}, direct={dev2_direct}"
        )

    def test_dev2_negative_for_synthetic(self, synthetic_null_model):
        """dev2 negative for synthetic data too (confirming general behavior)."""
        data = synthetic_null_model
        dev2 = reml_log_likelihood_dev2(
            data["lambda_remle"], data["eigenvalues"], data["Uab"], data["n_cvt"]
        )
        assert dev2 < 0
        # se(pve) should be a reasonable positive value
        se_lambda = np.sqrt(-1.0 / dev2)
        assert se_lambda > 0
        assert np.isfinite(se_lambda)

    def test_dev2_matches_finite_differences(self, synthetic_null_model):
        """Analytical dev2 matches numerical second derivative of log-likelihood."""
        data = synthetic_null_model
        lam = data["lambda_remle"]
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = data["n_cvt"]

        dev2_analytical = reml_log_likelihood_dev2(lam, eigenvalues, Uab, n_cvt)

        # Central finite differences with h ~ O(eps^{1/4}) * lambda
        # for optimal second-derivative accuracy
        h = lam * 1e-3
        f_plus = reml_log_likelihood_null(lam + h, eigenvalues, Uab, n_cvt)
        f_center = reml_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)
        f_minus = reml_log_likelihood_null(lam - h, eigenvalues, Uab, n_cvt)
        dev2_numerical = (f_plus - 2.0 * f_center + f_minus) / (h * h)

        np.testing.assert_allclose(
            dev2_analytical,
            dev2_numerical,
            rtol=1e-4,
            err_msg=(
                f"Analytical dev2={dev2_analytical:.8e} vs "
                f"numerical={dev2_numerical:.8e}"
            ),
        )

    def test_dev2_matches_finite_differences_mouse(self, mouse_null_model):
        """Finite-difference validation on mouse_hs1940 real data."""
        data = mouse_null_model
        lam = data["lambda_remle"]
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = data["n_cvt"]

        dev2_analytical = reml_log_likelihood_dev2(lam, eigenvalues, Uab, n_cvt)

        h = lam * 1e-3
        f_plus = reml_log_likelihood_null(lam + h, eigenvalues, Uab, n_cvt)
        f_center = reml_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)
        f_minus = reml_log_likelihood_null(lam - h, eigenvalues, Uab, n_cvt)
        dev2_numerical = (f_plus - 2.0 * f_center + f_minus) / (h * h)

        np.testing.assert_allclose(
            dev2_analytical,
            dev2_numerical,
            rtol=1e-4,
            err_msg=(
                f"Analytical dev2={dev2_analytical:.8e} vs "
                f"numerical={dev2_numerical:.8e}"
            ),
        )

    def test_dev2_nan_for_degenerate_pyy(self):
        """dev2 returns NaN when P_yy is degenerate (phenotype = covariates)."""
        n = 50
        n_cvt = 1
        rng = np.random.default_rng(77)
        eigenvalues = rng.uniform(0.1, 2.0, size=n)
        lam = 1.0

        # Construct data where y is exactly in the covariate span
        # so P_yy ≈ 0 after projection
        UtW = rng.standard_normal((n, 1))
        # y = W @ beta exactly — after projection, residual is zero
        Uty = UtW[:, 0] * 3.0
        Uab = compute_Uab(UtW, Uty, Utx=None)

        dev2 = reml_log_likelihood_dev2(lam, eigenvalues, Uab, n_cvt)
        assert np.isnan(dev2), f"Expected NaN for degenerate P_yy, got {dev2}"

    def test_dev2_nan_for_nonpositive_lambda(self):
        """dev2 returns NaN for lambda <= 0."""
        n = 50
        n_cvt = 1
        rng = np.random.default_rng(42)
        eigenvalues = rng.uniform(0.1, 2.0, size=n)
        UtW = rng.standard_normal((n, 1))
        Uty = rng.standard_normal(n)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        assert np.isnan(reml_log_likelihood_dev2(0.0, eigenvalues, Uab, n_cvt))
        assert np.isnan(reml_log_likelihood_dev2(-1.0, eigenvalues, Uab, n_cvt))

    def test_dev2_finite_at_lambda_boundary(self, synthetic_null_model):
        """dev2 is finite at lambda=1e-5 (optimizer lower bound)."""
        data = synthetic_null_model
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = data["n_cvt"]

        dev2 = reml_log_likelihood_dev2(1e-5, eigenvalues, Uab, n_cvt)
        assert np.isfinite(dev2), f"dev2 should be finite at lambda=1e-5, got {dev2}"

        # se_lambda should also be computable (dev2 should be negative)
        if dev2 < 0:
            se_lambda = np.sqrt(-1.0 / dev2)
            assert np.isfinite(se_lambda)


# --- finite_difference_dev2 tests ---


class TestFiniteDifferenceDev2:
    """Tests for finite_difference_dev2 (numerical second derivative fallback)."""

    @pytest.mark.parametrize("n_cvt", [2, 3, 4])
    def test_matches_direct_stencil(self, n_cvt):
        """finite_difference_dev2 agrees with a direct central stencil."""
        n = 200
        rng = np.random.default_rng(200)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = np.sqrt(eigenvalues) * rng.standard_normal(n) + rng.standard_normal(n)

        lam, _ = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        dev2 = finite_difference_dev2(lam, eigenvalues, Uab, n_cvt)

        # Independent verification with a different step size
        h = lam * 1e-3
        f_plus = reml_log_likelihood_null(lam + h, eigenvalues, Uab, n_cvt)
        f_center = reml_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)
        f_minus = reml_log_likelihood_null(lam - h, eigenvalues, Uab, n_cvt)
        dev2_check = (f_plus - 2.0 * f_center + f_minus) / (h * h)

        if dev2_check != 0:
            np.testing.assert_allclose(dev2, dev2_check, rtol=1e-2)

    def test_negative_at_optimum_ncvt4(self):
        """dev2 is negative at REML optimum for n_cvt=4."""
        n = 200
        n_cvt = 4
        rng = np.random.default_rng(200)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = np.sqrt(eigenvalues) * rng.standard_normal(n) + rng.standard_normal(n)

        lam, _ = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        dev2 = finite_difference_dev2(lam, eigenvalues, Uab, n_cvt)
        assert dev2 < 0, f"Expected negative dev2 at optimum, got {dev2}"

    def test_near_lower_bound(self):
        """Uses forward stencil when lambda is near l_min."""
        n = 100
        n_cvt = 2
        rng = np.random.default_rng(42)
        eigenvalues = rng.uniform(0.1, 2.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = rng.standard_normal(n)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        # Force lambda near lower bound
        dev2 = finite_difference_dev2(1e-5, eigenvalues, Uab, n_cvt, l_min=1e-5)
        assert np.isfinite(dev2), f"dev2 should be finite near l_min, got {dev2}"


# --- compute_and_log_pve multi-covariate tests ---


class TestComputeAndLogPveMultiCvt:
    """Tests that compute_and_log_pve produces correct pve_se for n_cvt > 1."""

    @pytest.mark.parametrize("n_cvt", [2, 3, 4])
    def test_pve_se_finite_for_multi_cvt(self, n_cvt):
        """pve_se is finite and positive for multi-covariate models."""
        n = 200
        rng = np.random.default_rng(200)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = np.sqrt(eigenvalues) * rng.standard_normal(n) + rng.standard_normal(n)

        pve, pve_se = compute_and_log_pve(eigenvalues, UtW, Uty, n_cvt)

        assert 0 < pve < 1, f"pve={pve} out of range"
        if pve_se is not None:
            assert pve_se > 0, f"pve_se={pve_se} should be positive"
            assert np.isfinite(pve_se), f"pve_se={pve_se} should be finite"

    @pytest.mark.parametrize("n_cvt", [1, 2, 4])
    def test_pve_se_matches_direct_finite_diff(self, n_cvt):
        """compute_and_log_pve pve_se matches manual finite-difference computation."""
        n = 200
        rng = np.random.default_rng(200)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = np.sqrt(eigenvalues) * rng.standard_normal(n) + rng.standard_normal(n)

        _, pve_se = compute_and_log_pve(eigenvalues, UtW, Uty, n_cvt)

        # Independent verification via direct stencil with different step size
        lam, _ = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
        Uab = compute_Uab(UtW, Uty, Utx=None)
        h = lam * 1e-3
        fp = reml_log_likelihood_null(lam + h, eigenvalues, Uab, n_cvt)
        fc = reml_log_likelihood_null(lam, eigenvalues, Uab, n_cvt)
        fm = reml_log_likelihood_null(lam - h, eigenvalues, Uab, n_cvt)
        dev2_check = (fp - 2.0 * fc + fm) / (h * h)

        if dev2_check < 0 and pve_se is not None:
            se_lam = np.sqrt(-1.0 / dev2_check)
            trace_G = np.sum(eigenvalues) / n
            denom = trace_G * lam + 1.0
            pve_se_check = trace_G / (denom * denom) * se_lam

            np.testing.assert_allclose(pve_se, pve_se_check, rtol=1e-2)
