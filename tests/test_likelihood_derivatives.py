"""Tests for REML second/third derivatives wrt lambda (the LogRL_dev2 family).

Covers ``calc_ppab`` (P''), ``calc_pppab`` (P'''), and
``reml_log_likelihood_dev2`` against GEMMA's exact algorithm. These are
the mathematical core for computing ``se(pve)`` via the delta method.

GEMMA function names use the suffix ``_dev2`` for the second derivative
of the log restricted likelihood; this file's symbols inherit that
naming, but the file itself is named for the *behavior* (derivatives)
rather than the GEMMA jargon.

GEMMA reference (mouse_hs1940_all.log.txt, intercept-only, n_cvt=1):
- 1410 of 1940 individuals analyzed, pve = 0.609672, se(pve) = 0.0327788

That log is the run whose kinship matrix is the one this file loads.
fixtures/lmm/mouse_hs1940.log.txt is a different run against a different
kinship matrix and reports se(pve) = 0.032753. The two are not
interchangeable.
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from jamma.io import load_plink_binary
from jamma.kinship.io import read_kinship_matrix
from jamma.lmm.eigen import eigendecompose_kinship
from jamma.lmm.likelihood import (
    calc_pab,
    compute_null_model_lambda,
    compute_Uab,
    finite_difference_dev2,
    reml_log_likelihood,
)
from jamma.lmm.prepare_common import compute_and_log_pve, compute_valid_mask
from tests.conftest import load_phenotypes_from_fam, require_fixture
from tests.reference.likelihood import (
    calc_ppab,
    calc_pppab,
    reml_log_likelihood_dev2,
)

pytestmark = pytest.mark.tier1

# GEMMA reference values, from mouse_hs1940_all.log.txt (intercept-only)
GEMMA_SE_PVE = 0.0327788
GEMMA_N_ANALYZED = 1410

# Fixture paths
MOUSE_FIXTURE = Path(__file__).parent / "fixtures" / "mouse_hs1940"
SYNTHETIC_FIXTURE = Path(__file__).parent / "fixtures" / "gemma_synthetic"


@pytest.fixture
def synthetic_null_model():
    """Load synthetic test data and compute null model quantities."""
    require_fixture(
        SYNTHETIC_FIXTURE / "test.bed",
        SYNTHETIC_FIXTURE / "test.fam",
        SYNTHETIC_FIXTURE / "gemma_kinship.cXX.txt",
    )

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
    require_fixture(
        MOUSE_FIXTURE / "mouse_hs1940.bed",
        MOUSE_FIXTURE / "mouse_hs1940.fam",
        MOUSE_FIXTURE / "mouse_hs1940_kinship.cXX.txt",
    )

    plink = load_plink_binary(MOUSE_FIXTURE / "mouse_hs1940")
    kinship = read_kinship_matrix(
        MOUSE_FIXTURE / "mouse_hs1940_kinship.cXX.txt", n_samples=plink.n_samples
    )
    phenotypes = load_phenotypes_from_fam(MOUSE_FIXTURE / "mouse_hs1940.fam")

    # 530 of the 1940 mouse_hs1940 phenotypes are missing. GEMMA drops those
    # samples before eigendecomposition; keeping them makes Uty, and then
    # every likelihood value derived from it, NaN.
    valid_mask = compute_valid_mask(phenotypes, None)
    n_samples = int(valid_mask.sum())
    assert n_samples == GEMMA_N_ANALYZED, (
        f"filtered to {n_samples} samples, GEMMA analyzed {GEMMA_N_ANALYZED}"
    )
    kinship = kinship[np.ix_(valid_mask, valid_mask)]
    phenotypes = phenotypes[valid_mask]

    eigenvalues, U = eigendecompose_kinship(kinship)

    W = np.ones((n_samples, 1))
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
        "n_samples": n_samples,
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
        """se(pve) derived from dev2 matches GEMMA's 0.0327788 within rtol=5e-4.

        The observed agreement is ~8.5e-5, consistent with the golden-section
        versus Brent lambda difference the delta method carries through. The
        tolerance is set tight enough to reject the se(pve) reported by the
        other, non-matching mouse_hs1940 log, which sits ~7.0e-4 away.
        """
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
            rtol=5e-4,
            err_msg=f"se(pve)={pve_se:.7f} vs GEMMA={GEMMA_SE_PVE}",
        )

    @pytest.mark.parametrize("n_cvt", [2, 3, 4])
    def test_analytical_dev2_vs_finite_difference_ncvt_general(self, n_cvt):
        """Analytical dev2 matches finite_difference_dev2 within rtol=1e-4 for n_cvt>1.

        Verifies that reml_log_likelihood_dev2 computes analytically (does NOT
        delegate to finite_difference_dev2) and produces results matching the
        finite-difference oracle.
        """
        n = 200
        rng = np.random.default_rng(42)
        eigenvalues = np.sort(rng.exponential(1.0, n))[::-1]
        UtW = rng.standard_normal((n, n_cvt))
        Uty = np.sqrt(eigenvalues) * rng.standard_normal(n) + rng.standard_normal(n)

        lam, _ = compute_null_model_lambda(eigenvalues, UtW, Uty, n_cvt)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        # Get the oracle value BEFORE patching
        dev2_oracle = finite_difference_dev2(lam, eigenvalues, Uab, n_cvt)

        # Now verify analytical path does NOT call finite_difference_dev2
        with patch(
            "jamma.lmm.likelihood.finite_difference_dev2",
            side_effect=AssertionError(
                "analytical path should not call finite_difference_dev2"
            ),
        ):
            dev2_analytical = reml_log_likelihood_dev2(lam, eigenvalues, Uab, n_cvt)

        np.testing.assert_allclose(
            dev2_analytical,
            dev2_oracle,
            rtol=1e-4,
            err_msg=(
                f"n_cvt={n_cvt}: analytical dev2={dev2_analytical:.8e} vs "
                f"oracle={dev2_oracle:.8e}"
            ),
        )

    def test_analytical_dev2_does_not_delegate_ncvt2(self):
        """Structural: dev2 does NOT delegate to finite_difference_dev2."""
        n = 100
        n_cvt = 2
        rng = np.random.default_rng(99)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = rng.standard_normal(n)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        with patch(
            "jamma.lmm.likelihood.finite_difference_dev2",
            side_effect=AssertionError("should not be called"),
        ):
            dev2 = reml_log_likelihood_dev2(0.5, eigenvalues, Uab, n_cvt)

        assert np.isfinite(dev2), f"Expected finite dev2, got {dev2}"

    def test_analytical_dev2_ncvt2_lambda_near_bounds(self):
        """n_cvt=2 with lambda near lower bound (1e-5) produces finite result."""
        n = 100
        n_cvt = 2
        rng = np.random.default_rng(77)
        eigenvalues = rng.uniform(0.5, 3.0, size=n)
        UtW = rng.standard_normal((n, n_cvt))
        Uty = rng.standard_normal(n)
        Uab = compute_Uab(UtW, Uty, Utx=None)

        dev2 = reml_log_likelihood_dev2(1e-5, eigenvalues, Uab, n_cvt)
        assert np.isfinite(dev2), f"dev2 should be finite at lambda=1e-5, got {dev2}"

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
        f_plus = reml_log_likelihood(lam + h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        f_center = reml_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        f_minus = reml_log_likelihood(lam - h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
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
        f_plus = reml_log_likelihood(lam + h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        f_center = reml_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        f_minus = reml_log_likelihood(lam - h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        dev2_numerical = (f_plus - 2.0 * f_center + f_minus) / (h * h)

        # assert_allclose treats NaN as equal to NaN, so without this the whole
        # test passes vacuously when the fixture feeds in unfiltered data.
        assert np.isfinite(dev2_analytical), f"analytical dev2={dev2_analytical}"
        assert np.isfinite(dev2_numerical), f"numerical dev2={dev2_numerical}"

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

    def test_analytical_dev2_vs_finite_difference_dev2_ncvt1(
        self, synthetic_null_model
    ):
        """Analytical dev2 (n_cvt=1) matches finite_difference_dev2 within rtol=5e-3.

        The analytical path (reml_log_likelihood_dev2 with n_cvt=1) omits the
        d^2(logdet_hiw)/dlambda^2 term (see likelihood.py near the n_cvt=1
        branch).  finite_difference_dev2 approximates the full second derivative
        numerically via a central stencil (h=lam*1e-4) on the null REML log-likelihood,
        implicitly capturing all terms.  This test pins the approximation error
        upper bound at 0.5%.

        This is distinct from test_dev2_matches_finite_differences (above),
        which uses a hand-rolled stencil with h=lam*1e-3.  The present test
        explicitly calls finite_difference_dev2 to compare the two code paths
        and document their agreement.
        """
        data = synthetic_null_model
        lam = data["lambda_remle"]
        eigenvalues = data["eigenvalues"]
        Uab = data["Uab"]
        n_cvt = 1  # analytical path; n_cvt=1 uses closed-form formula

        dev2_analytical = reml_log_likelihood_dev2(lam, eigenvalues, Uab, n_cvt)
        dev2_finite_diff = finite_difference_dev2(lam, eigenvalues, Uab, n_cvt)

        np.testing.assert_allclose(
            dev2_analytical,
            dev2_finite_diff,
            rtol=5e-3,
            err_msg=(
                f"Analytical dev2={dev2_analytical:.8e} vs "
                f"finite_difference_dev2={dev2_finite_diff:.8e}: "
                f"approximation error exceeds 0.5% bound"
            ),
        )


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
        f_plus = reml_log_likelihood(lam + h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        f_center = reml_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        f_minus = reml_log_likelihood(lam - h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
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
        fp = reml_log_likelihood(lam + h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        fc = reml_log_likelihood(lam, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        fm = reml_log_likelihood(lam - h, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
        dev2_check = (fp - 2.0 * fc + fm) / (h * h)

        if dev2_check < 0 and pve_se is not None:
            se_lam = np.sqrt(-1.0 / dev2_check)
            trace_G = np.sum(eigenvalues) / n
            denom = trace_G * lam + 1.0
            pve_se_check = trace_G / (denom * denom) * se_lam

            np.testing.assert_allclose(pve_se, pve_se_check, rtol=1e-2)
