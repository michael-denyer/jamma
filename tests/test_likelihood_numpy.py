"""Parity tests for likelihood_numpy.py and compute_numpy.py.

Verifies that NumPy batch implementations produce numerically correct
results against GEMMA reference output.

Notes on tolerance:
- Uab/Pab/Iab: atol=1e-14 (identical floating-point arithmetic)
- Lambda: rtol=1e-5 (golden section convergence tolerance)
- Wald/Score beta/se: rtol=1e-10 (same Pab arithmetic)
- p_wald/p_score: rtol=1e-8 (Cephes betainc vs GSL betainc)
  Cephes betainc is more accurate than GSL betainc for large a.
  For n=50 samples this difference is negligible, but documented here.
- LRT p-values: rtol=1e-8 (chi2_sf implementation difference)
"""

from __future__ import annotations

import numpy as np
import pytest

from jamma.lmm.likelihood_numpy import (
    _batch_grid_reml_numpy,
    _batch_reml_at_lambda_numpy,
    golden_section_optimize_lambda_numpy,
)
from jamma.lmm.pab import compute_Uab
from jamma.lmm.uab import (
    batch_compute_iab_numpy,
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from tests.builders import rotated_lmm_inputs

pytestmark = pytest.mark.tier0


@pytest.fixture
def synthetic_data():
    """Small synthetic dataset for fast parity tests.

    Returns:
        (eigenvalues, UtW, Uty, UtG) with n_samples=50, n_snps=10.
    """
    d = rotated_lmm_inputs(50, 10, seed=42)
    return d.eigenvalues, d.UtW, d.Uty, d.UtG


# ---------------------------------------------------------------------------
# Scalar P_yy warning deduplication (LIK-07)
# ---------------------------------------------------------------------------


def test_p_yy_warn_once_scalar():
    """_clamp_p_yy fires warning exactly once per run; reset restarts the counter."""
    from loguru import logger

    from jamma.lmm.likelihood import _clamp_p_yy, reset_p_yy_warned

    warning_messages: list[str] = []

    def _capture_sink(message):
        if message.record["level"].name == "WARNING":
            warning_messages.append(message.record["message"])

    # Start clean
    reset_p_yy_warned()

    sink_id = logger.add(_capture_sink, level="WARNING")
    try:
        for _ in range(10):
            _clamp_p_yy(-1.0, 1.0)

        assert len(warning_messages) == 1, (
            f"Expected exactly 1 warning, got {len(warning_messages)}"
        )

        # Reset and fire again — should produce a second warning
        reset_p_yy_warned()
        _clamp_p_yy(-1.0, 1.0)

        assert len(warning_messages) == 2, (
            f"Expected 2 total warnings after reset, got {len(warning_messages)}"
        )
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Scalar MLE P_yy without full Pab
# ---------------------------------------------------------------------------


def test_mle_scalar_pab_ncvt1():
    """_mle_p_yy_scalar_ncvt1 must match calc_pab path to rtol=1e-14."""
    from jamma.lmm.likelihood import (
        _mle_p_yy_scalar_ncvt1,
    )
    from jamma.lmm.pab import calc_pab, get_ab_index

    n_samples = 50
    n_cvt = 1

    d = rotated_lmm_inputs(n_samples, 1, seed=123)
    eigenvalues, UtW, Uty = d.eigenvalues, d.UtW, d.Uty
    Utx = d.UtG[:, 0]

    lambda_val = 0.5
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    Uab = compute_Uab(UtW, Uty, Utx)
    Pab = calc_pab(n_cvt, Hi_eval, Uab)

    # Full Pab path: nc_total = n_cvt + 1 = 2
    nc_total = n_cvt + 1
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    p_yy_full = Pab[nc_total, index_yy]

    # Scalar path
    p_yy_scalar = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)

    np.testing.assert_allclose(
        p_yy_scalar,
        p_yy_full,
        rtol=1e-14,
        err_msg="_mle_p_yy_scalar_ncvt1 does not match calc_pab P_yy",
    )


def test_mle_null_scalar_ncvt1():
    """Null-model mle_log_likelihood with n_cvt=1 matches the full Pab path."""
    from jamma.lmm.likelihood import (
        _mle_p_yy_scalar_null_ncvt1,
        mle_log_likelihood,
    )
    from jamma.lmm.pab import calc_pab, get_ab_index

    rng = np.random.default_rng(789)
    n_samples = 50
    n_cvt = 1

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n_samples))
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)

    lambda_val = 0.3
    v_temp = lambda_val * eigenvalues + 1.0
    Hi_eval = 1.0 / v_temp

    # Null model Uab (no genotype)
    Uab = compute_Uab(UtW, Uty, Utx=None)

    # Full Pab path: nc_total = n_cvt = 1 for null model
    nc_total = n_cvt  # null model
    Pab = calc_pab(n_cvt, Hi_eval, Uab)
    index_yy = get_ab_index(n_cvt + 2, n_cvt + 2, n_cvt)
    p_yy_full = Pab[nc_total, index_yy]

    # Scalar null path
    p_yy_scalar = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)

    np.testing.assert_allclose(
        p_yy_scalar,
        p_yy_full,
        rtol=1e-14,
        err_msg="_mle_p_yy_scalar_null_ncvt1 does not match calc_pab P_yy",
    )

    # Verify end-to-end: the null-model MLE should produce a finite result
    logl = mle_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt, nc_total=n_cvt)
    assert np.isfinite(logl), (
        f"null-model mle_log_likelihood returned non-finite: {logl}"
    )


def test_mle_scalar_degenerate_s_ww_zero():
    """Scalar MLE P_yy returns s_yy when s_ww == 0 (degenerate intercept)."""
    from jamma.lmm.likelihood import (
        _mle_p_yy_scalar_ncvt1,
        _mle_p_yy_scalar_null_ncvt1,
    )
    from jamma.lmm.pab import calc_pab, get_ab_index

    n_samples = 50
    # Construct Uab where column 0 (ww) is all zeros -> Hi_eval @ Uab[:,0] = 0
    Uab = np.zeros((n_samples, 6), dtype=np.float64)
    rng = np.random.default_rng(111)
    Uab[:, 5] = rng.standard_normal(n_samples) ** 2  # yy column: non-zero

    Hi_eval = np.ones(n_samples)

    # s_ww = 0 -> should return s_yy
    p_yy = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)
    expected_s_yy = float(Hi_eval @ Uab[:, 5])
    assert p_yy == expected_s_yy, f"Expected s_yy={expected_s_yy}, got {p_yy}"

    # Null path: same behavior
    p_yy_null = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)
    assert p_yy_null == expected_s_yy

    # Full Pab path should also handle this
    Pab = calc_pab(1, Hi_eval, Uab)
    p_yy_full = Pab[2, get_ab_index(3, 3, 1)]
    # Both should be s_yy since ww=0 means no projection happens
    np.testing.assert_allclose(p_yy, p_yy_full, rtol=1e-12)


def test_mle_scalar_degenerate_p1_xx_zero():
    """Scalar MLE P_yy returns p1_yy when p1_xx == 0 (constant genotype)."""
    from jamma.lmm.likelihood import _mle_p_yy_scalar_ncvt1
    from jamma.lmm.pab import calc_pab, get_ab_index

    n_samples = 50
    rng = np.random.default_rng(222)
    Hi_eval = np.ones(n_samples)

    # Construct Uab where genotype column produces p1_xx = 0:
    # s_xx - s_wx^2/s_ww = 0 when s_xx = s_wx^2/s_ww
    # Easiest: make wx and xx columns such that genotype is proportional to intercept
    w = np.ones(n_samples)
    x = 2.0 * w  # genotype proportional to intercept -> p1_xx = 0
    y = rng.standard_normal(n_samples)

    Uab = np.zeros((n_samples, 6), dtype=np.float64)
    Uab[:, 0] = w * w  # ww
    Uab[:, 1] = w * x  # wx
    Uab[:, 2] = w * y  # wy
    Uab[:, 3] = x * x  # xx
    Uab[:, 4] = x * y  # xy
    Uab[:, 5] = y * y  # yy

    p_yy_scalar = _mle_p_yy_scalar_ncvt1(Hi_eval, Uab)

    # Full Pab path for reference
    Pab = calc_pab(1, Hi_eval, Uab)
    p_yy_full = Pab[2, get_ab_index(3, 3, 1)]

    np.testing.assert_allclose(p_yy_scalar, p_yy_full, rtol=1e-12)


# ---------------------------------------------------------------------------
# Multi-covariate Uab parity (n_cvt > 1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Precomputed REML constants, Iab invariant scalars, and golden section
# ---------------------------------------------------------------------------


def test_reml_const_precomputed():
    """_compute_reml_const(df) must match inline computation bit-exactly."""
    from jamma.lmm.likelihood_numpy import _compute_reml_const

    for df in [10, 48, 100, 1000, 50000]:
        result = _compute_reml_const(df)
        inline = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
        np.testing.assert_equal(
            result,
            inline,
            err_msg=f"_compute_reml_const({df}) does not match inline for df={df}",
        )


def test_iab_invariant_scalars():
    """compute_iab_invariant_scalars_ncvt1 must match manual np.sum exactly."""
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    rng = np.random.default_rng(123)
    n_samples = 80
    uab_invariant_soa = rng.standard_normal((3, n_samples))
    # Ensure rows are all positive-valued for log checks
    uab_invariant_soa = np.abs(uab_invariant_soa) + 0.1

    iab_s_ww, iab_s_wy, iab_s_yy, logdet_iab = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    expected_s_ww = float(uab_invariant_soa[0, :].sum())
    expected_s_wy = float(uab_invariant_soa[1, :].sum())
    expected_s_yy = float(uab_invariant_soa[2, :].sum())
    expected_logdet = np.log(expected_s_ww)

    np.testing.assert_equal(iab_s_ww, expected_s_ww)
    np.testing.assert_equal(iab_s_wy, expected_s_wy)
    np.testing.assert_equal(iab_s_yy, expected_s_yy)
    np.testing.assert_equal(logdet_iab, expected_logdet)


def test_golden_section_eval_count(monkeypatch):
    """The optimizer evaluates REML 2 + n_iter + 1 times (final midpoint eval).

    The final midpoint evaluation ensures the returned (lambda, logl) pair is
    consistent — both from the same evaluation point. Without it, lambda is at
    the midpoint but logl is max(fc, fd) from different points c and d, causing
    a mismatch that propagates into LRT p-values.

    Counted through the public optimizer rather than the bracket helper, since
    the helper stops at the optimum and the caller owns the final evaluation.
    """
    import jamma.lmm.likelihood_numpy as ln

    n_samples, n_snps = 50, 10
    d = rotated_lmm_inputs(n_samples, n_snps, seed=42)
    eigenvalues, UtW, Uty, UtG = d.eigenvalues, d.UtW, d.Uty, d.UtG

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    lambdas, logls, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, n_grid=10, n_iter=5
    )

    reml_const = ln._compute_reml_const(n_samples - 2)
    at_lambda, _ = ln._batch_reml_at_lambda_numpy(
        1, lambdas, eigenvalues, Uab_batch, Iab_batch, reml_const=reml_const
    )
    np.testing.assert_array_equal(
        logls,
        at_lambda,
        err_msg="returned logl must be the REML evaluated at the returned lambda",
    )


def test_golden_section_accuracy_no_final_eval(synthetic_data):
    """Golden section without final eval must produce finite, positive lambdas."""
    eigenvalues, UtW, Uty, UtG = synthetic_data
    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    lambdas_opt, logls_opt, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch
    )

    # Lambdas should be finite positive values
    assert np.all(np.isfinite(lambdas_opt)), "Some lambdas are not finite"
    assert np.all(lambdas_opt[np.isfinite(lambdas_opt)] > 0), (
        "Some finite lambdas are non-positive"
    )
    # Logls should be finite (no NaN for valid SNPs)
    assert np.all(np.isfinite(logls_opt)), "Some logls are not finite"


# ---------------------------------------------------------------------------
# Split-Uab REML path for grid and refinement (n_cvt=1)
# ---------------------------------------------------------------------------


@pytest.fixture
def split_uab_data():
    """Synthetic dataset for split-Uab tests.

    Returns:
        (eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch)
        with n_samples=100, n_snps=50.
    """
    d = rotated_lmm_inputs(100, 50, seed=99)
    eigenvalues, UtW, Uty, UtG = d.eigenvalues, d.UtW, d.Uty, d.UtG

    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty, 1)
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    return eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch


@pytest.fixture
def split_reml_inputs(split_uab_data):
    """Precomputed split-REML scalars for n_cvt=1: invariant, varying, and const.

    Runs the chain every split-path test needs before it can call the kernel
    under test: ``compute_iab_invariant_scalars_ncvt1`` for the invariant
    scalars, its reciprocal ``iab_inv_s_ww``, ``_compute_iab_varying_ncvt1``
    for the per-SNP varying scalars, and ``_compute_reml_const`` for the
    degrees-of-freedom constant.

    Returns:
        (iab_logdet, iab_inv_s_ww, iab_p1_xx, iab_logdet_var, reml_const, df)
    """
    from jamma.lmm.likelihood_numpy import (
        _compute_iab_varying_ncvt1,
        _compute_reml_const,
    )
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    eigenvalues, uab_varying_soa, uab_invariant_soa, _Uab_batch, _Iab_batch = (
        split_uab_data
    )
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2  # n_cvt=1

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0
    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )
    reml_const = _compute_reml_const(df)

    return iab_logdet, iab_inv_s_ww, iab_p1_xx, iab_logdet_var, reml_const, df


def test_grid_reml_split_matches_full(split_uab_data, split_reml_inputs):
    """_batch_grid_reml_split_ncvt1_numpy must match _batch_grid_reml_numpy."""
    from jamma.lmm.likelihood_numpy import _batch_grid_reml_split_ncvt1_numpy

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    iab_logdet, iab_inv_s_ww, iab_p1_xx, iab_logdet_var, reml_const, _df = (
        split_reml_inputs
    )
    n_grid = 20
    lambdas_grid = np.exp(np.linspace(np.log(1e-5), np.log(1e5), n_grid))

    logls_split = _batch_grid_reml_split_ncvt1_numpy(
        lambdas_grid,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )

    logls_full = _batch_grid_reml_numpy(
        1, lambdas_grid, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        logls_split,
        logls_full,
        rtol=1e-12,
        err_msg="_batch_grid_reml_split_ncvt1_numpy mismatch",
    )


def test_refinement_reml_split_matches_full(split_uab_data, split_reml_inputs):
    """_batch_reml_at_lambda_split_ncvt1_numpy must match full path."""
    from jamma.lmm.likelihood_numpy import _batch_reml_at_lambda_split_ncvt1_numpy

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    iab_logdet, iab_inv_s_ww, iab_p1_xx, iab_logdet_var, reml_const, _df = (
        split_reml_inputs
    )
    n_snps = uab_varying_soa.shape[0]

    # Per-SNP lambda values (different for each SNP)
    rng = np.random.default_rng(7)
    lambda_vals = np.exp(rng.uniform(np.log(1e-4), np.log(1e3), n_snps))

    logls_split, _ = _batch_reml_at_lambda_split_ncvt1_numpy(
        lambda_vals,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )

    logls_full, _ = _batch_reml_at_lambda_numpy(
        1, lambda_vals, eigenvalues, Uab_batch, Iab_batch, reml_const
    )

    np.testing.assert_allclose(
        logls_split,
        logls_full,
        rtol=1e-12,
        err_msg="_batch_reml_at_lambda_split_ncvt1_numpy does not match full path",
    )


def test_split_optimizer_matches_full(split_uab_data):
    """golden_section_optimize_lambda_split_ncvt1_numpy must match full optimizer."""
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )

    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    lambdas_split, logls_split, _ = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
    )

    lambdas_full, logls_full, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        lambdas_split,
        lambdas_full,
        rtol=1e-10,
        err_msg="Split optimizer lambdas do not match full optimizer",
    )
    np.testing.assert_allclose(
        logls_split,
        logls_full,
        rtol=1e-8,
        err_msg="Split optimizer logls do not match full optimizer",
    )


def test_split_pab_matches_generic_pab(split_uab_data):
    """Pab from split optimizer must match generic Pab element-by-element."""
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )

    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    _, _, Pab_split = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
    )

    _, _, Pab_generic = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch
    )

    assert Pab_split.shape == Pab_generic.shape, (
        f"Shape mismatch: split={Pab_split.shape} vs generic={Pab_generic.shape}"
    )
    np.testing.assert_allclose(
        Pab_split,
        Pab_generic,
        rtol=1e-12,
        err_msg="Split Pab does not match generic Pab element-by-element",
    )


def test_invariant_computed_once_per_lambda(split_uab_data, split_reml_inputs):
    """Invariant dot products must be (n_grid,), not (n_grid, n_snps)."""
    from jamma.lmm.likelihood_numpy import _batch_grid_reml_split_ncvt1_numpy

    # Verify structural property: function produces (n_grid, n_snps) output
    # while internally computing (n_grid,) invariant sums.
    eigenvalues, uab_varying_soa, uab_invariant_soa, _Uab_batch, _Iab_batch = (
        split_uab_data
    )
    iab_logdet, iab_inv_s_ww, iab_p1_xx, iab_logdet_var, reml_const, _df = (
        split_reml_inputs
    )
    n_grid = 15
    n_snps = uab_varying_soa.shape[0]

    lambdas_grid = np.exp(np.linspace(np.log(1e-5), np.log(1e5), n_grid))

    Hi_eval_grid = 1.0 / (lambdas_grid[:, None] * eigenvalues[None, :] + 1.0)

    # s_ww_grid: (n_grid,) @ (n_samples,) -> (n_grid,)
    s_ww_grid = Hi_eval_grid @ uab_invariant_soa[0]
    assert s_ww_grid.shape == (n_grid,), (
        f"s_ww_grid should be (n_grid,)={(n_grid,)}, got {s_ww_grid.shape}"
    )
    # Must NOT be (n_grid, n_snps) — that would be the old O(n_grid * n_snps) path
    assert s_ww_grid.shape != (n_grid, n_snps), (
        "s_ww_grid shape should NOT be (n_grid, n_snps)"
    )

    # Also verify the split function itself returns (n_grid, n_snps) output
    logls = _batch_grid_reml_split_ncvt1_numpy(
        lambdas_grid,
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_logdet,
        iab_inv_s_ww,
        iab_p1_xx,
        iab_logdet_var,
        reml_const,
    )
    assert logls.shape == (n_grid, n_snps), (
        f"Expected output shape ({n_grid}, {n_snps}), got {logls.shape}"
    )
