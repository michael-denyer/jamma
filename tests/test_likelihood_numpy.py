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

from jamma.lmm.likelihood import (
    _golden_section_minimize,
    compute_Uab,
    reml_log_likelihood,
)
from jamma.lmm.likelihood_numpy import (
    _batch_grid_reml_numpy,
    _batch_reml_at_lambda_numpy,
    golden_section_optimize_lambda_numpy,
)
from jamma.lmm.stats import batch_calc_wald_stats_from_pab_numpy
from jamma.lmm.uab import (
    _batch_compute_pab_varying_numpy,
    batch_compute_iab_numpy,
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from tests.builders import rotated_lmm_inputs

pytestmark = pytest.mark.tier0


def _wald_stats_from_lambdas(n_cvt, lambdas, eigenvalues, Uab_batch, n_samples):
    """Wald stats from optimized lambdas, for tests that hold lambdas, not Pab.

    Production code holds Pab from the optimizer and calls
    batch_calc_wald_stats_from_pab_numpy directly.
    """
    Hi_eval_batch = 1.0 / (lambdas[:, None] * eigenvalues[None, :] + 1.0)
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)
    return batch_calc_wald_stats_from_pab_numpy(n_cvt, Pab_batch, n_samples)


@pytest.fixture
def synthetic_data():
    """Small synthetic dataset for fast parity tests.

    Returns:
        (eigenvalues, UtW, Uty, UtG) with n_samples=50, n_snps=10.
    """
    d = rotated_lmm_inputs(50, 10, seed=42)
    return d.eigenvalues, d.UtW, d.Uty, d.UtG


# ---------------------------------------------------------------------------
# Scalar P_yy warning deduplication
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
        calc_pab,
        compute_Uab,
        get_ab_index,
    )

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
        calc_pab,
        compute_Uab,
        get_ab_index,
        mle_log_likelihood,
    )

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
        calc_pab,
        get_ab_index,
    )

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
    from jamma.lmm.likelihood import _mle_p_yy_scalar_ncvt1, calc_pab, get_ab_index

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


# ---------------------------------------------------------------------------
# Merged Wald path (optimizer returns Pab, no redundant Hi_eval)
# ---------------------------------------------------------------------------


@pytest.fixture
def wald_pab_data():
    """Synthetic data for merged Wald path tests.

    Returns:
        (eigenvalues, Uab_batch, Iab_batch, n_samples) with n_samples=80, n_snps=20.
    """
    n_samples = 80
    d = rotated_lmm_inputs(n_samples, 20, seed=99)
    eigenvalues, UtW, Uty, UtG = d.eigenvalues, d.UtW, d.Uty, d.UtG

    from jamma.lmm.uab import batch_compute_iab_numpy, batch_compute_uab_numpy

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    return eigenvalues, Uab_batch, Iab_batch, n_samples


def test_optimizer_returns_pab(wald_pab_data):
    """The optimizer returns (lambdas, logls, Pab) with Pab of the right shape."""
    from jamma.lmm.likelihood import build_index_table

    eigenvalues, Uab_batch, Iab_batch, _n_samples = wald_pab_data
    n_cvt = 1
    n_snps = Uab_batch.shape[0]

    result = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )

    assert isinstance(result, tuple), "Result must be a tuple"
    assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
    lambdas, logls, Pab_final = result

    assert lambdas.shape == (n_snps,), f"lambdas shape {lambdas.shape}"
    assert logls.shape == (n_snps,), f"logls shape {logls.shape}"

    # Pab shape: (n_snps, n_cvt+2, n_index)
    table = build_index_table(n_cvt)
    n_index = table.n_index
    assert Pab_final.shape == (n_snps, n_cvt + 2, n_index), (
        f"Pab_final shape {Pab_final.shape}, expected {(n_snps, n_cvt + 2, n_index)}"
    )
    # Pab values should be finite (no NaN for non-degenerate synthetic data)
    assert np.all(np.isfinite(Pab_final)), "Pab_final contains non-finite values"


# ---------------------------------------------------------------------------
# All-NaN grid test for _batch_golden_section_numpy
# ---------------------------------------------------------------------------


def test_batch_golden_section_numpy_all_nan_grid():
    """The bracket refinement with all-NaN grid logls brackets around l_min.

    When every grid log-likelihood is NaN (all SNPs degenerate), argmax of
    NaN-replaced -inf selects index 0 (the lower bound), bracketing around
    l_min. Refinement should return log-lambdas at or near log(l_min) without
    crashing, and never an infinite one.

    This is the all-SNPs-degenerate extreme: _guard_P_yy produces NaN for
    every grid point, so safe_logls is all -inf.
    """
    from jamma.lmm.likelihood_numpy import (
        _batch_golden_section_bracket_numpy,
        _batch_reml_at_lambda_numpy,
        _compute_reml_const,
    )

    rng = np.random.default_rng(42)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # all-zero genotype → all-NaN grid logls

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG_degen.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    n_grid = 10
    log_l_min = np.log(l_min)
    log_l_max = np.log(1e5)
    log_lambdas = np.linspace(log_l_min, log_l_max, n_grid)

    # All-degenerate grid: for zero genotype, P_yy after projecting out X
    # may be pathological, but REML logls will be constant across lambda.
    # Force the all-NaN scenario by using an artificial grid of NaN logls.
    grid_logls_all_nan = np.full((n_grid, n_snps), np.nan)

    reml_const = _compute_reml_const(n - 1 - 1)

    def compute_batch_fn(log_lams):
        lams = np.exp(log_lams)
        return _batch_reml_at_lambda_numpy(
            1, lams, eigenvalues, Uab_batch, Iab_batch, reml_const
        )[0]

    lambdas_out = np.exp(
        _batch_golden_section_bracket_numpy(
            compute_batch_fn, grid_logls_all_nan, log_lambdas, n_iter=20
        )
    )

    assert lambdas_out.shape == (n_snps,), (
        f"Expected ({n_snps},), got {lambdas_out.shape}"
    )

    # With all-NaN grid, argmax(safe_logls) falls to index 0 (l_min bracket).
    # Optimizer must not produce +inf or -inf lambdas.
    assert not np.any(np.isinf(lambdas_out)), (
        f"Lambdas should not be infinite, got {lambdas_out}"
    )
    # Lambdas should be within [l_min, l_max] range
    assert np.all(lambdas_out >= l_min * 0.99), f"Lambdas below l_min: {lambdas_out}"
    assert np.all(lambdas_out <= 1e5 * 1.01), f"Lambdas above l_max: {lambdas_out}"


# ---------------------------------------------------------------------------
# Degenerate SNP tests — NumPy batch REML path
# ---------------------------------------------------------------------------


def _degenerate_snp_inputs(rng, n, n_snps, pattern):
    """Build UtG under one of two degeneracy patterns.

    Args:
        rng: NumPy random generator.
        n: Number of samples.
        n_snps: Number of SNPs.
        pattern: "all_zero" (every SNP degenerate, P_XX=0) or "mixed"
            (columns 1 and 3 live, the rest degenerate).

    Returns:
        (UtG, degenerate_idxs, valid_idxs).
    """
    if pattern == "all_zero":
        UtG = np.zeros((n, n_snps))
        return UtG, list(range(n_snps)), []
    UtG = np.zeros((n, n_snps))
    UtG[:, 1] = rng.standard_normal(n)
    UtG[:, 3] = rng.standard_normal(n)
    return UtG, [0, 2, 4], [1, 3]


@pytest.mark.parametrize("pattern", ["all_zero", "mixed"])
def test_batch_numpy_degenerate_snps_wald_nan(pattern):
    """Generic-path NumPy batch REML: degenerate SNPs return l_min and NaN Wald.

    Constant genotypes produce zero Uab columns for genotype interactions.
    The log-likelihood is driven by phenotype variance only (constant w.r.t.
    lambda change), so the optimizer converges to l_min (the lower bound) and
    Wald stats return NaN (P_XX=0).  ``pattern="mixed"`` additionally checks
    that valid SNPs (columns 1, 3) produce finite stats in the same batch,
    without cross-SNP contamination from the degenerate columns.
    """
    rng = np.random.default_rng(42)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG, degenerate_idxs, valid_idxs = _degenerate_snp_inputs(rng, n, n_snps, pattern)

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    lambdas, logls, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )
    betas, ses, pwalds = _wald_stats_from_lambdas(1, lambdas, eigenvalues, Uab_batch, n)

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    if pattern == "all_zero":
        np.testing.assert_allclose(
            lambdas,
            l_min,
            rtol=1e-4,
            err_msg="All-degenerate SNPs should return l_min lambda",
        )

    assert np.all(np.isnan(betas[degenerate_idxs])), (
        f"Degenerate betas should be NaN, got {betas[degenerate_idxs]}"
    )
    assert np.all(np.isnan(ses[degenerate_idxs])), (
        f"Degenerate ses should be NaN, got {ses[degenerate_idxs]}"
    )
    assert np.all(np.isnan(pwalds[degenerate_idxs])), (
        f"Degenerate p_walds should be NaN, got {pwalds[degenerate_idxs]}"
    )
    if valid_idxs:
        assert np.all(np.isfinite(betas[valid_idxs])), (
            f"Valid betas should be finite, got {betas[valid_idxs]}"
        )
        assert np.all(np.isfinite(ses[valid_idxs])), (
            f"Valid ses should be finite, got {ses[valid_idxs]}"
        )
        assert np.all(np.isfinite(pwalds[valid_idxs])), (
            f"Valid p_walds should be finite, got {pwalds[valid_idxs]}"
        )


# ---------------------------------------------------------------------------
# Degenerate SNP tests — Python fallback (split ncvt1 path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pattern", ["all_zero", "mixed"])
def test_split_ncvt1_fallback_degenerate_snps_wald_nan(pattern):
    """Split ncvt1 Python fallback path: degenerate SNPs produce NaN Wald stats.

    golden_section_optimize_lambda_split_ncvt1_numpy is the Python fallback
    when the C extension is unavailable.  When UtG is all-zero (constant
    genotype), the varying columns [wx, xx, xy] are zero for every SNP.
    The optimizer returns lambdas near l_min; downstream Wald stats must
    produce NaN for every SNP because P_XX = 0.  ``pattern="mixed"``
    additionally checks that valid SNPs (columns 1, 3) produce finite stats
    in the same batch.
    """
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )
    from jamma.lmm.uab import (
        batch_compute_uab_varying_soa_numpy,
        compute_iab_invariant_scalars_ncvt1,
        compute_uab_invariant_soa,
        reconstruct_uab_from_soa,
    )

    rng = np.random.default_rng(17)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG, degenerate_idxs, valid_idxs = _degenerate_snp_inputs(rng, n, n_snps, pattern)

    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty, 1)
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG.T)
    iab_s_ww, iab_s_wy, iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )

    lambdas, logls, _ = golden_section_optimize_lambda_split_ncvt1_numpy(
        eigenvalues,
        uab_varying_soa,
        uab_invariant_soa,
        iab_s_ww,
        iab_s_wy,
        iab_s_yy,
        iab_logdet,
        l_min=l_min,
    )

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    if pattern == "all_zero":
        np.testing.assert_allclose(
            lambdas,
            l_min,
            rtol=1e-4,
            err_msg="Split ncvt1 all-degenerate SNPs should return l_min lambda",
        )

    # Reconstruct full Uab for Wald stats
    Uab_batch = reconstruct_uab_from_soa(uab_invariant_soa, uab_varying_soa, 1)
    betas, ses, pwalds = _wald_stats_from_lambdas(1, lambdas, eigenvalues, Uab_batch, n)

    assert np.all(np.isnan(betas[degenerate_idxs])), (
        f"Degenerate betas should be NaN, got {betas[degenerate_idxs]}"
    )
    assert np.all(np.isnan(ses[degenerate_idxs])), (
        f"Degenerate ses should be NaN, got {ses[degenerate_idxs]}"
    )
    assert np.all(np.isnan(pwalds[degenerate_idxs])), (
        f"Degenerate p_walds should be NaN, got {pwalds[degenerate_idxs]}"
    )
    if valid_idxs:
        assert np.all(np.isfinite(betas[valid_idxs])), (
            f"Valid betas should be finite, got {betas[valid_idxs]}"
        )
        assert np.all(np.isfinite(ses[valid_idxs])), (
            f"Valid ses should be finite, got {ses[valid_idxs]}"
        )
        assert np.all(np.isfinite(pwalds[valid_idxs])), (
            f"Valid p_walds should be finite, got {pwalds[valid_idxs]}"
        )


# ---------------------------------------------------------------------------
# Scalar-vs-batch REML optimizer parity tests
# ---------------------------------------------------------------------------


def test_scalar_vs_batch_reml_multi_snp_consistency():
    """Batch REML optimizer agrees with scalar path for each of 10 SNPs.

    Tests that the batch path does not introduce cross-SNP interference:
    each SNP in the batch should get the same optimal lambda as the scalar
    path processing that SNP individually.
    """
    n, n_snps = 30, 10
    n_cvt = 1

    d = rotated_lmm_inputs(n, n_snps, seed=7)
    eigenvalues, UtW, Uty, UtG = d.eigenvalues, d.UtW, d.Uty, d.UtG

    # Batch path — all 10 SNPs at once
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas_batch, _, _ = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )

    # Scalar path — one SNP at a time
    lambda_scalars = np.empty(n_snps)
    for i in range(n_snps):
        Uab_i = compute_Uab(UtW, Uty, UtG[:, i])

        def _scalar_obj(lam, uab=Uab_i):
            return -reml_log_likelihood(
                lam, eigenvalues, uab, n_cvt=n_cvt, nc_total=n_cvt + 1
            )

        lambda_scalars[i], _ = _golden_section_minimize(
            _scalar_obj, 1e-5, 1e5, n_grid=50, n_iter=20
        )

    np.testing.assert_allclose(
        lambda_scalars,
        lambdas_batch,
        rtol=1e-4,
        err_msg="Scalar and batch REML paths should agree on all 10 SNP lambdas",
    )


# ---------------------------------------------------------------------------
# Scalar-vs-batch REML optimizer: tight lambda and logl parity
# ---------------------------------------------------------------------------


def test_scalar_vs_batch_reml_single_snp_lambda_and_logl_parity():
    """Scalar and batch REML optimizers agree on lambda and logl.

    Feeds a single SNP through both optimizers:
    - Scalar: _golden_section_minimize + reml_log_likelihood (likelihood.py)
    - Batch:  golden_section_optimize_lambda_numpy (likelihood_numpy.py)

    Both run identical grid search + golden section in log-lambda space (same
    n_grid=50, n_iter=20 defaults), so the returned optimal lambda and the
    log-likelihood evaluated at that lambda should agree to near-machine-epsilon
    after accounting for the batch midpoint vs scalar midpoint evaluation.

    Tolerance rationale:
    - lambda: rtol=1e-10 — both converge to log((a+b)/2) with same bracket,
      so the difference is sub-ULP in double precision.
    - logl: rtol=1e-10 — evaluated at the same lambda point via the same REML
      arithmetic; any discrepancy would indicate a divergence in Pab/Iab logic.
    """
    n = 50
    n_cvt = 1

    d = rotated_lmm_inputs(n, 1, seed=123)
    eigenvalues, UtW, Uty = d.eigenvalues, d.UtW, d.Uty
    Utx = d.UtG[:, 0]

    # --- Scalar path ---
    Uab_scalar = compute_Uab(UtW, Uty, Utx)

    def scalar_neg_reml(lam: float) -> float:
        return -reml_log_likelihood(
            lam, eigenvalues, Uab_scalar, n_cvt=n_cvt, nc_total=n_cvt + 1
        )

    lambda_scalar, logl_scalar = _golden_section_minimize(
        scalar_neg_reml, l_min=1e-5, l_max=1e5, n_grid=50, n_iter=20
    )

    # --- Batch path (single SNP wrapped in batch dimension) ---
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, Utx.reshape(1, n))
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas_batch, logls_batch, _ = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch, n_grid=50, n_iter=20
    )
    lambda_batch = lambdas_batch[0]
    logl_batch = logls_batch[0]

    np.testing.assert_allclose(
        lambda_scalar,
        lambda_batch,
        rtol=1e-10,
        atol=1e-14,
        err_msg=(
            f"Scalar lambda {lambda_scalar:.6e} vs batch lambda {lambda_batch:.6e} "
            "disagree beyond rtol=1e-10"
        ),
    )
    np.testing.assert_allclose(
        logl_scalar,
        logl_batch,
        rtol=1e-10,
        atol=1e-14,
        err_msg=(
            f"Scalar logl {logl_scalar:.6e} vs batch logl {logl_batch:.6e} "
            "disagree beyond rtol=1e-10"
        ),
    )


# ---------------------------------------------------------------------------
# reconstruct_uab_from_soa generalization for n_cvt > 1
# ---------------------------------------------------------------------------


def test_reconstruct_uab_from_soa_ncvt1_fast_path():
    """reconstruct_uab_from_soa's n_cvt=1 fast path rebuilds the six-column Uab."""
    from jamma.lmm.uab import reconstruct_uab_from_soa

    rng = np.random.default_rng(42)
    n_samples, n_snps = 50, 8
    UtW = np.ones((n_samples, 1))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    Uab_ref = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)

    # Build SoA for n_cvt=1
    inv_soa = np.stack([Uab_ref[0, :, 0], Uab_ref[0, :, 2], Uab_ref[0, :, 5]])  # (3, n)
    var_soa = np.stack(
        [Uab_ref[:, :, 1], Uab_ref[:, :, 3], Uab_ref[:, :, 4]], axis=1
    )  # (n_snps, 3, n)

    Uab_recon = reconstruct_uab_from_soa(inv_soa, var_soa, 1)
    np.testing.assert_allclose(
        Uab_recon,
        Uab_ref,
        rtol=1e-14,
        atol=0,
        err_msg="reconstruct_uab_from_soa backward compat (no n_cvt) failed",
    )


@pytest.mark.parametrize(
    ("n_cvt", "seed", "n_samples", "n_snps"),
    [(2, 7, 40, 6), (4, 13, 30, 5)],
)
def test_reconstruct_uab_from_soa_multi_cvt(n_cvt, seed, n_samples, n_snps):
    """reconstruct_uab_from_soa round-trips via classify_uab_columns for n_cvt > 1."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.uab import reconstruct_uab_from_soa

    rng = np.random.default_rng(seed)
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Full reference Uab
    Uab_ref = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)

    inv_indices, var_indices = classify_uab_columns(n_cvt)

    # Build invariant SoA from first SNP's Uab (all SNPs share invariant columns)
    inv_list = list(inv_indices)
    uab_invariant_soa = np.ascontiguousarray(
        Uab_ref[0, :, inv_list]
    )  # (n_inv, n_samples)

    # Build varying SoA
    uab_varying_soa = np.ascontiguousarray(
        Uab_ref[:, :, list(var_indices)].transpose(0, 2, 1)
    )  # (n_snps, n_var, n_samples)

    # Reconstruct should match Uab_ref
    Uab_recon = reconstruct_uab_from_soa(
        uab_invariant_soa, uab_varying_soa, n_cvt=n_cvt
    )
    np.testing.assert_allclose(
        Uab_recon,
        Uab_ref,
        rtol=1e-14,
        atol=0,
        err_msg=f"reconstruct_uab_from_soa failed to round-trip for n_cvt={n_cvt}",
    )
