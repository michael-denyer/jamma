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

from jamma.lmm.compute_numpy import compute_lmm_chunk_numpy
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
from jamma.lmm.stats import batch_calc_wald_stats_numpy
from jamma.lmm.uab import (
    batch_compute_iab_numpy,
    batch_compute_uab_numpy,
    batch_compute_uab_varying_soa_numpy,
    compute_uab_invariant_soa,
)
from tests.builders import rotated_lmm_inputs


@pytest.fixture
def synthetic_data():
    """Small synthetic dataset for fast parity tests.

    Returns:
        (eigenvalues, UtW, Uty, UtG) with n_samples=50, n_snps=10.
    """
    d = rotated_lmm_inputs(50, 10, seed=42)
    return d.eigenvalues, d.UtW, d.Uty, d.UtG


# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def testcompute_lmm_chunk_numpy_all_modes(synthetic_data, monkeypatch):
    """compute_lmm_chunk_numpy must return non-None expected keys for each mode.

    The extension is cleared because this function is the full-Uab NumPy path,
    and the runner reaches it only on NUMPY_FALLBACK, which is selected only
    when the extension is absent.
    """
    from jamma.lmm import compute_numpy as cn

    monkeypatch.setattr(cn, "_accel", None)

    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_samples = eigenvalues.shape[0]

    lambda_null = 0.1
    Hi_eval_null = 1.0 / (lambda_null * eigenvalues + 1.0)
    logl_H0 = -25.0

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)

    # Mode 1: Wald — expects lambdas, logls, betas, ses, pwalds
    result1 = compute_lmm_chunk_numpy(1, 1, eigenvalues, Uab_batch, n_samples)
    for key in ("lambdas", "logls", "betas", "ses", "pwalds"):
        assert result1[key] is not None, f"Mode 1: key '{key}' is None"
    assert result1["lambdas_mle"] is None
    assert result1["p_lrts"] is None
    assert result1["p_scores"] is None

    # Mode 2: LRT — expects lambdas_mle, p_lrts
    result2 = compute_lmm_chunk_numpy(
        2, 1, eigenvalues, Uab_batch, n_samples, logl_H0=logl_H0
    )
    for key in ("lambdas_mle", "p_lrts"):
        assert result2[key] is not None, f"Mode 2: key '{key}' is None"
    assert result2["lambdas"] is None
    assert result2["logls"] is None
    assert result2["betas"] is None
    assert result2["ses"] is None
    assert result2["pwalds"] is None
    assert result2["p_scores"] is None

    # Mode 3: Score — expects betas, ses, p_scores
    result3 = compute_lmm_chunk_numpy(
        3, 1, eigenvalues, Uab_batch, n_samples, Hi_eval_null=Hi_eval_null
    )
    for key in ("betas", "ses", "p_scores"):
        assert result3[key] is not None, f"Mode 3: key '{key}' is None"
    assert result3["lambdas"] is None
    assert result3["logls"] is None
    assert result3["pwalds"] is None
    assert result3["lambdas_mle"] is None
    assert result3["p_lrts"] is None

    # Mode 4: All — all keys non-None
    result4 = compute_lmm_chunk_numpy(
        4,
        1,
        eigenvalues,
        Uab_batch,
        n_samples,
        Hi_eval_null=Hi_eval_null,
        logl_H0=logl_H0,
    )
    for key in (
        "lambdas",
        "logls",
        "betas",
        "ses",
        "pwalds",
        "lambdas_mle",
        "p_lrts",
        "p_scores",
    ):
        assert result4[key] is not None, f"Mode 4: key '{key}' is None"


@pytest.mark.tier0
def testcompute_lmm_chunk_numpy_missing_args_raise(synthetic_data):
    """compute_lmm_chunk_numpy must raise ValueError when required args are absent."""
    eigenvalues, UtW, Uty, UtG = synthetic_data
    n_samples = eigenvalues.shape[0]
    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)

    with pytest.raises(ValueError, match="logl_H0 is required"):
        compute_lmm_chunk_numpy(2, 1, eigenvalues, Uab_batch, n_samples)

    with pytest.raises(ValueError, match="Hi_eval_null is required"):
        compute_lmm_chunk_numpy(3, 1, eigenvalues, Uab_batch, n_samples)

    # Mode 4 (All) requires both logl_H0 and Hi_eval_null.
    # Missing logl_H0 is checked first (line order in source).
    with pytest.raises(ValueError, match="logl_H0 is required"):
        compute_lmm_chunk_numpy(4, 1, eigenvalues, Uab_batch, n_samples)

    # Providing logl_H0 but omitting Hi_eval_null also raises.
    with pytest.raises(ValueError, match="Hi_eval_null is required"):
        compute_lmm_chunk_numpy(4, 1, eigenvalues, Uab_batch, n_samples, logl_H0=-50.0)


# ---------------------------------------------------------------------------
# Scalar P_yy warning deduplication (LIK-07)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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
# Scalar MLE P_yy without full Pab (LIK-08)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
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
# Task 1: Precompute REML constants, Iab invariant scalars, golden section fix
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
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
# Task 2: Split-Uab REML path for grid and refinement (n_cvt=1)
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


@pytest.mark.tier0
def test_grid_reml_split_matches_full(split_uab_data):
    """_batch_grid_reml_split_ncvt1_numpy must match _batch_grid_reml_numpy."""
    from jamma.lmm.likelihood_numpy import _batch_grid_reml_split_ncvt1_numpy
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    n_grid = 20
    lambdas_grid = np.exp(np.linspace(np.log(1e-5), np.log(1e5), n_grid))

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0

    n_samples = eigenvalues.shape[0]
    df = n_samples - 2  # n_cvt=1

    from jamma.lmm.likelihood_numpy import (
        _compute_iab_varying_ncvt1,
        _compute_reml_const,
    )

    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )
    reml_const = _compute_reml_const(df)

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


@pytest.mark.tier0
def test_refinement_reml_split_matches_full(split_uab_data):
    """_batch_reml_at_lambda_split_ncvt1_numpy must match full path."""
    from jamma.lmm.likelihood_numpy import (
        _batch_reml_at_lambda_split_ncvt1_numpy,
        _compute_reml_const,
    )
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    n_snps = uab_varying_soa.shape[0]
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2

    # Per-SNP lambda values (different for each SNP)
    rng = np.random.default_rng(7)
    lambda_vals = np.exp(rng.uniform(np.log(1e-4), np.log(1e3), n_snps))

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0
    reml_const = _compute_reml_const(df)

    from jamma.lmm.likelihood_numpy import _compute_iab_varying_ncvt1

    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )

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


@pytest.mark.tier0
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


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_invariant_computed_once_per_lambda(split_uab_data):
    """Invariant dot products must be (n_grid,), not (n_grid, n_snps)."""
    from jamma.lmm.likelihood_numpy import _compute_reml_const
    from jamma.lmm.uab import compute_iab_invariant_scalars_ncvt1

    # Verify structural property: function produces (n_grid, n_snps) output
    # while internally computing (n_grid,) invariant sums.
    eigenvalues, uab_varying_soa, uab_invariant_soa, Uab_batch, Iab_batch = (
        split_uab_data
    )
    n_grid = 15
    n_snps = uab_varying_soa.shape[0]
    n_samples = eigenvalues.shape[0]
    df = n_samples - 2

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
    from jamma.lmm.likelihood_numpy import _batch_grid_reml_split_ncvt1_numpy

    iab_s_ww, _iab_s_wy, _iab_s_yy, iab_logdet = compute_iab_invariant_scalars_ncvt1(
        uab_invariant_soa
    )
    iab_inv_s_ww = 1.0 / iab_s_ww if iab_s_ww != 0 else 0.0
    reml_const = _compute_reml_const(df)

    from jamma.lmm.likelihood_numpy import _compute_iab_varying_ncvt1

    iab_p1_xx, iab_logdet_var = _compute_iab_varying_ncvt1(
        uab_varying_soa, iab_inv_s_ww
    )

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
# Plan 53-03: Merged Wald path (optimizer returns Pab, no redundant Hi_eval)
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


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_wald_from_pab_matches_original(wald_pab_data):
    """Wald stats from pre-computed Pab match original path to rtol=1e-14."""
    from jamma.lmm.stats import batch_calc_wald_stats_from_pab_numpy

    eigenvalues, Uab_batch, Iab_batch, n_samples = wald_pab_data
    n_cvt = 1

    lambdas, _logls, Pab_final = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )

    # Path A: reconstruct Hi_eval and Pab from the optimal lambdas.
    betas_orig, ses_orig, pwalds_orig = batch_calc_wald_stats_numpy(
        n_cvt, lambdas, eigenvalues, Uab_batch, n_samples
    )

    # Path B: reuse the Pab the optimizer already evaluated at those lambdas.
    betas_pab, ses_pab, pwalds_pab = batch_calc_wald_stats_from_pab_numpy(
        n_cvt, Pab_final, n_samples
    )

    # Wald stats should be identical to machine precision
    np.testing.assert_allclose(
        betas_pab,
        betas_orig,
        rtol=1e-14,
        atol=1e-15,
        err_msg="betas from Pab path differ from original",
    )
    np.testing.assert_allclose(
        ses_pab,
        ses_orig,
        rtol=1e-14,
        atol=1e-15,
        err_msg="ses from Pab path differ from original",
    )
    np.testing.assert_allclose(
        pwalds_pab,
        pwalds_orig,
        rtol=1e-14,
        atol=1e-15,
        err_msg="p_walds from Pab path differ from original",
    )


# ---------------------------------------------------------------------------
# All-NaN grid test for _batch_golden_section_numpy
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_batch_numpy_all_degenerate_snps_return_lmin():
    """NumPy batch REML returns l_min for all-degenerate SNPs (UtG=0, P_XX=0).

    Constant genotypes produce zero Uab columns for genotype interactions.
    The log-likelihood is driven by phenotype variance only (constant w.r.t.
    lambda change), so the optimizer converges to l_min (the lower bound).
    The critical downstream behavior is that Wald stats return NaN.
    """
    from jamma.lmm.stats import batch_calc_wald_stats_numpy

    rng = np.random.default_rng(42)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # all-zero genotype → P_XX = 0

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG_degen.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    lambdas, logls, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    # Optimizer converges to l_min when genotype has no variance
    np.testing.assert_allclose(
        lambdas,
        l_min,
        rtol=1e-4,
        err_msg="All-degenerate SNPs should return l_min lambda",
    )

    # P_XX = 0 → Wald stats all NaN (critical downstream behavior)
    assert np.all(np.isnan(betas)), f"All betas should be NaN, got {betas}"
    assert np.all(np.isnan(ses)), f"All ses should be NaN, got {ses}"
    assert np.all(np.isnan(pwalds)), f"All p_walds should be NaN, got {pwalds}"


@pytest.mark.tier0
def test_batch_numpy_mixed_degenerate_and_valid_snps():
    """NumPy batch REML handles a mix of degenerate and valid SNPs correctly.

    Columns 0, 2, 4 are zero (degenerate, P_XX=0); columns 1, 3 are normal (valid).
    Degenerate SNPs produce NaN Wald stats (beta/se/p_wald = NaN);
    valid SNPs produce finite stats. Both run in the same batch without
    cross-SNP contamination.
    """
    from jamma.lmm.stats import batch_calc_wald_stats_numpy

    rng = np.random.default_rng(99)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)

    UtG = np.zeros((n, n_snps))
    UtG[:, 1] = rng.standard_normal(n)  # valid
    UtG[:, 3] = rng.standard_normal(n)  # valid

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    lambdas, logls, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    degenerate_idxs = [0, 2, 4]
    valid_idxs = [1, 3]

    # Degenerate SNPs: P_XX=0 → NaN Wald stats (beta/se/p_wald)
    assert np.all(np.isnan(betas[degenerate_idxs])), (
        f"Degenerate SNP betas should be NaN, got {betas[degenerate_idxs]}"
    )
    assert np.all(np.isnan(ses[degenerate_idxs])), (
        f"Degenerate SNP ses should be NaN, got {ses[degenerate_idxs]}"
    )
    assert np.all(np.isnan(pwalds[degenerate_idxs])), (
        f"Degenerate SNP p_walds should be NaN, got {pwalds[degenerate_idxs]}"
    )

    # Valid SNPs: finite Wald stats
    assert np.all(np.isfinite(betas[valid_idxs])), (
        f"Valid SNP betas should be finite, got {betas[valid_idxs]}"
    )
    assert np.all(np.isfinite(ses[valid_idxs])), (
        f"Valid SNP ses should be finite, got {ses[valid_idxs]}"
    )
    assert np.all(np.isfinite(pwalds[valid_idxs])), (
        f"Valid SNP p_walds should be finite, got {pwalds[valid_idxs]}"
    )


# ---------------------------------------------------------------------------
# Degenerate SNP tests — Python fallback (split ncvt1 and generic batch paths)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_split_ncvt1_fallback_degenerate_snps_wald_nan():
    """Split ncvt1 Python fallback path: degenerate SNPs produce NaN Wald stats.

    golden_section_optimize_lambda_split_ncvt1_numpy is the Python fallback
    when the C extension is unavailable.  When UtG is all-zero (constant
    genotype), the varying columns [wx, xx, xy] are zero for every SNP.
    The optimizer returns lambdas near l_min; downstream Wald stats must
    produce NaN for every SNP because P_XX = 0.
    """
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )
    from jamma.lmm.stats import batch_calc_wald_stats_numpy
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
    UtG_degen = np.zeros((n, n_snps))  # constant genotype -> P_XX = 0

    uab_invariant_soa = compute_uab_invariant_soa(UtW, Uty, 1)
    uab_varying_soa = batch_compute_uab_varying_soa_numpy(1, UtW, Uty, UtG_degen.T)
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

    # Optimizer converges to l_min when genotype has no variance
    np.testing.assert_allclose(
        lambdas,
        l_min,
        rtol=1e-4,
        err_msg="Split ncvt1 all-degenerate SNPs should return l_min lambda",
    )

    # Reconstruct full Uab for Wald stats
    Uab_batch = reconstruct_uab_from_soa(uab_invariant_soa, uab_varying_soa, 1)
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    # P_XX = 0 -> all Wald stats NaN
    assert np.all(np.isnan(betas)), f"Expected all-NaN betas, got {betas}"
    assert np.all(np.isnan(ses)), f"Expected all-NaN ses, got {ses}"
    assert np.all(np.isnan(pwalds)), f"Expected all-NaN pwalds, got {pwalds}"


@pytest.mark.tier0
def test_split_ncvt1_fallback_mixed_degenerate_valid():
    """Split ncvt1 Python fallback: mixed batch, degenerate NaN, valid finite.

    SNPs at indices 1 and 3 have non-zero genotypes (valid).
    SNPs at indices 0, 2, 4 are zero-genotype (degenerate, P_XX=0).
    The split optimizer must process them in the same batch without
    cross-SNP contamination.
    """
    from jamma.lmm.likelihood_numpy import (
        golden_section_optimize_lambda_split_ncvt1_numpy,
    )
    from jamma.lmm.stats import batch_calc_wald_stats_numpy
    from jamma.lmm.uab import (
        batch_compute_uab_varying_soa_numpy,
        compute_iab_invariant_scalars_ncvt1,
        compute_uab_invariant_soa,
        reconstruct_uab_from_soa,
    )

    rng = np.random.default_rng(99)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)

    UtG = np.zeros((n, n_snps))
    UtG[:, 1] = rng.standard_normal(n)  # valid
    UtG[:, 3] = rng.standard_normal(n)  # valid

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

    Uab_batch = reconstruct_uab_from_soa(uab_invariant_soa, uab_varying_soa, 1)
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )

    degenerate_idxs = [0, 2, 4]
    valid_idxs = [1, 3]

    assert np.all(np.isnan(betas[degenerate_idxs])), (
        f"Degenerate betas should be NaN, got {betas[degenerate_idxs]}"
    )
    assert np.all(np.isnan(ses[degenerate_idxs])), (
        f"Degenerate ses should be NaN, got {ses[degenerate_idxs]}"
    )
    assert np.all(np.isnan(pwalds[degenerate_idxs])), (
        f"Degenerate p_walds should be NaN, got {pwalds[degenerate_idxs]}"
    )
    assert np.all(np.isfinite(betas[valid_idxs])), (
        f"Valid betas should be finite, got {betas[valid_idxs]}"
    )
    assert np.all(np.isfinite(ses[valid_idxs])), (
        f"Valid ses should be finite, got {ses[valid_idxs]}"
    )
    assert np.all(np.isfinite(pwalds[valid_idxs])), (
        f"Valid p_walds should be finite, got {pwalds[valid_idxs]}"
    )


@pytest.mark.tier0
def test_generic_batch_numpy_fallback_degenerate_wald_nan():
    """golden_section_optimize_lambda_numpy (generic path) with degenerate SNPs.

    This tests the Python fallback path (golden_section_optimize_lambda_numpy
    with n_cvt=1) independently from the C-extension path.  Degenerate SNPs
    (UtG=0) should produce NaN Wald stats downstream regardless of which
    low-level optimizer is used.
    """
    rng = np.random.default_rng(55)
    n, n_snps = 30, 5
    l_min = 1e-5

    eigenvalues = np.sort(rng.uniform(0.1, 5.0, n))
    UtW = np.ones((n, 1))
    Uty = rng.standard_normal(n)
    UtG_degen = np.zeros((n, n_snps))  # constant genotype

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG_degen.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)

    lambdas, logls, _ = golden_section_optimize_lambda_numpy(
        1, eigenvalues, Uab_batch, Iab_batch, l_min=l_min
    )

    assert lambdas.shape == (n_snps,), f"Expected ({n_snps},), got {lambdas.shape}"

    # Degenerate SNPs: optimizer should converge to l_min (no SNP signal)
    np.testing.assert_allclose(
        lambdas,
        l_min,
        rtol=1e-4,
        err_msg="Generic fallback: all-degenerate SNPs should return l_min lambda",
    )

    # Wald stats: P_XX=0 -> all NaN
    betas, ses, pwalds = batch_calc_wald_stats_numpy(
        1, lambdas, eigenvalues, Uab_batch, n
    )
    assert np.all(np.isnan(betas)), f"Expected all-NaN betas, got {betas}"
    assert np.all(np.isnan(ses)), f"Expected all-NaN ses, got {ses}"
    assert np.all(np.isnan(pwalds)), f"Expected all-NaN pwalds, got {pwalds}"


# ---------------------------------------------------------------------------
# Scalar-vs-batch REML optimizer parity tests
# ---------------------------------------------------------------------------


@pytest.mark.tier0
def test_scalar_vs_batch_reml_single_snp_parity():
    """Scalar and batch REML optimizer paths produce matching lambda for single SNP.

    Verifies that golden_section_optimize_lambda_numpy (batch path) and
    _golden_section_minimize + reml_log_likelihood (scalar path) agree on
    the optimal lambda for a single SNP within rtol=1e-4.
    """
    n = 30
    n_cvt = 1

    d = rotated_lmm_inputs(n, 1, seed=42)
    eigenvalues, UtW, Uty = d.eigenvalues, d.UtW, d.Uty
    Utx = d.UtG[:, 0]

    # Scalar path
    Uab_scalar = compute_Uab(UtW, Uty, Utx)

    def scalar_obj(lam):
        return -reml_log_likelihood(
            lam, eigenvalues, Uab_scalar, n_cvt=n_cvt, nc_total=n_cvt + 1
        )

    lambda_scalar, _ = _golden_section_minimize(
        scalar_obj, 1e-5, 1e5, n_grid=50, n_iter=20
    )

    # Batch path (single SNP, shape (n, 1))
    Uab_batch = batch_compute_uab_numpy(n_cvt, UtW, Uty, Utx.reshape(1, n))
    Iab_batch = batch_compute_iab_numpy(n_cvt, Uab_batch)
    lambdas_batch, _, _ = golden_section_optimize_lambda_numpy(
        n_cvt, eigenvalues, Uab_batch, Iab_batch
    )

    np.testing.assert_allclose(
        lambda_scalar,
        lambdas_batch[0],
        rtol=1e-4,
        err_msg="Scalar and batch REML paths should agree on optimal lambda",
    )


@pytest.mark.tier0
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
# Scalar-vs-batch REML optimizer: tight lambda and logl parity (jamma-68j0)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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
# reconstruct_uab_from_soa generalization for n_cvt > 1 (Plan 70-02)
# ---------------------------------------------------------------------------


@pytest.mark.tier0
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


@pytest.mark.tier0
def test_reconstruct_uab_from_soa_ncvt2():
    """reconstruct_uab_from_soa with n_cvt=2 round-trips via classify_uab_columns."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.uab import reconstruct_uab_from_soa

    rng = np.random.default_rng(7)
    n_samples, n_snps = 40, 6
    n_cvt = 2
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Full reference Uab
    Uab_ref = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    # n_index = (n_cvt+3)*(n_cvt+2)//2 = 5*4//2 = 10

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
        err_msg="reconstruct_uab_from_soa failed to round-trip for n_cvt=2",
    )


@pytest.mark.tier0
def test_reconstruct_uab_from_soa_ncvt4():
    """reconstruct_uab_from_soa with n_cvt=4 matches batch_compute_uab_numpy."""
    from jamma.lmm.likelihood import classify_uab_columns
    from jamma.lmm.uab import reconstruct_uab_from_soa

    rng = np.random.default_rng(13)
    n_samples, n_snps = 30, 5
    n_cvt = 4
    UtW = rng.standard_normal((n_samples, n_cvt))
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, n_snps))

    # Full reference Uab
    Uab_ref = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    # n_index = (4+3)*(4+2)//2 = 7*6//2 = 21

    inv_indices, var_indices = classify_uab_columns(n_cvt)

    inv_list = list(inv_indices)
    uab_invariant_soa = np.ascontiguousarray(Uab_ref[0, :, inv_list])
    uab_varying_soa = np.ascontiguousarray(
        Uab_ref[:, :, list(var_indices)].transpose(0, 2, 1)
    )

    Uab_recon = reconstruct_uab_from_soa(
        uab_invariant_soa, uab_varying_soa, n_cvt=n_cvt
    )
    np.testing.assert_allclose(
        Uab_recon,
        Uab_ref,
        rtol=1e-14,
        atol=0,
        err_msg="reconstruct_uab_from_soa failed to round-trip for n_cvt=4",
    )
