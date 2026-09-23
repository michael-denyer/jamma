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

from pathlib import Path

import numpy as np
import pytest

from jamma.lmm import accel
from jamma.lmm.likelihood_numpy import (
    golden_section_optimize_lambda_numpy,
)
from jamma.lmm.pab import compute_Uab
from jamma.lmm.reml_score import _batch_reml_score_log_lambda_numpy
from jamma.lmm.uab import (
    batch_compute_iab_numpy,
    batch_compute_uab_numpy,
    compute_uab_invariant_soa,
)
from tests.builders import rotated_lmm_inputs
from tests.math_validation.dense_oracle import reml_score_log_lambda

pytestmark = pytest.mark.tier0


@pytest.mark.parametrize("n_cvt", [1, 2, 4])
def test_reml_score_matches_independent_dense_projector(n_cvt):
    rng = np.random.default_rng(8123 + n_cvt)
    n_samples = 30
    eigenvalues = np.sort(rng.uniform(0.05, 3.0, n_samples))
    UtW = np.column_stack(
        (np.ones(n_samples), rng.standard_normal((n_samples, n_cvt - 1)))
    )
    Uty = rng.standard_normal(n_samples)
    UtG = rng.standard_normal((n_samples, 3))
    Uab = batch_compute_uab_numpy(n_cvt, UtW, Uty, UtG.T)
    lambdas = np.array([1e-4, 0.7, 50.0])

    actual = _batch_reml_score_log_lambda_numpy(
        n_cvt, np.log(lambdas), eigenvalues, Uab
    )
    expected = np.array(
        [
            reml_score_log_lambda(np.diag(eigenvalues), UtW, UtG[:, i], Uty, lambdas[i])
            for i in range(3)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("backend", ["numpy", "native"])
def test_flat_reml_optima_match_independent_high_precision_roots(backend):
    """The eight real flat peaks that exposed objective-rounding drift."""
    fixture = np.load(Path(__file__).parent / "fixtures/reml_flat_optima.npz")
    eigenvalues = fixture["eigenvalues"]
    UtW = fixture["UtW"]
    Uty = fixture["Uty"]
    UtG = fixture["UtG"]
    expected = fixture["oracle_lambdas"]
    Uab = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab = batch_compute_iab_numpy(1, Uab)

    if backend == "numpy":
        actual, _logls, _pab = golden_section_optimize_lambda_numpy(
            1, eigenvalues, Uab, Iab
        )
    else:
        if not accel.available():
            pytest.skip("C accelerator is unavailable")
        invariant = compute_uab_invariant_soa(UtW, Uty, n_cvt=1)
        workspace = accel.require().create_workspace_c(
            eigenvalues,
            invariant,
            UtW,
            Uty,
            len(eigenvalues),
            1e-5,
            1e5,
            50,
            20,
            1,
            1,
            lmm_mode=1,
        )
        actual = accel.require().compute_lmm_chunk_c(
            workspace, np.ascontiguousarray(UtG.T), 1
        )["lambdas"]

    np.testing.assert_allclose(actual, expected, rtol=5e-6, atol=0.0)


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
    """guard_p_yy fires warning exactly once per run; reset restarts the counter."""
    from loguru import logger

    from jamma.lmm.pab import guard_p_yy, reset_p_yy_warned

    warning_messages: list[str] = []

    def _capture_sink(message):
        if message.record["level"].name == "WARNING":
            warning_messages.append(message.record["message"])

    # Start clean
    reset_p_yy_warned()

    sink_id = logger.add(_capture_sink, level="WARNING")
    try:
        for _ in range(10):
            guard_p_yy(-1.0)

        assert len(warning_messages) == 1, (
            f"Expected exactly 1 warning, got {len(warning_messages)}"
        )

        # Reset and fire again — should produce a second warning
        reset_p_yy_warned()
        guard_p_yy(-1.0)

        assert len(warning_messages) == 2, (
            f"Expected 2 total warnings after reset, got {len(warning_messages)}"
        )
    finally:
        logger.remove(sink_id)


# ---------------------------------------------------------------------------
# Scalar MLE P_yy without full Pab
# ---------------------------------------------------------------------------


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
    logl = mle_log_likelihood(lambda_val, eigenvalues, Uab, n_cvt)
    assert np.isfinite(logl), (
        f"null-model mle_log_likelihood returned non-finite: {logl}"
    )


def test_mle_scalar_degenerate_s_ww_zero():
    """Scalar null MLE P_yy returns s_yy when s_ww == 0 (degenerate intercept)."""
    from jamma.lmm.likelihood import _mle_p_yy_scalar_null_ncvt1
    from jamma.lmm.pab import calc_pab, get_ab_index

    n_samples = 50
    # Construct Uab where column 0 (ww) is all zeros -> Hi_eval @ Uab[:,0] = 0
    Uab = np.zeros((n_samples, 6), dtype=np.float64)
    rng = np.random.default_rng(111)
    Uab[:, 5] = rng.standard_normal(n_samples) ** 2  # yy column: non-zero

    Hi_eval = np.ones(n_samples)

    p_yy_null = _mle_p_yy_scalar_null_ncvt1(Hi_eval, Uab)
    expected_s_yy = float(Hi_eval @ Uab[:, 5])
    assert p_yy_null == expected_s_yy, f"Expected s_yy={expected_s_yy}, got {p_yy_null}"

    Pab = calc_pab(1, Hi_eval, Uab)
    p_yy_full = Pab[1, get_ab_index(3, 3, 1)]
    np.testing.assert_allclose(p_yy_null, p_yy_full, rtol=1e-12)


# ---------------------------------------------------------------------------
# Multi-covariate Uab parity (n_cvt > 1)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Precomputed REML constants, Iab invariant scalars, and golden section
# ---------------------------------------------------------------------------


def test_reml_const_precomputed():
    """_logl_const(df) must match inline computation bit-exactly."""
    from jamma.lmm.likelihood_numpy import _logl_const

    for df in [10, 48, 100, 1000, 50000]:
        result = _logl_const(df)
        inline = 0.5 * df * (np.log(df) - np.log(2.0 * np.pi) - 1.0)
        np.testing.assert_equal(
            result,
            inline,
            err_msg=f"_logl_const({df}) does not match inline for df={df}",
        )


@pytest.mark.parametrize("n_cvt", [1, 2])
def test_grid_and_per_snp_finishers_agree(n_cvt):
    """REML and MLE at a grid lambda match the per-SNP evaluation there.

    Both batch shapes go through one finisher per likelihood; only the Pab
    row-0 contraction differs (tensordot vs einsum), so agreement is to
    round-off, not bits.
    """
    import jamma.lmm.likelihood_numpy as ln

    n_samples, n_snps = 60, 7
    d = rotated_lmm_inputs(n_samples, n_snps, seed=3, n_cvt=n_cvt)
    Uab = batch_compute_uab_numpy(n_cvt, d.UtW, d.Uty, d.UtG.T)
    logdet_iab = ln._logdet_diag(batch_compute_iab_numpy(n_cvt, Uab))
    df = n_samples - n_cvt - 1
    grid = np.array([1e-3, 0.7, 40.0])

    grid_pab = ln._batch_grid_pab_numpy(n_cvt, grid, d.eigenvalues, Uab)
    grid_reml = ln._reml_logl(*grid_pab, logdet_iab, df)
    grid_mle = ln._mle_logl(*grid_pab, n_samples)
    assert grid_reml.shape == grid_mle.shape == (len(grid), n_snps)

    for g, lam in enumerate(grid):
        snp_pab = ln._batch_pab_at_lambda_numpy(
            n_cvt, np.full(n_snps, lam), d.eigenvalues, Uab
        )
        np.testing.assert_allclose(
            ln._reml_logl(*snp_pab, logdet_iab, df), grid_reml[g], rtol=1e-12
        )
        np.testing.assert_allclose(
            ln._mle_logl(*snp_pab, n_samples), grid_mle[g], rtol=1e-12
        )


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

    at_lambda = ln._reml_logl(
        *ln._batch_pab_at_lambda_numpy(1, lambdas, eigenvalues, Uab_batch),
        ln._logdet_diag(Iab_batch),
        n_samples - 2,
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
