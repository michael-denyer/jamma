"""Batch Wald and consistency contracts for NumPy likelihood code."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import brentq

from jamma.lmm.likelihood import _golden_section_minimize
from jamma.lmm.likelihood_numpy import (
    _batch_pab_at_lambda_numpy,
    _logdet_diag,
    _reml_logl,
    golden_section_optimize_lambda_numpy,
)
from jamma.lmm.pab import compute_Uab
from jamma.lmm.stats import batch_calc_wald_stats_from_pab_numpy
from jamma.lmm.uab import (
    _batch_compute_pab_varying_numpy,
    batch_compute_iab_numpy,
    batch_compute_uab_numpy,
)
from tests.builders import rotated_lmm_inputs
from tests.math_validation.dense_oracle import reml_score_log_lambda
from tests.reference.likelihood import reml_log_likelihood_alt

pytestmark = pytest.mark.tier0


def _wald_stats_from_lambdas(n_cvt, lambdas, eigenvalues, Uab_batch, n_samples):
    """Calculate Wald stats for tests that retain lambdas rather than Pab."""
    Hi_eval_batch = 1.0 / (lambdas[:, None] * eigenvalues[None, :] + 1.0)
    Pab_batch = _batch_compute_pab_varying_numpy(n_cvt, Hi_eval_batch, Uab_batch)
    return batch_calc_wald_stats_from_pab_numpy(n_cvt, Pab_batch, n_samples)


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

    Uab_batch = batch_compute_uab_numpy(1, UtW, Uty, UtG.T)
    Iab_batch = batch_compute_iab_numpy(1, Uab_batch)
    return eigenvalues, Uab_batch, Iab_batch, n_samples


def test_optimizer_returns_pab(wald_pab_data):
    """The optimizer returns (lambdas, logls, Pab) with Pab of the right shape."""
    from jamma.lmm.pab import build_index_table

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

    This is the all-SNPs-degenerate extreme: guard_p_yy produces NaN for
    every grid point, so safe_logls is all -inf.
    """
    from jamma.lmm.likelihood_numpy import _batch_golden_section_bracket_numpy

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

    logdet_iab = _logdet_diag(Iab_batch)

    def compute_batch_fn(log_lams):
        Pab, logdet_h = _batch_pab_at_lambda_numpy(
            1, np.exp(log_lams), eigenvalues, Uab_batch
        )
        return _reml_logl(Pab, logdet_h, logdet_iab, n - 1 - 1)

    lambdas_out = np.exp(
        _batch_golden_section_bracket_numpy(
            compute_batch_fn, grid_logls_all_nan, log_lambdas, n_iter=20
        )[0]
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
            return -reml_log_likelihood_alt(lam, eigenvalues, uab, n_cvt=n_cvt)

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
    """Batch refinement reaches the independent root and preserves likelihood.

    The generic scalar optimizer returns a golden-section midpoint. The REML
    batch optimizer also uses the analytic score, so lambda equality with that
    midpoint would require keeping its approximation error. Compare the refined
    lambda with a root of the independently implemented dense projector score.
    """
    n = 50
    n_cvt = 1

    d = rotated_lmm_inputs(n, 1, seed=123)
    eigenvalues, UtW, Uty = d.eigenvalues, d.UtW, d.Uty
    Utx = d.UtG[:, 0]

    # --- Scalar path ---
    Uab_scalar = compute_Uab(UtW, Uty, Utx)

    def scalar_neg_reml(lam: float) -> float:
        return -reml_log_likelihood_alt(lam, eigenvalues, Uab_scalar, n_cvt=n_cvt)

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

    def independent_score(log_lambda):
        return reml_score_log_lambda(
            np.diag(eigenvalues), UtW, Utx, Uty, np.exp(log_lambda)
        )

    root = brentq(
        independent_score,
        np.log(lambda_scalar) - 0.1,
        np.log(lambda_scalar) + 0.1,
        xtol=1e-13,
    )
    np.testing.assert_allclose(lambda_batch, np.exp(root), rtol=1e-10, atol=1e-14)
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
